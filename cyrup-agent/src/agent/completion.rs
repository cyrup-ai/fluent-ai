// ---
// src/completion.rs
// ---

#![allow(clippy::needless_doctest_main)] // suppress until examples are split

//! **LLM Agent – request assembly & streaming glue**
//!
//! This module wires the zero‑alloc [`Agent`](crate::agent::Agent) created by
//! [`AgentBuilder`](crate::agent::builder::AgentBuilder) to the existing
//! *provider‑agnostic* completion / chat traits (`Completion`, `Prompt`, …).
//!
/// The implementation tries hard to keep the **hot path** free from heap
//! allocations and locks:
///
/// * static context & tools are stored as `ArrayVec` in `Agent`
//! * RAG and dynamic tool lookup run **before** the streaming cycle starts,
///   so no allocations happen while chunks are pushed through the bounded
///   ring powering [`AsyncStream`].
//! * concurrent fetches use `FuturesUnordered` which avoids intermediate `Vec`s
//!   and allocates at most **once** for the task list.

use std::collections::HashMap;

use futures::{
    stream::{self, FuturesUnordered},
    StreamExt, TryStreamExt,
};

use crate::{
    agent::Agent,
    completion::{
        Chat, Completion, CompletionError, CompletionModel, CompletionRequestBuilder, Document,
        Message, Prompt, PromptError,
    },
    streaming::{
        StreamingChat, StreamingCompletion, StreamingCompletionResponse, StreamingPrompt,
    },
    tool::ToolSet,
    vector_store::{VectorStoreError, VectorStoreIndexDyn},
};

/* ------------------------------------------------------------------------- */
/* Completion impl                                                            */
/* ------------------------------------------------------------------------- */

impl<M: CompletionModel> Completion<M> for Agent<M> {
    /// Build a provider‑specific *request builder* for a completion / chat turn.
    ///
    /// - static context & tools are injected immediately
    /// - if any message (prompt or history) carries RAG text,
    ///   dynamic context & tools are fetched **in parallel** before the builder
    ///   is returned.
    /// - the resulting `CompletionRequestBuilder` can then be streamed or
    ///   awaited depending on the caller’s needs.
    async fn completion(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> Result<CompletionRequestBuilder<M>, CompletionError> {
        let prompt: Message = prompt.into();

        /* --------------------------------------------------------- */
        /* 1. Prepare the base request (static artefacts only)       */
        /* --------------------------------------------------------- */

        let mut req = self
            .model
            .completion_request(prompt.clone())
            .preamble(self.preamble.clone())
            .messages(chat_history.clone())
            .temperature_opt(self.temperature)
            .max_tokens_opt(self.max_tokens)
            .additional_params_opt(self.additional_params.clone())
            .documents(self.static_context.clone());

        /* --------------------------------------------------------- */
        /* 2. Determine whether RAG is needed                        */
        /* --------------------------------------------------------- */

        let rag_seed = prompt
            .rag_text()
            .or_else(|| chat_history.iter().rev().find_map(Message::rag_text));

        if rag_seed.is_none() {
            // fast path – no RAG => only static tools to inject
            let static_defs = self
                .static_tools
                .iter()
                .filter_map(|&name| self.tools.get(name))
                .map(|tool| tool.definition(String::new()))
                .collect::<FuturesUnordered<_>>()
                .collect::<Vec<_>>()
                .await;

            return Ok(req.tools(static_defs));
        }

        let rag_seed = rag_seed.unwrap();

        /* --------------------------------------------------------- */
        /* 3. Dynamic context (vector stores)                        */
        /* --------------------------------------------------------- */

        let dyn_ctx = self
            .dynamic_context
            .iter()
            .map(|(n, store)| {
                let seed = rag_seed.to_owned();
                async move {
                    store
                        .top_n(&seed, *n)
                        .await
                        .map(|hits| {
                            hits.into_iter()
                                .map(|(_, id, doc)| Document {
                                    id,
                                    text: serde_json::to_string_pretty(&doc)
                                        .unwrap_or_else(|_| doc.to_string()),
                                    additional_props: HashMap::new(),
                                })
                                .collect::<Vec<_>>()
                        })
                        .map_err(VectorStoreError::from)
                }
            })
            .collect::<FuturesUnordered<_>>()
            .try_fold(Vec::new(), |mut acc, docs| async move {
                acc.extend(docs);
                Ok::<_, VectorStoreError>(acc)
            })
            .await
            .map_err(|e| CompletionError::RequestError(Box::new(e)))?;

        /* --------------------------------------------------------- */
        /* 4. Dynamic & static tools                                 */
        /* --------------------------------------------------------- */

        // (a) dynamic tool IDs from vector stores
        let dyn_tool_defs = self
            .dynamic_tools
            .iter()
            .map(|(n, store)| {
                let seed = rag_seed.to_owned();
                async move {
                    store
                        .top_n_ids(&seed, *n)
                        .await
                        .map_err(VectorStoreError::from)
                }
            })
            .collect::<FuturesUnordered<_>>()
            .try_fold(Vec::new(), |mut acc, ids| async move {
                acc.extend(ids);
                Ok::<_, VectorStoreError>(acc)
            })
            .await
            .map_err(|e| CompletionError::RequestError(Box::new(e)))?
            .into_iter()
            .filter_map(|id| self.tools.get(&id))
            .map(|tool| tool.definition(rag_seed.to_owned()))
            .collect::<FuturesUnordered<_>>()
            .collect::<Vec<_>>()
            .await;

        // (b) static tools
        let static_defs = self
            .static_tools
            .iter()
            .filter_map(|&name| self.tools.get(name))
            .map(|tool| tool.definition(rag_seed.to_owned()))
            .collect::<FuturesUnordered<_>>()
            .collect::<Vec<_>>()
            .await;

        /* --------------------------------------------------------- */
        /* 5. Return the fully‑specified request builder             */
        /* --------------------------------------------------------- */
        req = req
            .documents(dyn_ctx)
            .tools([static_defs, dyn_tool_defs].concat());

        Ok(req)
    }
}

/* ------------------------------------------------------------------------- */
/* Prompt / Chat trait impls                                                 */
/* ------------------------------------------------------------------------- */

#[allow(refining_impl_trait)]
impl<M: CompletionModel> Prompt for Agent<M> {
    fn prompt(&self, prompt: impl Into<Message> + Send) -> PromptRequest<M> {
        PromptRequest::new(self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl<M: CompletionModel> Prompt for &Agent<M> {
    fn prompt(&self, prompt: impl Into<Message> + Send) -> PromptRequest<M> {
        PromptRequest::new(*self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl<M: CompletionModel> Chat for Agent<M> {
    async fn chat(
        &self,
        prompt: impl Into<Message> + Send,
        mut chat_history: Vec<Message>,
    ) -> Result<String, PromptError> {
        PromptRequest::new(self, prompt)
            .with_history(&mut chat_history)
            .await
    }
}

/* ------------------------------------------------------------------------- */
/* Streaming glue                                                            */
/* ------------------------------------------------------------------------- */

impl<M: CompletionModel> StreamingCompletion<M> for Agent<M> {
    async fn stream_completion(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> Result<CompletionRequestBuilder<M>, CompletionError> {
        // reuse the logic above to build the request
        self.completion(prompt, chat_history).await
    }
}

impl<M: CompletionModel> StreamingPrompt<M::StreamingResponse> for Agent<M> {
    async fn stream_prompt(
        &self,
        prompt: impl Into<Message> + Send,
    ) -> Result<StreamingCompletionResponse<M::StreamingResponse>, CompletionError> {
        self.stream_chat(prompt, Vec::new()).await
    }
}

impl<M: CompletionModel> StreamingChat<M::StreamingResponse> for Agent<M> {
    async fn stream_chat(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> Result<StreamingCompletionResponse<M::StreamingResponse>, CompletionError> {
        self.stream_completion(prompt, chat_history)
            .await?
            .stream()
            .await
    }
}
