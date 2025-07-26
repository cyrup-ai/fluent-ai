// ---
// src/completion.rs
// ---

#![allow(clippy::needless_doctest_main)] // suppress until examples are split

//! **LLM Agent – request assembly & streaming glue**
//!
//! This module wires the zero‑alloc [`Agent`](crate::agent::Agent) created by
//! [`AgentBuilder`](crate::agent::builder::AgentBuilder) to the existing
//! *provider‑agnostic* completion / chat traits (`Completion`, `Prompt`, …).
/// The implementation tries hard to keep the **hot path** free from heap
/// allocations and locks:
///
/// * static context & tools are stored as `ArrayVec` in `Agent`
/// * RAG and dynamic tool lookup run **before** the streaming cycle starts,
///   so no allocations happen while chunks are pushed through the bounded
///   ring powering [`AsyncStream`].
/// * concurrent fetches use `FuturesUnordered` which avoids intermediate `Vec`s
///   and allocates at most **once** for the task list.
use futures_util::{
    StreamExt, TryStreamExt,
    stream::{self, FuturesUnordered}};

use crate::{
    agent::Agent,
    client::completion::Chat,
    completion::{
        Completion, CompletionError, CompletionModelTrait, CompletionRequestBuilder, Document,
        Message, Prompt, PromptError},
    domain::tool::ToolSet,
    streaming::{StreamingChat, StreamingCompletion, StreamingCompletionResponse, StreamingPrompt},
    vector_store::{VectorStoreError, VectorStoreIndexDyn}};

// -------------------------------------------------------------------------
// Completion impl
// -------------------------------------------------------------------------

impl<M: CompletionModelTrait> Completion<M> for Agent<M> {
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

        // ---------------------------------------------------------
        // 1. Prepare the base request (static artefacts only)
        // ---------------------------------------------------------

        let mut req = self
            .model
            .completion_request(prompt.clone())
            .preamble(self.preamble.clone())
            .messages(chat_history.clone())
            .temperature_opt(self.temperature)
            .max_tokens_opt(self.max_tokens)
            .additional_params_opt(self.additional_params.clone())
            .documents(self.static_context.clone());

        // ---------------------------------------------------------
        // 2. Determine whether RAG is needed
        // ---------------------------------------------------------

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

        let rag_seed = match rag_seed {
            Some(seed) => seed,
            None => {
                return Err(CompletionError::RequestError(Box::new(
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, "Missing RAG seed"),
                )));
            }
        };

        // ---------------------------------------------------------
        // 3. Dynamic context (vector stores)
        // ---------------------------------------------------------

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
                                    additional_props: HashMap::new()})
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

        // ---------------------------------------------------------
        // 4. Dynamic & static tools
        // ---------------------------------------------------------

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

        // ---------------------------------------------------------
        // 5. Return the fully‑specified request builder
        // ---------------------------------------------------------
        req = req
            .documents(dyn_ctx)
            .tools([static_defs, dyn_tool_defs].concat());

        Ok(req)
    }
}

// -------------------------------------------------------------------------
// Prompt / Chat trait impls
// -------------------------------------------------------------------------

#[allow(refining_impl_trait)]
impl<M: CompletionModelTrait> Prompt for Agent<M> {
    type PromptedBuilder = PromptRequest<'static, M>;

    fn prompt(self, prompt: impl ToString) -> Result<Self::PromptedBuilder, PromptError> {
        // Create a PromptRequest with proper lifetime management
        let prompt_text = prompt.to_string();
        let prompt_request = PromptRequest {
            agent: self,
            prompt: prompt_text,
            _phantom: std::marker::PhantomData};
        Ok(prompt_request)
    }
}

#[allow(refining_impl_trait)]
impl<M: CompletionModelTrait> Prompt for &Agent<M> {
    type PromptedBuilder = PromptRequest<'static, M>;

    fn prompt(self, prompt: impl ToString) -> Result<Self::PromptedBuilder, PromptError> {
        Ok(PromptRequest::new(self, prompt.to_string()))
    }
}

#[allow(refining_impl_trait)]
impl<M: CompletionModelTrait> Chat for Agent<M> {
    fn chat(
        &self,
        prompt: impl Into<Message> + Send,
        mut chat_history: Vec<Message>,
    ) -> AsyncStream<ChatMessageChunk> {
        let prompt_request = PromptRequest::new(self, prompt).with_history(&mut chat_history);

        AsyncStream::with_channel(move |sender| {
            // Use the actual drive implementation directly
            let agent = prompt_request.agent;
            let mut local_hist = Vec::new();
            let hist = chat_history;
            let mut depth = 0usize;
            let mut prompt = prompt_request.prompt.clone();
            
            loop {
                depth += 1;
                // Build provider request (static + dyn context/tools)
                let completion_task = agent.completion(prompt.clone(), hist.clone());
                
                // Execute completion and get response
                let resp = match completion_task.and_then(|builder| builder.send()) {
                    Ok(response) => response,
                    Err(_) => break,
                };
                
                // Check for plain-text reply
                if let Some(text) = resp
                    .choice
                    .iter()
                    .filter_map(|c| c.as_text())
                    .map(|t| t.text.clone())
                    .reduce(|a, b| a + "\n" + &b)
                {
                    let chunk = ChatMessageChunk { text, done: true };
                    let _ = sender.send(chunk);
                    break;
                }
                
                // Handle tool calls
                prompt = match agent.tools.handle_tool_calls(&resp, &hist) {
                    Ok(new_prompt) => new_prompt,
                    Err(_) => break,
                };
                
                if depth > prompt_request.max_depth {
                    break;
                }
            }
        })
    }
}

// -------------------------------------------------------------------------
// Streaming glue
// -------------------------------------------------------------------------

impl<M: CompletionModelTrait> StreamingCompletion<M> for Agent<M> {
    fn stream_completion(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> crate::runtime::AsyncTask<Result<CompletionRequestBuilder<M>, CompletionError>> {
        let agent = self.clone();
        crate::runtime::spawn_async(async move { agent.completion(prompt, chat_history).await })
    }
}

impl<M: CompletionModelTrait> StreamingPrompt<M::StreamingResponse> for Agent<M> {
    fn stream_prompt(
        &self,
        prompt: impl Into<Message> + Send,
    ) -> crate::runtime::AsyncTask<
        Result<StreamingCompletionResponse<M::StreamingResponse>, CompletionError>,
    > {
        let agent = self.clone();
        crate::runtime::spawn_async(async move { agent.stream_chat(prompt, Vec::new()).await })
    }
}

impl<M: CompletionModelTrait> StreamingChat<M::StreamingResponse> for Agent<M> {
    fn stream_chat(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> crate::runtime::AsyncTask<
        Result<StreamingCompletionResponse<M::StreamingResponse>, CompletionError>,
    > {
        let agent = self.clone();
        crate::runtime::spawn_async(async move {
            agent
                .stream_completion(prompt, chat_history)
                .await?
                .stream()
                .await
        })
    }
}
