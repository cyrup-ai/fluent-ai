// ---
// src/completion.rs
// ---

#![allow(clippy::needless_doctest_main)] // suppress until examples are split

//! **LLM Agent – request assembly & streaming glue**
//!
//! This module wires the zero‑alloc [`Agent`](crate::agent::Agent) created by
//! [`AgentBuilder`](crate::agent::builder::AgentBuilder) to the existing
//! *provider‑agnostic* completion / chat traits (`Completion`, `Prompt`, …).
/// The implementation tries hard to keep the **hot path** free from heap
/// allocations and locks:
///
/// * static context & tools are stored as `ArrayVec` in `Agent`
/// * RAG and dynamic tool lookup run **before** the streaming cycle starts,
///   so no allocations happen while chunks are pushed through the bounded
///   ring powering [`AsyncStream`].
/// * NO FUTURES - all operations use AsyncStream patterns exclusively

use fluent_ai_async::AsyncStream;
use hashbrown::HashMap;

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
    ///   dynamic context & tools are fetched using streams before the builder
    ///   is returned.
    /// - the resulting `CompletionRequestBuilder` uses streaming patterns
    fn completion(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> AsyncStream<CompletionRequestBuilder<M>> {
        let prompt: Message = prompt.into();
        let agent = self.clone();
        
        AsyncStream::with_channel(move |sender| {
            // ---------------------------------------------------------
            // 1. Prepare the base request (static artefacts only)
            // ---------------------------------------------------------

            let mut req = agent
                .model
                .completion_request(prompt.clone())
                .preamble(agent.preamble.clone())
                .messages(chat_history.clone())
                .temperature_opt(agent.temperature)
                .max_tokens_opt(agent.max_tokens)
                .additional_params_opt(agent.additional_params.clone())
                .documents(agent.static_context.clone());

            // ---------------------------------------------------------
            // 2. Determine whether RAG is needed
            // ---------------------------------------------------------

            let rag_seed = prompt
                .rag_text()
                .or_else(|| chat_history.iter().rev().find_map(Message::rag_text));

            if rag_seed.is_none() {
                // fast path – no RAG => only static tools to inject
                let mut static_defs = Vec::new();
                for &name in &agent.static_tools {
                    if let Some(tool) = agent.tools.get(name) {
                        let mut def_stream = tool.definition(String::new());
                        if let Some(def) = def_stream.try_next() {
                            static_defs.push(def);
                        }
                    }
                }

                let final_req = req.tools(static_defs);
                let _ = sender.send(final_req);
                return;
            }

            let rag_seed = rag_seed.unwrap();

            // ---------------------------------------------------------
            // 3. Dynamic context (vector stores)
            // ---------------------------------------------------------

            let mut dyn_ctx = Vec::new();
            for (n, store) in &agent.dynamic_context {
                let mut hits_stream = store.top_n(&rag_seed, *n);
                if let Some(hits) = hits_stream.try_next() {
                    let docs = hits.into_iter()
                        .map(|(_, id, doc)| Document {
                            id,
                            text: serde_json::to_string_pretty(&doc)
                                .unwrap_or_else(|_| doc.to_string()),
                            additional_props: HashMap::new()})
                        .collect::<Vec<_>>();
                    dyn_ctx.extend(docs);
                }
            }

            // ---------------------------------------------------------
            // 4. Dynamic & static tools
            // ---------------------------------------------------------

            // (a) dynamic tool IDs from vector stores
            let mut dyn_tool_ids = Vec::new();
            for (n, store) in &agent.dynamic_tools {
                let mut ids_stream = store.top_n_ids(&rag_seed, *n);
                if let Some(ids) = ids_stream.try_next() {
                    dyn_tool_ids.extend(ids);
                }
            }
            
            let mut dyn_tool_defs = Vec::new();
            for id in dyn_tool_ids {
                if let Some(tool) = agent.tools.get(&id) {
                    let mut def_stream = tool.definition(rag_seed.to_owned());
                    if let Some(def) = def_stream.try_next() {
                        dyn_tool_defs.push(def);
                    }
                }
            }

            // (b) static tools
            let mut static_defs = Vec::new();
            for &name in &agent.static_tools {
                if let Some(tool) = agent.tools.get(name) {
                    let mut def_stream = tool.definition(rag_seed.to_owned());
                    if let Some(def) = def_stream.try_next() {
                        static_defs.push(def);
                    }
                }
            }

            // ---------------------------------------------------------
            // 5. Return the fully‑specified request builder
            // ---------------------------------------------------------
            req = req
                .documents(dyn_ctx)
                .tools([static_defs, dyn_tool_defs].concat());

            let _ = sender.send(req);
        })
    }
}

// -------------------------------------------------------------------------
// Prompt / Chat trait impls
// -------------------------------------------------------------------------

use super::prompt::PromptRequest;

#[allow(refining_impl_trait)]
impl<M: CompletionModelTrait> Prompt for Agent<M> {
    type PromptedBuilder = PromptRequest<'static, M>;

    fn prompt(self, prompt: impl ToString) -> Result<Self::PromptedBuilder, PromptError> {
        // Create a PromptRequest with proper lifetime management
        let prompt_text = prompt.to_string();
        let prompt_request = PromptRequest {
            agent: self,
            prompt: Message::user(prompt_text),
            chat_hist: None,
            max_depth: 0,
        };
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

/// Chat message chunk for streaming responses
#[derive(Debug, Clone)]
pub struct ChatMessageChunk {
    pub text: String,
    pub done: bool,
}

#[allow(refining_impl_trait)]
impl<M: CompletionModelTrait> Chat for Agent<M> {
    fn chat(
        &self,
        prompt: impl Into<Message> + Send,
        mut chat_history: Vec<Message>,
    ) -> AsyncStream<ChatMessageChunk> {
        let prompt_msg = prompt.into();
        let agent = self.clone();
        
        AsyncStream::with_channel(move |sender| {
            let mut depth = 0usize;
            let mut current_prompt = prompt_msg;
            
            loop {
                depth += 1;
                
                // Build provider request (static + dyn context/tools)
                let mut completion_stream = agent.completion(current_prompt.clone(), chat_history.clone());
                
                if let Some(builder) = completion_stream.try_next() {
                    let mut response_stream = builder.send();
                    
                    if let Some(resp) = response_stream.try_next() {
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
                        let mut tool_stream = agent.tools.handle_tool_calls(&resp, &chat_history);
                        if let Some(new_prompt) = tool_stream.try_next() {
                            current_prompt = new_prompt;
                        } else {
                            break;
                        }
                        
                        if depth > 10 { // max depth
                            break;
                        }
                    } else {
                        break;
                    }
                } else {
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
    ) -> AsyncStream<CompletionRequestBuilder<M>> {
        self.completion(prompt, chat_history)
    }
}

impl<M: CompletionModelTrait> StreamingPrompt<M::StreamingResponse> for Agent<M> {
    fn stream_prompt(
        &self,
        prompt: impl Into<Message> + Send,
    ) -> AsyncStream<StreamingCompletionResponse<M::StreamingResponse>> {
        self.stream_chat(prompt, Vec::new())
    }
}

impl<M: CompletionModelTrait> StreamingChat<M::StreamingResponse> for Agent<M> {
    fn stream_chat(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> AsyncStream<StreamingCompletionResponse<M::StreamingResponse>> {
        let agent = self.clone();
        let prompt_msg = prompt.into();
        
        AsyncStream::with_channel(move |sender| {
            let mut completion_stream = agent.stream_completion(prompt_msg, chat_history);
            
            if let Some(builder) = completion_stream.try_next() {
                let mut stream_resp = builder.stream();
                if let Some(response) = stream_resp.try_next() {
                    let _ = sender.send(response);
                }
            }
        })
    }
}