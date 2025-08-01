//! This file is part of the fluent-ai-rs project. It provides a client
//! for interacting with completion models, including streaming and non-streaming
//! responses. The design emphasizes a zero-allocation, streams-only architecture.

#![allow(clippy::type_complexity)]

use std::sync::Arc;

use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    OneOrMany,
    client::ProviderClient,
    completion::message::{AssistantContent, Message, ToolCall},
    runtime::AsyncTask,
    streaming::{StreamingCompletionResponse, StreamingResult}};

// ================================================================
// Error types
// ================================================================

#[derive(Debug, Error)]
pub enum CompletionError {
    #[error("http error: {0}")]
    Http(#[from] fluent_ai_http3::HttpError),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("request construction: {0}")]
    Request(Box<dyn std::error::Error + Send + Sync>),
    #[error("provider returned error: {0}")]
    Provider(String),
    #[error("response parsing error: {0}")]
    Response(String)}

// ================================================================
// Core traits and types
// ================================================================

pub trait CompletionModel: Sized + Send + Sync + 'static {
    type Response: Clone + Send + Sync + 'static;
    type StreamingResponse: Clone + Send + Sync + Unpin + 'static;

    fn completion(
        &self,
        req: CompletionRequest,
    ) -> AsyncTask<Result<CompletionResponse<Self::Response>, CompletionError>>;

    fn stream(
        &self,
        req: CompletionRequest,
    ) -> AsyncTask<Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>>;
}

pub trait CompletionClient: ProviderClient + Clone + Send + Sync + 'static {
    type Model: CompletionModel;
    fn completion_model(&self, model: &str) -> Self::Model;
}

// ================================================================
// Request and Response structures
// ================================================================

#[derive(Clone, Debug, Default, Serialize)]
pub struct CompletionRequest {
    pub preamble: Option<Message>,
    pub chat_history: OneOrMany<Message>,
    pub documents: OneOrMany<Message>,
    pub tools: OneOrMany<ToolCall>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub additional_params: serde_json::Value}

#[derive(Clone, Debug, Deserialize)]
pub struct CompletionResponse<R: Clone> {
    pub choice: OneOrMany<AssistantContent>,
    pub response: R}

// ================================================================
// Fluent Request Builder
// ================================================================

#[derive(Clone)]
pub struct CompletionRequestBuilder<'a, M: CompletionModel> {
    model: &'a M,
    preamble: Option<Message>,
    chat_history: OneOrMany<Message>,
    prompt: Message,
    documents: OneOrMany<Message>,
    tools: OneOrMany<ToolCall>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    additional_params: serde_json::Value}

impl<'a, M: CompletionModel> CompletionRequestBuilder<'a, M> {
    pub fn new(model: &'a M, prompt: impl Into<Message>) -> Self {
        Self {
            model,
            preamble: None,
            chat_history: OneOrMany::None,
            prompt: prompt.into(),
            documents: OneOrMany::None,
            tools: OneOrMany::None,
            temperature: None,
            max_tokens: None,
            additional_params: serde_json::Value::Null}
    }

    pub fn build(self) -> CompletionRequest {
        CompletionRequest {
            preamble: self.preamble,
            chat_history: self.chat_history.with_pushed(self.prompt),
            documents: self.documents,
            tools: self.tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            additional_params: self.additional_params}
    }

    pub fn send(self) -> AsyncTask<Result<CompletionResponse<M::Response>, CompletionError>> {
        let req = self.build();
        self.model.completion(req)
    }

    pub fn stream(
        self,
    ) -> AsyncTask<Result<StreamingCompletionResponse<M::StreamingResponse>, CompletionError>> {
        let req = self.build();
        self.model.stream(req)
    }
}

// ================================================================
// Blanket trait implementations
// ================================================================

impl<M: CompletionModel> CompletionModel for &M {
    type Response = M::Response;
    type StreamingResponse = M::StreamingResponse;

    fn completion(
        &self,
        req: CompletionRequest,
    ) -> AsyncTask<Result<CompletionResponse<Self::Response>, CompletionError>> {
        (*self).completion(req)
    }

    fn stream(
        &self,
        req: CompletionRequest,
    ) -> AsyncTask<Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>>
    {
        // This is the key change: directly pass through the stream from the underlying model.
        // No more boxing, no more `AsyncStreamDyn`.
        (*self).stream(req)
    }
}
