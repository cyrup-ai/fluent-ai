use std::fmt;
use std::marker::PhantomData;
use fluent_ai_async::AsyncStream;
use serde::de::DeserializeOwned;
// Removed unused import: use tokio_stream::StreamExt;

use super::error::{ExtractionError, _ExtractionResult as ExtractionResult};
use crate::domain::agent::types::CandleAgent as Agent;
use crate::domain::chat::message::types::CandleMessageRole as MessageRole;
use crate::domain::completion::CandleCompletionRequest as CompletionRequest;
use crate::domain::completion::model::CandleCompletionModel as CompletionModel;
use crate::domain::completion::chunk::{CandleCompletionChunk as CompletionChunk, CandleFinishReason as FinishReason};
use crate::prompt::CandlePrompt as Prompt;

/// Trait defining the core extraction interface
pub trait Extractor<T>: Send + Sync + fmt::Debug + Clone
where
    T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + Default + 'static,
{
    /// Get the agent used for extraction
    fn agent(&self) -> &Agent;

    /// Get the system prompt for extraction
    fn system_prompt(&self) -> Option<&str>;

    /// Extract structured data from text with comprehensive error handling
    fn extract_from(&self, text: &str) -> AsyncStream<T>;

    /// Create new extractor with agent
    fn new(agent: Agent) -> Self;

    /// Set system prompt for extraction guidance
    fn with_system_prompt(self, prompt: impl Into<String>) -> Self;
}

/// Implementation of the Extractor trait
#[derive(Debug, Clone)]
pub struct ExtractorImpl<T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static> {
    agent: Agent,
    system_prompt: Option<String>,
    _marker: PhantomData<T>}

impl<T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + Default + 'static> Extractor<T>
    for ExtractorImpl<T>
{
    fn agent(&self) -> &Agent {
        &self.agent
    }

    fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    fn new(agent: Agent) -> Self {
        Self {
            agent,
            system_prompt: None,
            _marker: PhantomData}
    }

    fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    fn extract_from(&self, text: &str) -> AsyncStream<T> {
        let _text = text.to_string();

        AsyncStream::with_channel(move |sender| {
            // TODO: Connect to execute_extraction method
            // For now, send default result to maintain compilation
            let default_result = T::default();
            let _ = sender.send(default_result);
        })
    }
}

impl<T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + Default + 'static> ExtractorImpl<T> {
    /// Execute extraction with agent (planned feature)
    pub async fn execute_extraction(
        agent: Agent,
        completion_request: CompletionRequest,
        _text_input: String,
    ) -> ExtractionResult<T> {
        let model = AgentCompletionModel::new(agent);
        let prompt = Prompt {
            content: completion_request.system.as_deref().unwrap_or("").to_string(),
            role: MessageRole::System};
        let params = CompletionParams {
            temperature: completion_request.temperature.unwrap_or(0.2),
            max_tokens: completion_request.max_tokens.and_then(|t| std::num::NonZeroU64::new(t as u64)),
            n: std::num::NonZeroU8::new(1).unwrap(),
            stream: true};
        let mut stream = model.prompt(prompt, &params);

        let mut full_response = String::new();
        let mut finish_reason = None;

        while let Some(chunk) = stream.try_next() {
            match chunk {
                CompletionChunk::Text(text) => {
                    full_response.push_str(&text);
                }
                CompletionChunk::Complete {
                    text,
                    finish_reason: reason,
                    ..
                } => {
                    full_response.push_str(&text);
                    finish_reason = reason;
                    break;
                }
                CompletionChunk::Error(error) => {
                    return Err(ExtractionError::CompletionError(format!(
                        "Completion error: {}",
                        error
                    )));
                }
                _ => {
                    // Handle other chunk types (tool calls, etc.) if needed
                }
            }
        }

        if finish_reason == Some(FinishReason::Stop) || !full_response.is_empty() {
            match Self::parse_json_response(&full_response) {
                Ok(result) => Ok(result),
                Err(e) => Err(e)}
        } else {
            Err(ExtractionError::CompletionError(
                "No valid response from model".to_string(),
            ))
        }
    }

    /// Parse JSON response (planned feature)
    pub fn parse_json_response(response: &str) -> ExtractionResult<T> {
        // First try to parse the whole response as JSON
        if let Ok(parsed) = serde_json::from_str::<T>(response) {
            return Ok(parsed);
        }

        // If that fails, try to find JSON in the response
        let json_start = response.find('{').unwrap_or(0);
        let json_end = response
            .rfind('}')
            .map(|i| i + 1)
            .unwrap_or_else(|| response.len());

        if json_start < json_end {
            let json_str = &response[json_start..json_end];
            serde_json::from_str(json_str).map_err(ExtractionError::from)
        } else {
            Err(ExtractionError::InvalidFormat {
                actual: response.to_string()})
        }
    }
}

/// Zero-allocation completion model wrapper for agents
#[derive(Debug, Clone)]
pub struct AgentCompletionModel {
    agent: Agent}

impl AgentCompletionModel {
    /// Create new completion model from agent
    pub fn new(agent: Agent) -> Self {
        Self { agent }
    }
}

impl CompletionModel for AgentCompletionModel {
    fn prompt<'a>(
        &'a self,
        prompt: Prompt,
        params: &'a CompletionParams,
    ) -> fluent_ai_async::AsyncStream<CompletionChunk> {
        let _agent = self.agent.clone();
        let _params = params.clone();

        AsyncStream::with_channel(move |sender| {
            // TODO: Replace with proper streams-only completion
            // For now, send default chunk to maintain compilation
            let default_chunk = CompletionChunk::Complete {
                text: format!("{:?}", prompt),
                finish_reason: Some(FinishReason::Stop),
                usage: None};
            let _ = sender.try_send(default_chunk);
        })
    }
}


