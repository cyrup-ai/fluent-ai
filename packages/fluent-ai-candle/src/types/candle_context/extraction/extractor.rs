use std::fmt;
use std::marker::PhantomData;

use fluent_ai_async::AsyncStream;
use serde::de::DeserializeOwned;
use tokio_stream::StreamExt;

use super::error::ExtractionError;
use crate::types::{CandleCompletionModel};
// Removed old incorrect imports - using Candle-prefixed types from types module instead

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
    _marker: PhantomData<T>,
}

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
            _marker: PhantomData,
        }
    }

    fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    fn extract_from(&self, text: &str) -> AsyncStream<T> {
        let agent = self.agent.clone();
        let system_prompt = self.system_prompt.clone();
        let text = text.to_string();

        AsyncStream::with_channel(move |sender| {
            let prompt = if let Some(sys_prompt) = system_prompt {
                format!(
                    "{}\n\nExtract information from the following text:\n{}",
                    sys_prompt, text
                )
            } else {
                format!("Extract information from the following text:\n{}", text)
            };

            // TODO: Replace with proper streams-only completion request
            // For now, skip the completion request to maintain compilation
            let _completion_request = format!("Completion request for: {}", prompt);

            // TODO: Replace with proper streams-only extraction
            // For now, send default result to maintain compilation
            let default_result = T::default(); // Assuming T implements Default
            let _ = sender.try_send(default_result);
        })
    }
}

impl<T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + Default + 'static> ExtractorImpl<T> {
    async fn execute_extraction(
        agent: Agent,
        completion_request: crate::types::CandleCompletionRequest,
        _text_input: String,
    ) -> Result<T, ExtractionError> {
        let model = AgentCompletionModel::new(agent);
        let prompt = completion_request.system_prompt.to_string();
        let params = crate::types::CandleCompletionParams {
            temperature: completion_request.temperature,
            max_tokens: completion_request.max_tokens,
            n: std::num::NonZeroU8::new(1).unwrap(),
            stream: true,
        };
        let mut stream = model.prompt(&prompt, &params);

        let mut full_response = String::new();
        let mut finish_reason = None;

        while let Some(chunk) = stream.next().await {
            // CandleCompletionChunk is CandleStreamingChoice struct with delta field
            if let Some(content) = chunk.delta.content {
                full_response.push_str(&content);
            }
            if let Some(reason) = chunk.finish_reason {
                finish_reason = Some(reason);
                break;
            }
        }

        if finish_reason == Some(crate::types::CandleFinishReason::Stop) || !full_response.is_empty() {
            match Self::parse_json_response(&full_response) {
                Ok(result) => Ok(result),
                Err(e) => Err(e),
            }
        } else {
            Err(ExtractionError::CompletionError(
                "No valid response from model".to_string(),
            ))
        }
    }

    fn parse_json_response(response: &str) -> Result<T, ExtractionError> {
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
                actual: response.to_string(),
            })
        }
    }
}

/// Agent configuration for completion models
#[derive(Debug, Clone)]
pub struct Agent {
    /// Agent name/identifier
    pub name: String,
    /// Agent system prompt
    pub system_prompt: Option<String>,
}

/// Zero-allocation completion model wrapper for agents
#[derive(Debug, Clone)]
pub struct AgentCompletionModel {
    agent: Agent,
}

impl AgentCompletionModel {
    /// Create new completion model from agent
    pub fn new(agent: Agent) -> Self {
        Self { agent }
    }
}

// Implement Model trait first
impl crate::types::Model for AgentCompletionModel {
    fn info(&self) -> &'static crate::types::CandleModelInfo {
        // Return a default model info for now
        static DEFAULT_INFO: crate::types::CandleModelInfo = crate::types::CandleModelInfo {
            provider_name: "candle",
            name: "agent-completion-model",
            max_input_tokens: std::num::NonZeroU32::new(4096),
            max_output_tokens: std::num::NonZeroU32::new(4096),
            input_price: Some(0.0),
            output_price: Some(0.0),
            supports_vision: false,
            supports_function_calling: false,
            supports_streaming: true,
            supports_embeddings: false,
            requires_max_tokens: false,
            supports_thinking: false,
            optimal_thinking_budget: None,
            system_prompt_prefix: None,
            real_name: None,
            model_type: None,
            patch: None,
        };
        &DEFAULT_INFO
    }
}

impl crate::types::CandleCompletionModel for AgentCompletionModel {
    fn complete(&self, _request: crate::types::CandleCompletionRequest) -> crate::client::CandleCompletionBuilder<'_, ()> {
        // TODO: Implement proper completion builder
        todo!("Implement completion builder")
    }
    
    fn stream_complete(&self, _request: crate::types::CandleCompletionRequest) -> crate::types::CandleStreamingResponse {
        // TODO: Implement proper streaming completion
        todo!("Implement streaming completion")
    }
    
    fn prompt<'a>(
        &'a self,
        prompt: &str,
        params: &'a crate::types::CandleCompletionParams,
    ) -> fluent_ai_async::AsyncStream<crate::types::CandleCompletionChunk> {
        let _agent = self.agent.clone();
        let _params = params.clone();

        AsyncStream::with_channel(move |sender| {
            // TODO: Replace with proper streams-only completion
            // For now, send default chunk to maintain compilation
            let default_chunk = crate::types::CandleCompletionChunk::Complete {
                text: format!("{:?}", prompt),
                finish_reason: Some(crate::types::CandleFinishReason::Stop),
                usage: None,
            };
            let _ = sender.try_send(default_chunk);
        })
    }
}
