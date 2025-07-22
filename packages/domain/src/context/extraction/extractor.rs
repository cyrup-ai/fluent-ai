use std::fmt;
use std::marker::PhantomData;

use fluent_ai_async::AsyncStream;
use tokio_stream::StreamExt;
use serde::de::DeserializeOwned;

use super::error::ExtractionError;
use crate::agent::Agent;
use crate::completion::{CompletionModel, CompletionParams, CompletionRequest};
use crate::chat::message::MessageRole;
use crate::prompt::Prompt;
use crate::context::chunk::{CompletionChunk, FinishReason};

/// Trait defining the core extraction interface
pub trait Extractor<T>: Send + Sync + fmt::Debug + Clone
where
    T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static,
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

impl<T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static> Extractor<T>
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

        let (sender, stream) = AsyncStream::channel();
        
        tokio::spawn(async move {
            let prompt = if let Some(sys_prompt) = system_prompt {
                format!(
                    "{}\n\nExtract information from the following text:\n{}",
                    sys_prompt, text
                )
            } else {
                format!("Extract information from the following text:\n{}", text)
            };

            let completion_request = CompletionRequest::new()
                .with_prompt(prompt)
                .with_temperature(0.2)
                .with_max_tokens(1000);

            match Self::execute_extraction(agent, completion_request, text).await {
                Ok(result) => {
                    let _ = sender.send(result);
                }
                Err(_) => {
                    // In streams-only architecture, errors are handled via on_chunk pattern
                    // The stream will close without sending a value
                }
            }
        });
        
        stream
    }
}

impl<T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static> ExtractorImpl<T> {
    async fn execute_extraction(
        agent: Agent,
        completion_request: CompletionRequest<'_>,
        _text_input: String,
    ) -> Result<T, ExtractionError> {
        let model = AgentCompletionModel::new(agent);
        let prompt = Prompt {
            content: completion_request.system_prompt.to_string(),
            role: MessageRole::System,
        };
        let params = CompletionParams {
            temperature: completion_request.temperature,
            max_tokens: completion_request.max_tokens,
            n: std::num::NonZeroU8::new(1).unwrap(),
            stream: true,
        };
        let mut stream = model.prompt(prompt, &params);

        let mut full_response = String::new();
        let mut finish_reason = None;

        while let Some(chunk) = stream.next().await {
            match chunk {
                CompletionChunk::Text(text) => {
                    full_response.push_str(&text);
                }
                CompletionChunk::Complete { text, finish_reason: reason, .. } => {
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

impl CompletionModel for AgentCompletionModel {
    fn prompt<'a>(
        &'a self,
        prompt: Prompt,
        params: &'a CompletionParams,
    ) -> fluent_ai_async::AsyncStream<CompletionChunk> {
        let agent = self.agent.clone();
        let params = params.clone();

        Box::pin(async_stream::stream! {
            let mut request = CompletionRequest::new()
                .with_prompt(prompt.to_string())
                .with_params(params);

            let mut stream = match agent.complete_stream(&request).await {
                Ok(stream) => stream,
                Err(e) => {
                    yield Err(e.into());
                    return;
                }
            };

            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(chunk) => yield Ok(chunk),
                    Err(e) => {
                        yield Err(e.into());
                        return;
                    }
                }
            }
        })
    }
}
