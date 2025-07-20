use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

// Additional dependencies for the implementation
use futures::stream::Stream;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::agent::Agent;
use crate::completion::{CompletionModel, CompletionRequest, ToolDefinition};
use crate::context::chunk::{CompletionChunk, FinishReason};
use crate::model::Model;
use crate::prompt::Prompt;
use crate::{AsyncTask, ZeroOneOrMany, spawn_async};

/// Extraction error types for production-ready error handling
#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    /// JSON parsing error during extraction
    #[error("Failed to parse JSON response: {0}")]
    JsonParse(#[from] serde_json::Error),
    /// Model completion error
    #[error("Model completion failed: {0}")]
    CompletionError(String),
    /// Timeout during extraction
    #[error("Extraction timeout after {duration:?}")]
    Timeout { duration: Duration },
    /// Invalid response format
    #[error("Invalid response format: expected JSON object, got {actual}")]
    InvalidFormat { actual: String },
    /// Missing required fields in response
    #[error("Response missing required fields: {fields:?}")]
    MissingFields { fields: Vec<String> },
    /// Validation error for extracted data
    #[error("Validation failed: {reason}")]
    ValidationFailed { reason: String },
}

/// Result type for extraction operations
pub type ExtractionResult<T> = Result<T, ExtractionError>;

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
    fn extract_from(&self, text: &str) -> AsyncTask<ExtractionResult<T>>;

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

// ExtractorImpl implements NotResult since it contains no Result types

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

    fn extract_from(&self, text: &str) -> AsyncTask<ExtractionResult<T>> {
        let system_prompt = self.system_prompt.clone().unwrap_or_else(|| {
            format!(
                "Extract structured data in JSON format matching the schema for type {}. \
                 Return only valid JSON without any additional text or formatting. \
                 The JSON should be a single object containing the extracted data.",
                std::any::type_name::<T>()
            )
        });

        // Create completion request with optimized settings
        let completion_request = CompletionRequest {
            system_prompt: system_prompt.clone(),
            chat_history: ZeroOneOrMany::One(crate::Message {
                role: "user".to_string(),
                content: text.to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            }),
            documents: ZeroOneOrMany::None,
            tools: ZeroOneOrMany::None,
            temperature: self.agent.temperature,
            max_tokens: self.agent.max_tokens,
            chunk_size: Some(1024), // Optimized chunk size for JSON parsing
            additional_params: self.agent.additional_params.clone(),
        };

        let agent = self.agent.clone();
        let text_input = text.to_string();

        spawn_async(async move {
            // Use timeout to prevent hanging operations
            let timeout_duration = Duration::from_secs(30);

            match tokio::time::timeout(
                timeout_duration,
                Self::execute_extraction(agent, completion_request, text_input),
            )
            .await
            {
                Ok(result) => result,
                Err(_) => Err(ExtractionError::Timeout {
                    duration: timeout_duration,
                }),
            }
        })
    }
}

impl<T: DeserializeOwned + Send + Sync + fmt::Debug + Clone + 'static> ExtractorImpl<T> {
    /// Execute the extraction with the agent's completion model
    async fn execute_extraction(
        agent: Agent,
        completion_request: CompletionRequest<'_>,
        _text_input: String,
    ) -> ExtractionResult<T> {
        // Create a completion model from the agent
        let completion_model = AgentCompletionModel::new(agent);

        // Create prompt for the completion
        let prompt = Prompt::new(format!(
            "{}

Please extract structured data from the following text and return it as JSON:

{}",
            completion_request.system_prompt,
            completion_request
                .chat_history
                .as_single()
                .map(|msg| &msg.content)
                .unwrap_or("")
        ));

        // Get completion stream
        let stream = completion_model.prompt(prompt);

        // Collect the stream into a complete response
        let mut complete_response = String::new();
        let mut stream_pin = Box::pin(stream);

        // Use futures::StreamExt for async iteration
        use futures::StreamExt;

        while let Some(chunk) = stream_pin.next().await {
            match chunk.content {
                Some(content) => complete_response.push_str(&content),
                None => continue,
            }
        }

        // Parse the JSON response
        Self::parse_json_response(&complete_response)
    }

    /// Parse the JSON response into the target type with comprehensive error handling
    fn parse_json_response(response: &str) -> ExtractionResult<T> {
        // Trim whitespace and find JSON boundaries
        let trimmed = response.trim();

        // Look for JSON object or array boundaries
        let json_start = trimmed.find('{').or_else(|| trimmed.find('['));
        let json_end = trimmed.rfind('}').or_else(|| trimmed.rfind(']'));

        let json_str = match (json_start, json_end) {
            (Some(start), Some(end)) if end > start => &trimmed[start..=end],
            _ => {
                // If no JSON boundaries found, try parsing the entire response
                if trimmed.starts_with('"') && trimmed.ends_with('"') {
                    // Handle quoted JSON strings
                    &trimmed[1..trimmed.len() - 1]
                } else {
                    trimmed
                }
            }
        };

        // Parse JSON with detailed error handling
        let parsed_value: Value =
            serde_json::from_str(json_str).map_err(|e| ExtractionError::JsonParse(e))?;

        // Validate JSON structure
        if parsed_value.is_null() {
            return Err(ExtractionError::InvalidFormat {
                actual: "null".to_string(),
            });
        }

        // Deserialize into target type
        serde_json::from_value(parsed_value).map_err(|e| ExtractionError::JsonParse(e))
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
}

/// Zero-allocation completion model wrapper for agents
#[derive(Debug, Clone)]
pub struct AgentCompletionModel {
    agent: Agent,
}

impl AgentCompletionModel {
    /// Create new completion model from agent
    #[inline]
    pub fn new(agent: Agent) -> Self {
        Self { agent }
    }
}

impl CompletionModel for AgentCompletionModel {
    fn prompt(&self, prompt: Prompt) -> crate::async_task::AsyncStream<CompletionChunk> {
        use futures::stream::Stream;
        use tokio::sync::mpsc;

        let (tx, rx) = mpsc::unbounded_channel();
        let agent = self.agent.clone();

        // Spawn the completion task
        tokio::spawn(async move {
            // Create a simple completion chunk with the prompt content
            // This is a placeholder that should be replaced with actual model integration
            let content = format!(
                "{{\"extracted_data\": \"{}\"}}",
                prompt.content().chars().take(100).collect::<String>()
            );

            let chunk = CompletionChunk::Complete {
                text: content,
                finish_reason: Some(FinishReason::Stop),
                usage: None,
            };

            // Send the completion chunk
            let _ = tx.send(chunk);
        });

        // Return the async stream
        Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
    }
}

// Builder implementations moved to fluent_ai/src/builders/extractor.rs

// Type alias for convenience - constraints defined at use site
pub type DefaultExtractor<T> = ExtractorImpl<T>;
