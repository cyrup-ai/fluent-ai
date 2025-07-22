// src/cognitive/common/models.rs
//! Defines the Model and ModelType for completion provider interactions.

use std::sync::Arc;

// Removed fluent_ai_domain import to break circular dependency
// Define local types instead of importing from domain

// Import response types from cyrup_sugars
use cyrup_sugars::{CompletionResponse, ResponseMetadata, TokenUsage};
use fluent_ai_domain::async_task::AsyncStream;
// Use domain types for traits and models
use fluent_ai_domain::{chat::Message, completion::CompletionProvider, model::ModelConfig};
// Use provider clients for completion services
use fluent_ai_provider::{anthropic::AnthropicClient, openai::OpenAIClient};
use serde::{Deserialize, Serialize};

/// Message type for completion interactions
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Core completion error type
#[derive(Debug, Clone, thiserror::Error)]
pub enum CompletionCoreError {
    #[error("Provider unavailable: {0}")]
    ProviderUnavailable(String),
    #[error("Request failed: {0}")]
    RequestFailed(String),
    #[error("Parse error: {0}")]
    ParseError(String),
}

/// Model type for completion provider interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Gpt35Turbo,
    Gpt4,
    Gpt4O,
    Gpt4Turbo,
    Claude3Opus,
    Claude3Sonnet,
    Claude3Haiku,
    GeminiPro,
    Mixtral8x7B,
    Llama270B,
    Llama3,
}

impl ModelType {
    pub fn display_name(&self) -> &'static str {
        match self {
            ModelType::Gpt35Turbo => "gpt-3.5-turbo",
            ModelType::Gpt4 => "gpt-4",
            ModelType::Gpt4O => "gpt-4o",
            ModelType::Gpt4Turbo => "gpt-4-turbo",
            ModelType::Claude3Opus => "claude-3-opus-20240229",
            ModelType::Claude3Sonnet => "claude-3-sonnet-20240229",
            ModelType::Claude3Haiku => "claude-3-haiku-20240307",
            ModelType::GeminiPro => "gemini-pro",
            ModelType::Mixtral8x7B => "mixtral-8x7b-instruct",
            ModelType::Llama270B => "llama-2-70b-chat",
            ModelType::Llama3 => "llama-3",
        }
    }

    pub fn provider_name(&self) -> &'static str {
        match self {
            ModelType::Gpt35Turbo | ModelType::Gpt4 | ModelType::Gpt4O | ModelType::Gpt4Turbo => {
                "openai"
            }
            ModelType::Claude3Opus | ModelType::Claude3Sonnet | ModelType::Claude3Haiku => {
                "anthropic"
            }
            ModelType::GeminiPro => "google",
            ModelType::Mixtral8x7B | ModelType::Llama270B | ModelType::Llama3 => "huggingface",
        }
    }
}

/// Simple model wrapper for completion providers using Provider package
#[derive(Debug, Clone)]
pub struct Model {
    model_type: ModelType,
    provider: Arc<dyn CompletionProvider>,
}

impl Model {
    pub async fn create(
        model_type: ModelType,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Create REAL completion clients using Provider package
        let provider: Arc<dyn CompletionProvider> = match model_type {
            ModelType::Gpt35Turbo | ModelType::Gpt4 | ModelType::Gpt4O | ModelType::Gpt4Turbo => {
                let api_key = std::env::var("OPENAI_API_KEY")
                    .map_err(|_| "OPENAI_API_KEY environment variable not set")?;

                let client = OpenAIClient::new(api_key, model_type.display_name())
                    .map_err(|e| format!("Failed to create OpenAI client: {}", e))?;
                Arc::new(client)
            }
            ModelType::Claude3Opus | ModelType::Claude3Sonnet | ModelType::Claude3Haiku => {
                let api_key = std::env::var("ANTHROPIC_API_KEY")
                    .map_err(|_| "ANTHROPIC_API_KEY environment variable not set")?;

                let client = AnthropicClient::new(api_key, model_type.display_name())
                    .map_err(|e| format!("Failed to create Anthropic client: {}", e))?;
                Arc::new(client)
            }
            ModelType::GeminiPro
            | ModelType::Mixtral8x7B
            | ModelType::Llama270B
            | ModelType::Llama3 => {
                return Err(format!("Model type {:?} is not yet implemented", model_type).into());
            }
        };

        Ok(Self {
            model_type,
            provider,
        })
    }

    pub fn available_types() -> Vec<ModelType> {
        vec![
            ModelType::Gpt35Turbo,
            ModelType::Gpt4,
            ModelType::Gpt4O,
            ModelType::Claude3Opus,
            ModelType::Claude3Sonnet,
            ModelType::Claude3Haiku,
        ]
    }

    /// Complete a request using the Provider package streaming interface
    pub async fn complete(
        &self,
        messages: Vec<Message>,
    ) -> Result<CompletionResponse<String>, Box<dyn std::error::Error + Send + Sync>> {
        // Convert messages to the last user message for simple completion
        let user_message = messages
            .iter()
            .filter(|msg| msg.role == "user")
            .last()
            .ok_or("No user message found in completion request")?;

        // Use completion provider streaming interface
        let stream = self.provider.prompt(&user_message.content);

        // Collect all chunks to form complete response
        let mut collected_text = String::new();
        let mut stream = stream;

        use futures_util::StreamExt;
        while let Some(chunk) = stream.next().await {
            if let Some(content) = chunk.content {
                collected_text.push_str(&content);
            }
        }

        Ok(CompletionResponse::new(
            collected_text.clone(),
            cyrup_sugars::ZeroOneOrMany::One(collected_text),
        )
        .with_token_usage(TokenUsage::new(0, 0)) // Placeholder - would need actual tracking
        .with_metadata(
            ResponseMetadata::new().with_model(self.model_type.display_name().to_string()),
        ))
    }

    /// Simple prompt interface
    pub async fn prompt(
        &self,
        prompt: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Use provider streaming interface and collect result
        let stream = self.provider.prompt(prompt);

        let mut result = String::new();
        let mut stream = stream;

        use futures_util::StreamExt;
        while let Some(chunk) = stream.next().await {
            if let Some(content) = chunk.content {
                result.push_str(&content);
            }
        }

        Ok(result)
    }

    /// Get the model type
    pub fn model_type(&self) -> &ModelType {
        &self.model_type
    }

    /// Get the model display name
    pub fn display_name(&self) -> &'static str {
        self.model_type.display_name()
    }

    /// Get the provider name
    pub fn provider_name(&self) -> &'static str {
        self.model_type.provider_name()
    }
}

/// Completion request wrapper for backward compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub messages: Vec<Message>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
}

impl CompletionRequest {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            temperature: None,
            max_tokens: None,
        }
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

/// Usage statistics for backward compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub processing_time_ms: Option<u64>,
    pub cache_hit_ratio: Option<f64>,
}

impl From<TokenUsage> for Usage {
    fn from(token_usage: TokenUsage) -> Self {
        Self {
            prompt_tokens: token_usage.prompt_tokens,
            completion_tokens: token_usage.completion_tokens,
            total_tokens: token_usage.total_tokens,
            processing_time_ms: None,
            cache_hit_ratio: None,
        }
    }
}
