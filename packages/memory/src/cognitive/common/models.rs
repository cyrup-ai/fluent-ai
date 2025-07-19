// src/cognitive/common/models.rs
//! Defines the Model and ModelType for LLM interactions.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::llm::{CompletionRequest, CompletionResponse, LLMProvider, Usage};

/// Model type for LLM evaluation
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
            ModelType::Claude3Opus => "claude-3-opus",
            ModelType::Claude3Sonnet => "claude-3-sonnet",
            ModelType::Claude3Haiku => "claude-3-haiku",
            ModelType::GeminiPro => "gemini-pro",
            ModelType::Mixtral8x7B => "mixtral-8x7b-instruct",
            ModelType::Llama270B => "llama-2-70b-chat",
            ModelType::Llama3 => "llama-3",
        }
    }
}

/// Simple model wrapper for LLM providers
#[derive(Debug, Clone)]
pub struct Model {
    model_type: ModelType,
    provider: Arc<dyn LLMProvider>,
}

impl Model {
    pub async fn create(
        model_type: ModelType,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Create appropriate provider based on model type
        let provider: Arc<dyn LLMProvider> = match model_type {
            ModelType::Gpt35Turbo | ModelType::Gpt4 | ModelType::Gpt4O | ModelType::Gpt4Turbo => {
                let provider = crate::llm::OpenAIProvider::new(
                    "".to_string(),
                    Some(model_type.display_name().to_string()),
                )
                .map_err(|e| e.to_string())?;
                Arc::new(provider)
            }
            ModelType::Claude3Opus | ModelType::Claude3Sonnet | ModelType::Claude3Haiku => {
                let provider = crate::llm::AnthropicProvider::new(
                    "".to_string(),
                    Some(model_type.display_name().to_string()),
                )
                .map_err(|e| e.to_string())?;
                Arc::new(provider)
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

    pub async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, Box<dyn std::error::Error + Send + Sync>> {
        let completion = self.provider.complete(&request.prompt).await?;
        Ok(CompletionResponse {
            text: completion,
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
            model: self.model_type.display_name().to_string(),
        })
    }

    pub async fn prompt(
        &self,
        prompt: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let completion = self.provider.complete(prompt).await?;
        Ok(completion)
    }
}
