//! Fluent AI Rig - CLI integration for fluent-ai
//!
//! This crate provides factory functions and utilities for using the fluent-ai
//! chat system from command-line applications.

use fluent_ai_provider::{Models, Providers};

/// Re-export fluent-ai types for CLI usage
pub use fluent_ai::{
    agent::ChunkHandler,
    domain::completion::CompletionBackend,
    engine::{Engine, FluentEngine},
    async_task::AsyncTask,
};

/// Utility functions for CLI integration
pub mod cli {
    use super::*;
    use fluent_ai_provider::{Models, Providers};
    
    /// Validate that a provider supports a given model
    pub fn validate_provider_model_combination(
        provider: Providers,
        model: Models,
    ) -> Result<(), String> {
        // Use the generated provider/model validation logic
        match (provider, model) {
            // OpenAI models
            (Providers::Openai, Models::OpenaiGpt4o) => Ok(()),
            (Providers::Openai, Models::OpenaiGpt4oMini) => Ok(()),
            (Providers::Openai, Models::OpenaiGpt35Turbo) => Ok(()),
            // Anthropic models
            (Providers::Anthropic, Models::AnthropicClaude35Sonnet) => Ok(()),
            (Providers::Anthropic, Models::AnthropicClaude3Haiku) => Ok(()),
            // Mistral models
            (Providers::Mistral, Models::MistralLarge) => Ok(()),
            (Providers::Mistral, Models::MistralSmall) => Ok(()),
            // Invalid combinations
            _ => Err(format!(
                "Model {:?} is not supported by provider {:?}",
                model, provider
            )),
        }
    }
    
    /// Get the default model for a given provider
    pub fn get_default_model_for_provider(provider: Providers) -> Models {
        match provider {
            Providers::Openai => Models::OpenaiGpt4oMini,
            Providers::Anthropic => Models::AnthropicClaude35Sonnet,
            Providers::Mistral => Models::MistralLarge,
            Providers::Cohere => Models::CohereCommandRPlus,
            Providers::Google => Models::GoogleGemini15Pro,
            Providers::Groq => Models::GroqLlama370b8192,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::cli::*;
    use fluent_ai_provider::{Models, Providers};

    #[test]
    fn test_validate_openai_models() {
        assert!(validate_provider_model_combination(
            Providers::Openai,
            Models::OpenaiGpt4oMini
        ).is_ok());
        
        assert!(validate_provider_model_combination(
            Providers::Openai,
            Models::MistralLarge // Wrong provider
        ).is_err());
    }

    #[test]
    fn test_default_models() {
        assert_eq!(
            get_default_model_for_provider(Providers::Openai),
            Models::OpenaiGpt4oMini
        );
        
        assert_eq!(
            get_default_model_for_provider(Providers::Anthropic),
            Models::AnthropicClaude35Sonnet
        );
    }
}
