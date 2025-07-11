//! Fluent AI Rig - CLI integration for fluent-ai
//!
//! This crate provides factory functions and utilities for using the fluent-ai
//! chat system from command-line applications.

use fluent_ai::engine::{engine_builder, FluentEngine};
use fluent_ai::async_task::AsyncTask;
use fluent_ai::domain::completion::CompletionBackend;
use fluent_ai_provider::{Models, Providers, Provider as ProviderTrait, Model as ModelTrait};
use rig::providers::{openai, anthropic};
use rig::completion::Prompt;
use rig::client::CompletionClient;
use std::env;
use std::sync::Arc;
use tracing::error;

/// Rig-based CompletionBackend implementation
pub struct RigCompletionBackend {
    provider: Providers,
    model: Models,
}

impl RigCompletionBackend {
    pub fn new(provider: Providers, model: Models) -> Self {
        Self { provider, model }
    }
}

impl CompletionBackend for RigCompletionBackend {
    fn submit_completion(&self, prompt: &str, tools: &[String]) -> AsyncTask<String> {
        let provider = self.provider.clone();
        let model = self.model.clone();
        let prompt_text = prompt.to_string();
        let _tools = tools.to_vec(); // Store for future tool support
        AsyncTask::from_future(async move {
            // Get rig client by provider name
            let provider_name = provider.name();
            let model_name = model.name();
            
            let result = match provider_name {
                "openai" => {
                    let api_key = env::var("OPENAI_API_KEY")
                        .expect("OPENAI_API_KEY environment variable must be set");
                    let client = openai::Client::new(&api_key);
                    let agent = client.agent(model_name).build();
                    match agent.prompt(&prompt_text).await {
                        Ok(response) => response.to_string(),
                        Err(e) => {
                            error!("OpenAI completion failed: {}", e);
                            format!("Error: {}", e)
                        }
                    }
                },
                "claude" => {
                    let api_key = env::var("ANTHROPIC_API_KEY")
                        .expect("ANTHROPIC_API_KEY environment variable must be set");
                    let client = anthropic::ClientBuilder::new(&api_key).build();
                    let agent = client.agent(model_name).build();
                    match agent.prompt(&prompt_text).await {
                        Ok(response) => response.to_string(),
                        Err(e) => {
                            error!("Anthropic completion failed: {}", e);
                            format!("Error: {}", e)
                        }
                    }
                },
                _ => {
                    error!("Unsupported provider: {}", provider_name);
                    format!("Unsupported provider: {}", provider_name)
                }
            };
            
            result
        })
    }
}

/// Create a FluentEngine with the specified provider and model
pub fn create_fluent_engine_with_model(provider: Providers, model: Models) -> Result<Arc<FluentEngine>, Box<dyn std::error::Error + Send + Sync>> {
    let backend = Arc::new(RigCompletionBackend::new(provider, model.clone()));
    let engine = Arc::new(FluentEngine::new(backend, model));
    
    engine_builder()
        .engine(engine.clone())
        .name("fluent-ai-rig-engine")
        .build_and_register()?;
    
    Ok(engine)
}

/// Utility functions for CLI integration
pub mod cli {
    use fluent_ai_provider::{Models, Providers};
    
    /// Validate that a provider supports a given model
    pub fn validate_provider_model_combination(
        provider: Providers,
        model: Models,
    ) -> Result<(), String> {
        use fluent_ai_provider::{Provider, Model};
        
        // Get all models supported by this provider
        let supported_models = provider.models();
        
        // Check if the requested model is supported
        let model_name = model.name();
        let is_supported = supported_models.iter()
            .any(|supported| supported.name() == model_name);
            
        if is_supported {
            Ok(())
        } else {
            Err(format!(
                "Model '{}' is not supported by provider '{}'",
                model_name, provider.name()
            ))
        }
    }
    
    /// Get the default model for a given provider (first model in the list)
    pub fn get_default_model_for_provider(provider: Providers) -> Option<Models> {
        use fluent_ai_provider::Provider;
        
        provider.models().first().and_then(|model| {
            Models::from_name(model.name())
        })
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
