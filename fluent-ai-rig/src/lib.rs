use fluent_ai::domain::completion::CompletionBackend;
use fluent_ai::async_task::AsyncTask;
use fluent_ai::engine::Engine;
use std::env;
use std::sync::Arc;
use tracing::info;

// Import rig-core types to discover API
use rig_core::providers::openai;

/// RigBackend implements CompletionBackend using rig-core for real LLM completions
pub struct RigBackend {
    client: openai::Client,
    model: String,
}

impl RigBackend {
    /// Create a new RigBackend with OpenAI API key
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let api_key = env::var("MISTRAL_API_KEY")
            .map_err(|_| "MISTRAL_API_KEY environment variable not set")?;
        
        let client = openai::Client::new(&api_key);
        
        info!("RigBackend initialized with OpenAI API key");
        
        Ok(RigBackend {
            client,
            model: Model::MagistralSmall.to_string(),
        })
    }
    
    /// Create RigBackend with custom model
    pub fn with_model(model_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let api_key = env::var("MISTRAL_API_KEY")
            .map_err(|_| "MISTRAL_API_KEY environment variable not set")?;
        
        let client = openai::Client::new(&api_key);
        
        info!("RigBackend initialized with custom model: {}", model_name);
        
        Ok(RigBackend {
            client,
            model: model_name.to_string(),
        })
    }
}

impl CompletionBackend for RigBackend {
    fn submit_completion(
        &self,
        prompt: &str,
        tools: &[String],
    ) -> AsyncTask<String> {
        let prompt = prompt.to_string();
        let client = self.client.clone();
        let model = self.model.clone();
        
        AsyncTask::from_future(async move {
            use rig_core::completion::{Prompt, CompletionModel};
            
            info!("Submitting completion request to OpenAI via rig-core");
            
            // Create the completion model
            let gpt = client.completion_model(&model);
            
            // Create and execute the prompt
            match gpt.prompt(&prompt).await {
                Ok(response) => {
                    info!("Received response from OpenAI: {} chars", response.len());
                    response
                }
                Err(e) => {
                    let error_msg = format!("Rig completion error: {}", e);
                    tracing::error!("{}", error_msg);
                    error_msg
                }
            }
        })
    }
}

/// Create a FluentEngine with RigBackend for production use
pub fn create_fluent_engine() -> Result<fluent_ai::engine::FluentEngine, Box<dyn std::error::Error + Send + Sync>> {
    let backend = RigBackend::new()?;
    let engine = fluent_ai::engine::FluentEngine::new(
        Arc::new(backend),
        "gpt-4o-mini"
    );
    Ok(engine)
}

/// Create a FluentEngine with custom model
pub fn create_fluent_engine_with_model(model: &str) -> Result<fluent_ai::engine::FluentEngine, Box<dyn std::error::Error + Send + Sync>> {
    let backend = RigBackend::with_model(model)?;
    let engine = fluent_ai::engine::FluentEngine::new(
        Arc::new(backend),
        model
    );
    Ok(engine)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_engine_without_api_key() {
        // Temporarily unset the API key
        let original = env::var("MISTRAL_API_KEY").ok();
        env::remove_var("MISTRAL_API_KEY");
        
        let result = create_fluent_engine();
        assert!(result.is_err());
        
        // Restore original value if it existed
        if let Some(key) = original {
            env::set_var("MISTRAL_API_KEY", key);
        }
    }
}
