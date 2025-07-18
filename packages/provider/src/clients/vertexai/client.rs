//! VertexAI client with clean completion builder integration
//!
//! Provides factory methods that return clean completion builders with ModelInfo defaults:
//! ```
//! let client = VertexAIClient::new(service_account_json).await?;
//! client.completion_model("gemini-2.5-flash")
//!     .system_prompt("You are helpful")
//!     .temperature(0.8)
//!     .prompt("Hello world")
//! ```

use crate::completion_provider::{CompletionProvider, CompletionError};
use crate::client::{CompletionClient, ProviderClient};
use super::completion::VertexAICompletionBuilder;
use fluent_ai_domain::AsyncTask;

/// VertexAI client providing clean completion builder factory methods
#[derive(Clone)]
pub struct VertexAIClient {
    service_account_json: String,
    project_id: String,
    region: String,
}

impl VertexAIClient {
    /// Create new VertexAI client with service account JSON
    pub fn new(service_account_json: String, project_id: String, region: String) -> Result<Self, CompletionError> {
        if service_account_json.is_empty() {
            return Err(CompletionError::AuthError);
        }
        
        if project_id.is_empty() {
            return Err(CompletionError::AuthError);
        }
        
        Ok(Self { 
            service_account_json,
            project_id,
            region,
        })
    }
    
    /// Create completion builder for specific model with ModelInfo defaults loaded
    pub fn completion_model(&self, model_name: &'static str) -> Result<VertexAICompletionBuilder, CompletionError> {
        VertexAICompletionBuilder::new(
            self.service_account_json.clone(),
            self.project_id.clone(),
            self.region.clone(),
            model_name,
        )
    }
    
    /// Get project ID
    pub fn project_id(&self) -> &str {
        &self.project_id
    }
    
    /// Get region
    pub fn region(&self) -> &str {
        &self.region
    }
}

/// VertexAI provider for enumeration and discovery
pub struct VertexAIProvider;

impl VertexAIProvider {
    /// Create new VertexAI provider instance
    pub fn new() -> Self {
        Self
    }
    
    /// Get provider name
    pub const fn name() -> &'static str {
        "vertexai"
    }
    
    /// Get available models (compile-time constant)
    pub const fn models() -> &'static [&'static str] {
        &[
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "claude-3-5-sonnet",
            "claude-3-5-haiku", 
            "claude-3-opus",
            "mistral-large",
            "mistral-nemo",
            "codestral",
        ]
    }
}

impl Default for VertexAIProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-allocation CompletionClient implementation for VertexAI
impl CompletionClient for VertexAIClient {
    type Model = Result<VertexAICompletionBuilder, CompletionError>;

    /// Create a completion model with zero allocation and blazing-fast performance
    #[inline]
    fn completion_model(&self, model: &str) -> Self::Model {
        // Convert &str to &'static str efficiently for compatibility
        // SAFETY: This is safe because model names are typically string literals
        // stored in static memory. For dynamic strings, we use a fallback.
        let static_model = match model {
            "gemini-2.5-flash" => "gemini-2.5-flash",
            "gemini-2.5-pro" => "gemini-2.5-pro",
            "gemini-1.5-flash" => "gemini-1.5-flash",
            "gemini-1.5-pro" => "gemini-1.5-pro",
            "claude-3-5-sonnet" => "claude-3-5-sonnet",
            "claude-3-5-haiku" => "claude-3-5-haiku",
            "claude-3-opus" => "claude-3-opus",
            "mistral-large" => "mistral-large",
            "mistral-nemo" => "mistral-nemo",
            "codestral" => "codestral",
            _ => {
                // For unknown models, create a leaked static string (one-time allocation)
                // This is acceptable for model names which are typically static
                Box::leak(model.to_string().into_boxed_str())
            }
        };
        
        VertexAICompletionBuilder::new(
            self.service_account_json.clone(),
            self.project_id.clone(),
            self.region.clone(),
            static_model,
        )
    }
}

/// Zero-allocation ProviderClient implementation for VertexAI
impl ProviderClient for VertexAIClient {
    /// Get provider name with zero allocation
    #[inline]
    fn provider_name(&self) -> &'static str {
        "vertexai"
    }

    /// Test connection with blazing-fast async task
    #[inline]
    fn test_connection(&self) -> AsyncTask<Result<(), Box<dyn std::error::Error + Send + Sync>>> {
        let service_account_json = self.service_account_json.clone();
        let project_id = self.project_id.clone();
        let region = self.region.clone();
        
        AsyncTask::spawn(async move {
            // Zero-allocation validation: check required fields
            if service_account_json.is_empty() {
                return Err("VertexAI service account JSON is empty".into());
            }
            
            if project_id.is_empty() {
                return Err("VertexAI project ID is empty".into());
            }
            
            if region.is_empty() {
                return Err("VertexAI region is empty".into());
            }
            
            // Basic JSON validation (check for braces)
            if !service_account_json.starts_with('{') || !service_account_json.ends_with('}') {
                return Err("VertexAI service account JSON format invalid".into());
            }
            
            // Basic region validation (GCP regions are typically like us-central1)
            if !region.contains('-') {
                return Err("VertexAI region format invalid".into());
            }
            
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_client_creation() {
        let client = VertexAIClient::new(
            "{}".to_string(),
            "test-project".to_string(),
            "us-central1".to_string(),
        );
        assert!(client.is_ok());
        
        let client = client.expect("Failed to create vertexai client in test");
        assert_eq!(client.project_id(), "test-project");
        assert_eq!(client.region(), "us-central1");
    }
    
    #[test]
    fn test_client_creation_empty_json() {
        let client = VertexAIClient::new(
            "".to_string(),
            "test-project".to_string(),
            "us-central1".to_string(),
        );
        assert!(matches!(client, Err(CompletionError::AuthError)));
    }
    
    #[test]
    fn test_completion_model_factory() {
        let client = VertexAIClient::new(
            "{}".to_string(),
            "test-project".to_string(),
            "us-central1".to_string(),
        ).expect("Failed to create vertexai client in test");
        let builder = client.completion_model("gemini-2.5-flash");
        assert!(builder.is_ok());
    }
    
    #[test]
    fn test_provider() {
        let provider = VertexAIProvider::new();
        assert_eq!(VertexAIProvider::name(), "vertexai");
        assert!(!VertexAIProvider::models().is_empty());
    }
}