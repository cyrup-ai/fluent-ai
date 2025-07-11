use crate::domain::completion::{CompletionBackend, CompletionRequest, ToolDefinition};
use serde_json::Value;
use std::error::Error as StdError;
use std::collections::HashMap;
use std::sync::Arc;
use std::future::Future;
use std::pin::Pin;
// AgentConfig, Agent trait, and CompletionResponse are defined in engine.rs
use crate::engine::{Engine, ExtractionConfig, AgentConfig, Agent, CompletionResponse};
use crate::providers::Model;

/// A concrete engine implementation that integrates with the existing fluent-ai domain system
pub struct FluentEngine {
    /// The backend implementation for completions
    backend: Arc<dyn CompletionBackend + Send + Sync>,
    /// Engine configuration
    model_name: String,
    /// Default temperature for requests
    default_temperature: Option<f64>,
    /// Default max tokens for requests
    default_max_tokens: Option<u64>,
}

impl FluentEngine {
    /// Create a new FluentEngine with a completion backend
    pub fn new(
        backend: Arc<dyn CompletionBackend + Send + Sync>,
        model_name: impl Into<String>,
    ) -> Self {
        Self {
            backend,
            model_name: model_name.into(),
            default_temperature: None,
            default_max_tokens: None,
        }
    }

    /// Set the default temperature for this engine
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.default_temperature = Some(temperature);
        self
    }

    /// Set the default max tokens for this engine
    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.default_max_tokens = Some(max_tokens);
        self
    }



    /// Convert ExtractionConfig to CompletionRequest
    fn extraction_config_to_completion_request(&self, config: &ExtractionConfig) -> CompletionRequest {
        let system_prompt = if let Some(schema) = &config.schema {
            format!("{}\n\nPlease respond with valid JSON matching this schema: {}", 
                   config.prompt, schema)
        } else {
            format!("{}\n\nPlease respond with valid JSON.", config.prompt)
        };

        CompletionRequest {
            system_prompt,
            chat_history: Vec::new(),
            documents: Vec::new(),  
            tools: Vec::new(),
            temperature: config.temperature.or(self.default_temperature),
            max_tokens: self.default_max_tokens,
            chunk_size: None,
            additional_params: None,
        }
    }
}

/// A simple agent implementation for identification and configuration storage
pub struct FluentAgent {
    config: AgentConfig,
}

impl FluentAgent {
    pub fn new(config: AgentConfig) -> Self {
        Self { config }
    }
}

impl Agent for FluentAgent {
    fn name(&self) -> &str {
        &self.config.model
    }
}

impl Engine for FluentEngine {
    fn create_agent(&self, config: AgentConfig) -> Pin<Box<dyn Future<Output = Result<Box<dyn Agent + Send>, Box<dyn StdError + Send + Sync>>> + Send + '_>> {
        Box::pin(async move {
            let agent = FluentAgent::new(config);
            Ok(Box::new(agent) as Box<dyn Agent + Send>)
        })
    }

    fn complete(&self, request: CompletionRequest) -> Pin<Box<dyn Future<Output = Result<CompletionResponse, Box<dyn StdError + Send + Sync>>> + Send + '_>> {
        let backend = Arc::clone(&self.backend);
        let model_name = self.model_name.clone();
        let default_temperature = self.default_temperature;
        let default_max_tokens = self.default_max_tokens;
        
        Box::pin(async move {
            // Convert CompletionRequest to the format expected by the backend
            let prompt = format!(
                "System: {}\n\nChat History:\n{}\n\nDocuments:\n{}\n\nPlease provide a response.",
                request.system_prompt,
                request.chat_history.iter()
                    .map(|msg| format!("{:?}: {}", msg.role, msg.content))
                    .collect::<Vec<_>>()
                    .join("\n"),
                request.documents.iter()
                    .map(|doc| format!("Document: {}", doc.content()))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            
            let tools: Vec<String> = request.tools.iter().map(|t| t.name.clone()).collect();
            
            // Submit completion to backend
            let result = backend.submit_completion(&prompt, &tools).await;
            
            match result {
                Ok(content) => {
                    Ok(CompletionResponse {
                        content,
                        usage: Some(crate::engine::Usage {
                            prompt_tokens: 0, // Backend doesn't provide this info
                            completion_tokens: 0, // Backend doesn't provide this info
                            total_tokens: 0, // Backend doesn't provide this info
                        }),
                    })
                },
                Err(e) => Err(format!("Completion failed: {}", e).into()),
            }
        })
    }

    fn extract_json(&self, config: ExtractionConfig) -> Pin<Box<dyn Future<Output = Result<Value, Box<dyn StdError + Send + Sync>>> + Send + '_>> {
        Box::pin(async move {
            let completion_request = self.extraction_config_to_completion_request(&config);
            let response = self.complete(completion_request).await?;
            
            // Try to parse the response as JSON
            match serde_json::from_str(&response.content) {
                Ok(json) => Ok(json),
                Err(e) => Err(format!("Failed to parse JSON response: {}", e).into()),
            }
        })
    }

    fn execute_tool(&self, tool_name: &str, args: Value) -> Pin<Box<dyn Future<Output = Result<Value, Box<dyn StdError + Send + Sync>>> + Send + '_>> {
        let backend = Arc::clone(&self.backend);
        let tool_name = tool_name.to_string();
        Box::pin(async move {
            // For now, we'll create a completion request that asks the backend to execute the tool
            let prompt = format!(
                "Execute tool '{}' with arguments: {}. Please provide the result.",
                tool_name,
                args
            );
            
            let result = backend.submit_completion(&prompt, &[tool_name.clone()]).await;
            
            match result {
                Ok(content) => {
                    // Try to parse as JSON, or return as string value
                    match serde_json::from_str(&content) {
                        Ok(json) => Ok(json),
                        Err(_) => Ok(Value::String(content)),
                    }
                },
                Err(e) => Err(format!("Tool execution failed: {}", e).into()),
            }
        })
    }

    fn available_tools(&self) -> Pin<Box<dyn Future<Output = Result<Vec<String>, Box<dyn StdError + Send + Sync>>> + Send + '_>> {
        Box::pin(async move {
            // For now, return a basic set of tools
            // In a real implementation, this would query the backend for available tools
            Ok(vec![
                "web_search".to_string(),
                "calculator".to_string(),
                "file_reader".to_string(),
                "code_executor".to_string(),
            ])
        })
    }
}

impl Clone for FluentEngine {
    fn clone(&self) -> Self {
        Self {
            backend: self.backend.clone(),
            model_name: self.model_name.clone(),  
            default_temperature: self.default_temperature,
            default_max_tokens: self.default_max_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::async_task::AsyncTask;
    use std::sync::Arc;

    /// Mock completion backend for testing
    struct MockCompletionBackend {
        response: String,
    }

    impl MockCompletionBackend {
        fn new(response: impl Into<String>) -> Self {
            Self {
                response: response.into(),
            }
        }
    }

    impl CompletionBackend for MockCompletionBackend {
        fn submit_completion(
            &self,
            _prompt: &str,
            _tools: &[String],
        ) -> AsyncTask<String> {
            let response = self.response.clone();
            AsyncTask::from_future(async move { response })
        }
    }

    #[tokio::test]
    async fn test_fluent_engine_complete() {
        let backend = Arc::new(MockCompletionBackend::new("Hello, world!"));
        let engine = FluentEngine::new(backend, "test-model");
        
        let request = CompletionRequest {
            system_prompt: "Say hello".to_string(),
            chat_history: Vec::new(),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            chunk_size: None,
            additional_params: None,
        };
        
        let response = engine.complete(request).await.unwrap();
        assert_eq!(response.content, "Hello, world!");
        assert!(response.usage.is_some());
    }

    #[tokio::test]
    async fn test_fluent_engine_extract_json() {
        let backend = Arc::new(MockCompletionBackend::new(r#"{"key": "value"}"#));
        let engine = FluentEngine::new(backend, "test-model");
        
        let config = ExtractionConfig {
            model: "test-model".to_string(),
            prompt: "Extract data".to_string(),
            schema: None,
            temperature: None,
        };
        
        let result = engine.extract_json(config).await.unwrap();
        assert_eq!(result, serde_json::json!({"key": "value"}));
    }

    #[tokio::test]
    async fn test_fluent_engine_create_agent() {
        let backend = Arc::new(MockCompletionBackend::new("Hello"));
        let engine = FluentEngine::new(backend, "test-model");
        
        let config = AgentConfig {
            model: Model::OpenaiGpt4o, // Using a real model enum variant instead of string
            system_prompt: Some("You are a helpful assistant".to_string()),
            temperature: Some(0.8),
            max_tokens: Some(500),
            tools: vec![],
        };
        
        let agent = engine.create_agent(config).await.unwrap();
        // Agent is created successfully - just verify it has the correct name
        assert_eq!(agent.name(), "test_model");
    }

    #[tokio::test]
    async fn test_fluent_engine_available_tools() {
        let backend = Arc::new(MockCompletionBackend::new("tools"));
        let engine = FluentEngine::new(backend, "test-model");
        
        let tools = engine.available_tools().await.unwrap();
        assert!(!tools.is_empty());
        assert!(tools.contains(&"web_search".to_string()));
    }
}
