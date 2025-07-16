use crate::async_task::{AsyncTask, spawn_async};
use crate::domain::completion::{CompletionBackend, CompletionRequest};
use crate::engine::{Agent, AgentConfig, CompletionResponse, Engine, ExtractionConfig};
use crate::ZeroOneOrMany;
use fluent_ai_provider::Models;
use serde_json::Value;
use std::sync::Arc;

/// A concrete engine implementation that integrates with the existing fluent-ai domain system
pub struct FluentEngine {
    /// The backend implementation for completions
    backend: Arc<dyn CompletionBackend + Send + Sync>,
    /// Engine configuration
    model: Models,
    /// Default temperature for requests
    default_temperature: Option<f64>,
    /// Default max tokens for requests
    default_max_tokens: Option<u64>,
}

impl FluentEngine {
    /// Create a new FluentEngine with a completion backend
    pub fn new(backend: Arc<dyn CompletionBackend + Send + Sync>, model: Models) -> Self {
        Self {
            backend,
            model,
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
    fn extraction_config_to_completion_request(
        &self,
        config: &ExtractionConfig,
    ) -> CompletionRequest {
        let system_prompt = if let Some(schema) = &config.schema {
            format!(
                "{}\n\nPlease respond with valid JSON matching this schema: {}",
                config.prompt, schema
            )
        } else {
            format!("{}\n\nPlease respond with valid JSON.", config.prompt)
        };

        CompletionRequest {
            system_prompt,
            chat_history: crate::ZeroOneOrMany::None,
            documents: crate::ZeroOneOrMany::None,
            tools: crate::ZeroOneOrMany::None,
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
    fn model(&self) -> &Models {
        &self.config.model
    }
}

impl Engine for FluentEngine {
    fn create_agent(
        &self,
        config: AgentConfig,
    ) -> AsyncTask<Box<dyn Agent + Send>>
    where
        Box<dyn Agent + Send>: crate::async_task::NotResult,
    {
        spawn_async(async move {
            let agent = FluentAgent::new(config);
            Box::new(agent) as Box<dyn Agent + Send>
        })
    }

    fn complete(
        &self,
        request: CompletionRequest,
    ) -> AsyncTask<CompletionResponse>
    where
        CompletionResponse: crate::async_task::NotResult,
    {
        let backend = Arc::clone(&self.backend);
        let _model = self.model.clone();
        let _default_temperature = self.default_temperature;
        let _default_max_tokens = self.default_max_tokens;

        spawn_async(async move {
            // Convert CompletionRequest to the format expected by the backend
            let prompt = format!(
                "System: {}\n\nChat History:\n{}\n\nDocuments:\n{}\n\nPlease provide a response.",
                request.system_prompt,
                request
                    .chat_history
                    .iter()
                    .map(|msg| format!("{:?}: {}", msg.role, msg.content))
                    .collect::<Vec<_>>()
                    .join("\n"),
                request
                    .documents
                    .iter()
                    .map(|doc| format!("Document: {}", doc.content()))
                    .collect::<Vec<_>>()
                    .join("\n")
            );

            let tools: Vec<String> = match &request.tools {
                ZeroOneOrMany::None => vec![],
                ZeroOneOrMany::One(tool) => vec![tool.name.clone()],
                ZeroOneOrMany::Many(tools) => tools.iter().map(|t| t.name.clone()).collect(),
            };

            // Submit completion to backend
            let result = backend.submit_completion(&prompt, &tools).await;

            // result is already a String since AsyncTask handles errors internally
            CompletionResponse {
                content: result,
                usage: Some(crate::engine::Usage {
                    prompt_tokens: 0,     // Backend doesn't provide this info
                    completion_tokens: 0, // Backend doesn't provide this info
                    total_tokens: 0,      // Backend doesn't provide this info
                }),
            }
        })
    }

    fn extract_json(
        &self,
        config: ExtractionConfig,
    ) -> AsyncTask<Value>
    where
        Value: crate::async_task::NotResult,
    {
        let completion_request = self.extraction_config_to_completion_request(&config);
        let complete_task = self.complete(completion_request);
        
        spawn_async(async move {
            let response = complete_task.await; // AsyncTask now returns T directly, not Result<T, E>

            // Try to parse the response as JSON, return default on error
            match serde_json::from_str(&response.content) {
                Ok(json) => json,
                Err(_) => Value::String(response.content), // Return original content as string value
            }
        })
    }

    fn execute_tool(
        &self,
        tool_name: &str,
        args: Value,
    ) -> AsyncTask<Value>
    where
        Value: crate::async_task::NotResult,
    {
        let backend = Arc::clone(&self.backend);
        let tool_name = tool_name.to_string();
        
        spawn_async(async move {
            // For now, we'll create a completion request that asks the backend to execute the tool
            let prompt = format!(
                "Execute tool '{}' with arguments: {}. Please provide the result.",
                tool_name, args
            );

            let result = backend
                .submit_completion(&prompt, &[tool_name.clone()])
                .await;

            // result is already a String since AsyncTask handles errors internally
            // Try to parse as JSON, or return as string value
            match serde_json::from_str(&result) {
                Ok(json) => json,
                Err(_) => Value::String(result),
            }
        })
    }

    fn available_tools(
        &self,
    ) -> AsyncTask<ZeroOneOrMany<String>>
    where
        ZeroOneOrMany<String>: crate::async_task::NotResult,
    {
        spawn_async(async move {
            // For now, return a basic set of tools
            // In a real implementation, this would query the backend for available tools
            ZeroOneOrMany::from_vec(vec![
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
            model: self.model.clone(),
            default_temperature: self.default_temperature,
            default_max_tokens: self.default_max_tokens,
        }
    }
}