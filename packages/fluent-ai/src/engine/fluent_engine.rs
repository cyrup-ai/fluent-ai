use std::sync::Arc;

use fluent_ai_domain::agent_role::AgentRole;
use fluent_ai_provider::Models;
use serde_json::Value;

use crate::ZeroOneOrMany;
use crate::async_task::{AsyncTask, spawn_async};
use crate::domain::completion::{CompletionBackend, CompletionRequest};
use crate::engine::{Agent, CompletionResponse, Engine, ExtractionConfig};

/// A concrete engine implementation that integrates with the existing fluent-ai domain system
pub struct FluentEngine {
    /// The backend implementation for completions
    backend: Arc<dyn CompletionBackend + Send + Sync>,
    /// Engine configuration
    model: Models,
    /// Default temperature for requests
    default_temperature: Option<f64>,
    /// Default max tokens for requests
    default_max_tokens: Option<u64>}

impl FluentEngine {
    /// Create a new FluentEngine with a completion backend
    pub fn new(backend: Arc<dyn CompletionBackend + Send + Sync>, model: Models) -> Self {
        Self {
            backend,
            model,
            default_temperature: None,
            default_max_tokens: None}
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
            additional_params: None}
    }
}

/// A simple agent implementation for identification and configuration storage
pub struct FluentAgent {
    role: Arc<dyn AgentRole>}

impl FluentAgent {
    pub fn new(role: Arc<dyn AgentRole>) -> Self {
        Self { role }
    }
}

impl Agent for FluentAgent {
    fn name(&self) -> &str {
        self.role.name()
    }
}

impl Engine for FluentEngine {
    fn name(&self) -> &str {
        "fluent-engine"
    }

    fn complete(
        &self,
        request: &crate::domain::completion::CompletionRequest,
    ) -> crate::runtime::AsyncTask<Result<CompletionResponse, crate::completion::CompletionError>>
    {
        let backend = self.backend.clone();
        let request = request.clone();
        spawn_async(async move {
            // Use backend to complete the request
            let response = backend.complete(request).await?;
            Ok(response)
        })
    }
}

impl Clone for FluentEngine {
    fn clone(&self) -> Self {
        Self {
            backend: self.backend.clone(),
            model: self.model.clone(),
            default_temperature: self.default_temperature,
            default_max_tokens: self.default_max_tokens}
    }
}
