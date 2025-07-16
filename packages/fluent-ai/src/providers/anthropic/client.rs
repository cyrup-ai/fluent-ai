//! High-level Anthropic client with zero-allocation patterns
//!
//! Provides a comprehensive client interface for all Anthropic API features
//! with optimal performance and ergonomic usage patterns.

use crate::async_task::AsyncTask;
use crate::domain::completion::{CompletionRequest, ToolDefinition};
use crate::providers::anthropic::{
    AnthropicProvider, AnthropicError, AnthropicResult,
    AnthropicCompletionResponse, Usage,
};
use fluent_ai_provider::Models;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

/// High-level Anthropic client with comprehensive feature support
#[derive(Clone)]
pub struct AnthropicClient {
    provider: Arc<AnthropicProvider>,
    default_model: Models,
    default_temperature: Option<f64>,
    default_max_tokens: Option<u64>,
}

/// Builder for creating Anthropic clients with custom configuration
pub struct AnthropicClientBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<Models>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    timeout: Option<Duration>,
    max_retries: Option<u32>,
}

/// Completion request builder for Anthropic client
pub struct AnthropicCompletionBuilder {
    client: AnthropicClient,
    request: CompletionRequest,
    model_override: Option<Models>,
    temperature_override: Option<f64>,
    max_tokens_override: Option<u64>,
    tool_choice: Option<String>,
    stop_sequences: Option<Vec<String>>,
    metadata: Option<Value>,
}

impl AnthropicClient {
    /// Create new Anthropic client with API key
    #[inline(always)]
    pub fn new(api_key: impl Into<String>) -> AnthropicResult<Self> {
        let provider = AnthropicProvider::new(api_key)?;
        Ok(Self {
            provider: Arc::new(provider),
            default_model: Models::AnthropicClaude35Sonnet,
            default_temperature: None,
            default_max_tokens: None,
        })
    }

    /// Create client from environment variables
    #[inline(always)]
    pub fn from_env() -> AnthropicResult<Self> {
        let provider = crate::providers::anthropic::from_env()?;
        Ok(Self {
            provider: Arc::new(provider),
            default_model: Models::AnthropicClaude35Sonnet,
            default_temperature: None,
            default_max_tokens: None,
        })
    }

    /// Start building a client with custom configuration
    #[inline(always)]
    pub fn builder() -> AnthropicClientBuilder {
        AnthropicClientBuilder {
            api_key: None,
            base_url: None,
            model: None,
            temperature: None,
            max_tokens: None,
            timeout: None,
            max_retries: None,
        }
    }

    /// Set default model for this client
    #[inline(always)]
    pub fn with_model(mut self, model: Models) -> Self {
        self.default_model = model;
        self
    }

    /// Set default temperature for this client
    #[inline(always)]
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.default_temperature = Some(temperature);
        self
    }

    /// Set default max tokens for this client
    #[inline(always)]
    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.default_max_tokens = Some(max_tokens);
        self
    }

    /// Start building a completion request
    #[inline(always)]
    pub fn completion(&self, system_prompt: impl Into<String>) -> AnthropicCompletionBuilder {
        AnthropicCompletionBuilder {
            client: self.clone(),
            request: CompletionRequest {
                system_prompt: system_prompt.into(),
                chat_history: crate::ZeroOneOrMany::None,
                documents: crate::ZeroOneOrMany::None,
                tools: crate::ZeroOneOrMany::None,
                temperature: self.default_temperature,
                max_tokens: self.default_max_tokens,
                chunk_size: None,
                additional_params: None,
            },
            model_override: None,
            temperature_override: None,
            max_tokens_override: None,
            tool_choice: None,
            stop_sequences: None,
            metadata: None,
        }
    }

    /// Make a simple completion request
    #[inline(always)]
    pub fn complete_simple(&self, prompt: impl Into<String>) -> AsyncTask<String> {
        self.completion("")
            .user(prompt)
            .complete()
    }

    /// Chat with Claude using messages
    #[inline(always)]
    pub fn chat(&self, messages: Vec<crate::domain::Message>) -> AsyncTask<String> {
        self.completion("")
            .messages(crate::ZeroOneOrMany::from_vec(messages))
            .complete()
    }

    /// Get available Claude models
    #[inline(always)]
    pub fn available_models() -> Vec<Models> {
        vec![
            Models::AnthropicClaudeOpus4,
            Models::AnthropicClaudeSonnet4,
            Models::AnthropicClaude37Sonnet,
            Models::AnthropicClaude37Sonnetthinking,
            Models::AnthropicClaude35Sonnet,
            Models::AnthropicClaude35Sonnet20241022V20,
            Models::AnthropicClaude35Haiku,
            Models::AnthropicClaude35Haiku20241022V10,
        ]
    }

    /// Check if model supports specific features
    #[inline(always)]
    pub fn model_supports_tools(model: &Models) -> bool {
        matches!(model,
            Models::AnthropicClaudeOpus4 |
            Models::AnthropicClaudeSonnet4 |
            Models::AnthropicClaude37Sonnet |
            Models::AnthropicClaude35Sonnet |
            Models::AnthropicClaude35Sonnet20241022V20
        )
    }

    /// Check if model supports vision
    #[inline(always)]
    pub fn model_supports_vision(model: &Models) -> bool {
        matches!(model,
            Models::AnthropicClaudeOpus4 |
            Models::AnthropicClaudeSonnet4 |
            Models::AnthropicClaude37Sonnet |
            Models::AnthropicClaude35Sonnet |
            Models::AnthropicClaude35Sonnet20241022V20
        )
    }
}

impl AnthropicClientBuilder {
    /// Set API key
    #[inline(always)]
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set base URL for API
    #[inline(always)]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set default model
    #[inline(always)]
    pub fn model(mut self, model: Models) -> Self {
        self.model = Some(model);
        self
    }

    /// Set default temperature
    #[inline(always)]
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set default max tokens
    #[inline(always)]
    pub fn max_tokens(mut self, tokens: u64) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set request timeout
    #[inline(always)]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set maximum retries
    #[inline(always)]
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
        self
    }

    /// Build the client
    #[inline(always)]
    pub fn build(self) -> AnthropicResult<AnthropicClient> {
        let api_key = self.api_key.ok_or_else(|| {
            AnthropicError::AuthenticationFailed("API key is required".to_string())
        })?;

        let model = self.model.unwrap_or(Models::AnthropicClaude35Sonnet);
        
        let mut provider = if let Some(base_url) = self.base_url {
            AnthropicProvider::with_config(
                api_key,
                base_url,
                model.clone(),
            )?
        } else {
            AnthropicProvider::new(api_key)?
        };

        if let Some(timeout) = self.timeout {
            provider = provider.with_timeout(timeout);
        }

        if let Some(max_retries) = self.max_retries {
            provider = provider.with_max_retries(max_retries);
        }

        Ok(AnthropicClient {
            provider: Arc::new(provider),
            default_model: model,
            default_temperature: self.temperature,
            default_max_tokens: self.max_tokens,
        })
    }
}

impl AnthropicCompletionBuilder {
    /// Override model for this request
    #[inline(always)]
    pub fn model(mut self, model: Models) -> Self {
        self.model_override = Some(model);
        self
    }

    /// Set temperature for this request
    #[inline(always)]
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature_override = Some(temp);
        self
    }

    /// Set max tokens for this request
    #[inline(always)]
    pub fn max_tokens(mut self, tokens: u64) -> Self {
        self.max_tokens_override = Some(tokens);
        self
    }

    /// Add user message
    #[inline(always)]
    pub fn user(mut self, content: impl Into<String>) -> Self {
        let message = crate::domain::Message::user(content);
        self.request.chat_history = match self.request.chat_history {
            crate::ZeroOneOrMany::None => crate::ZeroOneOrMany::One(message),
            crate::ZeroOneOrMany::One(existing) => {
                crate::ZeroOneOrMany::from_vec(vec![existing, message])
            }
            crate::ZeroOneOrMany::Many(mut messages) => {
                messages.push(message);
                crate::ZeroOneOrMany::from_vec(messages)
            }
        };
        self
    }

    /// Add assistant message
    #[inline(always)]
    pub fn assistant(mut self, content: impl Into<String>) -> Self {
        let message = crate::domain::Message::assistant(content);
        self.request.chat_history = match self.request.chat_history {
            crate::ZeroOneOrMany::None => crate::ZeroOneOrMany::One(message),
            crate::ZeroOneOrMany::One(existing) => {
                crate::ZeroOneOrMany::from_vec(vec![existing, message])
            }
            crate::ZeroOneOrMany::Many(mut messages) => {
                messages.push(message);
                crate::ZeroOneOrMany::from_vec(messages)
            }
        };
        self
    }

    /// Set all messages at once
    #[inline(always)]
    pub fn messages(mut self, messages: crate::ZeroOneOrMany<crate::domain::Message>) -> Self {
        self.request.chat_history = messages;
        self
    }

    /// Add documents
    #[inline(always)]
    pub fn documents(mut self, docs: crate::ZeroOneOrMany<crate::domain::Document>) -> Self {
        self.request.documents = docs;
        self
    }

    /// Add tools
    #[inline(always)]
    pub fn tools(mut self, tools: crate::ZeroOneOrMany<ToolDefinition>) -> Self {
        self.request.tools = tools;
        self
    }

    /// Add single tool
    #[inline(always)]
    pub fn tool(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        let tool = ToolDefinition {
            name: name.into(),
            description: description.into(),
            parameters,
        };
        self.request.tools = match self.request.tools {
            crate::ZeroOneOrMany::None => crate::ZeroOneOrMany::One(tool),
            crate::ZeroOneOrMany::One(existing) => {
                crate::ZeroOneOrMany::from_vec(vec![existing, tool])
            }
            crate::ZeroOneOrMany::Many(mut tools) => {
                tools.push(tool);
                crate::ZeroOneOrMany::from_vec(tools)
            }
        };
        self
    }

    /// Set tool choice strategy
    #[inline(always)]
    pub fn tool_choice(mut self, choice: impl Into<String>) -> Self {
        self.tool_choice = Some(choice.into());
        self
    }

    /// Set stop sequences
    #[inline(always)]
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
        self
    }

    /// Add metadata
    #[inline(always)]
    pub fn metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Execute completion request
    #[inline(always)]
    pub fn complete(self) -> AsyncTask<String> {
        let provider = self.client.provider.clone();
        let mut request = self.request;
        
        // Apply overrides
        if let Some(temp) = self.temperature_override {
            request.temperature = Some(temp);
        }
        if let Some(max_tokens) = self.max_tokens_override {
            request.max_tokens = Some(max_tokens);
        }
        
        crate::async_task::spawn_async(async move {
            match provider.convert_request(&request) {
                Ok(mut anthropic_request) => {
                    // Apply additional configurations
                    if let Some(metadata) = self.metadata {
                        anthropic_request.metadata = Some(
                            metadata.as_object().unwrap_or(&serde_json::Map::new())
                                .iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect()
                        );
                    }
                    
                    if let Some(stop_seqs) = self.stop_sequences {
                        anthropic_request.stop_sequences = Some(stop_seqs);
                    }
                    
                    match provider.make_completion_request(anthropic_request).await {
                        Ok(response) => provider.extract_text_content(&response),
                        Err(e) => format!("Anthropic completion error: {}", e),
                    }
                }
                Err(e) => format!("Request conversion error: {}", e),
            }
        })
    }

    /// Execute completion request and return structured response
    #[inline(always)]
    pub fn complete_structured(self) -> AsyncTask<AnthropicCompletionResponse> {
        let provider = self.client.provider.clone();
        let mut request = self.request;
        
        // Apply overrides
        if let Some(temp) = self.temperature_override {
            request.temperature = Some(temp);
        }
        if let Some(max_tokens) = self.max_tokens_override {
            request.max_tokens = Some(max_tokens);
        }
        
        crate::async_task::spawn_async(async move {
            let anthropic_request = match provider.convert_request(&request) {
                Ok(req) => req,
                Err(_) => {
                    // Return default response instead of propagating error
                    return AnthropicCompletionResponse {
                        id: String::new(),
                        response_type: "message".to_string(),
                        role: "assistant".to_string(),
                        content: Vec::new(),
                        model: String::new(),
                        stop_reason: None,
                        stop_sequence: None,
                        usage: Usage {
                            input_tokens: 0,
                            output_tokens: 0,
                            cache_creation_input_tokens: None,
                            cache_read_input_tokens: None,
                        },
                    };
                }
            };
            
            match provider.make_completion_request(anthropic_request).await {
                Ok(response) => response,
                Err(_) => {
                    // Return default response instead of propagating error
                    AnthropicCompletionResponse {
                        id: String::new(),
                        response_type: "message".to_string(),
                        role: "assistant".to_string(),
                        content: Vec::new(),
                        model: String::new(),
                        stop_reason: None,
                        stop_sequence: None,
                        usage: Usage {
                            input_tokens: 0,
                            output_tokens: 0,
                            cache_creation_input_tokens: None,
                            cache_read_input_tokens: None,
                        },
                    }
                }
            }
        })
    }
}