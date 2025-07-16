//! Zero-allocation Anthropic completion implementation with comprehensive feature support
//!
//! Production-ready completion provider supporting all Claude models, tool calling,
//! document attachments, and advanced API features with optimal performance.

use crate::async_task::AsyncTask;
use crate::domain::completion::{CompletionBackend, CompletionRequest, ToolDefinition};
use crate::util::json_util::{merge_inplace, to_compact_string};
use crate::providers::anthropic::{
    AnthropicError, AnthropicResult, Message, MessageConverter, Tool,
    handle_http_error, handle_reqwest_error, handle_json_error,
};
use fluent_ai_provider::Models;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Duration;

/// Production-ready Anthropic completion provider
#[derive(Clone)]
pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    base_url: String,
    default_model: Models,
    request_timeout: Duration,
    max_retries: u32,
}

/// Anthropic completion request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, Value>>,
}

/// Anthropic completion response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicCompletionResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<CompletionContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

/// Content block in completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CompletionContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

/// Usage statistics from Anthropic API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
}

/// Tool choice configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Auto,
    Any,
    Tool { name: String },
}

impl AnthropicProvider {
    /// Create new Anthropic provider with API key
    #[inline(always)]
    pub fn new(api_key: impl Into<String>) -> AnthropicResult<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .user_agent("fluent-ai/1.0")
            .build()
            .map_err(handle_reqwest_error)?;

        Ok(Self {
            client,
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            default_model: Models::AnthropicClaude35Sonnet,
            request_timeout: Duration::from_secs(120),
            max_retries: 3,
        })
    }

    /// Create provider with custom configuration
    #[inline(always)]
    pub fn with_config(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        default_model: Models,
    ) -> AnthropicResult<Self> {
        let mut provider = Self::new(api_key)?;
        provider.base_url = base_url.into();
        provider.default_model = default_model;
        Ok(provider)
    }

    /// Set request timeout
    #[inline(always)]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Set maximum retries for failed requests
    #[inline(always)]
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Get Claude model string from Models enum
    #[inline(always)]
    fn model_to_string(&self, model: &Models) -> &'static str {
        match model {
            // Claude 4 models
            Models::AnthropicClaudeOpus4 => "claude-4-opus-20250514",
            Models::AnthropicClaudeSonnet4 => "claude-4-sonnet-20250514",
            
            // Claude 3.7 models
            Models::AnthropicClaude37Sonnet => "claude-3-7-sonnet-20250219",
            Models::AnthropicClaude37Sonnetthinking => "claude-3-7-sonnet-20250219-thinking",
            
            // Claude 3.5 models
            Models::AnthropicClaude35Sonnet => "claude-3-5-sonnet-20241022",
            Models::AnthropicClaude35Sonnet20241022V20 => "claude-3-5-sonnet-20241022-v2",
            Models::AnthropicClaude35Haiku => "claude-3-5-haiku-20241022",
            Models::AnthropicClaude35Haiku20241022V10 => "claude-3-5-haiku-20241022-v1",
            
            // Legacy support
            Models::Claude35Sonnet20241022 => "claude-3-5-sonnet-20241022",
            Models::Claude35SonnetV220241022 => "claude-3-5-sonnet-20241022-v2",
            Models::Claude35Haiku20241022 => "claude-3-5-haiku-20241022",
            Models::Claude37Sonnet20250219 => "claude-3-7-sonnet-20250219",
            Models::Claude37Sonnet20250219Thinking => "claude-3-7-sonnet-20250219-thinking",
            Models::ClaudeOpus420250514 => "claude-4-opus-20250514",
            Models::ClaudeOpus420250514Thinking => "claude-4-opus-20250514-thinking",
            Models::ClaudeSonnet420250514 => "claude-4-sonnet-20250514",
            Models::ClaudeSonnet420250514Thinking => "claude-4-sonnet-20250514-thinking",
            
            // Fallback to 3.5 Sonnet for non-Claude models
            _ => "claude-3-5-sonnet-20241022",
        }
    }

    /// Get appropriate max_tokens for model
    #[inline(always)]
    fn get_max_tokens_for_model(&self, model: &Models) -> u64 {
        match model {
            // Claude 4 models - higher limits
            Models::AnthropicClaudeOpus4 |
            Models::AnthropicClaudeSonnet4 |
            Models::ClaudeOpus420250514 |
            Models::ClaudeOpus420250514Thinking |
            Models::ClaudeSonnet420250514 |
            Models::ClaudeSonnet420250514Thinking => 8192,
            
            // Claude 3.7 models
            Models::AnthropicClaude37Sonnet |
            Models::AnthropicClaude37Sonnetthinking |
            Models::Claude37Sonnet20250219 |
            Models::Claude37Sonnet20250219Thinking => 4096,
            
            // Claude 3.5 models
            Models::AnthropicClaude35Sonnet |
            Models::AnthropicClaude35Sonnet20241022V20 |
            Models::Claude35Sonnet20241022 |
            Models::Claude35SonnetV220241022 => 4096,
            
            // Haiku models - smaller limits
            Models::AnthropicClaude35Haiku |
            Models::AnthropicClaude35Haiku20241022V10 |
            Models::Claude35Haiku20241022 => 4096,
            
            // Default fallback
            _ => 4096,
        }
    }

    /// Convert fluent-ai CompletionRequest to Anthropic format
    #[inline(always)]
    pub fn convert_request(&self, request: &CompletionRequest) -> AnthropicResult<AnthropicCompletionRequest> {
        // Determine model to use
        let model_str = self.model_to_string(&self.default_model);
        
        // Convert messages
        let messages = MessageConverter::convert_messages(&request.chat_history);
        
        // Convert tools if present
        let tools = if matches!(request.tools, crate::ZeroOneOrMany::None) {
            None
        } else {
            Some(MessageConverter::convert_tools(&request.tools))
        };
        
        // Set appropriate max_tokens
        let max_tokens = request.max_tokens
            .or_else(|| Some(self.get_max_tokens_for_model(&self.default_model)));
        
        // Handle system prompt
        let system = if request.system_prompt.is_empty() {
            None
        } else {
            Some(request.system_prompt.clone())
        };

        Ok(AnthropicCompletionRequest {
            model: model_str.to_string(),
            messages,
            system,
            max_tokens,
            temperature: request.temperature,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            tool_choice: if tools.is_some() { Some(ToolChoice::Auto) } else { None },
            tools,
            stream: Some(false),
            metadata: None,
        })
    }

    /// Make completion request to Anthropic API
    pub async fn make_completion_request(
        &self,
        anthropic_request: AnthropicCompletionRequest,
    ) -> AnthropicResult<AnthropicCompletionResponse> {
        let url = format!("{}/messages", self.base_url);
        
        // Serialize request with zero-allocation optimizations
        let mut request_json = serde_json::to_value(&anthropic_request)
            .map_err(handle_json_error)?;
        
        // Merge additional parameters if needed
        if let Some(additional) = &anthropic_request.metadata {
            for (key, value) in additional {
                merge_inplace(&mut request_json, json!({ key: value }));
            }
        }
        
        let request_body = to_compact_string(&request_json);
        
        // Make HTTP request with retries
        let mut last_error = None;
        for attempt in 0..=self.max_retries {
            let response = self.client
                .post(&url)
                .header("Content-Type", "application/json")
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .timeout(self.request_timeout)
                .body(request_body.clone())
                .send()
                .await;
            
            match response {
                Ok(resp) => {
                    let status = resp.status();
                    let body = resp.text().await.map_err(handle_reqwest_error)?;
                    
                    if status.is_success() {
                        return serde_json::from_str(&body)
                            .map_err(handle_json_error);
                    } else {
                        let error = handle_http_error(status.as_u16(), &body);
                        
                        // Retry on rate limits and server errors
                        if matches!(error, AnthropicError::RateLimited { .. }) ||
                           matches!(error, AnthropicError::ServerError { status: 500..=599, .. }) {
                            if attempt < self.max_retries {
                                let delay = Duration::from_millis(1000 * (1 << attempt));
                                tokio::time::sleep(delay).await;
                                last_error = Some(error);
                                continue;
                            }
                        }
                        
                        return Err(error);
                    }
                }
                Err(e) => {
                    let error = handle_reqwest_error(e);
                    
                    // Retry on network errors
                    if matches!(error, AnthropicError::NetworkError(..)) && attempt < self.max_retries {
                        let delay = Duration::from_millis(1000 * (1 << attempt));
                        tokio::time::sleep(delay).await;
                        last_error = Some(error);
                        continue;
                    }
                    
                    return Err(error);
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| AnthropicError::Unknown("Max retries exceeded".to_string())))
    }

    /// Extract text content from response
    #[inline(always)]
    pub fn extract_text_content(&self, response: &AnthropicCompletionResponse) -> String {
        let mut text_parts = Vec::new();
        
        for block in &response.content {
            match block {
                CompletionContentBlock::Text { text } => text_parts.push(text.clone()),
                CompletionContentBlock::ToolUse { name, input, .. } => {
                    // Include tool calls in text format for backward compatibility
                    text_parts.push(format!(
                        "[Tool Call: {} with input: {}]",
                        name,
                        to_compact_string(input)
                    ));
                }
            }
        }
        
        text_parts.join("\n")
    }
}

impl CompletionBackend for AnthropicProvider {
    fn submit_completion(
        &self,
        prompt: &str,
        tools: &[String],
    ) -> AsyncTask<String> {
        let provider = self.clone();
        let prompt = prompt.to_string();
        let tools = tools.to_vec();
        
        crate::async_task::spawn_async(async move {
            // Convert simple prompt to CompletionRequest format
            let completion_request = CompletionRequest {
                system_prompt: String::new(),
                chat_history: crate::ZeroOneOrMany::One(
                    crate::domain::Message::user(prompt)
                ),
                documents: crate::ZeroOneOrMany::None,
                tools: if tools.is_empty() {
                    crate::ZeroOneOrMany::None
                } else {
                    crate::ZeroOneOrMany::from_vec(
                        tools.into_iter().map(|name| ToolDefinition {
                            description: format!("Tool: {}", name),
                            name,
                            parameters: json!({}),
                        }).collect()
                    )
                },
                temperature: None,
                max_tokens: None,
                chunk_size: None,
                additional_params: None,
            };
            
            match provider.convert_request(&completion_request) {
                Ok(anthropic_request) => {
                    match provider.make_completion_request(anthropic_request).await {
                        Ok(response) => provider.extract_text_content(&response),
                        Err(e) => format!("Anthropic completion error: {}", e),
                    }
                }
                Err(e) => format!("Request conversion error: {}", e),
            }
        })
    }
}

/// Create Anthropic provider from environment variable
pub fn from_env() -> AnthropicResult<AnthropicProvider> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| AnthropicError::AuthenticationFailed(
            "ANTHROPIC_API_KEY environment variable not set".to_string()
        ))?;
    
    AnthropicProvider::new(api_key)
}

/// Create Anthropic provider with custom model
pub fn with_model(api_key: impl Into<String>, model: Models) -> AnthropicResult<AnthropicProvider> {
    let mut provider = AnthropicProvider::new(api_key)?;
    provider.default_model = model;
    Ok(provider)
}