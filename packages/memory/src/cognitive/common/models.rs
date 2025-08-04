// src/cognitive/common/models.rs
//! Defines the Model and ModelType for completion provider interactions using HTTP3 + model-info architecture.

use std::collections::HashMap;
use std::env;

use fluent_ai_async::AsyncStream;
use fluent_ai_domain::{
    chat::{Message, MessageRole}, 
    completion::CompletionResponse,
    http::common::CommonUsage as TokenUsage};
use fluent_ai_http3::{Http3, HttpError, HttpClient, HttpConfig, HttpChunk};
use model_info::{DiscoveryProvider as Provider, ModelInfo, ModelInfoBuilder};
use model_info::providers::anthropic::AnthropicProvider;
use serde::{Deserialize, Serialize};
use serde_json::json;

// REMOVED: Local Message struct - using fluent_ai_domain::chat::Message instead

/// Core completion error type
#[derive(Debug, Clone, thiserror::Error)]
pub enum CompletionCoreError {
    #[error("Provider unavailable: {0}")]
    ProviderUnavailable(String),
    #[error("Request failed: {0}")]
    RequestFailed(String),
    #[error("Parse error: {0}")]
    ParseError(String),
}

/// Model type for completion provider interactions
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
            ModelType::Claude3Opus => "claude-3-opus-20240229",
            ModelType::Claude3Sonnet => "claude-3-sonnet-20240229",
            ModelType::Claude3Haiku => "claude-3-haiku-20240307",
            ModelType::GeminiPro => "gemini-pro",
            ModelType::Mixtral8x7B => "mixtral-8x7b-instruct",
            ModelType::Llama270B => "llama-2-70b-chat",
            ModelType::Llama3 => "llama-3",
        }
    }

    pub fn provider_name(&self) -> &'static str {
        match self {
            ModelType::Gpt35Turbo | ModelType::Gpt4 | ModelType::Gpt4O | ModelType::Gpt4Turbo => {
                "openai"
            }
            ModelType::Claude3Opus | ModelType::Claude3Sonnet | ModelType::Claude3Haiku => {
                "anthropic"
            }
            ModelType::GeminiPro => "google",
            ModelType::Mixtral8x7B | ModelType::Llama270B | ModelType::Llama3 => "huggingface",
        }
    }

    pub fn to_provider(&self) -> Provider {
        match self {
            ModelType::Gpt35Turbo | ModelType::Gpt4 | ModelType::Gpt4O | ModelType::Gpt4Turbo => {
                Provider::OpenAI
            }
            ModelType::Claude3Opus | ModelType::Claude3Sonnet | ModelType::Claude3Haiku => {
                Provider::Anthropic
            }
            ModelType::GeminiPro => Provider::Google,
            ModelType::Mixtral8x7B | ModelType::Llama270B | ModelType::Llama3 => Provider::HuggingFace,
        }
    }
    
    pub fn from_name_and_provider(name: &str, provider: Provider) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        match (provider, name) {
            (Provider::OpenAI, "gpt-3.5-turbo") => Ok(ModelType::Gpt35Turbo),
            (Provider::OpenAI, "gpt-4") => Ok(ModelType::Gpt4),
            (Provider::OpenAI, "gpt-4o") => Ok(ModelType::Gpt4O),
            (Provider::OpenAI, "gpt-4-turbo") => Ok(ModelType::Gpt4Turbo),
            (Provider::Anthropic, "claude-3-opus-20240229") => Ok(ModelType::Claude3Opus),
            (Provider::Anthropic, "claude-3-sonnet-20240229") => Ok(ModelType::Claude3Sonnet),
            (Provider::Anthropic, "claude-3-haiku-20240307") => Ok(ModelType::Claude3Haiku),
            _ => Err(format!("Unsupported model: {} for provider: {:?}", name, provider).into()),
        }
    }
}

/// Model wrapper using HTTP3 + model-info architecture
#[derive(Debug, Clone)]
pub struct Model {
    provider: Provider,
    model_info: ModelInfo,
    api_key: String,
    http_client: HttpClient}

impl Model {
    pub async fn create(
        model_name: &'static str,
        provider: Provider,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let api_key_env = match provider {
            Provider::OpenAI => "OPENAI_API_KEY",
            Provider::Anthropic => "ANTHROPIC_API_KEY",  
            _ => return Err(format!("Provider {:?} is not yet implemented", provider).into()),
        };
        
        // Alternative constructor for backward compatibility with ModelType
        Self::create_with_model_type(ModelType::from_name_and_provider(model_name, provider)?).await
    }
    
    pub async fn create_with_model_type(
        model_type: ModelType,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let provider = match model_type {
            ModelType::Gpt35Turbo | ModelType::Gpt4 | ModelType::Gpt4O | ModelType::Gpt4Turbo => {
                Provider::OpenAI
            }
            ModelType::Claude3Opus | ModelType::Claude3Sonnet | ModelType::Claude3Haiku => {
                Provider::Anthropic
            }
            _ => return Err(format!("Model type {:?} is not yet implemented", model_type).into()),
        };
        
        let api_key_env = match provider {
            Provider::OpenAI => "OPENAI_API_KEY",
            Provider::Anthropic => "ANTHROPIC_API_KEY",  
            _ => return Err(format!("Provider {:?} is not yet implemented", provider).into()),
        };

        let api_key = env::var(api_key_env)
            .map_err(|_| format!("{} environment variable not set", api_key_env))?;

        // Create HTTP3 client optimized for AI operations
        let http_client = HttpClient::with_config(HttpConfig::ai_optimized())
            .map_err(|e| format!("Failed to create HTTP3 client: {}", e))?;

        // Create model info using model-info package
        let model_info = ModelInfoBuilder::new()
            .provider_name(model_type.provider_name())
            .name(model_type.display_name())
            .with_streaming(true) // All models support streaming
            .build()
            .map_err(|e| format!("Failed to create model info: {}", e))?;

        Ok(Self {
            provider,
            model_info,
            api_key,
            http_client})
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
    
    pub fn available_models() -> Vec<(&'static str, Provider)> {
        vec![
            ("gpt-3.5-turbo", Provider::OpenAI),
            ("gpt-4", Provider::OpenAI),
            ("gpt-4o", Provider::OpenAI),
            ("claude-3-opus-20240229", Provider::Anthropic),
            ("claude-3-sonnet-20240229", Provider::Anthropic),
            ("claude-3-haiku-20240307", Provider::Anthropic),
        ]
    }

    /// Complete a request using HTTP3 + model-info architecture
    pub async fn complete(
        &self,
        messages: Vec<Message>,
    ) -> Result<CompletionResponse<'static>, Box<dyn std::error::Error + Send + Sync>> {
        let base_url = self.provider.default_base_url();
        let endpoint = match self.provider {
            Provider::OpenAI | Provider::Anthropic => "/chat/completions",
            _ => return Err("Provider not implemented".into()),
        };
        
        let url = format!("{}{}", base_url, endpoint);
        
        // Build request payload based on provider
        let request_body = match self.provider {
            Provider::OpenAI => {
                json!({
                    "model": self.model_info.name,
                    "messages": messages.iter().map(|m| {
                        json!({"role": m.role, "content": m.content})
                    }).collect::<Vec<_>>(),
                    "stream": false
                })
            }
            Provider::Anthropic => {
                let system_msgs: Vec<_> = messages.iter().filter(|m| m.role == "system").collect();
                let user_msgs: Vec<_> = messages.iter().filter(|m| m.role != "system").collect();
                
                json!({
                    "model": self.model_info.name,
                    "max_tokens": 4096,
                    "messages": user_msgs.iter().map(|m| {
                        json!({"role": m.role, "content": m.content})
                    }).collect::<Vec<_>>(),
                    "system": system_msgs.first().map(|m| &m.content).unwrap_or(""),
                })
            }
            _ => return Err("Provider not implemented".into()),
        };

        // Make HTTP3 request
        let response = Http3::json()
            .api_key(&self.api_key)
            .body(&request_body)
            .post(&url)
            .collect::<serde_json::Value>()
            .await
            .map_err(|e| format!("HTTP3 request failed: {}", e))?;

        // Parse response based on provider
        let content = match self.provider {
            Provider::OpenAI => {
                response["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .to_string()
            }
            Provider::Anthropic => {
                response["content"][0]["text"]
                    .as_str()
                    .unwrap_or("")
                    .to_string()
            }
            _ => return Err("Provider not implemented".into()),
        };

        // Extract usage information if available
        let usage = if let Some(usage_data) = response.get("usage") {
            TokenUsage::new(
                usage_data["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                usage_data["completion_tokens"].as_u64().unwrap_or(0) as u32,
            )
        } else {
            TokenUsage::new(0, 0)
        };

        // Create completion response using domain types
        let mut completion_response = CompletionResponse::text(content.clone());
        completion_response.set_token_usage(usage);
        
        Ok(completion_response)
    }

    /// Simple prompt interface using HTTP3 + model-info
    pub async fn prompt(
        &self,
        prompt: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Convert prompt to messages format
        let messages = vec![Message {
            role: MessageRole::User,
            content: prompt.to_string(),
            id: Some(uuid::Uuid::new_v4().to_string()),
            timestamp: Some(chrono::Utc::now().timestamp() as u64),
        }];

        // Use complete method and extract text
        let response = self.complete(messages).await?;
        Ok(response.text().unwrap_or_default().to_string())
    }

    /// Stream completions using AsyncStream
    pub fn stream_complete(&self, messages: Vec<Message>) -> AsyncStream<String> {
        let model_info = self.model_info.clone();
        let provider = self.provider.clone();
        let api_key = self.api_key.clone();
        let http_client = self.http_client.clone();
        
        AsyncStream::with_channel(move |sender| {
            Box::pin(async move {
                let base_url = provider.default_base_url();
                let endpoint = match provider {
                    Provider::OpenAI | Provider::Anthropic => "/chat/completions",
                    _ => {
                        eprintln!("Provider not implemented for streaming");
                        return Ok(());
                    }
                };
                
                let url = format!("{}{}", base_url, endpoint);
                
                // Build streaming request payload
                let request_body = match provider {
                    Provider::OpenAI => {
                        json!({
                            "model": model_info.name,
                            "messages": messages.iter().map(|m| {
                                json!({"role": m.role, "content": m.content})
                            }).collect::<Vec<_>>(),
                            "stream": true
                        })
                    }
                    Provider::Anthropic => {
                        let system_msgs: Vec<_> = messages.iter().filter(|m| m.role == "system").collect();
                        let user_msgs: Vec<_> = messages.iter().filter(|m| m.role != "system").collect();
                        
                        json!({
                            "model": model_info.name,
                            "max_tokens": 4096,
                            "messages": user_msgs.iter().map(|m| {
                                json!({"role": m.role, "content": m.content})
                            }).collect::<Vec<_>>(),
                            "system": system_msgs.first().map(|m| &m.content).unwrap_or(""),
                            "stream": true
                        })
                    }
                    _ => {
                        eprintln!("Provider not implemented for streaming");
                        return Ok(());
                    }
                };

                // Stream HTTP3 response
                let mut stream = Http3::json()
                    .api_key(&api_key)
                    .body(&request_body)
                    .post(&url)
                    .stream();

                while let Some(chunk_result) = stream.next().await {
                    match chunk_result {
                        Ok(HttpChunk::Body(chunk_bytes)) => {
                            // Parse SSE chunk and extract content
                            if let Some(content) = Self::parse_streaming_chunk(&chunk_bytes, &provider) {
                                let _ = sender.send(content).await;
                            }
                        }
                        Ok(HttpChunk::Head(_, _)) => {
                            // Headers received, continue
                            continue;
                        }
                        Err(e) => {
                            eprintln!("Stream error: {}", e);
                            break;
                        }
                    }
                }
                
                Ok(())
            })
        })
    }

    /// Parse streaming chunk based on provider format
    fn parse_streaming_chunk(chunk_bytes: &[u8], provider: &Provider) -> Option<String> {
        let chunk_str = std::str::from_utf8(chunk_bytes).ok()?;
        
        // Handle Server-Sent Events format
        for line in chunk_str.lines() {
            if line.starts_with("data: ") {
                let data = &line[6..];
                if data == "[DONE]" {
                    break;
                }
                
                if let Ok(json_data) = serde_json::from_str::<serde_json::Value>(data) {
                    return match provider {
                        Provider::OpenAI => {
                            json_data["choices"][0]["delta"]["content"]
                                .as_str()
                                .map(|s| s.to_string())
                        }
                        Provider::Anthropic => {
                            json_data["delta"]["text"]
                                .as_str()
                                .map(|s| s.to_string())
                        }
                        _ => None,
                    };
                }
            }
        }
        
        None
    }

    /// Get the model info
    pub fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    /// Get the model display name
    pub fn display_name(&self) -> &'static str {
        self.model_info.name
    }

    /// Get the provider name
    pub fn provider_name(&self) -> &'static str {
        self.provider.provider_name()
    }
}

/// Completion request wrapper for backward compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub messages: Vec<Message>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>}

impl CompletionRequest {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            temperature: None,
            max_tokens: None}
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

// REMOVED: Usage struct - using fluent_ai_domain::http::common::CommonUsage (TokenUsage) instead
