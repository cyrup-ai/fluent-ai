//! Anthropic Claude LLM provider implementation

use std::sync::Arc;

use fluent_ai_http3::{HttpClient, HttpConfig};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::llm::{LLMError, LLMProvider, PendingCompletion, PendingEmbedding};

/// Anthropic provider
pub struct AnthropicProvider {
    client: Arc<HttpClient>,
    api_key: String,
    model: String,
    api_base: String,
}

impl std::fmt::Debug for AnthropicProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicProvider")
            .field("model", &self.model)
            .field("api_base", &self.api_base)
            .field("api_key", &"[REDACTED]")
            .finish()
    }
}

impl AnthropicProvider {
    /// Create a new Anthropic provider
    pub fn new(api_key: String, model: Option<String>) -> Result<Self, LLMError> {
        let client = HttpClient::with_config(HttpConfig::ai_optimized()).map_err(|e| {
            LLMError::InitializationError(format!("Failed to create HTTP3 client: {}", e))
        })?;

        Ok(Self {
            client: Arc::new(client),
            api_key,
            model: model.unwrap_or_else(|| "claude-3-opus-20240229".to_string()),
            api_base: "https://api.anthropic.com/v1".to_string(),
        })
    }

    /// Set custom API base URL
    pub fn with_api_base(mut self, api_base: String) -> Self {
        self.api_base = api_base;
        self
    }
}

impl LLMProvider for AnthropicProvider {
    fn complete(&self, prompt: &str) -> PendingCompletion {
        let (tx, rx) = oneshot::channel();

        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let model = self.model.clone();
        let api_base = self.api_base.clone();
        let prompt = prompt.to_string();

        tokio::spawn(async move {
            let request = CompletionRequest {
                model,
                messages: vec![Message {
                    role: "user".to_string(),
                    content: prompt,
                }],
                max_tokens: 1024,
                temperature: 0.7,
            };

            let result = async {
                let request_body = serde_json::to_vec(&request).map_err(|e| {
                    LLMError::InvalidRequest(format!("Failed to serialize request: {}", e))
                })?;

                let http_request = client
                    .post(&format!("{api_base}/messages"))
                    .header("x-api-key", api_key)
                    .header("anthropic-version", "2023-06-01")
                    .header("content-type", "application/json")
                    .with_body(request_body);

                let response = client
                    .send(http_request)
                    .await
                    .map_err(|e| LLMError::NetworkError(format!("HTTP request failed: {}", e)))?;

                if response.status().is_success() {
                    let response_body = response.bytes().map_err(|e| {
                        LLMError::NetworkError(format!("Failed to read response body: {}", e))
                    })?;

                    let completion: CompletionResponse = serde_json::from_slice(&response_body)
                        .map_err(|e| {
                            LLMError::InvalidResponse(format!("Failed to parse response: {}", e))
                        })?;

                    completion
                        .content
                        .first()
                        .and_then(|content| {
                            if content.content_type == "text" {
                                Some(content.text.clone())
                            } else {
                                None
                            }
                        })
                        .ok_or_else(|| {
                            LLMError::InvalidResponse("No text content in response".to_string())
                        })
                } else if response.status().as_u16() == 401 {
                    Err(LLMError::AuthenticationFailed(
                        "Invalid API key".to_string(),
                    ))
                } else if response.status().as_u16() == 429 {
                    Err(LLMError::RateLimitExceeded)
                } else {
                    let error_body = response
                        .bytes()
                        .map(|body| String::from_utf8_lossy(&body).into_owned())
                        .unwrap_or_else(|_| "Unknown error".to_string());
                    Err(LLMError::ApiError(error_body))
                }
            }
            .await;

            let _ = tx.send(result);
        });

        PendingCompletion::new(rx)
    }

    fn embed(&self, _text: &str) -> PendingEmbedding {
        let (tx, rx) = oneshot::channel();

        // Anthropic doesn't provide embeddings API
        tokio::spawn(async move {
            let _ = tx.send(Err(LLMError::ApiError(
                "Anthropic does not provide embedding API".to_string(),
            )));
        });

        PendingEmbedding::new(rx)
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// Request/Response types

#[derive(Serialize)]
struct CompletionRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct CompletionResponse {
    content: Vec<Content>,
}

#[derive(Deserialize)]
struct Content {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}
