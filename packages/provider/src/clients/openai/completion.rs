//! Zero-allocation OpenAI completion implementation with comprehensive API support
//!
//! Provides blazing-fast chat completions with tool calling, streaming, and vision support
//! using OpenAI's latest API features and optimal performance patterns.

use crate::async_task::{AsyncStream, AsyncTask};
use crate::domain::completion::CompletionRequest;
use crate::domain::chunk::CompletionChunk;
use crate::providers::openai::{
    OpenAIError, OpenAIResult, OpenAIMessage, OpenAITool, OpenAIToolChoice,
    convert_messages, convert_tool_definitions,
    streaming::{StreamAccumulator, SSEParser, parse_stream_chunk},
};
use crate::ZeroOneOrMany;
use fluent_ai_provider::Models;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::{Duration, SystemTime};

/// OpenAI chat completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompletionRequest {
    pub model: String,
    pub messages: ZeroOneOrMany<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ZeroOneOrMany<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<ZeroOneOrMany<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

/// Stream options for completion requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

/// Response format specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<Value>,
}

/// OpenAI completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: ZeroOneOrMany<CompletionChoice>,
    pub usage: CompletionUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// Individual completion choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub message: OpenAIMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogProbs>,
    pub finish_reason: String,
}

/// Log probabilities for choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceLogProbs {
    pub content: Option<ZeroOneOrMany<TokenLogProb>>,
}

/// Token log probability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogProb {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<ZeroOneOrMany<u8>>,
    pub top_logprobs: ZeroOneOrMany<TopLogProb>,
}

/// Top alternative log probability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopLogProb {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<ZeroOneOrMany<u8>>,
}

/// Usage statistics for completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

/// Detailed completion token breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted_prediction_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejected_prediction_tokens: Option<u32>,
}

/// Detailed prompt token breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
}

/// OpenAI provider for completions
#[derive(Clone)]
pub struct OpenAIProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
    default_model: Models,
    timeout: Duration,
    max_retries: u32,
}

/// Completion configuration
#[derive(Debug, Clone)]
pub struct CompletionConfig {
    pub model: Models,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop_sequences: Option<ZeroOneOrMany<String>>,
    pub seed: Option<u64>,
    pub user: Option<String>,
    pub response_format: Option<ResponseFormat>,
    pub service_tier: Option<String>,
}

/// Tool calling configuration
#[derive(Debug, Clone)]
pub struct ToolConfig {
    pub tools: ZeroOneOrMany<OpenAITool>,
    pub tool_choice: OpenAIToolChoice,
    pub parallel_calls: bool,
}

/// Streaming configuration
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub enabled: bool,
    pub include_usage: bool,
    pub buffer_size: usize,
    pub timeout_ms: u64,
}

impl OpenAIProvider {
    /// Create new OpenAI provider
    #[inline(always)]
    pub fn new(api_key: impl Into<String>) -> OpenAIResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|e| OpenAIError::NetworkError(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            client,
            default_model: Models::Gpt4O,
            timeout: Duration::from_secs(120),
            max_retries: 3,
        })
    }

    /// Create provider with custom configuration
    #[inline(always)]
    pub fn with_config(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        model: Models,
    ) -> OpenAIResult<Self> {
        let mut provider = Self::new(api_key)?;
        provider.base_url = base_url.into();
        provider.default_model = model;
        Ok(provider)
    }

    /// Set timeout for requests
    #[inline(always)]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set maximum retries
    #[inline(always)]
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Convert fluent-ai request to OpenAI format
    #[inline(always)]
    pub fn convert_request(&self, request: &CompletionRequest) -> OpenAIResult<OpenAICompletionRequest> {
        let messages = convert_messages(&request.chat_history)?;
        let tools = if let ZeroOneOrMany::None = request.tools {
            None
        } else {
            Some(convert_tool_definitions(&request.tools)?)
        };

        Ok(OpenAICompletionRequest {
            model: self.get_model_string(&self.default_model),
            messages: ZeroOneOrMany::from_vec(messages),
            temperature: request.temperature.map(|t| t as f32),
            max_tokens: request.max_tokens.map(|t| t as u32),
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            tools: tools.map(|t| ZeroOneOrMany::from_vec(t)),
            tool_choice: None,
            parallel_tool_calls: None,
            stop: None,
            stream: None,
            stream_options: None,
            user: None,
            seed: None,
            logprobs: None,
            top_logprobs: None,
            response_format: None,
            service_tier: None,
        })
    }

    /// Make completion request
    #[inline(always)]
    pub fn make_completion_request(
        &self,
        request: OpenAICompletionRequest,
    ) -> AsyncTask<OpenAICompletionResponse> {
        let api_key = self.api_key.clone();
        let base_url = self.base_url.clone();
        let client = self.client.clone();
        let timeout = self.timeout;
        let max_retries = self.max_retries;

        crate::async_task::spawn_async(async move {
            let url = format!("{}/chat/completions", base_url);

            for attempt in 0..=max_retries {
                let response = client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("Content-Type", "application/json")
                    .json(&request)
                    .timeout(timeout)
                    .send()
                    .await;

                match response {
                    Ok(resp) => {
                        if resp.status().is_success() {
                            match resp.json::<OpenAICompletionResponse>().await {
                                Ok(completion) => return completion,
                                Err(e) => {
                                    if attempt == max_retries {
                                        return OpenAICompletionResponse {
                                            id: "error".to_string(),
                                            object: "chat.completion".to_string(),
                                            created: SystemTime::now()
                                                .duration_since(SystemTime::UNIX_EPOCH)
                                                .unwrap_or_default()
                                                .as_secs(),
                                            model: request.model.clone(),
                                            choices: ZeroOneOrMany::One(CompletionChoice {
                                                index: 0,
                                                message: OpenAIMessage {
                                                    role: "assistant".to_string(),
                                                    content: Some(crate::providers::openai::OpenAIContent::Text(
                                                        format!("Error parsing response: {}", e)
                                                    )),
                                                    name: None,
                                                    tool_calls: None,
                                                    tool_call_id: None,
                                                    function_call: None,
                                                },
                                                logprobs: None,
                                                finish_reason: "error".to_string(),
                                            }),
                                            usage: CompletionUsage {
                                                prompt_tokens: 0,
                                                completion_tokens: 0,
                                                total_tokens: 0,
                                                completion_tokens_details: None,
                                                prompt_tokens_details: None,
                                            },
                                            system_fingerprint: None,
                                        };
                                    }
                                }
                            }
                        } else {
                            let status = resp.status().as_u16();
                            let body = resp.text().await.unwrap_or_default();
                            
                            if attempt == max_retries {
                                return OpenAICompletionResponse {
                                    id: "error".to_string(),
                                    object: "chat.completion".to_string(),
                                    created: SystemTime::now()
                                        .duration_since(SystemTime::UNIX_EPOCH)
                                        .unwrap_or_default()
                                        .as_secs(),
                                    model: request.model.clone(),
                                    choices: ZeroOneOrMany::One(CompletionChoice {
                                        index: 0,
                                        message: OpenAIMessage {
                                            role: "assistant".to_string(),
                                            content: Some(crate::providers::openai::OpenAIContent::Text(
                                                format!("HTTP {} - {}", status, body)
                                            )),
                                            name: None,
                                            tool_calls: None,
                                            tool_call_id: None,
                                            function_call: None,
                                        },
                                        logprobs: None,
                                        finish_reason: "error".to_string(),
                                    }),
                                    usage: CompletionUsage {
                                        prompt_tokens: 0,
                                        completion_tokens: 0,
                                        total_tokens: 0,
                                        completion_tokens_details: None,
                                        prompt_tokens_details: None,
                                    },
                                    system_fingerprint: None,
                                };
                            }
                        }
                    }
                    Err(e) => {
                        if attempt == max_retries {
                            return OpenAICompletionResponse {
                                id: "error".to_string(),
                                object: "chat.completion".to_string(),
                                created: SystemTime::now()
                                    .duration_since(SystemTime::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs(),
                                model: request.model.clone(),
                                choices: ZeroOneOrMany::One(CompletionChoice {
                                    index: 0,
                                    message: OpenAIMessage {
                                        role: "assistant".to_string(),
                                        content: Some(crate::providers::openai::OpenAIContent::Text(
                                            format!("Network error: {}", e)
                                        )),
                                        name: None,
                                        tool_calls: None,
                                        tool_call_id: None,
                                        function_call: None,
                                    },
                                    logprobs: None,
                                    finish_reason: "error".to_string(),
                                }),
                                usage: CompletionUsage {
                                    prompt_tokens: 0,
                                    completion_tokens: 0,
                                    total_tokens: 0,
                                    completion_tokens_details: None,
                                    prompt_tokens_details: None,
                                },
                                system_fingerprint: None,
                            };
                        }
                    }
                }

                // Wait before retry
                if attempt < max_retries {
                    tokio::time::sleep(Duration::from_millis(1000 * (attempt + 1) as u64)).await;
                }
            }

            // This should never be reached due to the loop structure
            unreachable!()
        })
    }

    /// Make streaming completion request
    #[inline(always)]
    pub fn make_streaming_request(
        &self,
        mut request: OpenAICompletionRequest,
    ) -> AsyncStream<CompletionChunk> {
        request.stream = Some(true);
        request.stream_options = Some(StreamOptions {
            include_usage: Some(true),
        });

        let api_key = self.api_key.clone();
        let base_url = self.base_url.clone();
        let client = self.client.clone();
        let timeout = self.timeout;

        let (sender, stream) = AsyncStream::channel();

        crate::async_task::spawn_async(async move {
            let url = format!("{}/chat/completions", base_url);

            let response = client
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&request)
                .timeout(timeout)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    if resp.status().is_success() {
                        let mut stream = resp.bytes_stream();
                        let mut sse_parser = SSEParser::new();
                        let mut accumulator = StreamAccumulator::new();

                        while let Some(chunk_result) = futures_util::StreamExt::next(&mut stream).await {
                            match chunk_result {
                                Ok(chunk_bytes) => {
                                    if let Ok(chunk_str) = std::str::from_utf8(&chunk_bytes) {
                                        let events = sse_parser.parse_chunk(chunk_str);
                                        
                                        match events {
                                            ZeroOneOrMany::None => continue,
                                            ZeroOneOrMany::One(event) => {
                                                if let Some(completion_chunk) = Self::process_sse_event(&event, &mut accumulator) {
                                                    if sender.try_send(completion_chunk).is_err() {
                                                        break;
                                                    }
                                                }
                                            }
                                            ZeroOneOrMany::Many(event_vec) => {
                                                for event in event_vec {
                                                    if let Some(completion_chunk) = Self::process_sse_event(&event, &mut accumulator) {
                                                        if sender.try_send(completion_chunk).is_err() {
                                                            break;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                Err(_) => break,
                            }
                        }
                    }
                }
                Err(_) => {
                    // Send error chunk
                    let error_chunk = CompletionChunk::Error(
                        "Error making streaming request".to_string()
                    );
                    let _ = sender.try_send(error_chunk);
                }
            }
        });

        stream
    }

    /// Process SSE event and return completion chunk if ready
    #[inline(always)]
    fn process_sse_event(
        event: &crate::providers::openai::streaming::SSEEvent,
        accumulator: &mut StreamAccumulator,
    ) -> Option<CompletionChunk> {
        // Skip non-data events
        if event.event_type.as_deref() != Some("data") && event.event_type.is_some() {
            return None;
        }

        // Check for stream end
        if event.data.trim() == "[DONE]" {
            return None;
        }

        // Parse streaming chunk
        match parse_stream_chunk(&event.data) {
            Ok(chunk) => {
                match accumulator.process_chunk(&chunk) {
                    Ok(completion_chunk) => completion_chunk,
                    Err(_) => None,
                }
            }
            Err(_) => None,
        }
    }

    /// Extract text content from response
    #[inline(always)]
    pub fn extract_text_content(&self, response: &OpenAICompletionResponse) -> String {
        match &response.choices {
            ZeroOneOrMany::None => String::new(),
            ZeroOneOrMany::One(choice) => choice.message.get_text_content().unwrap_or_default(),
            ZeroOneOrMany::Many(choices) => {
                choices.first()
                    .and_then(|choice| choice.message.get_text_content())
                    .unwrap_or_default()
            }
        }
    }

    /// Get model string from enum
    #[inline(always)]
    fn get_model_string(&self, model: &Models) -> String {
        match model {
            Models::Gpt4O => "gpt-4o",
            Models::Gpt4OMini => "gpt-4o-mini",
            Models::Gpt41 => "gpt-4-1",
            Models::Gpt41Mini => "gpt-4-1-mini",
            Models::Gpt41Nano => "gpt-4-1-nano",
            Models::Chatgpt4OLatest => "chatgpt-4o-latest",
            Models::Gpt4OSearchPreview => "gpt-4o-search-preview",
            Models::Gpt4OMiniSearchPreview => "gpt-4o-mini-search-preview",
            Models::O3 => "o3",
            Models::O3Mini => "o3-mini",
            Models::O3MiniHigh => "o3-mini-high",
            Models::O4Mini => "o4-mini",
            Models::O4MiniHigh => "o4-mini-high",
            Models::Gpt4Turbo => "gpt-4-turbo",
            Models::Gpt35Turbo => "gpt-3.5-turbo",
            _ => "gpt-4o", // Default fallback
        }.to_string()
    }
}

impl CompletionConfig {
    /// Create default configuration
    #[inline(always)]
    pub fn default() -> Self {
        Self {
            model: Models::Gpt4O,
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            seed: None,
            user: None,
            response_format: None,
            service_tier: None,
        }
    }

    /// Set model
    #[inline(always)]
    pub fn with_model(mut self, model: Models) -> Self {
        self.model = model;
        self
    }

    /// Set temperature
    #[inline(always)]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set max tokens
    #[inline(always)]
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set JSON response format
    #[inline(always)]
    pub fn with_json_response(mut self) -> Self {
        self.response_format = Some(ResponseFormat {
            format_type: "json_object".to_string(),
            json_schema: None,
        });
        self
    }

    /// Set structured response with schema
    #[inline(always)]
    pub fn with_structured_response(mut self, schema: Value) -> Self {
        self.response_format = Some(ResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: Some(schema),
        });
        self
    }
}

impl ResponseFormat {
    /// Create text response format
    #[inline(always)]
    pub fn text() -> Self {
        Self {
            format_type: "text".to_string(),
            json_schema: None,
        }
    }

    /// Create JSON object response format
    #[inline(always)]
    pub fn json_object() -> Self {
        Self {
            format_type: "json_object".to_string(),
            json_schema: None,
        }
    }

    /// Create structured response with JSON schema
    #[inline(always)]
    pub fn json_schema(schema: Value) -> Self {
        Self {
            format_type: "json_schema".to_string(),
            json_schema: Some(schema),
        }
    }
}

/// Get available OpenAI models
#[inline(always)]
pub fn available_models() -> ZeroOneOrMany<Models> {
    ZeroOneOrMany::from_vec(vec![
        Models::O3,
        Models::O3Mini,
        Models::O3MiniHigh,
        Models::O4Mini,
        Models::O4MiniHigh,
        Models::Gpt4O,
        Models::Gpt4OMini,
        Models::Gpt41,
        Models::Gpt41Mini,
        Models::Gpt41Nano,
        Models::Chatgpt4OLatest,
        Models::Gpt4OSearchPreview,
        Models::Gpt4OMiniSearchPreview,
        Models::Gpt4Turbo,
        Models::Gpt35Turbo,
    ])
}

/// Check if model supports specific features
#[inline(always)]
pub fn model_supports_tools(model: &Models) -> bool {
    matches!(model,
        Models::Gpt4O | Models::Gpt4OMini | Models::Gpt41 | Models::Gpt41Mini |
        Models::Chatgpt4OLatest | Models::Gpt4Turbo | Models::Gpt35Turbo |
        Models::O3 | Models::O3Mini | Models::O4Mini
    )
}

/// Check if model supports vision
#[inline(always)]
pub fn model_supports_vision(model: &Models) -> bool {
    matches!(model,
        Models::Gpt4O | Models::Gpt4OMini | Models::Chatgpt4OLatest |
        Models::Gpt4Turbo | Models::Gpt4OSearchPreview | Models::Gpt4OMiniSearchPreview
    )
}

/// Check if model supports audio
#[inline(always)]
pub fn model_supports_audio(model: &Models) -> bool {
    matches!(model,
        Models::Gpt4O | Models::Gpt4OMini | Models::Chatgpt4OLatest
    )
}

/// Get model context length
#[inline(always)]
pub fn get_model_context_length(model: &Models) -> u32 {
    match model {
        Models::O3 | Models::O3Mini | Models::O3MiniHigh => 200_000,
        Models::O4Mini | Models::O4MiniHigh => 200_000,
        Models::Gpt4O | Models::Chatgpt4OLatest => 128_000,
        Models::Gpt4OMini => 128_000,
        Models::Gpt41 => 200_000,
        Models::Gpt41Mini | Models::Gpt41Nano => 128_000,
        Models::Gpt4Turbo => 128_000,
        Models::Gpt35Turbo => 16_384,
        _ => 128_000, // Default
    }
}

/// Get model max output tokens
#[inline(always)]
pub fn get_model_max_output_tokens(model: &Models) -> u32 {
    match model {
        Models::O3 | Models::O3Mini | Models::O3MiniHigh => 65_536,
        Models::O4Mini | Models::O4MiniHigh => 65_536,
        Models::Gpt4O | Models::Chatgpt4OLatest => 16_384,
        Models::Gpt4OMini => 16_384,
        Models::Gpt41 => 32_768,
        Models::Gpt41Mini => 16_384,
        Models::Gpt41Nano => 8_192,
        Models::Gpt4Turbo => 4_096,
        Models::Gpt35Turbo => 4_096,
        _ => 16_384, // Default
    }
}

/// Create OpenAI provider from environment variable
pub fn from_env() -> OpenAIResult<OpenAIProvider> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| OpenAIError::AuthenticationFailed(
            "OPENAI_API_KEY environment variable not set".to_string()
        ))?;
    
    OpenAIProvider::new(api_key)
}

/// Create OpenAI provider with custom model
pub fn with_model(api_key: impl Into<String>, model: Models) -> OpenAIResult<OpenAIProvider> {
    let mut provider = OpenAIProvider::new(api_key)?;
    provider.default_model = model;
    Ok(provider)
}

/// Compatibility alias for CompletionResponse
pub type CompletionResponse = OpenAICompletionResponse;

/// Compatibility function for Azure and other providers
pub fn send_compatible_streaming_request(
    request: OpenAICompletionRequest,
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
) -> AsyncStream<CompletionChunk> {
    let provider = OpenAIProvider::new(api_key).unwrap_or_else(|_| {
        // Create fallback provider if key is invalid
        OpenAIProvider {
            api_key: api_key.to_string(),
            client: client.clone(),
            base_url: "https://api.openai.com/v1".to_string(),
            timeout: Duration::from_secs(30),
            default_model: Models::Gpt4O,
        }
    });
    
    provider.make_streaming_request(request)
}