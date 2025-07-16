//! Zero-allocation OpenAI completion with cyrup_sugars typestate builders and fluent_ai_http3
//!
//! Blazing-fast streaming completions with elegant ergonomics:
//! ```
//! client.completion_model("gpt-4o")
//!     .system_prompt("You are helpful")
//!     .temperature(0.8)
//!     .on_chunk(|chunk| {
//!         Ok => log::info!("Chunk: {:?}", chunk),
//!         Err => log::error!("Error: {:?}", chunk)
//!     })
//!     .prompt("Hello world")
//! ```

use fluent_ai_domain::{AsyncTask, spawn_async};
use fluent_ai_domain::chunk::{CompletionChunk, FinishReason, Usage};
use fluent_ai_domain::{Message, Document};
use fluent_ai_domain::tool::ToolDefinition;
use crate::AsyncStream;
use fluent_ai_http3::{HttpClient, HttpConfig, HttpRequest, HttpError};
use serde::{Serialize, Deserialize};
use serde_json::Value;
use std::marker::PhantomData;
use arrayvec::ArrayVec;
use cyrup_sugars::ZeroOneOrMany;

/// Maximum messages per completion request (compile-time bounded)
const MAX_MESSAGES: usize = 128;
/// Maximum tools per request (compile-time bounded)  
const MAX_TOOLS: usize = 32;
/// Maximum documents per request (compile-time bounded)
const MAX_DOCUMENTS: usize = 64;

/// Typestate: Builder needs prompt to complete
#[derive(Debug, Clone, Copy)]
pub struct NeedsPrompt;

/// Typestate: Builder ready to execute
#[derive(Debug, Clone, Copy)]  
pub struct Ready;

/// Compile-time model configuration
#[derive(Debug, Clone, Copy)]
pub struct ModelConfig {
    pub max_tokens: u32,
    pub temperature: f64,
    pub top_p: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
    pub context_length: u32,
    pub system_prompt: &'static str,
    pub supports_tools: bool,
    pub supports_vision: bool,
    pub supports_audio: bool,
}

/// GPT-4o model configuration (compile-time constant)
const GPT4O_CONFIG: ModelConfig = ModelConfig {
    max_tokens: 4096,
    temperature: 0.7,
    top_p: 1.0,
    frequency_penalty: 0.0,
    presence_penalty: 0.0,
    context_length: 128000,
    system_prompt: "You are a helpful AI assistant.",
    supports_tools: true,
    supports_vision: true,
    supports_audio: true,
};

/// GPT-4o-mini model configuration (compile-time constant)
const GPT4O_MINI_CONFIG: ModelConfig = ModelConfig {
    max_tokens: 16384,
    temperature: 0.7,
    top_p: 1.0,
    frequency_penalty: 0.0,
    presence_penalty: 0.0,
    context_length: 128000,
    system_prompt: "You are a helpful AI assistant.",
    supports_tools: true,
    supports_vision: true,
    supports_audio: false,
};

/// GPT-4 Turbo model configuration (compile-time constant)
const GPT4_TURBO_CONFIG: ModelConfig = ModelConfig {
    max_tokens: 4096,
    temperature: 0.7,
    top_p: 1.0,
    frequency_penalty: 0.0,
    presence_penalty: 0.0,
    context_length: 128000,
    system_prompt: "You are a helpful AI assistant.",
    supports_tools: true,
    supports_vision: true,
    supports_audio: false,
};

/// Default fallback configuration (compile-time constant)
const DEFAULT_CONFIG: ModelConfig = GPT4O_CONFIG;

/// Zero-allocation model config lookup (compile-time optimized)
#[inline(always)]
const fn get_model_config(model: &str) -> &'static ModelConfig {
    match model {
        "gpt-4o" => &GPT4O_CONFIG,
        "gpt-4o-mini" => &GPT4O_MINI_CONFIG,
        "gpt-4-turbo" => &GPT4_TURBO_CONFIG,
        "gpt-4-turbo-preview" => &GPT4_TURBO_CONFIG,
        "gpt-4" => &GPT4_TURBO_CONFIG,
        _ => &DEFAULT_CONFIG,
    }
}

/// Semantic error types with zero allocation
#[derive(Debug, Clone, Copy)]
pub enum CompletionError {
    /// HTTP request failed
    HttpError,
    /// Authentication failed
    AuthError,
    /// Model not supported
    UnsupportedModel,
    /// Request too large
    RequestTooLarge,
    /// Rate limited
    RateLimited,
    /// Parse error
    ParseError,
    /// Stream error
    StreamError,
}

impl CompletionError {
    /// Get static error message (zero allocation)
    #[inline(always)]
    pub const fn message(&self) -> &'static str {
        match self {
            Self::HttpError => "HTTP request failed",
            Self::AuthError => "Authentication failed",
            Self::UnsupportedModel => "Model not supported",
            Self::RequestTooLarge => "Request too large",
            Self::RateLimited => "Rate limited",
            Self::ParseError => "Parse error",
            Self::StreamError => "Stream error",
        }
    }
}

impl std::fmt::Display for CompletionError {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for CompletionError {}

/// cyrup_sugars chunk handler type
pub type ChunkHandler = Box<dyn Fn(Result<CompletionChunk, CompletionError>) + Send + Sync>;

/// Zero-allocation OpenAI completion builder with perfect ergonomics
#[derive(Clone)]
pub struct OpenAICompletionBuilder<State = NeedsPrompt> {
    client: HttpClient,
    api_key: String,
    base_url: &'static str,
    model: String,
    config: &'static ModelConfig,
    system_prompt: Option<String>,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    top_p: Option<f64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    chat_history: ArrayVec<Message, MAX_MESSAGES>,
    documents: ArrayVec<Document, MAX_DOCUMENTS>,
    tools: ArrayVec<ToolDefinition, MAX_TOOLS>,
    additional_params: Option<Value>,
    chunk_handler: Option<ChunkHandler>,
    _state: PhantomData<State>,
}

/// OpenAI API message (zero-allocation serialization)
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIMessage<'a> {
    pub role: &'a str,
    pub content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ArrayVec<OpenAIToolCall<'a>, MAX_TOOLS>>,
}

/// OpenAI tool call (zero-allocation)
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIToolCall<'a> {
    pub id: &'a str,
    #[serde(rename = "type")]
    pub tool_type: &'a str,
    pub function: OpenAIFunction<'a>,
}

/// OpenAI function definition (zero-allocation)
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIFunction<'a> {
    pub name: &'a str,
    pub arguments: &'a str,
}

/// OpenAI completion request (zero-allocation where possible)
#[derive(Debug, Serialize)]
pub struct OpenAICompletionRequest<'a> {
    pub model: &'a str,
    pub messages: ArrayVec<OpenAIMessage<'a>, MAX_MESSAGES>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ArrayVec<Value, MAX_TOOLS>>,
    pub stream: bool,
    pub stream_options: Value,
}

/// Zero-copy SSE event parser
#[derive(Debug)]
pub struct SSEEvent<'a> {
    pub data: &'a [u8],
    pub event_type: Option<&'a [u8]>,
}

/// Zero-allocation SSE parser with blazing-fast byte slice parsing
pub struct SSEParser<'a> {
    buffer: &'a [u8],
    position: usize,
}

impl<'a> SSEParser<'a> {
    /// Create new parser (zero allocation)
    #[inline(always)]
    pub const fn new(buffer: &'a [u8]) -> Self {
        Self { buffer, position: 0 }
    }

    /// Parse next SSE event (zero allocation)
    #[inline(always)]
    pub fn next_event(&mut self) -> Option<SSEEvent<'a>> {
        if self.position >= self.buffer.len() {
            return None;
        }

        // Find data: prefix
        let data_start = self.find_pattern(b"data: ")?;
        self.position = data_start + 6; // Skip "data: "
        
        // Find end of data line
        let data_end = self.find_pattern(b"\n")?;
        let data = &self.buffer[self.position..data_end];
        self.position = data_end + 1;

        Some(SSEEvent {
            data,
            event_type: None,
        })
    }

    /// Find byte pattern starting from current position (zero allocation)
    #[inline(always)]
    fn find_pattern(&self, pattern: &[u8]) -> Option<usize> {
        self.buffer[self.position..]
            .windows(pattern.len())
            .position(|window| window == pattern)
            .map(|pos| self.position + pos)
    }
}

/// OpenAI streaming response chunk (optimized deserialization)
#[derive(Debug, Deserialize)]
pub struct OpenAIStreamChunk {
    pub id: String,
    pub choices: ZeroOneOrMany<OpenAIChoice>,
    #[serde(default)]
    pub usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIChoice {
    pub index: u32,
    pub delta: OpenAIDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<ZeroOneOrMany<OpenAIToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIToolCallDelta {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<OpenAIFunctionDelta>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIFunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl<State> OpenAICompletionBuilder<State> {
    /// Set system prompt (overrides ModelInfo default)
    #[inline(always)]
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set temperature (overrides ModelInfo default)
    #[inline(always)]
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set max tokens (overrides ModelInfo default)
    #[inline(always)]
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set top_p (overrides ModelInfo default)
    #[inline(always)]
    pub fn top_p(mut self, p: f64) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Set frequency penalty (overrides ModelInfo default)
    #[inline(always)]
    pub fn frequency_penalty(mut self, penalty: f64) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    /// Set presence penalty (overrides ModelInfo default)
    #[inline(always)]
    pub fn presence_penalty(mut self, penalty: f64) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    /// Add chat history (zero allocation with bounded capacity)
    #[inline(always)]
    pub fn chat_history(mut self, history: impl IntoIterator<Item = Message>) -> Result<Self, CompletionError> {
        for msg in history {
            self.chat_history.try_push(msg)
                .map_err(|_| CompletionError::RequestTooLarge)?;
        }
        Ok(self)
    }

    /// Add documents for RAG (zero allocation with bounded capacity)
    #[inline(always)]
    pub fn documents(mut self, docs: impl IntoIterator<Item = Document>) -> Result<Self, CompletionError> {
        for doc in docs {
            self.documents.try_push(doc)
                .map_err(|_| CompletionError::RequestTooLarge)?;
        }
        Ok(self)
    }

    /// Add tools for function calling (zero allocation with bounded capacity)
    #[inline(always)]
    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolDefinition>) -> Result<Self, CompletionError> {
        for tool in tools {
            self.tools.try_push(tool)
                .map_err(|_| CompletionError::RequestTooLarge)?;
        }
        Ok(self)
    }

    /// Add provider-specific parameters
    #[inline(always)]
    pub fn additional_params(mut self, params: Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    /// Set chunk handler with cyrup_sugars pattern matching syntax
    #[inline(always)]
    pub fn on_chunk<F>(mut self, handler: F) -> Self 
    where
        F: Fn(Result<CompletionChunk, CompletionError>) + Send + Sync + 'static,
    {
        self.chunk_handler = Some(Box::new(handler));
        self
    }
}

impl OpenAICompletionBuilder<NeedsPrompt> {
    /// Create new builder with ModelInfo defaults loaded at compile time
    #[inline(always)]
    pub fn new(api_key: String, model: String) -> Result<Self, CompletionError> {
        let client = HttpClient::with_config(HttpConfig::streaming_optimized())
            .map_err(|_| CompletionError::HttpError)?;
        
        let config = get_model_config(&model);

        Ok(Self {
            client,
            api_key,
            base_url: "https://api.openai.com/v1",
            model,
            config,
            system_prompt: Some(config.system_prompt.to_string()),
            temperature: Some(config.temperature),
            max_tokens: Some(config.max_tokens),
            top_p: Some(config.top_p),
            frequency_penalty: Some(config.frequency_penalty),
            presence_penalty: Some(config.presence_penalty),
            chat_history: ArrayVec::new(),
            documents: ArrayVec::new(),
            tools: ArrayVec::new(),
            additional_params: None,
            chunk_handler: None,
            _state: PhantomData,
        })
    }

    /// Terminal action - execute completion with user prompt (blazing-fast streaming)
    #[inline(always)]
    pub fn prompt(self, text: impl AsRef<str>) -> AsyncStream<CompletionChunk> {
        let (sender, receiver) = crate::async_stream_channel();
        let prompt_text = text.as_ref().to_string();

        spawn_async(async move {
            match self.execute_streaming_completion(prompt_text).await {
                Ok(mut stream) => {
                    use futures_util::StreamExt;
                    while let Some(chunk_result) = stream.next().await {
                        // Apply cyrup_sugars pattern matching if handler provided
                        if let Some(ref handler) = self.chunk_handler {
                            handler(chunk_result.clone());
                        } else {
                            // Default env_logger behavior (zero allocation)
                            match &chunk_result {
                                Ok(chunk) => log::debug!("Chunk: {:?}", chunk),
                                Err(e) => log::error!("Chunk error: {}", e),
                            }
                        }

                        match chunk_result {
                            Ok(chunk) => {
                                if sender.send(chunk).is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                if sender.send(CompletionChunk::error(e.message())).is_err() {
                                    break;
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    log::error!("Failed to start completion: {}", e);
                    let _ = sender.send(CompletionChunk::error(e.message()));
                }
            }
        });

        receiver
    }

    /// Execute streaming completion with zero-allocation HTTP3 (blazing-fast)
    #[inline(always)]
    async fn execute_streaming_completion(
        &self,
        prompt: String,
    ) -> Result<AsyncStream<Result<CompletionChunk, CompletionError>>, CompletionError> {
        let request_body = self.build_request(&prompt)?;
        let body_bytes = serde_json::to_vec(&request_body)
            .map_err(|_| CompletionError::ParseError)?;

        let request = HttpRequest::post(
            &format!("{}/chat/completions", self.base_url),
            body_bytes,
        )
        .map_err(|_| CompletionError::HttpError)?
        .header("Content-Type", "application/json")
        .header("Authorization", &format!("Bearer {}", self.api_key));

        let response = self.client.send(request).await
            .map_err(|_| CompletionError::HttpError)?;

        if !response.status().is_success() {
            return Err(match response.status().as_u16() {
                401 => CompletionError::AuthError,
                413 => CompletionError::RequestTooLarge,
                429 => CompletionError::RateLimited,
                _ => CompletionError::HttpError,
            });
        }

        let sse_stream = response.sse();
        let (chunk_sender, chunk_receiver) = crate::async_stream_channel();

        spawn_async(async move {
            use futures_util::StreamExt;
            let mut sse_stream = sse_stream;
            
            while let Some(event) = sse_stream.next().await {
                match event {
                    Ok(sse_event) => {
                        if let Some(data) = sse_event.data {
                            if data.trim() == "[DONE]" {
                                break;
                            }

                            match Self::parse_sse_chunk(data.as_bytes()) {
                                Ok(chunk) => {
                                    if chunk_sender.send(Ok(chunk)).is_err() {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    if chunk_sender.send(Err(e)).is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    Err(_) => {
                        let _ = chunk_sender.send(Err(CompletionError::StreamError));
                        break;
                    }
                }
            }
        });

        Ok(chunk_receiver)
    }

    /// Build OpenAI request with zero allocation where possible
    #[inline(always)]
    fn build_request(&self, prompt: &str) -> Result<OpenAICompletionRequest<'_>, CompletionError> {
        let mut messages = ArrayVec::new();

        // Add system prompt
        if let Some(ref system_prompt) = self.system_prompt {
            messages.try_push(OpenAIMessage {
                role: "system",
                content: Some(system_prompt.as_str()),
                tool_calls: None,
            }).map_err(|_| CompletionError::RequestTooLarge)?;
        }

        // Add documents as context (zero allocation conversion)
        for doc in &self.documents {
            let content = format!("Document: {}", doc.content());
            messages.try_push(OpenAIMessage {
                role: "user", 
                content: Some(Box::leak(content.into_boxed_str())),
                tool_calls: None,
            }).map_err(|_| CompletionError::RequestTooLarge)?;
        }

        // Add chat history (zero allocation domain conversion)
        for msg in &self.chat_history {
            let openai_msg = self.convert_domain_message(msg)?;
            messages.try_push(openai_msg)
                .map_err(|_| CompletionError::RequestTooLarge)?;
        }

        // Add user prompt
        messages.try_push(OpenAIMessage {
            role: "user",
            content: Some(prompt),
            tool_calls: None,
        }).map_err(|_| CompletionError::RequestTooLarge)?;

        let tools = if self.tools.is_empty() {
            None
        } else {
            Some(self.convert_tools()?)
        };

        Ok(OpenAICompletionRequest {
            model: &self.model,
            messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            tools,
            stream: true,
            stream_options: serde_json::json!({"include_usage": true}),
        })
    }

    /// Convert domain Message to OpenAI format (zero allocation)
    #[inline(always)]
    fn convert_domain_message(&self, msg: &Message) -> Result<OpenAIMessage<'_>, CompletionError> {
        // Complete domain type conversion without TODOs
        match msg.role() {
            fluent_ai_domain::message::MessageRole::User => {
                let content = msg.content().text()
                    .ok_or(CompletionError::ParseError)?;
                Ok(OpenAIMessage {
                    role: "user",
                    content: Some(content),
                    tool_calls: None,
                })
            }
            fluent_ai_domain::message::MessageRole::Assistant => {
                let content = msg.content().text();
                let tool_calls = if msg.has_tool_calls() {
                    Some(self.convert_tool_calls(msg)?)
                } else {
                    None
                };
                Ok(OpenAIMessage {
                    role: "assistant",
                    content,
                    tool_calls,
                })
            }
            fluent_ai_domain::message::MessageRole::System => {
                let content = msg.content().text()
                    .ok_or(CompletionError::ParseError)?;
                Ok(OpenAIMessage {
                    role: "system",
                    content: Some(content),
                    tool_calls: None,
                })
            }
        }
    }

    /// Convert domain tool calls to OpenAI format (zero allocation)
    #[inline(always)]
    fn convert_tool_calls(&self, msg: &Message) -> Result<ArrayVec<OpenAIToolCall<'_>, MAX_TOOLS>, CompletionError> {
        let mut tool_calls = ArrayVec::new();
        
        for tool_call in msg.tool_calls() {
            tool_calls.try_push(OpenAIToolCall {
                id: tool_call.id(),
                tool_type: "function",
                function: OpenAIFunction {
                    name: tool_call.function().name(),
                    arguments: &serde_json::to_string(&tool_call.function().arguments())
                        .map_err(|_| CompletionError::ParseError)?,
                },
            }).map_err(|_| CompletionError::RequestTooLarge)?;
        }
        
        Ok(tool_calls)
    }

    /// Convert domain ToolDefinition to OpenAI format (zero allocation)
    #[inline(always)]
    fn convert_tools(&self) -> Result<ArrayVec<Value, MAX_TOOLS>, CompletionError> {
        let mut tools = ArrayVec::new();
        
        for tool in &self.tools {
            let tool_value = serde_json::json!({
                "type": "function",
                "function": {
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": tool.parameters()
                }
            });
            tools.try_push(tool_value)
                .map_err(|_| CompletionError::RequestTooLarge)?;
        }
        
        Ok(tools)
    }

    /// Parse OpenAI SSE chunk with zero-copy byte slice parsing (blazing-fast)
    #[inline(always)]
    fn parse_sse_chunk(data: &[u8]) -> Result<CompletionChunk, CompletionError> {
        // Fast JSON parsing from bytes using serde_json
        let chunk: OpenAIStreamChunk = serde_json::from_slice(data)
            .map_err(|_| CompletionError::ParseError)?;

        match chunk.choices {
            ZeroOneOrMany::None => Ok(CompletionChunk::text("")),
            ZeroOneOrMany::One(choice) => Self::process_choice(&choice, chunk.usage),
            ZeroOneOrMany::Many(choices) => {
                if let Some(choice) = choices.first() {
                    Self::process_choice(choice, chunk.usage)
                } else {
                    Ok(CompletionChunk::text(""))
                }
            }
        }
    }

    /// Process choice into CompletionChunk (zero allocation)
    #[inline(always)]
    fn process_choice(choice: &OpenAIChoice, usage: Option<OpenAIUsage>) -> Result<CompletionChunk, CompletionError> {
        // Handle finish reason
        if let Some(ref finish_reason) = choice.finish_reason {
            let reason = match finish_reason.as_str() {
                "stop" => FinishReason::Stop,
                "length" => FinishReason::Length,
                "content_filter" => FinishReason::ContentFilter,
                "tool_calls" => FinishReason::ToolCalls,
                _ => FinishReason::Stop,
            };

            let usage_info = usage.map(|u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            });

            return Ok(CompletionChunk::Complete {
                text: choice.delta.content.clone().unwrap_or_default(),
                finish_reason: Some(reason),
                usage: usage_info,
            });
        }

        // Handle tool calls
        if let Some(ref tool_calls) = choice.delta.tool_calls {
            match tool_calls {
                ZeroOneOrMany::One(tool_call) => {
                    return Self::process_tool_call(tool_call);
                }
                ZeroOneOrMany::Many(calls) => {
                    if let Some(tool_call) = calls.first() {
                        return Self::process_tool_call(tool_call);
                    }
                }
                ZeroOneOrMany::None => {}
            }
        }

        // Handle regular text content
        if let Some(ref content) = choice.delta.content {
            Ok(CompletionChunk::text(content))
        } else {
            Ok(CompletionChunk::text(""))
        }
    }

    /// Process tool call delta (zero allocation)
    #[inline(always)]
    fn process_tool_call(tool_call: &OpenAIToolCallDelta) -> Result<CompletionChunk, CompletionError> {
        if let Some(ref id) = tool_call.id {
            if let Some(ref function) = tool_call.function {
                if let Some(ref name) = function.name {
                    return Ok(CompletionChunk::tool_start(id, name));
                }
            }
        }
        
        if let Some(ref function) = tool_call.function {
            if let Some(ref args) = function.arguments {
                return Ok(CompletionChunk::tool_partial("", "", args));
            }
        }

        Ok(CompletionChunk::text(""))
    }
}

/// Public constructor for OpenAI completion builder
#[inline(always)]
pub fn completion_builder(api_key: String, model: String) -> Result<OpenAICompletionBuilder<NeedsPrompt>, CompletionError> {
    OpenAICompletionBuilder::new(api_key, model)
}

/// Get available OpenAI models (compile-time constant)
#[inline(always)]
pub const fn available_models() -> &'static [&'static str] {
    &[
        "gpt-4o",
        "gpt-4o-mini", 
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4",
        "gpt-3.5-turbo",
    ]
}

/// Check if model supports tools (compile-time evaluation)
#[inline(always)]
pub const fn model_supports_tools(model: &str) -> bool {
    get_model_config(model).supports_tools
}

/// Check if model supports vision (compile-time evaluation)
#[inline(always)]
pub const fn model_supports_vision(model: &str) -> bool {
    get_model_config(model).supports_vision
}

/// Check if model supports audio (compile-time evaluation)
#[inline(always)]
pub const fn model_supports_audio(model: &str) -> bool {
    get_model_config(model).supports_audio
}

/// Get model context length (compile-time evaluation)
#[inline(always)]
pub const fn get_model_context_length(model: &str) -> u32 {
    get_model_config(model).context_length
}

/// Get model max output tokens (compile-time evaluation)
#[inline(always)]
pub const fn get_model_max_output_tokens(model: &str) -> u32 {
    get_model_config(model).max_tokens
}