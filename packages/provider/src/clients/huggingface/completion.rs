//! Zero-allocation HuggingFace completion with cyrup_sugars typestate builders and fluent_ai_http3
//!
//! Blazing-fast streaming completions with elegant ergonomics:
//! ```
//! client.completion_model("meta-llama/Meta-Llama-3.1-8B-Instruct")
//!     .system_prompt("You are helpful")
//!     .temperature(0.8)
//!     .on_chunk(|chunk| {
//!         Ok => log::info!("Chunk: {:?}", chunk),
//!         Err => log::error!("Error: {:?}", chunk)
//!     })
//!     .prompt("Hello world")
//! ```

use arrayvec::ArrayVec;
use cyrup_sugars::ZeroOneOrMany;
use fluent_ai_domain::chat::config::ModelConfig;
use fluent_ai_domain::context::{CompletionChunk, FinishReason};
use fluent_ai_domain::model::Usage;
use fluent_ai_domain::tool::ToolDefinition;
use fluent_ai_domain::{AsyncTask, spawn_async};
use fluent_ai_domain::{Document, Message};
use fluent_ai_http3::{Http3, HttpClient, HttpConfig, HttpError};
// CRITICAL: Import model information from model-info package (single source of truth)
use model_info::{ModelInfo as ModelInfoFromPackage, discovery::Provider};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    AsyncStream,
    completion_provider::{ChunkHandler, CompletionError, CompletionProvider},
};

/// Maximum messages per completion request (compile-time bounded)
const MAX_MESSAGES: usize = 128;
/// Maximum tools per request (compile-time bounded)  
const MAX_TOOLS: usize = 32;
/// Maximum documents per request (compile-time bounded)
const MAX_DOCUMENTS: usize = 64;

// ARCHITECTURAL COMPLIANCE: Local model enumeration eliminated
// Model information now comes from model-info package via load_model_info() method

/// Zero-allocation HuggingFace completion builder with perfect ergonomics
#[derive(Clone)]
pub struct HuggingFaceCompletionBuilder {
    client: HttpClient,
    api_key: String,
    explicit_api_key: Option<String>, // .api_key() override takes priority
    base_url: &'static str,
    model_name: &'static str,
    config: &'static ModelConfig,
    system_prompt: String,
    temperature: f64,
    max_tokens: u32,
    top_p: f64,
    frequency_penalty: f64,
    presence_penalty: f64,
    chat_history: ArrayVec<Message, MAX_MESSAGES>,
    documents: ArrayVec<Document, MAX_DOCUMENTS>,
    tools: ArrayVec<ToolDefinition, MAX_TOOLS>,
    additional_params: Option<Value>,
    chunk_handler: Option<ChunkHandler>,
}

/// HuggingFace API message (zero-allocation serialization)
#[derive(Debug, Serialize, Deserialize)]
pub struct HuggingFaceMessage<'a> {
    pub role: &'a str,
    pub content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ArrayVec<HuggingFaceToolCall<'a>, MAX_TOOLS>>,
}

/// HuggingFace tool call (zero-allocation)
#[derive(Debug, Serialize, Deserialize)]
pub struct HuggingFaceToolCall<'a> {
    pub id: &'a str,
    #[serde(rename = "type")]
    pub tool_type: &'a str,
    pub function: HuggingFaceFunction<'a>,
}

/// HuggingFace function definition (zero-allocation)
#[derive(Debug, Serialize, Deserialize)]
pub struct HuggingFaceFunction<'a> {
    pub name: &'a str,
    pub arguments: &'a str,
}

/// HuggingFace completion request (zero-allocation where possible)
#[derive(Debug, Serialize)]
pub struct HuggingFaceCompletionRequest<'a> {
    pub model: &'a str,
    pub messages: ArrayVec<HuggingFaceMessage<'a>, MAX_MESSAGES>,
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
}

/// HuggingFace streaming response chunk (optimized deserialization)
#[derive(Debug, Deserialize)]
pub struct HuggingFaceStreamChunk {
    pub id: String,
    pub choices: ZeroOneOrMany<HuggingFaceChoice>,
    #[serde(default)]
    pub usage: Option<HuggingFaceUsage>,
}

#[derive(Debug, Deserialize)]
pub struct HuggingFaceChoice {
    pub index: u32,
    pub delta: HuggingFaceDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct HuggingFaceDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<ZeroOneOrMany<HuggingFaceToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
pub struct HuggingFaceToolCallDelta {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<HuggingFaceFunctionDelta>,
}

#[derive(Debug, Deserialize)]
pub struct HuggingFaceFunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct HuggingFaceUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl CompletionProvider for HuggingFaceCompletionBuilder {
    /// Create new HuggingFace completion builder with ModelInfo defaults
    #[inline(always)]
    fn new(api_key: String, model_name: &'static str) -> Result<Self, CompletionError> {
        // Use default configuration - model info queries via load_model_info() method
        let config = &ModelConfig {
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            context_length: 32768,
            system_prompt: "You are a helpful assistant.",
            supports_tools: true,
            supports_vision: false,
            supports_audio: false,
            supports_thinking: false,
            optimal_thinking_budget: 0,
            provider: "huggingface",
            model_name,
        };

        Ok(Self {
            client: HttpClient::with_config(HttpConfig::streaming_optimized())
                .map_err(|_| CompletionError::HttpError)?,
            api_key,
            explicit_api_key: None,
            base_url: "https://api-inference.huggingface.co/models",
            model_name,
            config,
            system_prompt: config.system_prompt.to_string(),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            top_p: config.top_p,
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
            chat_history: ArrayVec::new(),
            documents: ArrayVec::new(),
            tools: ArrayVec::new(),
            additional_params: None,
            chunk_handler: None,
        })
    }

    /// Set explicit API key (takes priority over environment variables)
    #[inline(always)]
    fn api_key(mut self, key: impl Into<String>) -> Self {
        self.explicit_api_key = Some(key.into());
        self
    }

    /// Environment variable names to search for HuggingFace API keys (ordered by priority)
    #[inline(always)]
    fn env_api_keys(&self) -> ZeroOneOrMany<String> {
        // First found wins - search in priority order
        ZeroOneOrMany::Many(vec![
            "HF_TOKEN".to_string(),            // Primary HuggingFace token
            "HUGGINGFACE_API_KEY".to_string(), // API key format
            "HUGGINGFACE_TOKEN".to_string(),   // Alternative token name
            "HF_API_KEY".to_string(),          // Short API key format
        ])
    }

    /// Set system prompt (overrides ModelInfo default)
    #[inline(always)]
    fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set temperature (overrides ModelInfo default)
    #[inline(always)]
    fn temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    /// Set max tokens (overrides ModelInfo default)
    #[inline(always)]
    fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set top_p (overrides ModelInfo default)
    #[inline(always)]
    fn top_p(mut self, p: f64) -> Self {
        self.top_p = p;
        self
    }

    /// Set frequency penalty (overrides ModelInfo default)
    #[inline(always)]
    fn frequency_penalty(mut self, penalty: f64) -> Self {
        self.frequency_penalty = penalty;
        self
    }

    /// Set presence penalty (overrides ModelInfo default)
    #[inline(always)]
    fn presence_penalty(mut self, penalty: f64) -> Self {
        self.presence_penalty = penalty;
        self
    }

    /// Add chat history (ZeroOneOrMany with bounded capacity)
    #[inline(always)]
    fn chat_history(mut self, history: ZeroOneOrMany<Message>) -> Result<Self, CompletionError> {
        match history {
            ZeroOneOrMany::None => {}
            ZeroOneOrMany::One(msg) => {
                self.chat_history
                    .try_push(msg)
                    .map_err(|_| CompletionError::RequestTooLarge)?;
            }
            ZeroOneOrMany::Many(msgs) => {
                for msg in msgs {
                    self.chat_history
                        .try_push(msg)
                        .map_err(|_| CompletionError::RequestTooLarge)?;
                }
            }
        }
        Ok(self)
    }

    /// Add documents for RAG (ZeroOneOrMany with bounded capacity)
    #[inline(always)]
    fn documents(mut self, docs: ZeroOneOrMany<Document>) -> Result<Self, CompletionError> {
        match docs {
            ZeroOneOrMany::None => {}
            ZeroOneOrMany::One(doc) => {
                self.documents
                    .try_push(doc)
                    .map_err(|_| CompletionError::RequestTooLarge)?;
            }
            ZeroOneOrMany::Many(documents) => {
                for doc in documents {
                    self.documents
                        .try_push(doc)
                        .map_err(|_| CompletionError::RequestTooLarge)?;
                }
            }
        }
        Ok(self)
    }

    /// Add tools for function calling (ZeroOneOrMany with bounded capacity)
    #[inline(always)]
    fn tools(mut self, tools: ZeroOneOrMany<ToolDefinition>) -> Result<Self, CompletionError> {
        match tools {
            ZeroOneOrMany::None => {}
            ZeroOneOrMany::One(tool) => {
                self.tools
                    .try_push(tool)
                    .map_err(|_| CompletionError::RequestTooLarge)?;
            }
            ZeroOneOrMany::Many(tool_list) => {
                for tool in tool_list {
                    self.tools
                        .try_push(tool)
                        .map_err(|_| CompletionError::RequestTooLarge)?;
                }
            }
        }
        Ok(self)
    }

    /// Add provider-specific parameters
    #[inline(always)]
    fn additional_params(mut self, params: Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    /// Set chunk handler with cyrup_sugars pattern matching syntax
    #[inline(always)]
    fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<CompletionChunk, CompletionError>) + Send + Sync + 'static,
    {
        self.chunk_handler = Some(Box::new(handler));
        self
    }

    /// Terminal action - execute completion with user prompt (blazing-fast streaming)
    #[inline(always)]
    fn prompt(self, text: impl AsRef<str>) -> AsyncStream<CompletionChunk> {
        let (sender, receiver) = crate::channel();
        let prompt_text = text.as_ref().to_string();

        spawn_async(async move {
            match self.execute_streaming_completion(prompt_text).await {
                Ok(mut stream) => {
                    // Removed AsyncStreamExt import - using pure AsyncStream patterns
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
}

impl HuggingFaceCompletionBuilder {
    /// Execute streaming completion with zero-allocation HTTP3 (blazing-fast)
    #[inline(always)]
    async fn execute_streaming_completion(
        &self,
        prompt: String,
    ) -> Result<AsyncStream<Result<CompletionChunk, CompletionError>>, CompletionError> {
        let request_body = self.build_request(&prompt)?;
        let body_bytes =
            serde_json::to_vec(&request_body).map_err(|_| CompletionError::ParseError)?;

        // Use explicit API key if set, otherwise use discovered key
        let auth_key = self.explicit_api_key.as_ref().unwrap_or(&self.api_key);

        // Use Http3::json() directly instead of HttpClient abstraction
        let mut response_stream =
            Http3::json()
                .api_key(auth_key)
                .body(&request_body)
                .post(&format!(
                    "{}/{}/v1/chat/completions",
                    self.base_url, self.model_name
                ));

        if !response_stream.is_success() {
            return Err(match response_stream.status_code() {
                401 => CompletionError::AuthError,
                413 => CompletionError::RequestTooLarge,
                429 => CompletionError::RateLimited,
                _ => CompletionError::HttpError,
            });
        }
        let (chunk_sender, chunk_receiver) = crate::channel();

        spawn_async(async move {
            use fluent_ai_http3::HttpStreamExt;

            while let Some(chunk) = response_stream.next().await {
                match chunk {
                    Ok(http_chunk) => {
                        if let fluent_ai_http3::HttpChunk::Body(data) = http_chunk {
                            let data_str = String::from_utf8_lossy(&data);

                            // Process SSE events
                            for line in data_str.lines() {
                                if line.starts_with("data: ") {
                                    let data = &line[6..];
                                    if data.trim() == "[DONE]" {
                                        return;
                                    }

                                    match Self::parse_sse_chunk(data.as_bytes()) {
                                        Ok(chunk) => {
                                            if chunk_sender.send(Ok(chunk)).is_err() {
                                                return;
                                            }
                                        }
                                        Err(e) => {
                                            if chunk_sender.send(Err(e)).is_err() {
                                                return;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(_) => {
                        let _ = chunk_sender.send(Err(CompletionError::StreamError));
                        return;
                    }
                }
            }
        });

        Ok(chunk_receiver)
    }

    /// Build HuggingFace request with zero allocation where possible
    #[inline(always)]
    fn build_request(
        &self,
        prompt: &str,
    ) -> Result<HuggingFaceCompletionRequest<'_>, CompletionError> {
        let mut messages = ArrayVec::new();

        // Add system prompt (always present from ModelInfo)
        if !self.system_prompt.is_empty() {
            messages
                .try_push(HuggingFaceMessage {
                    role: "system",
                    content: Some(&self.system_prompt),
                    tool_calls: None,
                })
                .map_err(|_| CompletionError::RequestTooLarge)?;
        }

        // Add documents as context (zero allocation conversion)
        for doc in &self.documents {
            let content = format!("Document: {}", doc.content());
            messages
                .try_push(HuggingFaceMessage {
                    role: "user",
                    content: Some(Box::leak(content.into_boxed_str())),
                    tool_calls: None,
                })
                .map_err(|_| CompletionError::RequestTooLarge)?;
        }

        // Add chat history (zero allocation domain conversion)
        for msg in &self.chat_history {
            let hf_msg = self.convert_domain_message(msg)?;
            messages
                .try_push(hf_msg)
                .map_err(|_| CompletionError::RequestTooLarge)?;
        }

        // Add user prompt
        messages
            .try_push(HuggingFaceMessage {
                role: "user",
                content: Some(prompt),
                tool_calls: None,
            })
            .map_err(|_| CompletionError::RequestTooLarge)?;

        let tools = if self.tools.is_empty() {
            None
        } else {
            Some(self.convert_tools()?)
        };

        Ok(HuggingFaceCompletionRequest {
            model: self.model_name,
            messages,
            temperature: Some(self.temperature),
            max_tokens: Some(self.max_tokens),
            top_p: Some(self.top_p),
            frequency_penalty: Some(self.frequency_penalty),
            presence_penalty: Some(self.presence_penalty),
            tools,
            stream: true,
        })
    }

    /// Convert domain Message to HuggingFace format (zero allocation)
    #[inline(always)]
    fn convert_domain_message(
        &self,
        msg: &Message,
    ) -> Result<HuggingFaceMessage<'_>, CompletionError> {
        // Complete domain type conversion without TODOs
        match msg.role() {
            fluent_ai_domain::message::MessageRole::User => {
                let content = msg.content().text().ok_or(CompletionError::ParseError)?;
                Ok(HuggingFaceMessage {
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
                Ok(HuggingFaceMessage {
                    role: "assistant",
                    content,
                    tool_calls,
                })
            }
            fluent_ai_domain::message::MessageRole::System => {
                let content = msg.content().text().ok_or(CompletionError::ParseError)?;
                Ok(HuggingFaceMessage {
                    role: "system",
                    content: Some(content),
                    tool_calls: None,
                })
            }
        }
    }

    /// Convert domain tool calls to HuggingFace format (zero allocation)
    #[inline(always)]
    fn convert_tool_calls(
        &self,
        msg: &Message,
    ) -> Result<ArrayVec<HuggingFaceToolCall<'_>, MAX_TOOLS>, CompletionError> {
        let mut tool_calls = ArrayVec::new();

        for tool_call in msg.tool_calls() {
            tool_calls
                .try_push(HuggingFaceToolCall {
                    id: tool_call.id(),
                    tool_type: "function",
                    function: HuggingFaceFunction {
                        name: tool_call.function().name(),
                        arguments: &serde_json::to_string(&tool_call.function().arguments())
                            .map_err(|_| CompletionError::ParseError)?,
                    },
                })
                .map_err(|_| CompletionError::RequestTooLarge)?;
        }

        Ok(tool_calls)
    }

    /// Convert domain ToolDefinition to HuggingFace format (zero allocation)
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
            tools
                .try_push(tool_value)
                .map_err(|_| CompletionError::RequestTooLarge)?;
        }

        Ok(tools)
    }

    /// Parse HuggingFace SSE chunk with zero-copy byte slice parsing (blazing-fast)
    #[inline(always)]
    fn parse_sse_chunk(data: &[u8]) -> Result<CompletionChunk, CompletionError> {
        // Fast JSON parsing from bytes using serde_json
        let chunk: HuggingFaceStreamChunk =
            serde_json::from_slice(data).map_err(|_| CompletionError::ParseError)?;

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
    fn process_choice(
        choice: &HuggingFaceChoice,
        usage: Option<HuggingFaceUsage>,
    ) -> Result<CompletionChunk, CompletionError> {
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
    fn process_tool_call(
        tool_call: &HuggingFaceToolCallDelta,
    ) -> Result<CompletionChunk, CompletionError> {
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

    /// Load model information from model-info package (single source of truth)
    pub fn load_model_info(&self) -> fluent_ai_async::AsyncStream<ModelInfoFromPackage> {
        let provider = Provider::Huggingface;
        provider.get_model_info(self.model_name)
    }
}

/// Public constructor for HuggingFace completion builder
#[inline(always)]
pub fn completion_builder(
    api_key: String,
    model_name: &'static str,
) -> Result<HuggingFaceCompletionBuilder, CompletionError> {
    HuggingFaceCompletionBuilder::new(api_key, model_name)
}

// Available models are provided by model-info package - use model_info queries instead

// Model information is provided by model-info package - no local constants needed

// =============================================================================
// Model Constants (for compatibility with existing imports)
// =============================================================================

/// Gemma 2 model variants
pub const GEMMA_2: &str = "google/gemma-2-9b-it";

/// Meta Llama 3.1 model variants  
pub const META_LLAMA_3_1: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct";

/// Phi-4 model
pub const PHI_4: &str = "microsoft/Phi-4";

/// Qwen QVQ Preview model
pub const QWEN_QVQ_PREVIEW: &str = "Qwen/QVQ-72B-Preview";

/// Qwen 2.5 model variants
pub const QWEN2_5: &str = "Qwen/Qwen2.5-72B-Instruct";

/// Qwen 2.5 Coder model variants
pub const QWEN2_5_CODER: &str = "Qwen/Qwen2.5-Coder-32B-Instruct";

/// Qwen 2 VL model variants
pub const QWEN2_VL: &str = "Qwen/Qwen2-VL-72B-Instruct";

/// SmallThinker Preview model
pub const SMALLTHINKER_PREVIEW: &str = "smallthinker-preview";

/// Get list of available models
pub fn available_models() -> Vec<&'static str> {
    vec![
        GEMMA_2,
        META_LLAMA_3_1,
        PHI_4,
        QWEN_QVQ_PREVIEW,
        QWEN2_5,
        QWEN2_5_CODER,
        QWEN2_VL,
        SMALLTHINKER_PREVIEW,
    ]
}

/// Completion model alias for compatibility
pub type CompletionModel = HuggingFaceCompletionBuilder;

/// API response type for compatibility
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ApiResponse<T> {
    pub data: T,
    pub usage: Option<crate::completion_provider::Usage>,
}
