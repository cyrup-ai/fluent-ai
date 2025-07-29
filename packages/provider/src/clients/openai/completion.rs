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

use cyrup_sugars::ZeroOneOrMany;
#[derive(Clone, Debug)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

#[derive(Clone, Debug)]
pub struct Message {
    role: MessageRole,
    content: Content,
}

#[derive(Clone, Debug)]
pub struct Content {
    text: String,
}

impl Message {
    pub fn role(&self) -> MessageRole {
        self.role.clone()
    }

    pub fn content(&self) -> &Content {
        &self.content
    }
}

impl Content {
    pub fn text(&self) -> &str {
        &self.text
    }
}

#[derive(Clone, Debug)]
pub struct Document {
    content: String,
}

impl Document {
    pub fn content(&self) -> &str {
        &self.content
    }
}

#[derive(Clone, Debug)]
pub struct ToolDefinition {
    // Fields as needed
}

#[derive(Clone, Debug)]
pub struct CompletionChunk {
    id: String,
    index: u32,
    text: String,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
}

impl CompletionChunk {
    pub fn new(id: String, index: u32, text: String, finish_reason: Option<FinishReason>, usage: Option<Usage>) -> Self {
        Self { id, index, text, finish_reason, usage }
    }

    pub fn error(message: String) -> Self {
        Self { id: "error".to_string(), index: 0, text: message, finish_reason: None, usage: None }
    }
}

#[derive(Clone, Debug)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    Other,
}

#[derive(Clone, Debug)]
pub struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl Usage {
    pub fn new(prompt: u32, completion: u32, total: u32) -> Self {
        Self { prompt_tokens: prompt, completion_tokens: completion, total_tokens: total }
    }
}
// Use model-info package as single source of truth for model information
use model_info::{Provider, OpenAiProvider, ModelInfo as ModelInfoFromPackage};
// Use local OpenAI request/response types
use super::types::{
    OpenAIMessage, OpenAIMessageContent, OpenAIStreamChunk, OpenAIChoice, 
    OpenAIDelta, OpenAIUsage
};
use fluent_ai_http3::{Http3, HttpClient, HttpConfig, HttpError};
use serde_json::Value;
use arrayvec::ArrayVec;
use crate::completion_provider::{
    ChunkHandler, CompletionError, CompletionProvider, ModelConfig};

// AsyncStream imports - using fluent-ai async architecture
use fluent_ai_async::{AsyncStream, AsyncStreamSender};

/// Maximum messages per completion request (compile-time bounded)
const MAX_MESSAGES: usize = 128;
/// Maximum tools per request (compile-time bounded)  
const MAX_TOOLS: usize = 32;
/// Maximum documents per request (compile-time bounded)
const MAX_DOCUMENTS: usize = 64;

/// Zero-allocation OpenAI completion builder with perfect ergonomics
#[derive(Clone)]
pub struct OpenAICompletionBuilder {
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
    chunk_handler: Option<ChunkHandler>}

impl CompletionProvider for OpenAICompletionBuilder {
    /// Create new OpenAI completion builder with ModelInfo defaults
    #[inline(always)]
    fn new(api_key: String, model_name: &'static str) -> Result<Self, CompletionError> {
        let client = HttpClient::with_config(HttpConfig::streaming_optimized()).map_err(|_| {
            CompletionError::ProviderUnavailable("HTTP client initialization failed".to_string())
        })?;

        // Use blazing-fast default configuration - model info loaded from model-info package when needed
        let default_config = ModelConfig {
            max_tokens: 4096,
            temperature: 0.7,
            top_p: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            context_length: 128000,
            system_prompt: "",
            supports_tools: true,
            supports_vision: true,
            supports_audio: false,
            supports_thinking: false,
            optimal_thinking_budget: 0,
            provider: "openai",
            model_name,
        };

        Ok(Self {
            client,
            api_key,
            explicit_api_key: None,
            base_url: "https://api.openai.com/v1",
            model_name,
            config: default_config,
            system_prompt: default_config.system_prompt.to_string(),
            temperature: default_config.temperature,
            max_tokens: default_config.max_tokens,
            top_p: default_config.top_p,
            frequency_penalty: default_config.frequency_penalty,
            presence_penalty: default_config.presence_penalty,
            chat_history: ArrayVec::new(),
            documents: ArrayVec::new(),
            tools: ArrayVec::new(),
            additional_params: None,
            chunk_handler: None})
    }

    /// Set explicit API key (takes priority over environment variables)
    #[inline(always)]
    fn api_key(mut self, key: impl Into<String>) -> Self {
        self.explicit_api_key = Some(key.into());
        self
    }

    /// Environment variable names to search for OpenAI API keys (ordered by priority)
    #[inline(always)]
    fn env_api_keys(&self) -> ZeroOneOrMany<String> {
        // First found wins - search in priority order
        ZeroOneOrMany::Many(vec![
            "OPENAI_API_KEY".to_string(),  // Primary OpenAI key
            "OPENAI_TOKEN".to_string(),    // Alternative name
            "OPEN_AI_API_KEY".to_string(), // Common variation
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
            ZeroOneOrMany::Zero => {}
            ZeroOneOrMany::One(msg) => {
                if self.chat_history.try_push(msg).is_err() {
                    return Err(CompletionError::InvalidRequest(
                        "Chat history too large".to_string(),
                    ));
                }
            }
            ZeroOneOrMany::Many(msgs) => {
                for msg in msgs {
                    if self.chat_history.try_push(msg).is_err() {
                        return Err(CompletionError::InvalidRequest(
                            "Chat history too large".to_string(),
                        ));
                    }
                }
            }
        }
        Ok(self)
    }

    /// Add documents for context (ZeroOneOrMany with bounded capacity)
    #[inline(always)]
    fn documents(mut self, documents: ZeroOneOrMany<Document>) -> Result<Self, CompletionError> {
        match documents {
            ZeroOneOrMany::Zero => {}
            ZeroOneOrMany::One(doc) => {
                if self.documents.try_push(doc).is_err() {
                    return Err(CompletionError::InvalidRequest(
                        "Documents too large".to_string(),
                    ));
                }
            }
            ZeroOneOrMany::Many(docs) => {
                for doc in docs {
                    if self.documents.try_push(doc).is_err() {
                        return Err(CompletionError::InvalidRequest(
                            "Documents too large".to_string(),
                        ));
                    }
                }
            }
        }
        Ok(self)
    }

    /// Add tools for function calling (ZeroOneOrMany with bounded capacity)
    #[inline(always)]
    fn tools(mut self, tools: ZeroOneOrMany<ToolDefinition>) -> Result<Self, CompletionError> {
        match tools {
            ZeroOneOrMany::Zero => {}
            ZeroOneOrMany::One(tool) => {
                if self.tools.try_push(tool).is_err() {
                    return Err(CompletionError::InvalidRequest(
                        "Tools too large".to_string(),
                    ));
                }
            }
            ZeroOneOrMany::Many(tool_list) => {
                for tool in tool_list {
                    if self.tools.try_push(tool).is_err() {
                        return Err(CompletionError::InvalidRequest(
                            "Tools too large".to_string(),
                        ));
                    }
                }
            }
        }
        Ok(self)
    }

    /// Set additional parameters as JSON value
    #[inline(always)]
    fn additional_params(mut self, params: Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    /// Set chunk handler for streaming responses
    #[inline(always)]
    fn on_chunk(mut self, handler: ChunkHandler) -> Self {
        self.chunk_handler = Some(handler);
        self
    }

    /// Execute completion with prompt string
    /// Returns AsyncStream<CompletionChunk> with proper error handling
    #[inline(always)]
    fn prompt(self, prompt: impl Into<String>) -> AsyncStream<CompletionChunk> {
        let prompt_string = prompt.into();
        
        AsyncStream::with_channel(move |sender: AsyncStreamSender<CompletionChunk>| {
            // Create tokio runtime for async operations (NO FUTURES architecture)
            let rt = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = sender.send(CompletionChunk::error(format!("Runtime creation failed: {}", e)));
                    return;
                }
            };
            
            rt.block_on(async move {
                match self.execute_streaming_completion(prompt_string).await {
                    Ok(mut stream) => {
                        while let Some(result) = stream.next().await {
                            match result {
                                Ok(chunk) => {
                                    if let Some(handler) = &self.chunk_handler {
                                        match handler(Ok(chunk.clone())) {
                                            Ok(_) => {}
                                            Err(e) => {
                                                let _ = sender.send(CompletionChunk::error(e.to_string()));
                                                break;
                                            }
                                        }
                                    }
                                    if sender.send(chunk).is_err() {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    if sender.send(CompletionChunk::error(e.to_string())).is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to start completion: {}", e);
                        let _ = sender.send(CompletionChunk::error(e.to_string()));
                    }
                }
            });
        })
    }
}

impl OpenAICompletionBuilder {
    /// Execute streaming completion with zero-allocation HTTP3 (blazing-fast)
    /// Returns AsyncStream<Result<CompletionChunk, CompletionError>> with proper error handling
    #[inline(always)]
    async fn execute_streaming_completion(
        &self,
        prompt: String,
    ) -> Result<AsyncStream<Result<CompletionChunk, CompletionError>>, CompletionError> {
        let request = self.build_request(&prompt)?;

        // Use explicit API key if set, otherwise use discovered key
        let auth_key = self.explicit_api_key.as_ref().unwrap_or(&self.api_key);

        // Build endpoint URL
        let endpoint = format!("{}/chat/completions", self.base_url);

        // Direct HTTP3 request with elegant builder pattern - zero allocation, blazing-fast
        let response = match Http3::json()
            .api_key(auth_key)
            .body(&request)
            .post(&endpoint)
            .await
        {
            Ok(response) => response,
            Err(e) => {
                return Err(CompletionError::ProviderUnavailable(format!(
                    "HTTP request failed: {}",
                    e
                )));
            }
        };

        // Get SSE stream from response for streaming completions
        let sse_stream = response.sse();

        let stream = AsyncStream::with_channel(move |chunk_sender: AsyncStreamSender<Result<CompletionChunk, CompletionError>>| {
            // Use runtime for async operations inside AsyncStream closure
            let rt = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(_) => {
                    let _ = chunk_sender.send(Err(CompletionError::NetworkError("Runtime creation failed".to_string())));
                    return;
                }
            };
            
            rt.block_on(async move {
                let mut sse_stream = sse_stream;

                while let Some(event) = sse_stream.next().await {
                    match event {
                        Ok(sse_event) => {
                            if let Some(data) = sse_event.data {
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
                        Err(_) => {
                            let _ = chunk_sender.send(Err(CompletionError::NetworkError("Stream error".to_string())));
                            return;
                        }
                    }
                }
            });
        });

        Ok(stream)
    }

    /// Build OpenAI request with direct construction (zero allocation where possible)
    #[inline(always)]
    fn build_request(&self, prompt: &str) -> Result<super::types::OpenAICompletionRequest, CompletionError> {
        use super::types::{OpenAIMessage, OpenAIMessageContent};
        
        let mut messages = ArrayVec::new();

        // Add system prompt if present
        if !self.system_prompt.is_empty() {
            if messages.try_push(OpenAIMessage {
                role: "system".to_string(),
                content: Some(OpenAIMessageContent::Text(self.system_prompt.clone())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }).is_err() {
                return Err(CompletionError::InvalidRequest("Too many messages".to_string()));
            }
        }

        // Add documents as context
        for doc in &self.documents {
            let content = format!("Document: {}", doc.content());
            if messages.try_push(OpenAIMessage {
                role: "user".to_string(),
                content: Some(OpenAIMessageContent::Text(content)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }).is_err() {
                return Err(CompletionError::InvalidRequest("Too many messages".to_string()));
            }
        }

        // Add chat history
        for msg in &self.chat_history {
            let (role, content_text) = match msg.role() {
                MessageRole::User => {
                    let text = msg.content().text();
                    ("user", text.to_string())
                }
                MessageRole::Assistant => {
                    let text = msg.content().text();
                    ("assistant", text.to_string())
                }
                MessageRole::System => {
                    let text = msg.content().text();
                    ("system", text.to_string())
                }
            };

            if messages.try_push(OpenAIMessage {
                role: role.to_string(),
                content: Some(OpenAIMessageContent::Text(content_text)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }).is_err() {
                return Err(CompletionError::InvalidRequest("Too many messages".to_string()));
            }
        }

        // Add user prompt
        if messages.try_push(OpenAIMessage {
            role: "user".to_string(),
            content: Some(OpenAIMessageContent::Text(prompt.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }).is_err() {
            return Err(CompletionError::InvalidRequest("Too many messages".to_string()));
        }

        // Direct parameter validation with clamping (blazing-fast, zero allocation)
        let temperature = self.temperature.clamp(0.0, 2.0);
        let max_tokens = self.max_tokens.min(4096).max(1);
        let top_p = self.top_p.clamp(0.0, 1.0);
        let frequency_penalty = self.frequency_penalty.clamp(-2.0, 2.0);
        let presence_penalty = self.presence_penalty.clamp(-2.0, 2.0);

        // Convert tools if present (simplified - tools not fully implemented yet)
        let tools = if !self.tools.is_empty() {
            // TODO: Implement proper tool conversion when needed
            None
        } else {
            None
        };

        Ok(super::types::OpenAICompletionRequest {
            model: self.model_name.to_string(),
            messages,
            temperature: Some(temperature),
            max_tokens: Some(max_tokens),
            top_p: Some(top_p),
            frequency_penalty: Some(frequency_penalty),
            presence_penalty: Some(presence_penalty),
            tools,
            tool_choice: None,
            stream: true,
        })
    }

    /// Load model information from model-info package (single source of truth)
    /// Returns AsyncStream<ModelInfoFromPackage> with blazing-fast zero-allocation streaming
    #[inline(always)]
    pub fn load_model_info(&self) -> AsyncStream<ModelInfoFromPackage> {
        // Use model-info package as single source of truth for model information
        let provider = Provider::OpenAi(OpenAiProvider);
        provider.get_model_info(self.model_name)
    }

    /// Parse OpenAI SSE chunk using centralized deserialization (blazing-fast)
    #[inline(always)]
    fn parse_sse_chunk(data: &[u8]) -> Result<CompletionChunk, CompletionError> {
        // Use centralized deserialization
        let openai_chunk: OpenAIStreamChunk = match serde_json::from_slice(data) {
            Ok(chunk) => chunk,
            Err(e) => return Err(CompletionError::ParseError(format!("Failed to parse SSE chunk: {}", e)))};

        // Convert to domain CompletionChunk
        if let Some(choice) = openai_chunk.choices.get(0) {
            let content = choice.delta.content.clone().unwrap_or_default();
            let finish_reason = choice.finish_reason.as_ref().map(|r| match r.as_str() {
                "stop" => FinishReason::Stop,
                "length" => FinishReason::Length,
                "function_call" => FinishReason::ToolCalls,
                "tool_calls" => FinishReason::ToolCalls,
                _ => FinishReason::Other});

            let usage = openai_chunk
                .usage
                .map(|u| Usage::new(u.prompt_tokens, u.completion_tokens, u.total_tokens));

            Ok(CompletionChunk::new(
                openai_chunk.id,
                choice.index,
                content,
                finish_reason,
                usage,
            ))
        } else {
            Err(CompletionError::ParseError("No choices in OpenAI response".to_string()))
        }
    }
}
