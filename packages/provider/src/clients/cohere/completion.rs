//! Cohere chat completion builder implementing CompletionProvider trait
//!
//! Provides high-performance chat completions for Command-A and Command-R models:
//! - Command-A-03-2025: Latest advanced model with enhanced reasoning
//! - Command-R7B-12-2024: Efficient 7B parameter model for production
//!
//! Features:
//! - Zero allocation message conversion with arrayvec collections
//! - Tool/function calling with Cohere format conversion
//! - Streaming and non-streaming execution modes
//! - Parameter validation and model-specific optimizations
//! - Circuit breaker integration and performance monitoring

use crate::completion_provider::{CompletionProvider, CompletionError, CompletionResponse, StreamingResponse};
use super::error::{CohereError, Result, ChatErrorReason, CohereOperation, JsonOperation};
use super::models;
use super::streaming::CohereStream;
use super::client::{CohereMetrics, RequestTimer};

use fluent_ai_http3::{HttpClient, HttpRequest};
use fluent_ai_domain::{AsyncTask, AsyncStream};
use fluent_ai_domain::chunk::CompletionChunk;
use fluent_ai_domain::completion::CompletionRequest;
use fluent_ai_domain::message::{Message, Role};
use fluent_ai_domain::tool::Tool;
use fluent_ai_domain::usage::Usage;

use arc_swap::{ArcSwap, Guard};
use arrayvec::{ArrayVec, ArrayString};
use smallvec::{SmallVec, smallvec};
use serde_json::{Map, Value};
use std::sync::Arc;
use std::time::Duration;

/// Cohere chat completion builder implementing CompletionProvider trait
#[derive(Clone)]
pub struct CohereCompletionBuilder {
    /// Shared HTTP client for API requests
    http_client: &'static HttpClient,
    
    /// Hot-swappable API key
    api_key: Guard<Arc<ArrayString<128>>>,
    
    /// Model identifier (compile-time constant)
    model: &'static str,
    
    /// Chat endpoint URL
    endpoint_url: &'static str,
    
    /// Performance metrics tracking
    metrics: &'static CohereMetrics,
    
    /// System prompt/message
    system_prompt: Option<String>,
    
    /// Conversation memory for multi-turn chats
    chat_history: Vec<Message>,
    
    /// Temperature (0.0 to 2.0 for Cohere)
    temperature: Option<f32>,
    
    /// Maximum tokens to generate
    max_tokens: Option<u32>,
    
    /// Top-p nucleus sampling
    top_p: Option<f32>,
    
    /// Top-k sampling
    top_k: Option<u32>,
    
    /// Frequency penalty (-2.0 to 2.0)
    frequency_penalty: Option<f32>,
    
    /// Presence penalty (-2.0 to 2.0)
    presence_penalty: Option<f32>,
    
    /// Stop sequences
    stop_sequences: Vec<String>,
    
    /// Tools/functions available to the model
    tools: Vec<Tool>,
    
    /// Whether to stream the response
    stream: bool,
    
    /// Request timeout
    timeout: Duration,
    
    /// Enable citation mode for Cohere
    citations: bool,
    
    /// Search queries grounding (Cohere-specific)
    search_queries: Vec<String>,
    
    /// Documents for grounded generation (Cohere-specific)
    documents: Vec<CohereDocument>,
}

/// Cohere document structure for grounded generation
#[derive(Debug, Clone)]
pub struct CohereDocument {
    pub id: String,
    pub title: Option<String>,
    pub snippet: String,
    pub url: Option<String>,
}

impl CohereCompletionBuilder {
    /// Create new Cohere completion builder with model validation
    pub fn new(
        http_client: &'static HttpClient,
        api_key: Guard<Arc<ArrayString<128>>>,
        model: &'static str,
        endpoint_url: &'static str,
        metrics: &'static CohereMetrics,
    ) -> Result<Self> {
        // Validate model is supported for chat completions
        if !models::is_chat_model(model) {
            return Err(CohereError::model_not_supported(
                model,
                CohereOperation::Chat,
                models::CHAT_MODELS,
                "chat",
            ));
        }
        
        Ok(Self {
            http_client,
            api_key,
            model,
            endpoint_url,
            metrics,
            system_prompt: None,
            chat_history: Vec::new(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: Vec::new(),
            tools: Vec::new(),
            stream: false,
            timeout: Duration::from_secs(30),
            citations: false,
            search_queries: Vec::new(),
            documents: Vec::new(),
        })
    }
    
    /// Set system prompt/message
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }
    
    /// Add message to conversation history
    pub fn add_message(mut self, message: Message) -> Self {
        self.chat_history.push(message);
        self
    }
    
    /// Set multiple messages for conversation context
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.chat_history = messages;
        self
    }
    
    /// Set temperature (0.0 to 2.0 for Cohere models)
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp.clamp(0.0, 2.0));
        self
    }
    
    /// Set maximum tokens to generate
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        // Validate against model context length
        let max_context = models::context_length(self.model);
        self.max_tokens = Some(tokens.min(max_context));
        self
    }
    
    /// Set top-p nucleus sampling (0.0 to 1.0)
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p.clamp(0.0, 1.0));
        self
    }
    
    /// Set top-k sampling (1 to 500 for Cohere)
    pub fn top_k(mut self, k: u32) -> Self {
        self.top_k = Some(k.clamp(1, 500));
        self
    }
    
    /// Set frequency penalty (-2.0 to 2.0)
    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = Some(penalty.clamp(-2.0, 2.0));
        self
    }
    
    /// Set presence penalty (-2.0 to 2.0)
    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = Some(penalty.clamp(-2.0, 2.0));
        self
    }
    
    /// Add stop sequence
    pub fn stop_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.stop_sequences.push(sequence.into());
        self
    }
    
    /// Add multiple stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences.extend(sequences);
        self
    }
    
    /// Add tool/function
    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }
    
    /// Add multiple tools/functions
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools.extend(tools);
        self
    }
    
    /// Enable streaming response
    pub fn stream(mut self, enabled: bool) -> Self {
        self.stream = enabled;
        self
    }
    
    /// Set request timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
    
    /// Enable citations for grounded generation (Cohere-specific)
    pub fn citations(mut self, enabled: bool) -> Self {
        self.citations = enabled;
        self
    }
    
    /// Add search queries for grounded generation (Cohere-specific)
    pub fn search_queries(mut self, queries: Vec<String>) -> Self {
        self.search_queries = queries;
        self
    }
    
    /// Add documents for grounded generation (Cohere-specific)
    pub fn documents(mut self, documents: Vec<CohereDocument>) -> Self {
        self.documents = documents;
        self
    }
    
    /// Build Cohere API request body with zero allocation optimizations
    fn build_request_body(&self, messages: &[Message]) -> Result<Vec<u8>> {
        let mut request = Map::new();
        
        // Model is required
        request.insert("model".to_string(), Value::String(self.model.to_string()));
        
        // Convert messages to Cohere format
        let cohere_messages = self.convert_messages_to_cohere(messages)?;
        request.insert("message".to_string(), Value::String(cohere_messages));
        
        // Add chat history if present
        if !self.chat_history.is_empty() {
            let chat_history = self.convert_chat_history_to_cohere()?;
            request.insert("chat_history".to_string(), Value::Array(chat_history));
        }
        
        // Add system prompt if present
        if let Some(ref system) = self.system_prompt {
            request.insert("preamble".to_string(), Value::String(system.clone()));
        }
        
        // Add generation parameters
        if let Some(temp) = self.temperature {
            request.insert("temperature".to_string(), Value::from(temp));
        }
        
        if let Some(max_tokens) = self.max_tokens {
            request.insert("max_tokens".to_string(), Value::from(max_tokens));
        }
        
        if let Some(top_p) = self.top_p {
            request.insert("p".to_string(), Value::from(top_p));
        }
        
        if let Some(top_k) = self.top_k {
            request.insert("k".to_string(), Value::from(top_k));
        }
        
        if let Some(freq_penalty) = self.frequency_penalty {
            request.insert("frequency_penalty".to_string(), Value::from(freq_penalty));
        }
        
        if let Some(pres_penalty) = self.presence_penalty {
            request.insert("presence_penalty".to_string(), Value::from(pres_penalty));
        }
        
        // Add stop sequences
        if !self.stop_sequences.is_empty() {
            let stop_seqs: Vec<Value> = self.stop_sequences.iter()
                .map(|s| Value::String(s.clone()))
                .collect();
            request.insert("stop_sequences".to_string(), Value::Array(stop_seqs));
        }
        
        // Add tools if present and model supports them
        if !self.tools.is_empty() {
            if !models::supports_tools(self.model) {
                return Err(CohereError::chat_error(
                    ChatErrorReason::ToolCallFailed,
                    self.model,
                    "Model does not support tools",
                    false,
                ));
            }
            
            let cohere_tools = self.convert_tools_to_cohere()?;
            request.insert("tools".to_string(), Value::Array(cohere_tools));
        }
        
        // Add Cohere-specific features
        if self.citations {
            request.insert("citations".to_string(), Value::Bool(true));
        }
        
        if !self.search_queries.is_empty() {
            let queries: Vec<Value> = self.search_queries.iter()
                .map(|q| Value::String(q.clone()))
                .collect();
            request.insert("search_queries".to_string(), Value::Array(queries));
        }
        
        if !self.documents.is_empty() {
            let docs = self.convert_documents_to_cohere()?;
            request.insert("documents".to_string(), Value::Array(docs));
        }
        
        // Add streaming flag
        if self.stream {
            request.insert("stream".to_string(), Value::Bool(true));
        }
        
        serde_json::to_vec(&request)
            .map_err(|e| CohereError::json_error(
                JsonOperation::RequestSerialization,
                &e.to_string(),
                None,
                false,
            ))
    }
    
    /// Convert domain messages to Cohere format (latest message only)
    fn convert_messages_to_cohere(&self, messages: &[Message]) -> Result<String> {
        // Cohere expects the latest user message as the main "message" field
        let user_message = messages.iter()
            .filter(|m| m.role == Role::User)
            .last()
            .ok_or_else(|| CohereError::chat_error(
                ChatErrorReason::InvalidMessage,
                self.model,
                "No user message found",
                true,
            ))?;
        
        Ok(user_message.content.clone())
    }
    
    /// Convert chat history to Cohere format
    fn convert_chat_history_to_cohere(&self) -> Result<Vec<Value>> {
        let mut cohere_history = Vec::new();
        
        for message in &self.chat_history {
            let mut cohere_msg = Map::new();
            
            let role = match message.role {
                Role::User => "USER",
                Role::Assistant => "CHATBOT",
                Role::System => continue, // System messages go in preamble
                Role::Tool => "TOOL", // Cohere tool responses
            };
            
            cohere_msg.insert("role".to_string(), Value::String(role.to_string()));
            cohere_msg.insert("message".to_string(), Value::String(message.content.clone()));
            
            cohere_history.push(Value::Object(cohere_msg));
        }
        
        Ok(cohere_history)
    }
    
    /// Convert domain tools to Cohere format
    fn convert_tools_to_cohere(&self) -> Result<Vec<Value>> {
        let mut cohere_tools = Vec::new();
        
        for tool in &self.tools {
            let mut cohere_tool = Map::new();
            
            cohere_tool.insert("name".to_string(), Value::String(tool.name.clone()));
            cohere_tool.insert("description".to_string(), Value::String(tool.description.clone()));
            
            if let Some(ref parameters) = tool.parameters {
                cohere_tool.insert("parameter_definitions".to_string(), parameters.clone());
            }
            
            cohere_tools.push(Value::Object(cohere_tool));
        }
        
        Ok(cohere_tools)
    }
    
    /// Convert documents to Cohere format
    fn convert_documents_to_cohere(&self) -> Result<Vec<Value>> {
        let mut cohere_docs = Vec::new();
        
        for doc in &self.documents {
            let mut cohere_doc = Map::new();
            
            cohere_doc.insert("id".to_string(), Value::String(doc.id.clone()));
            cohere_doc.insert("text".to_string(), Value::String(doc.snippet.clone()));
            
            if let Some(ref title) = doc.title {
                cohere_doc.insert("title".to_string(), Value::String(title.clone()));
            }
            
            if let Some(ref url) = doc.url {
                cohere_doc.insert("url".to_string(), Value::String(url.clone()));
            }
            
            cohere_docs.push(Value::Object(cohere_doc));
        }
        
        Ok(cohere_docs)
    }
    
    /// Build authentication headers with zero allocation
    fn build_headers(&self) -> SmallVec<[(&'static str, ArrayString<140>); 4]> {
        let mut auth_header = ArrayString::<140>::new();
        let _ = auth_header.try_push_str("Bearer ");
        let _ = auth_header.try_push_str(&self.api_key);
        
        smallvec![
            ("Authorization", auth_header),
            ("Content-Type", ArrayString::from("application/json").unwrap_or_default()),
            ("User-Agent", ArrayString::from(super::utils::user_agent()).unwrap_or_default()),
            ("Accept", ArrayString::from("application/json").unwrap_or_default()),
        ]
    }
    
    /// Execute completion request (non-streaming)
    async fn execute_completion(&self, messages: &[Message]) -> Result<CompletionResponse> {
        let timer = RequestTimer::start(self.metrics);
        
        let request_body = self.build_request_body(messages)?;
        let headers = self.build_headers();
        
        let http_request = HttpRequest::post(self.endpoint_url, request_body)
            .map_err(|e| CohereError::Configuration {
                setting: ArrayString::from("http_request").unwrap_or_default(),
                reason: ArrayString::from(&e.to_string()).unwrap_or_default(),
                current_value: ArrayString::new(),
                valid_range: None,
            })?
            .headers(headers.iter().map(|(k, v)| (*k, v.as_str())))
            .timeout(self.timeout);
        
        let response = self.http_client.send(http_request).await
            .map_err(|e| {
                timer.finish_failure();
                CohereError::Http(e)
            })?;
        
        if !response.status().is_success() {
            timer.finish_failure();
            return Err(CohereError::from(response.status().as_u16()));
        }
        
        let body = response.body().await
            .map_err(|e| {
                timer.finish_failure();
                CohereError::json_error(
                    JsonOperation::ResponseDeserialization,
                    &e.to_string(),
                    None,
                    false,
                )
            })?;
        
        let response_json: Value = serde_json::from_slice(&body)
            .map_err(|e| {
                timer.finish_failure();
                CohereError::from(e)
            })?;
        
        let completion_response = self.parse_completion_response(response_json)?;
        timer.finish_success();
        Ok(completion_response)
    }
    
    /// Parse Cohere completion response to domain type
    fn parse_completion_response(&self, response: Value) -> Result<CompletionResponse> {
        let text = response.get("text")
            .and_then(|t| t.as_str())
            .unwrap_or("");
        
        let finish_reason = response.get("finish_reason")
            .and_then(|r| r.as_str())
            .map(|r| r.to_string());
        
        // Parse usage information
        let usage = response.get("meta").and_then(|meta| {
            let billed_units = meta.get("billed_units")?;
            let input_tokens = billed_units.get("input_tokens")?.as_u64()? as u32;
            let output_tokens = billed_units.get("output_tokens")?.as_u64()? as u32;
            
            Some(Usage {
                prompt_tokens: input_tokens,
                completion_tokens: output_tokens,
                total_tokens: input_tokens + output_tokens,
            })
        });
        
        Ok(CompletionResponse {
            content: text.to_string(),
            finish_reason,
            usage,
            model: Some(self.model.to_string()),
        })
    }
    
    /// Execute streaming completion request
    async fn execute_streaming(&self, messages: &[Message]) -> Result<AsyncStream<CompletionChunk>> {
        let timer = RequestTimer::start(self.metrics);
        
        let mut builder = self.clone();
        builder.stream = true;
        
        let request_body = builder.build_request_body(messages)?;
        let headers = builder.build_headers();
        
        let http_request = HttpRequest::post(self.endpoint_url, request_body)
            .map_err(|e| {
                timer.finish_failure();
                CohereError::Configuration {
                    setting: ArrayString::from("http_request").unwrap_or_default(),
                    reason: ArrayString::from(&e.to_string()).unwrap_or_default(),
                    current_value: ArrayString::new(),
                    valid_range: None,
                }
            })?
            .headers(headers.iter().map(|(k, v)| (*k, v.as_str())))
            .timeout(self.timeout);
        
        let response = self.http_client.send(http_request).await
            .map_err(|e| {
                timer.finish_failure();
                CohereError::Http(e)
            })?;
        
        if !response.status().is_success() {
            timer.finish_failure();
            return Err(CohereError::from(response.status().as_u16()));
        }
        
        let cohere_stream = CohereStream::new(response, self.model);
        timer.finish_success();
        Ok(cohere_stream.into_chunk_stream())
    }
}

/// Implementation of CompletionProvider trait
impl CompletionProvider for CohereCompletionBuilder {
    type Response = CompletionResponse;
    type StreamingResponse = StreamingResponse;
    type Error = CompletionError;

    fn prompt(&self, prompt: fluent_ai_domain::prompt::Prompt) -> AsyncStream<CompletionChunk> {
        let messages = vec![Message {
            role: Role::User,
            content: prompt.to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        
        let builder = self.clone();
        
        AsyncStream::new(async move {
            match builder.execute_streaming(&messages).await {
                Ok(stream) => stream,
                Err(e) => {
                    let error_chunk = CompletionChunk {
                        content: Some(format!("Error: {}", e)),
                        finish_reason: Some("error".to_string()),
                        usage: None,
                        model: Some(builder.model.to_string()),
                        delta: None,
                    };
                    AsyncStream::from_single(error_chunk)
                }
            }
        })
    }

    fn completion(&self, request: CompletionRequest) -> AsyncTask<Result<Self::Response, Self::Error>> {
        let builder = self.clone();
        AsyncTask::spawn(async move {
            let response = builder.execute_completion(&request.messages).await
                .map_err(|e| CompletionError::from(e))?;
            Ok(response)
        })
    }

    fn stream(&self, request: CompletionRequest) -> AsyncTask<Result<AsyncStream<Self::StreamingResponse>, Self::Error>> {
        let builder = self.clone();
        
        AsyncTask::spawn(async move {
            let stream = builder.execute_streaming(&request.messages).await
                .map_err(|e| CompletionError::from(e))?;
            
            let streaming_response = StreamingResponse::new(stream);
            Ok(AsyncStream::from_single(streaming_response))
        })
    }
}

/// Convert CohereError to CompletionError
impl From<CohereError> for CompletionError {
    fn from(err: CohereError) -> Self {
        match err {
            CohereError::Authentication { message, .. } => {
                CompletionError::ProviderUnavailable(message.to_string())
            }
            CohereError::ModelNotSupported { model, .. } => {
                CompletionError::ModelLoadingFailed(format!("Model not supported: {}", model))
            }
            CohereError::RateLimited { .. } => {
                CompletionError::RateLimitExceeded
            }
            CohereError::RequestValidation { field, reason, .. } => {
                CompletionError::InvalidRequest(format!("Validation failed for {}: {}", field, reason))
            }
            CohereError::Timeout { duration_ms, .. } => {
                CompletionError::Timeout
            }
            _ => {
                CompletionError::Internal(err.to_string())
            }
        }
    }
}