//! Shared HTTP Types for AI Provider Integration
//!
//! This module defines reusable types and utilities that are common across all AI providers.
//! Designed for zero-allocation patterns, type safety, and comprehensive validation.
//!
//! # Core Types
//!
//! - [`BaseMessage`] - Universal message structure for chat conversations
//! - [`CommonUsage`] - Standardized token usage tracking
//! - [`FinishReason`] - Unified completion termination reasons
//! - [`ModelParameters`] - Validated model parameters with range checking
//! - [`HttpContentType`] - HTTP content types for API requests
//! - [`HttpMethod`] - HTTP methods with compile-time constants
//! - [`ProviderMetadata`] - Provider identification and capabilities
//! - [`StreamingMode`] - Streaming configuration options
//!
//! # Zero-Allocation Design
//!
//! Uses `ArrayVec` for bounded collections and stack-allocated strings where possible.
//! All validation is performed at compile-time or with minimal runtime overhead.
//!
//! # Usage Example
//!
//! ```rust
//! use fluent_ai_domain::http::common::{BaseMessage, ModelParameters, MessageRole};
//! //!
//! // Create messages with zero allocation for small collections
//! let mut messages = ArrayVec::<BaseMessage, 128>::new();
//! messages.push(BaseMessage::user("Hello, assistant!"));
//! messages.push(BaseMessage::assistant("Hello! How can I help you?"));
//!
//! // Validate model parameters
//! let params = ModelParameters::new()
//!     .with_temperature(0.7)?
//!     .with_max_tokens(1000)?
//!     .with_top_p(0.9)?;
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use arrayvec::{ArrayString, ArrayVec};
use serde::{Deserialize, Serialize};
use std::fmt;
use crate::model::Provider;

// Import and re-export canonical MessageRole from chat module
pub use crate::chat::message::MessageRole;

/// Maximum number of messages in a conversation (compile-time bounded)
pub const MAX_MESSAGES: usize = 128;

/// Maximum number of tools per request (compile-time bounded)
pub const MAX_TOOLS: usize = 32;

/// Maximum number of stop sequences (compile-time bounded)
pub const MAX_STOP_SEQUENCES: usize = 8;

/// Maximum length for model names and other identifiers
pub const MAX_IDENTIFIER_LEN: usize = 64;

/// Maximum length for stop sequences
pub const MAX_STOP_SEQUENCE_LEN: usize = 32;

/// Universal message structure for chat conversations
/// 
/// Designed to work across all AI providers while maintaining type safety
/// and zero-allocation patterns for small message collections.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaseMessage {
    /// Message role (user, assistant, system, tool)
    pub role: MessageRole,
    /// Message content text
    pub content: String,
    /// Optional name/identifier for the message sender
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<ArrayString<MAX_IDENTIFIER_LEN>>,
    /// Tool call identifier if this message contains tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<ArrayString<MAX_IDENTIFIER_LEN>>,
    /// Tool calls made by the assistant
    #[serde(skip_serializing_if = "ArrayVec::is_empty")]
    pub tool_calls: ArrayVec<ToolCall, MAX_TOOLS>}

impl BaseMessage {
    /// Create a user message
    #[inline]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: ArrayVec::new()}
    }

    /// Create an assistant message
    #[inline]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: ArrayVec::new()}
    }

    /// Create a system message
    #[inline]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: ArrayVec::new()}
    }

    /// Create a tool result message
    #[inline]
    pub fn tool_result(tool_call_id: &str, content: impl Into<String>) -> Result<Self, ValidationError> {
        let id = ArrayString::from(tool_call_id)
            .map_err(|_| ValidationError::ToolCallIdTooLong(tool_call_id.len()))?;
        
        Ok(Self {
            role: MessageRole::Tool,
            content: content.into(),
            name: None,
            tool_call_id: Some(id),
            tool_calls: ArrayVec::new()})
    }

    /// Add a tool call to this message
    #[inline]
    pub fn with_tool_call(mut self, tool_call: ToolCall) -> Result<Self, ValidationError> {
        self.tool_calls.try_push(tool_call)
            .map_err(|_| ValidationError::TooManyTools)?;
        Ok(self)
    }

    /// Set the name/identifier for this message
    #[inline]
    pub fn with_name(mut self, name: &str) -> Result<Self, ValidationError> {
        self.name = Some(ArrayString::from(name)
            .map_err(|_| ValidationError::NameTooLong(name.len()))?);
        Ok(self)
    }

    /// Check if this message has tool calls
    #[inline]
    pub const fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Get the content as a string slice
    #[inline]
    pub fn content_str(&self) -> &str {
        &self.content
    }
}



/// Tool call representation for function calling
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call
    pub id: ArrayString<MAX_IDENTIFIER_LEN>,
    /// Type of tool call (typically "function")
    #[serde(rename = "type")]
    pub tool_type: ToolCallType,
    /// Function details
    pub function: FunctionCall}

impl ToolCall {
    /// Create a new function tool call
    #[inline]
    pub fn function(id: &str, name: &str, arguments: &str) -> Result<Self, ValidationError> {
        let call_id = ArrayString::from(id)
            .map_err(|_| ValidationError::ToolCallIdTooLong(id.len()))?;
        let func_name = ArrayString::from(name)
            .map_err(|_| ValidationError::FunctionNameTooLong(name.len()))?;

        Ok(Self {
            id: call_id,
            tool_type: ToolCallType::Function,
            function: FunctionCall {
                name: func_name,
                arguments: arguments.to_string()}})
    }
}

/// Type of tool call
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolCallType {
    /// Function call
    Function}

/// Function call details
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Function name
    pub name: ArrayString<MAX_IDENTIFIER_LEN>,
    /// Function arguments as JSON string
    pub arguments: String}

/// Standardized token usage tracking across all providers
/// 
/// Provides a unified interface for token counting that maps to different
/// provider-specific field names while maintaining consistency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommonUsage {
    /// Tokens used in the input/prompt
    pub input_tokens: u32,
    /// Tokens generated in the output/completion
    pub output_tokens: u32,
    /// Total tokens used (input + output)
    pub total_tokens: u32}

impl CommonUsage {
    /// Create new usage tracking
    #[inline]
    pub const fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens}
    }

    /// Create from OpenAI-style field names
    #[inline]
    pub const fn from_openai(prompt_tokens: u32, completion_tokens: u32, total_tokens: u32) -> Self {
        Self {
            input_tokens: prompt_tokens,
            output_tokens: completion_tokens,
            total_tokens}
    }

    /// Create from Anthropic-style field names
    #[inline]
    pub const fn from_anthropic(input_tokens: u32, output_tokens: u32) -> Self {
        Self::new(input_tokens, output_tokens)
    }

    /// Get cost estimation in USD (approximate, based on token prices)
    #[inline]
    pub fn estimate_cost_usd(&self, input_price_per_1k: f64, output_price_per_1k: f64) -> f64 {
        let input_cost = f64::from(self.input_tokens) / 1000.0 * input_price_per_1k;
        let output_cost = f64::from(self.output_tokens) / 1000.0 * output_price_per_1k;
        input_cost + output_cost
    }
}

/// Unified completion termination reasons across all providers
/// 
/// Maps provider-specific finish reasons to a common set of values
/// for consistent handling across different AI services.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Normal completion (reached natural stopping point)
    Stop,
    /// Reached maximum token limit
    Length,
    /// Content filtered due to safety policies
    ContentFilter,
    /// Completed tool/function calls
    ToolCalls,
    /// Stopped due to provided stop sequence
    StopSequence,
    /// Provider-specific error or timeout
    Error,
    /// Unknown or unrecognized finish reason
    Unknown}

impl FinishReason {
    /// Create from OpenAI finish reason string
    #[inline]
    pub fn from_openai(reason: &str) -> Self {
        match reason {
            "stop" => Self::Stop,
            "length" => Self::Length,
            "content_filter" => Self::ContentFilter,
            "tool_calls" => Self::ToolCalls,
            _ => Self::Unknown}
    }

    /// Create from Anthropic stop reason string
    #[inline]
    pub fn from_anthropic(reason: &str) -> Self {
        match reason {
            "end_turn" | "stop_sequence" => Self::Stop,
            "max_tokens" => Self::Length,
            "tool_use" => Self::ToolCalls,
            _ => Self::Unknown}
    }

    /// Create from generic string (attempts intelligent mapping)
    #[inline]
    pub fn from_str(reason: &str) -> Self {
        match reason.to_lowercase().as_str() {
            "stop" | "end_turn" | "stop_sequence" => Self::Stop,
            "length" | "max_tokens" | "max_length" => Self::Length,
            "content_filter" | "safety" | "filtered" => Self::ContentFilter,
            "tool_calls" | "tool_use" | "function_call" => Self::ToolCalls,
            "error" | "timeout" | "failed" => Self::Error,
            _ => Self::Unknown}
    }
}

impl fmt::Display for FinishReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ContentFilter => "content_filter",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::StopSequence => "stop_sequence",
            FinishReason::Error => "error",
            FinishReason::Unknown => "unknown"};
        write!(f, "{s}")
    }
}

impl From<&str> for FinishReason {
    fn from(s: &str) -> Self {
        match s {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            "content_filter" => FinishReason::ContentFilter,
            "tool_calls" => FinishReason::ToolCalls,
            "stop_sequence" => FinishReason::StopSequence,
            "error" => FinishReason::Error,
            _ => FinishReason::Unknown, // Default fallback
        }
    }
}

/// Validated model parameters with range checking
/// 
/// Ensures all parameters are within valid ranges for AI model inference.
/// Provides compile-time and runtime validation to prevent invalid API calls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelParameters {
    /// Model identifier
    pub model: ArrayString<MAX_IDENTIFIER_LEN>,
    /// Temperature for randomness (0.0 = deterministic, 2.0 = very random)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Top-p nucleus sampling (0.0 to 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Top-k sampling (1 or higher)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Frequency penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// Presence penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Stop sequences
    #[serde(skip_serializing_if = "ArrayVec::is_empty")]
    pub stop_sequences: ArrayVec<ArrayString<MAX_STOP_SEQUENCE_LEN>, MAX_STOP_SEQUENCES>,
    /// Enable streaming response
    pub stream: bool}

impl ModelParameters {
    /// Create new model parameters with defaults
    #[inline]
    pub fn new(model: &str) -> Result<Self, ValidationError> {
        let model_name = ArrayString::from(model)
            .map_err(|_| ValidationError::ModelNameTooLong(model.len()))?;
        
        Ok(Self {
            model: model_name,
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: ArrayVec::new(),
            stream: false})
    }

    /// Set temperature with validation (0.0 to 2.0)
    #[inline]
    pub fn with_temperature(mut self, temperature: f64) -> Result<Self, ValidationError> {
        if !(0.0..=2.0).contains(&temperature) {
            return Err(ValidationError::TemperatureOutOfRange(temperature));
        }
        self.temperature = Some(temperature);
        Ok(self)
    }

    /// Set max tokens with validation
    #[inline]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Result<Self, ValidationError> {
        if max_tokens == 0 {
            return Err(ValidationError::MaxTokensZero);
        }
        if max_tokens > 128_000 {
            return Err(ValidationError::MaxTokensTooLarge(max_tokens));
        }
        self.max_tokens = Some(max_tokens);
        Ok(self)
    }

    /// Set top-p with validation (0.0 to 1.0)
    #[inline]
    pub fn with_top_p(mut self, top_p: f64) -> Result<Self, ValidationError> {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(ValidationError::TopPOutOfRange(top_p));
        }
        self.top_p = Some(top_p);
        Ok(self)
    }

    /// Set top-k with validation (must be positive)
    #[inline]
    pub fn with_top_k(mut self, top_k: u32) -> Result<Self, ValidationError> {
        if top_k == 0 {
            return Err(ValidationError::TopKZero);
        }
        self.top_k = Some(top_k);
        Ok(self)
    }

    /// Set frequency penalty with validation (-2.0 to 2.0)
    #[inline]
    pub fn with_frequency_penalty(mut self, penalty: f64) -> Result<Self, ValidationError> {
        if !(-2.0..=2.0).contains(&penalty) {
            return Err(ValidationError::FrequencyPenaltyOutOfRange(penalty));
        }
        self.frequency_penalty = Some(penalty);
        Ok(self)
    }

    /// Set presence penalty with validation (-2.0 to 2.0)
    #[inline]
    pub fn with_presence_penalty(mut self, penalty: f64) -> Result<Self, ValidationError> {
        if !(-2.0..=2.0).contains(&penalty) {
            return Err(ValidationError::PresencePenaltyOutOfRange(penalty));
        }
        self.presence_penalty = Some(penalty);
        Ok(self)
    }

    /// Add a stop sequence with validation
    #[inline]
    pub fn with_stop_sequence(mut self, sequence: &str) -> Result<Self, ValidationError> {
        if sequence.is_empty() {
            return Err(ValidationError::EmptyStopSequence);
        }
        
        let stop_seq = ArrayString::from(sequence)
            .map_err(|_| ValidationError::StopSequenceTooLong(sequence.len()))?;
        
        self.stop_sequences.try_push(stop_seq)
            .map_err(|_| ValidationError::TooManyStopSequences)?;
        
        Ok(self)
    }

    /// Enable streaming response
    #[inline]
    pub fn with_streaming(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Validate all parameters are consistent
    #[inline]
    pub fn validate(&self) -> Result<(), ValidationError> {
        // Check that temperature and top_p aren't both specified (some providers don't allow this)
        if self.temperature.is_some() && self.top_p.is_some() {
            if let (Some(temp), Some(top_p)) = (self.temperature, self.top_p) {
                // Allow both if they're reasonable values
                if temp > 1.5 && top_p < 0.5 {
                    return Err(ValidationError::ConflictingParameters);
                }
            }
        }
        
        Ok(())
    }
}

/// HTTP content types for API requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HttpContentType {
    /// application/json
    ApplicationJson,
    /// multipart/form-data
    MultipartFormData,
    /// text/plain
    TextPlain,
    /// application/octet-stream
    ApplicationOctetStream}

impl HttpContentType {
    /// Get the MIME type string
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            HttpContentType::ApplicationJson => "application/json",
            HttpContentType::MultipartFormData => "multipart/form-data",
            HttpContentType::TextPlain => "text/plain",
            HttpContentType::ApplicationOctetStream => "application/octet-stream"}
    }
}

impl fmt::Display for HttpContentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// HTTP methods for API requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HttpMethod {
    /// GET method
    Get,
    /// POST method (most common for AI APIs)
    Post,
    /// PUT method
    Put,
    /// PATCH method
    Patch,
    /// DELETE method
    Delete}

impl HttpMethod {
    /// Get the method string
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            HttpMethod::Get => "GET",
            HttpMethod::Post => "POST",
            HttpMethod::Put => "PUT",
            HttpMethod::Patch => "PATCH",
            HttpMethod::Delete => "DELETE"}
    }
}

impl fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Provider identification and capabilities metadata
/// 
/// Contains information about AI provider capabilities and configuration
/// for optimal request routing and feature detection.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderMetadata {
    /// Provider identifier
    pub provider: Provider,
    /// Base URL for API requests
    pub base_url: String,
    /// Maximum tokens supported by this provider
    pub max_tokens_supported: u32,
    /// Whether streaming is supported
    pub supports_streaming: bool,
    /// Whether function calling is supported
    pub supports_function_calling: bool,
    /// Whether vision/image input is supported
    pub supports_vision: bool,
    /// Default timeout for requests (in milliseconds)
    pub default_timeout_ms: u64}

impl ProviderMetadata {
    /// Create new provider metadata
    #[inline]
    pub fn new(provider: Provider) -> Self {
        let base_url = provider.default_base_url().to_string();
        let max_tokens_supported = match &provider {
            Provider::OpenAI | Provider::Azure => 128_000,
            Provider::Anthropic => 200_000,
            Provider::VertexAI | Provider::Gemini => 2_000_000,
            _ => 32_000, // Conservative default
        };
        let supports_streaming = true; // All providers support streaming in this architecture
        let supports_function_calling = provider.supports_function_calling();
        let supports_vision = matches!(&provider, 
            Provider::OpenAI | Provider::Azure | 
            Provider::VertexAI | Provider::Gemini |
            Provider::Anthropic
        );
        
        Self {
            provider: provider.clone(),
            base_url,
            max_tokens_supported,
            supports_streaming,
            supports_function_calling,
            supports_vision,
            default_timeout_ms: 60_000, // 60 seconds default
        }
    }

    /// Set custom base URL
    #[inline]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set custom timeout
    #[inline]
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.default_timeout_ms = timeout_ms;
        self
    }
}

/// Streaming configuration options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingMode {
    /// No streaming, return complete response
    None,
    /// Server-Sent Events (SSE) streaming
    ServerSentEvents,
    /// JSON Lines streaming
    JsonLines,
    /// Provider-specific streaming format
    ProviderSpecific}

impl StreamingMode {
    /// Check if streaming is enabled
    #[inline]
    pub const fn is_streaming(&self) -> bool {
        !matches!(self, StreamingMode::None)
    }

    /// Get the appropriate content type for this streaming mode
    #[inline]
    pub const fn content_type(&self) -> HttpContentType {
        match self {
            StreamingMode::None => HttpContentType::ApplicationJson,
            StreamingMode::ServerSentEvents => HttpContentType::TextPlain,
            StreamingMode::JsonLines => HttpContentType::ApplicationJson,
            StreamingMode::ProviderSpecific => HttpContentType::ApplicationJson}
    }
}

/// Validation errors for parameter and input validation
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// Model name is too long
    ModelNameTooLong(usize),
    /// Temperature is out of valid range (0.0 to 2.0)
    TemperatureOutOfRange(f64),
    /// Max tokens is zero
    MaxTokensZero,
    /// Max tokens is too large
    MaxTokensTooLarge(u32),
    /// Top-p is out of valid range (0.0 to 1.0)
    TopPOutOfRange(f64),
    /// Top-k is zero
    TopKZero,
    /// Frequency penalty is out of valid range (-2.0 to 2.0)
    FrequencyPenaltyOutOfRange(f64),
    /// Presence penalty is out of valid range (-2.0 to 2.0)
    PresencePenaltyOutOfRange(f64),
    /// Stop sequence is empty
    EmptyStopSequence,
    /// Stop sequence is too long
    StopSequenceTooLong(usize),
    /// Too many stop sequences
    TooManyStopSequences,
    /// Name/identifier is too long
    NameTooLong(usize),
    /// Tool call ID is too long
    ToolCallIdTooLong(usize),
    /// Function name is too long
    FunctionNameTooLong(usize),
    /// Too many tools in request
    TooManyTools,
    /// Conflicting parameters specified
    ConflictingParameters}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::ModelNameTooLong(len) => {
                write!(f, "Model name too long: {len} characters (max {MAX_IDENTIFIER_LEN})")
            }
            ValidationError::TemperatureOutOfRange(temp) => {
                write!(f, "Temperature {temp} out of range (0.0 to 2.0)")
            }
            ValidationError::MaxTokensZero => {
                write!(f, "Max tokens cannot be zero")
            }
            ValidationError::MaxTokensTooLarge(tokens) => {
                write!(f, "Max tokens {tokens} too large (max 128,000)")
            }
            ValidationError::TopPOutOfRange(top_p) => {
                write!(f, "Top-p {top_p} out of range (0.0 to 1.0)")
            }
            ValidationError::TopKZero => {
                write!(f, "Top-k cannot be zero")
            }
            ValidationError::FrequencyPenaltyOutOfRange(penalty) => {
                write!(f, "Frequency penalty {penalty} out of range (-2.0 to 2.0)")
            }
            ValidationError::PresencePenaltyOutOfRange(penalty) => {
                write!(f, "Presence penalty {penalty} out of range (-2.0 to 2.0)")
            }
            ValidationError::EmptyStopSequence => {
                write!(f, "Stop sequence cannot be empty")
            }
            ValidationError::StopSequenceTooLong(len) => {
                write!(f, "Stop sequence too long: {len} characters (max {MAX_STOP_SEQUENCE_LEN})")
            }
            ValidationError::TooManyStopSequences => {
                write!(f, "Too many stop sequences (max {MAX_STOP_SEQUENCES})")
            }
            ValidationError::NameTooLong(len) => {
                write!(f, "Name too long: {len} characters (max {MAX_IDENTIFIER_LEN})")
            }
            ValidationError::ToolCallIdTooLong(len) => {
                write!(f, "Tool call ID too long: {len} characters (max {MAX_IDENTIFIER_LEN})")
            }
            ValidationError::FunctionNameTooLong(len) => {
                write!(f, "Function name too long: {len} characters (max {MAX_IDENTIFIER_LEN})")
            }
            ValidationError::TooManyTools => {
                write!(f, "Too many tools (max {MAX_TOOLS})")
            }
            ValidationError::ConflictingParameters => {
                write!(f, "Conflicting parameters specified (high temperature with low top-p)")
            }
        }
    }
}

impl std::error::Error for ValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_message_creation() {
        let user_msg = BaseMessage::user("Hello");
        assert_eq!(user_msg.role, MessageRole::User);
        assert_eq!(user_msg.content, "Hello");
        assert!(!user_msg.has_tool_calls());

        let assistant_msg = BaseMessage::assistant("Hi there!");
        assert_eq!(assistant_msg.role, MessageRole::Assistant);
        assert_eq!(assistant_msg.content, "Hi there!");
    }

    #[test]
    fn test_common_usage() {
        let usage = CommonUsage::new(100, 50);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);

        let openai_usage = CommonUsage::from_openai(100, 50, 150);
        assert_eq!(openai_usage, usage);

        let cost = usage.estimate_cost_usd(0.01, 0.02);
        assert!((cost - 0.002).abs() < f64::EPSILON);
    }

    #[test]
    fn test_finish_reason_mapping() {
        assert_eq!(FinishReason::from_openai("stop"), FinishReason::Stop);
        assert_eq!(FinishReason::from_openai("length"), FinishReason::Length);
        assert_eq!(FinishReason::from_openai("tool_calls"), FinishReason::ToolCalls);
        
        assert_eq!(FinishReason::from_anthropic("end_turn"), FinishReason::Stop);
        assert_eq!(FinishReason::from_anthropic("max_tokens"), FinishReason::Length);
        assert_eq!(FinishReason::from_anthropic("tool_use"), FinishReason::ToolCalls);
    }

    #[test]
    fn test_model_parameters_validation() {
        let params = ModelParameters::new("gpt-4").expect("Valid model name");
        
        let params_with_temp = params.with_temperature(0.7).expect("Valid temperature");
        assert_eq!(params_with_temp.temperature, Some(0.7));
        
        let invalid_temp = ModelParameters::new("gpt-4")
            .expect("Valid model")
            .with_temperature(3.0);
        assert!(invalid_temp.is_err());

        let params_with_tokens = ModelParameters::new("gpt-4")
            .expect("Valid model")
            .with_max_tokens(1000)
            .expect("Valid max tokens");
        assert_eq!(params_with_tokens.max_tokens, Some(1000));

        let invalid_tokens = ModelParameters::new("gpt-4")
            .expect("Valid model")
            .with_max_tokens(0);
        assert!(invalid_tokens.is_err());
    }

    #[test]
    fn test_tool_call_creation() {
        let tool_call = ToolCall::function("call_123", "get_weather", r#"{"location": "NYC"}"#)
            .expect("Valid tool call");
        
        assert_eq!(tool_call.id.as_str(), "call_123");
        assert_eq!(tool_call.function.name.as_str(), "get_weather");
        assert_eq!(tool_call.function.arguments, r#"{"location": "NYC"}"#);
    }

    #[test]
    fn test_provider_metadata() {
        let metadata = ProviderMetadata::new(super::super::Provider::OpenAI);
        assert_eq!(metadata.provider, super::super::Provider::OpenAI);
        assert_eq!(metadata.max_tokens_supported, 128_000);
        assert!(metadata.supports_streaming);
        assert!(metadata.supports_function_calling);
        assert!(metadata.supports_vision);
    }

    #[test]
    fn test_streaming_mode() {
        assert!(!StreamingMode::None.is_streaming());
        assert!(StreamingMode::ServerSentEvents.is_streaming());
        assert!(StreamingMode::JsonLines.is_streaming());
        
        assert_eq!(StreamingMode::ServerSentEvents.content_type(), HttpContentType::TextPlain);
        assert_eq!(StreamingMode::JsonLines.content_type(), HttpContentType::ApplicationJson);
    }
}