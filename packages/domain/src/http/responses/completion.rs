//! Completion Response Models for AI Provider Integration
//
//! This module provides unified completion response structures that work across all 17 AI providers
//! while maintaining type safety, zero-allocation patterns, and efficient streaming capabilities.
//
//! # Response Types
//
//! - [`CompletionResponse`] - Complete response for non-streaming calls
//! - [`CompletionChunk`] - Streaming response chunk for real-time completion
//! - [`CompletionChoice`] - Individual completion candidate
//! - [`StreamingResponse`] - Unified streaming interface across providers
//
//! # Supported Providers
//
//! - **OpenAI**: Choices with deltas, tool calls, usage tracking
//! - **Anthropic**: Type-based chunks with content blocks and tool use
//! - **Google**: Candidates with safety ratings and function calling
//! - **AWS Bedrock**: Converse responses with message deltas
// - **Cohere**: Event-based streaming with text generation
// - **Azure OpenAI**: All OpenAI response formats via Azure

// - **Local/OSS**: Ollama, HuggingFace, Together response formats
// - **Commercial**: AI21, Groq, Mistral, Perplexity, OpenRouter, xAI, DeepSeek
//
// # Streaming Architecture
//
// The streaming system uses a unified `CompletionChunk` format that all providers
// convert to, enabling consistent handling regardless of the underlying provider.
//
// # Usage Examples
//
// ```rust
// use fluent_ai_domain::http::responses::completion::{CompletionResponse, CompletionChunk};
// use fluent_ai_domain::Provider;
//
// // Parse provider-specific response to unified format
// let response: CompletionResponse = CompletionResponse::from_provider_json(
//     Provider::OpenAI,
//     &openai_json
// )?;
//
// // Handle streaming chunks
// let chunk: CompletionChunk = CompletionChunk::from_provider_chunk(
//     Provider::Anthropic,
//     &anthropic_sse_data
// )?;
// ```
#[warn(missing_docs)]
#[forbid(unsafe_code)]
#[deny(clippy::all)]
#[deny(clippy::pedantic)]
#[allow(clippy::module_name_repetitions)]
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use arrayvec::{ArrayString, ArrayVec};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Provider;
use crate::http::common::{
    BaseMessage, CommonUsage, FinishReason, MAX_IDENTIFIER_LEN, MAX_TOOLS, ToolCall,
    ValidationError,
};

/// Maximum number of choices in a completion response
pub const MAX_CHOICES: usize = 16;

/// Maximum number of candidates in a response (Google)
pub const MAX_CANDIDATES: usize = 8;

/// Maximum length for content chunks
pub const MAX_CHUNK_CONTENT_LEN: usize = 4096;

/// Maximum length for event types and IDs
pub const MAX_EVENT_LEN: usize = 64;

/// Maximum number of log probabilities
pub const MAX_LOGPROBS: usize = 100;

/// Universal completion response for non-streaming calls
///
/// This struct provides a unified interface for completion responses across all providers
/// while preserving provider-specific metadata and features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Response ID for tracking and debugging
    pub id: ArrayString<MAX_IDENTIFIER_LEN>,

    /// Object type (typically "chat.completion")
    pub object: ArrayString<32>,

    /// Model that generated the response
    pub model: ArrayString<MAX_IDENTIFIER_LEN>,

    /// Response creation timestamp (Unix seconds)
    pub created: u64,

    /// Completion choices with bounded capacity
    pub choices: ArrayVec<CompletionChoice, MAX_CHOICES>,

    /// Token usage statistics
    pub usage: CommonUsage,

    /// System fingerprint for reproducibility
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<ArrayString<64>>,

    /// Provider-specific metadata
    #[serde(skip_serializing_if = "ProviderMetadata::is_empty")]
    pub provider_metadata: ProviderMetadata,
}

impl CompletionResponse {
    /// Create a new completion response
    #[inline]
    pub fn new(
        id: &str,
        model: &str,
        choices: Vec<CompletionChoice>,
        usage: CommonUsage,
    ) -> Result<Self, CompletionResponseError> {
        let response_id =
            ArrayString::from(id).map_err(|_| CompletionResponseError::IdTooLong(id.len()))?;
        let model_name = ArrayString::from(model)
            .map_err(|_| CompletionResponseError::ModelNameTooLong(model.len()))?;

        let mut choice_array = ArrayVec::new();
        for choice in choices {
            choice_array
                .try_push(choice)
                .map_err(|_| CompletionResponseError::TooManyChoices)?;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| CompletionResponseError::SystemTimeError)?
            .as_secs();

        Ok(Self {
            id: response_id,
            object: ArrayString::from("chat.completion")
                .map_err(|_| CompletionResponseError::InternalError)?,
            model: model_name,
            created: now,
            choices: choice_array,
            usage,
            system_fingerprint: None,
            provider_metadata: ProviderMetadata::new(),
        })
    }

    /// Parse provider-specific JSON to unified response format
    #[inline]
    pub fn from_provider_json(
        provider: Provider,
        json: &Value,
    ) -> Result<Self, CompletionResponseError> {
        match provider {
            Provider::OpenAI | Provider::Azure => Self::from_openai_json(json),
            Provider::Anthropic => Self::from_anthropic_json(json),
            Provider::VertexAI | Provider::Gemini => Self::from_google_json(json),
            Provider::Bedrock => Self::from_bedrock_json(json),
            Provider::Cohere => Self::from_cohere_json(json),
            Provider::Ollama => Self::from_ollama_json(json),
            Provider::Groq | Provider::OpenRouter | Provider::Together => {
                Self::from_openai_compatible_json(json)
            }
            Provider::AI21 => Self::from_ai21_json(json),
            Provider::Mistral => Self::from_mistral_json(json),
            Provider::HuggingFace => Self::from_huggingface_json(json),
            Provider::Perplexity => Self::from_perplexity_json(json),
            Provider::XAI => Self::from_xai_json(json),
            Provider::DeepSeek => Self::from_deepseek_json(json),
        }
    }

    /// Parse OpenAI response format
    #[inline]
    fn from_openai_json(json: &Value) -> Result<Self, CompletionResponseError> {
        let id = json.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
        let model = json
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let usage = if let Some(usage_obj) = json.get("usage") {
            let prompt_tokens = usage_obj
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let completion_tokens = usage_obj
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let total_tokens = usage_obj
                .get("total_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            CommonUsage::from_openai(prompt_tokens, completion_tokens, total_tokens)
        } else {
            CommonUsage::new(0, 0)
        };

        let mut choices = Vec::new();
        if let Some(choices_array) = json.get("choices").and_then(|v| v.as_array()) {
            for (index, choice_obj) in choices_array.iter().enumerate() {
                let choice = CompletionChoice::from_openai_choice(choice_obj, index)?;
                choices.push(choice);
            }
        }

        let mut response = Self::new(id, model, choices, usage)?;

        // Add system fingerprint if present
        if let Some(fingerprint) = json.get("system_fingerprint").and_then(|v| v.as_str()) {
            response.system_fingerprint = Some(ArrayString::from(fingerprint).unwrap_or_default());
        }

        Ok(response)
    }

    /// Parse Anthropic response format
    #[inline]
    fn from_anthropic_json(json: &Value) -> Result<Self, CompletionResponseError> {
        let id = json.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
        let model = json
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let usage = if let Some(usage_obj) = json.get("usage") {
            let input_tokens = usage_obj
                .get("input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let output_tokens = usage_obj
                .get("output_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            CommonUsage::from_anthropic(input_tokens, output_tokens)
        } else {
            CommonUsage::new(0, 0)
        };

        // Anthropic has content array instead of choices
        let mut choices = Vec::new();
        if let Some(content_array) = json.get("content").and_then(|v| v.as_array()) {
            let mut full_text = String::new();
            let mut tool_calls = Vec::new();

            for content_block in content_array {
                if let Some(text) = content_block.get("text").and_then(|v| v.as_str()) {
                    full_text.push_str(text);
                } else if content_block.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
                    // Handle tool use blocks
                    if let Some(tool_call) = Self::parse_anthropic_tool_use(content_block)? {
                        tool_calls.push(tool_call);
                    }
                }
            }

            let finish_reason = json
                .get("stop_reason")
                .and_then(|v| v.as_str())
                .map(FinishReason::from_anthropic)
                .unwrap_or(FinishReason::Stop);

            let choice = CompletionChoice {
                index: 0,
                message: BaseMessage {
                    role: crate::http::common::MessageRole::Assistant,
                    content: full_text,
                    name: None,
                    tool_call_id: None,
                    tool_calls: {
                        let mut array_vec = ArrayVec::new();
                        for call in tool_calls.into_iter().take(32) {
                            if array_vec.try_push(call).is_err() {
                                break;
                            }
                        }
                        array_vec
                    },
                },
                finish_reason: Some(finish_reason),
                logprobs: None,
            };

            choices.push(choice);
        }

        Self::new(id, model, choices, usage)
    }

    /// Parse Google response format
    #[inline]
    fn from_google_json(json: &Value) -> Result<Self, CompletionResponseError> {
        let model = json
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        // Google doesn't provide traditional usage in the same format
        let usage = CommonUsage::new(0, 0); // Would need to be calculated from metadata

        let mut choices = Vec::new();
        if let Some(candidates) = json.get("candidates").and_then(|v| v.as_array()) {
            for (index, candidate) in candidates.iter().enumerate() {
                let choice = CompletionChoice::from_google_candidate(candidate, index)?;
                choices.push(choice);
            }
        }

        Self::new("google-response", model, choices, usage)
    }

    /// Parse Bedrock response format
    #[inline]
    fn from_bedrock_json(json: &Value) -> Result<Self, CompletionResponseError> {
        let model = json
            .get("modelId")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let usage = if let Some(usage_obj) = json.get("usage") {
            let input_tokens = usage_obj
                .get("inputTokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let output_tokens = usage_obj
                .get("outputTokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            CommonUsage::new(input_tokens, output_tokens)
        } else {
            CommonUsage::new(0, 0)
        };

        let mut choices = Vec::new();
        if let Some(output) = json.get("output") {
            let choice = CompletionChoice::from_bedrock_output(output, 0)?;
            choices.push(choice);
        }

        Self::new("bedrock-response", model, choices, usage)
    }

    /// Parse Cohere response format
    #[inline]
    fn from_cohere_json(json: &Value) -> Result<Self, CompletionResponseError> {
        let model = json
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let usage = if let Some(meta) = json.get("meta") {
            if let Some(billed_units) = meta.get("billed_units") {
                let input_tokens = billed_units
                    .get("input_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
                let output_tokens = billed_units
                    .get("output_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
                CommonUsage::new(input_tokens, output_tokens)
            } else {
                CommonUsage::new(0, 0)
            }
        } else {
            CommonUsage::new(0, 0)
        };

        let mut choices = Vec::new();
        if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
            let finish_reason = json
                .get("finish_reason")
                .and_then(|v| v.as_str())
                .map(FinishReason::from)
                .unwrap_or(FinishReason::Stop);

            let choice = CompletionChoice {
                index: 0,
                message: BaseMessage::assistant(text),
                finish_reason: Some(finish_reason),
                logprobs: None,
            };
            choices.push(choice);
        }

        Self::new("cohere-response", model, choices, usage)
    }

    /// Parse Ollama response format
    #[inline]
    fn from_ollama_json(json: &Value) -> Result<Self, CompletionResponseError> {
        let model = json
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        // Ollama provides eval_count and prompt_eval_count
        let usage = if let Some(prompt_eval_count) =
            json.get("prompt_eval_count").and_then(|v| v.as_u64())
        {
            let eval_count = json.get("eval_count").and_then(|v| v.as_u64()).unwrap_or(0);
            CommonUsage::new(prompt_eval_count as u32, eval_count as u32)
        } else {
            CommonUsage::new(0, 0)
        };

        let mut choices = Vec::new();
        if let Some(message) = json.get("message") {
            let content = message
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let choice = CompletionChoice {
                index: 0,
                message: BaseMessage::assistant(content),
                finish_reason: Some(FinishReason::Stop),
                logprobs: None,
            };
            choices.push(choice);
        }

        Self::new("ollama-response", model, choices, usage)
    }

    /// Parse OpenAI-compatible response format (Groq, OpenRouter, Together)
    #[inline]
    fn from_openai_compatible_json(json: &Value) -> Result<Self, CompletionResponseError> {
        // Most providers are OpenAI-compatible
        Self::from_openai_json(json)
    }

    /// Parse AI21 response format
    #[inline]
    fn from_ai21_json(json: &Value) -> Result<Self, CompletionResponseError> {
        let model = json
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let usage = CommonUsage::new(0, 0); // AI21 may have different usage format

        let mut choices = Vec::new();
        if let Some(completions) = json.get("completions").and_then(|v| v.as_array()) {
            for (index, completion) in completions.iter().enumerate() {
                if let Some(text) = completion
                    .get("data")
                    .and_then(|d| d.get("text"))
                    .and_then(|v| v.as_str())
                {
                    let choice = CompletionChoice {
                        index: index as u32,
                        message: BaseMessage::assistant(text),
                        finish_reason: Some(FinishReason::Stop),
                        logprobs: None,
                    };
                    choices.push(choice);
                }
            }
        }

        Self::new("ai21-response", model, choices, usage)
    }

    /// Parse Mistral response format
    #[inline]
    fn from_mistral_json(json: &Value) -> Result<Self, CompletionResponseError> {
        // Mistral is mostly OpenAI-compatible
        Self::from_openai_json(json)
    }

    /// Parse HuggingFace response format
    #[inline]
    fn from_huggingface_json(json: &Value) -> Result<Self, CompletionResponseError> {
        // HuggingFace can vary, but often follows OpenAI format
        Self::from_openai_json(json)
    }

    /// Parse Perplexity response format
    #[inline]
    fn from_perplexity_json(json: &Value) -> Result<Self, CompletionResponseError> {
        // Perplexity is OpenAI-compatible
        Self::from_openai_json(json)
    }

    /// Parse xAI response format
    #[inline]
    fn from_xai_json(json: &Value) -> Result<Self, CompletionResponseError> {
        // xAI is OpenAI-compatible
        Self::from_openai_json(json)
    }

    /// Parse DeepSeek response format
    #[inline]
    fn from_deepseek_json(json: &Value) -> Result<Self, CompletionResponseError> {
        // DeepSeek is OpenAI-compatible
        Self::from_openai_json(json)
    }

    /// Parse Anthropic tool use block
    #[inline]
    fn parse_anthropic_tool_use(
        content_block: &Value,
    ) -> Result<Option<ToolCall>, CompletionResponseError> {
        let tool_id = content_block
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let function_name = content_block
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let arguments = content_block
            .get("input")
            .map(|v| serde_json::to_string(v).unwrap_or_default())
            .unwrap_or_default();

        ToolCall::function(tool_id, function_name, &arguments)
            .map(Some)
            .map_err(|e| CompletionResponseError::ValidationError(e))
    }

    /// Get the primary completion text
    #[inline]
    pub fn primary_text(&self) -> Option<&str> {
        self.choices
            .first()
            .map(|choice| choice.message.content_str())
    }

    /// Get all completion texts
    #[inline]
    pub fn all_texts(&self) -> Vec<&str> {
        self.choices
            .iter()
            .map(|choice| choice.message.content_str())
            .collect()
    }

    /// Get the finish reason for the primary choice
    #[inline]
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.choices.first().and_then(|choice| choice.finish_reason)
    }

    /// Check if the response contains tool calls
    #[inline]
    pub fn has_tool_calls(&self) -> bool {
        self.choices
            .iter()
            .any(|choice| choice.message.has_tool_calls())
    }

    /// Get all tool calls from all choices
    #[inline]
    pub fn tool_calls(&self) -> Vec<&ToolCall> {
        self.choices
            .iter()
            .flat_map(|choice| choice.message.tool_calls.iter())
            .collect()
    }
}

/// Individual completion choice within a response
///
/// Each choice represents a different completion candidate, with providers like OpenAI
/// supporting multiple choices per request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    /// Choice index within the response
    pub index: u32,

    /// The completion message
    pub message: BaseMessage,

    /// Reason the generation finished
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,

    /// Log probabilities information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
}

impl CompletionChoice {
    /// Create a new completion choice
    #[inline]
    pub fn new(index: u32, message: BaseMessage, finish_reason: Option<FinishReason>) -> Self {
        Self {
            index,
            message,
            finish_reason,
            logprobs: None,
        }
    }

    /// Create from OpenAI choice object
    #[inline]
    fn from_openai_choice(
        choice_obj: &Value,
        index: usize,
    ) -> Result<Self, CompletionResponseError> {
        let message_obj = choice_obj
            .get("message")
            .ok_or(CompletionResponseError::MissingField("message"))?;

        let role = message_obj
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("assistant");
        let content = message_obj
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let mut tool_calls = ArrayVec::new();
        if let Some(calls_array) = message_obj.get("tool_calls").and_then(|v| v.as_array()) {
            for call_obj in calls_array {
                if let Some(tool_call) = Self::parse_openai_tool_call(call_obj)? {
                    tool_calls
                        .try_push(tool_call)
                        .map_err(|_| CompletionResponseError::TooManyToolCalls)?;
                }
            }
        }

        let message = BaseMessage {
            role: match role {
                "user" => crate::http::common::MessageRole::User,
                "assistant" => crate::http::common::MessageRole::Assistant,
                "system" => crate::http::common::MessageRole::System,
                "tool" => crate::http::common::MessageRole::Tool,
                _ => crate::http::common::MessageRole::Assistant, // Default fallback
            },
            content: content.to_string(),
            name: None,
            tool_call_id: None,
            tool_calls,
        };

        let finish_reason = choice_obj
            .get("finish_reason")
            .and_then(|v| v.as_str())
            .map(FinishReason::from);

        Ok(Self::new(index as u32, message, finish_reason))
    }

    /// Create from Google candidate object
    #[inline]
    fn from_google_candidate(
        candidate: &Value,
        index: usize,
    ) -> Result<Self, CompletionResponseError> {
        let content = candidate
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array())
            .and_then(|parts| parts.first())
            .and_then(|part| part.get("text"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let finish_reason = candidate
            .get("finishReason")
            .and_then(|v| v.as_str())
            .map(|s| FinishReason::from(s)); // Explicit closure to resolve type ambiguity

        let message = BaseMessage::assistant(content);
        Ok(Self::new(index as u32, message, finish_reason))
    }

    /// Create from Bedrock output object
    #[inline]
    fn from_bedrock_output(output: &Value, index: usize) -> Result<Self, CompletionResponseError> {
        let message_obj = output
            .get("message")
            .ok_or(CompletionResponseError::MissingField("message"))?;

        let content = message_obj
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|parts| parts.first())
            .and_then(|part| part.get("text"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let message = BaseMessage::assistant(content);
        Ok(Self::new(index as u32, message, Some(FinishReason::Stop)))
    }

    /// Parse OpenAI tool call object
    #[inline]
    fn parse_openai_tool_call(
        call_obj: &Value,
    ) -> Result<Option<ToolCall>, CompletionResponseError> {
        let id = call_obj
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let function_obj = call_obj
            .get("function")
            .ok_or(CompletionResponseError::MissingField("function"))?;
        let name = function_obj
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let arguments = function_obj
            .get("arguments")
            .and_then(|v| v.as_str())
            .unwrap_or("{}");

        ToolCall::function(id, name, arguments)
            .map(Some)
            .map_err(|e| CompletionResponseError::ValidationError(e))
    }

    /// Get the text content of this choice
    #[inline]
    pub fn text(&self) -> &str {
        self.message.content_str()
    }

    /// Check if this choice contains tool calls
    #[inline]
    pub fn has_tool_calls(&self) -> bool {
        self.message.has_tool_calls()
    }
}

/// Log probabilities information for tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogProbs {
    /// Token log probabilities
    pub tokens: ArrayVec<TokenLogProb, MAX_LOGPROBS>,

    /// Alternative tokens and their probabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<ArrayVec<ArrayVec<TokenLogProb, 5>, MAX_LOGPROBS>>,
}

/// Individual token log probability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogProb {
    /// The token string
    pub token: ArrayString<64>,

    /// Log probability of the token
    pub logprob: f64,

    /// Byte positions of the token
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<[u32; 2]>,
}

/// Streaming completion chunk for real-time responses
///
/// This provides a unified format for streaming chunks across all providers,
/// enabling consistent handling of real-time completion data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChunk {
    /// Chunk ID for tracking
    pub id: ArrayString<MAX_IDENTIFIER_LEN>,

    /// Object type (typically "chat.completion.chunk")
    pub object: ArrayString<32>,

    /// Model that generated the chunk
    pub model: ArrayString<MAX_IDENTIFIER_LEN>,

    /// Chunk creation timestamp (Unix seconds)
    pub created: u64,

    /// Chunk choices with deltas
    pub choices: ArrayVec<ChunkChoice, MAX_CHOICES>,

    /// Token usage statistics (final chunk only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<CommonUsage>,

    /// System fingerprint for reproducibility
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<ArrayString<64>>,
}

impl CompletionChunk {
    /// Create a new completion chunk
    #[inline]
    pub fn new(
        id: &str,
        model: &str,
        choices: Vec<ChunkChoice>,
    ) -> Result<Self, CompletionResponseError> {
        let chunk_id =
            ArrayString::from(id).map_err(|_| CompletionResponseError::IdTooLong(id.len()))?;
        let model_name = ArrayString::from(model)
            .map_err(|_| CompletionResponseError::ModelNameTooLong(model.len()))?;

        let mut choice_array = ArrayVec::new();
        for choice in choices {
            choice_array
                .try_push(choice)
                .map_err(|_| CompletionResponseError::TooManyChoices)?;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| CompletionResponseError::SystemTimeError)?
            .as_secs();

        Ok(Self {
            id: chunk_id,
            object: ArrayString::from("chat.completion.chunk")
                .map_err(|_| CompletionResponseError::InternalError)?,
            model: model_name,
            created: now,
            choices: choice_array,
            usage: None,
            system_fingerprint: None,
        })
    }

    /// Parse provider-specific chunk data to unified format
    #[inline]
    pub fn from_provider_chunk(
        provider: Provider,
        data: &[u8],
    ) -> Result<Self, CompletionResponseError> {
        match provider {
            Provider::OpenAI | Provider::Azure => Self::from_openai_chunk(data),
            Provider::Anthropic => Self::from_anthropic_chunk(data),
            Provider::VertexAI | Provider::Gemini => Self::from_google_chunk(data),
            Provider::Bedrock => Self::from_bedrock_chunk(data),
            Provider::Cohere => Self::from_cohere_chunk(data),
            Provider::Ollama => Self::from_ollama_chunk(data),
            Provider::Groq | Provider::OpenRouter | Provider::Together => {
                Self::from_openai_chunk(data)
            }
            Provider::AI21 => Self::from_ai21_chunk(data),
            Provider::Mistral => Self::from_mistral_chunk(data),
            Provider::HuggingFace => Self::from_huggingface_chunk(data),
            Provider::Perplexity => Self::from_perplexity_chunk(data),
            Provider::XAI => Self::from_xai_chunk(data),
            Provider::DeepSeek => Self::from_deepseek_chunk(data),
        }
    }

    /// Parse OpenAI streaming chunk
    #[inline]
    fn from_openai_chunk(data: &[u8]) -> Result<Self, CompletionResponseError> {
        let json: Value =
            serde_json::from_slice(data).map_err(|_| CompletionResponseError::InvalidJson)?;

        let id = json.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
        let model = json
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let mut choices = Vec::new();
        if let Some(choices_array) = json.get("choices").and_then(|v| v.as_array()) {
            for (index, choice_obj) in choices_array.iter().enumerate() {
                let choice = ChunkChoice::from_openai_delta(choice_obj, index)?;
                choices.push(choice);
            }
        }

        let mut chunk = Self::new(id, model, choices)?;

        // Add usage if present (final chunk)
        if let Some(usage_obj) = json.get("usage") {
            let prompt_tokens = usage_obj
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let completion_tokens = usage_obj
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let total_tokens = usage_obj
                .get("total_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            chunk.usage = Some(CommonUsage::from_openai(
                prompt_tokens,
                completion_tokens,
                total_tokens,
            ));
        }

        Ok(chunk)
    }

    /// Parse Anthropic streaming chunk
    #[inline]
    fn from_anthropic_chunk(data: &[u8]) -> Result<Self, CompletionResponseError> {
        let json: Value =
            serde_json::from_slice(data).map_err(|_| CompletionResponseError::InvalidJson)?;

        let event_type = json
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        match event_type {
            "content_block_delta" => {
                let delta = json
                    .get("delta")
                    .ok_or(CompletionResponseError::MissingField("delta"))?;
                let text = delta.get("text").and_then(|v| v.as_str()).unwrap_or("");

                let choice = ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: Some(text.to_string()),
                        tool_calls: ArrayVec::new(),
                    },
                    finish_reason: None,
                    logprobs: None,
                };

                Self::new("anthropic-chunk", "claude", vec![choice])
            }
            "message_stop" => {
                let choice = ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: None,
                        tool_calls: ArrayVec::new(),
                    },
                    finish_reason: Some(FinishReason::Stop),
                    logprobs: None,
                };

                Self::new("anthropic-chunk", "claude", vec![choice])
            }
            _ => {
                // For other event types, create empty chunk
                Self::new("anthropic-chunk", "claude", vec![])
            }
        }
    }

    /// Parse Google streaming chunk
    #[inline]
    fn from_google_chunk(data: &[u8]) -> Result<Self, CompletionResponseError> {
        let json: Value =
            serde_json::from_slice(data).map_err(|_| CompletionResponseError::InvalidJson)?;

        let mut choices = Vec::new();
        if let Some(candidates) = json.get("candidates").and_then(|v| v.as_array()) {
            for (index, candidate) in candidates.iter().enumerate() {
                let choice = ChunkChoice::from_google_candidate(candidate, index)?;
                choices.push(choice);
            }
        }

        Self::new("google-chunk", "gemini", choices)
    }

    /// Parse Bedrock streaming chunk
    #[inline]
    fn from_bedrock_chunk(data: &[u8]) -> Result<Self, CompletionResponseError> {
        let json: Value =
            serde_json::from_slice(data).map_err(|_| CompletionResponseError::InvalidJson)?;

        let mut choices = Vec::new();
        if let Some(content_block_delta) = json.get("contentBlockDelta") {
            let text = content_block_delta
                .get("delta")
                .and_then(|d| d.get("text"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let choice = ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some(text.to_string()),
                    tool_calls: ArrayVec::new(),
                },
                finish_reason: None,
                logprobs: None,
            };
            choices.push(choice);
        }

        Self::new("bedrock-chunk", "bedrock", choices)
    }

    /// Parse Cohere streaming chunk
    #[inline]
    fn from_cohere_chunk(data: &[u8]) -> Result<Self, CompletionResponseError> {
        let json: Value =
            serde_json::from_slice(data).map_err(|_| CompletionResponseError::InvalidJson)?;

        let event_type = json
            .get("event_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        match event_type {
            "text-generation" => {
                let text = json.get("text").and_then(|v| v.as_str()).unwrap_or("");

                let choice = ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: Some(text.to_string()),
                        tool_calls: ArrayVec::new(),
                    },
                    finish_reason: None,
                    logprobs: None,
                };

                Self::new("cohere-chunk", "command", vec![choice])
            }
            "stream-end" => {
                let choice = ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: None,
                        tool_calls: ArrayVec::new(),
                    },
                    finish_reason: Some(FinishReason::Stop),
                    logprobs: None,
                };

                Self::new("cohere-chunk", "command", vec![choice])
            }
            _ => Self::new("cohere-chunk", "command", vec![]),
        }
    }

    /// Parse Ollama streaming chunk
    #[inline]
    fn from_ollama_chunk(data: &[u8]) -> Result<Self, CompletionResponseError> {
        let json: Value =
            serde_json::from_slice(data).map_err(|_| CompletionResponseError::InvalidJson)?;

        let model = json
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let mut choices = Vec::new();
        if let Some(message) = json.get("message") {
            let content = message
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let done = json.get("done").and_then(|v| v.as_bool()).unwrap_or(false);

            let choice = ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some(content.to_string()),
                    tool_calls: ArrayVec::new(),
                },
                finish_reason: if done { Some(FinishReason::Stop) } else { None },
                logprobs: None,
            };
            choices.push(choice);
        }

        Self::new("ollama-chunk", model, choices)
    }

    /// Parse AI21 streaming chunk
    #[inline]
    fn from_ai21_chunk(data: &[u8]) -> Result<Self, CompletionResponseError> {
        // AI21 may not support streaming or have different format
        Self::from_openai_chunk(data) // Fallback to OpenAI format
    }

    /// Parse Mistral streaming chunk
    #[inline]
    fn from_mistral_chunk(data: &[u8]) -> Result<Self, CompletionResponseError> {
        // Mistral is mostly OpenAI-compatible
        Self::from_openai_chunk(data)
    }

    /// Parse HuggingFace streaming chunk
    #[inline]
    fn from_huggingface_chunk(data: &[u8]) -> Result<Self, CompletionResponseError> {
        // HuggingFace often follows OpenAI format
        Self::from_openai_chunk(data)
    }

    /// Parse Perplexity streaming chunk
    #[inline]
    fn from_perplexity_chunk(data: &[u8]) -> Result<Self, CompletionResponseError> {
        // Perplexity is OpenAI-compatible
        Self::from_openai_chunk(data)
    }

    /// Parse xAI streaming chunk
    #[inline]
    fn from_xai_chunk(data: &[u8]) -> Result<Self, CompletionResponseError> {
        // xAI is OpenAI-compatible
        Self::from_openai_chunk(data)
    }

    /// Parse DeepSeek streaming chunk
    #[inline]
    fn from_deepseek_chunk(data: &[u8]) -> Result<Self, CompletionResponseError> {
        // DeepSeek is OpenAI-compatible
        Self::from_openai_chunk(data)
    }

    /// Get the text content from the first choice
    #[inline]
    pub fn text(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|choice| choice.delta.content.as_deref())
    }

    /// Get the finish reason from the first choice
    #[inline]
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.choices.first().and_then(|choice| choice.finish_reason)
    }

    /// Check if this is the final chunk
    #[inline]
    pub fn is_final(&self) -> bool {
        self.finish_reason().is_some() || self.usage.is_some()
    }
}

/// Individual choice within a streaming chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkChoice {
    /// Choice index within the chunk
    pub index: u32,

    /// Delta containing the incremental changes
    pub delta: ChunkDelta,

    /// Reason the generation finished (only in final chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,

    /// Log probabilities information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
}

impl ChunkChoice {
    /// Create from OpenAI choice delta
    #[inline]
    fn from_openai_delta(
        choice_obj: &Value,
        index: usize,
    ) -> Result<Self, CompletionResponseError> {
        let delta_obj = choice_obj
            .get("delta")
            .ok_or(CompletionResponseError::MissingField("delta"))?;

        let role = delta_obj.get("role").and_then(|v| v.as_str());
        let content = delta_obj.get("content").and_then(|v| v.as_str());

        let mut tool_calls = ArrayVec::new();
        if let Some(calls_array) = delta_obj.get("tool_calls").and_then(|v| v.as_array()) {
            for call_obj in calls_array {
                if let Some(tool_call) = Self::parse_openai_tool_call_delta(call_obj)? {
                    tool_calls
                        .try_push(tool_call)
                        .map_err(|_| CompletionResponseError::TooManyToolCalls)?;
                }
            }
        }

        let delta = ChunkDelta {
            role: role.map(|r| match r {
                "user" => crate::http::common::MessageRole::User,
                "assistant" => crate::http::common::MessageRole::Assistant,
                "system" => crate::http::common::MessageRole::System,
                "tool" => crate::http::common::MessageRole::Tool,
                _ => crate::http::common::MessageRole::Assistant, // Default fallback
            }),
            content: content.map(String::from),
            tool_calls,
        };

        let finish_reason = choice_obj
            .get("finish_reason")
            .and_then(|v| v.as_str())
            .map(FinishReason::from);

        Ok(Self {
            index: index as u32,
            delta,
            finish_reason,
            logprobs: None,
        })
    }

    /// Create from Google candidate delta
    #[inline]
    fn from_google_candidate(
        candidate: &Value,
        index: usize,
    ) -> Result<Self, CompletionResponseError> {
        let content = candidate
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array())
            .and_then(|parts| parts.first())
            .and_then(|part| part.get("text"))
            .and_then(|v| v.as_str());

        let finish_reason = candidate
            .get("finishReason")
            .and_then(|v| v.as_str())
            .map(|s| FinishReason::from(s)); // Explicit closure to resolve type ambiguity

        let delta = ChunkDelta {
            role: None,
            content: content.map(String::from),
            tool_calls: ArrayVec::new(),
        };

        Ok(Self {
            index: index as u32,
            delta,
            finish_reason,
            logprobs: None,
        })
    }

    /// Parse OpenAI tool call delta
    #[inline]
    fn parse_openai_tool_call_delta(
        call_obj: &Value,
    ) -> Result<Option<ToolCall>, CompletionResponseError> {
        let id = call_obj.get("id").and_then(|v| v.as_str());
        let function_obj = call_obj.get("function");

        if let Some(function) = function_obj {
            let name = function.get("name").and_then(|v| v.as_str());
            let arguments = function.get("arguments").and_then(|v| v.as_str());

            if let (Some(tool_id), Some(func_name)) = (id, name) {
                let args = arguments.unwrap_or("{}");
                return ToolCall::function(tool_id, func_name, args)
                    .map(Some)
                    .map_err(|e| CompletionResponseError::ValidationError(e));
            }
        }

        Ok(None)
    }
}

/// Delta containing incremental changes in a streaming chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkDelta {
    /// Role update (only in first chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<crate::http::common::MessageRole>,

    /// Content update
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Tool call updates
    #[serde(skip_serializing_if = "ArrayVec::is_empty")]
    pub tool_calls: ArrayVec<ToolCall, MAX_TOOLS>,
}

/// Unified streaming response interface
///
/// This provides a consistent way to handle streaming responses across all providers,
/// abstracting away provider-specific streaming protocols.
#[derive(Debug)]
pub struct StreamingResponse {
    /// Provider that generated this stream
    pub provider: Provider,

    /// Model being used
    pub model: ArrayString<MAX_IDENTIFIER_LEN>,

    /// Request ID for tracking
    pub request_id: ArrayString<MAX_IDENTIFIER_LEN>,

    /// Stream start timestamp
    pub started_at: u64,

    /// Total chunks received
    pub chunk_count: u32,

    /// Whether the stream has completed
    pub completed: bool,

    /// Final usage statistics (if available)
    pub usage: Option<CommonUsage>,

    /// Error that terminated the stream (if any)
    pub error: Option<CompletionResponseError>,
}

impl StreamingResponse {
    /// Create a new streaming response
    #[inline]
    pub fn new(
        provider: Provider,
        model: &str,
        request_id: &str,
    ) -> Result<Self, CompletionResponseError> {
        let model_name = ArrayString::from(model)
            .map_err(|_| CompletionResponseError::ModelNameTooLong(model.len()))?;
        let req_id = ArrayString::from(request_id)
            .map_err(|_| CompletionResponseError::IdTooLong(request_id.len()))?;

        let started_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| CompletionResponseError::SystemTimeError)?
            .as_secs();

        Ok(Self {
            provider,
            model: model_name,
            request_id: req_id,
            started_at,
            chunk_count: 0,
            completed: false,
            usage: None,
            error: None,
        })
    }

    /// Record a new chunk
    #[inline]
    pub fn record_chunk(&mut self, chunk: &CompletionChunk) {
        self.chunk_count += 1;

        if chunk.is_final() {
            self.completed = true;
            if let Some(usage) = &chunk.usage {
                self.usage = Some(usage.clone());
            }
        }
    }

    /// Mark the stream as completed with error
    #[inline]
    pub fn complete_with_error(&mut self, error: CompletionResponseError) {
        self.completed = true;
        self.error = Some(error);
    }

    /// Get stream duration in milliseconds
    #[inline]
    pub fn duration_ms(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(self.started_at);
        (now - self.started_at) * 1000
    }

    /// Calculate average chunks per second
    #[inline]
    pub fn chunks_per_second(&self) -> f64 {
        let duration_secs = self.duration_ms() as f64 / 1000.0;
        if duration_secs > 0.0 {
            self.chunk_count as f64 / duration_secs
        } else {
            0.0
        }
    }
}

/// Provider-specific metadata for completion responses
///
/// Contains additional information that providers may include beyond the standard response format.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderMetadata {
    /// OpenAI-specific metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai: Option<OpenAIMetadata>,

    /// Anthropic-specific metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anthropic: Option<AnthropicMetadata>,

    /// Google-specific metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google: Option<GoogleMetadata>,

    /// AWS Bedrock-specific metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bedrock: Option<BedrockMetadata>,

    /// Cohere-specific metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cohere: Option<CohereMetadata>,
}

impl ProviderMetadata {
    /// Create empty metadata
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if metadata is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.openai.is_none()
            && self.anthropic.is_none()
            && self.google.is_none()
            && self.bedrock.is_none()
            && self.cohere.is_none()
    }
}

/// OpenAI-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIMetadata {
    /// System fingerprint for reproducibility
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<ArrayString<64>>,

    /// Organization ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub organization: Option<ArrayString<64>>,
}

/// Anthropic-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMetadata {
    /// Stop reason details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<ArrayString<32>>,

    /// Stop sequence that triggered completion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<ArrayString<64>>,
}

/// Google-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleMetadata {
    /// Safety ratings for the response
    #[serde(skip_serializing_if = "ArrayVec::is_empty")]
    pub safety_ratings: ArrayVec<SafetyRating, 8>,

    /// Citation metadata for responses
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citation_metadata: Option<CitationMetadata>,
}

/// Safety rating from Google
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyRating {
    /// Safety category
    pub category: ArrayString<32>,

    /// Probability of harmful content
    pub probability: ArrayString<16>,

    /// Whether content was blocked
    pub blocked: bool,
}

/// Citation metadata from Google
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationMetadata {
    /// Citation sources
    pub citation_sources: ArrayVec<CitationSource, 16>,
}

/// Individual citation source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationSource {
    /// Start index in the response
    pub start_index: u32,

    /// End index in the response
    pub end_index: u32,

    /// Source URI
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<ArrayString<256>>,

    /// License information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<ArrayString<64>>,
}

/// AWS Bedrock-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockMetadata {
    /// Metrics for the request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<BedrockMetrics>,
}

/// Bedrock performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockMetrics {
    /// Latency in milliseconds
    pub latency_ms: u64,
}

/// Cohere-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereMetadata {
    /// Generation ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_id: Option<ArrayString<64>>,

    /// Warning messages
    #[serde(skip_serializing_if = "ArrayVec::is_empty")]
    pub warnings: ArrayVec<ArrayString<128>, 8>,
}

/// Errors that can occur during completion response processing
#[derive(Debug, Clone, PartialEq)]
pub enum CompletionResponseError {
    /// Response ID is too long for ArrayString
    IdTooLong(usize),

    /// Model name is too long for ArrayString
    ModelNameTooLong(usize),

    /// Too many choices in response
    TooManyChoices,

    /// Too many tool calls in choice
    TooManyToolCalls,

    /// Required field is missing from JSON
    MissingField(&'static str),

    /// Invalid JSON format
    InvalidJson,

    /// System time error
    SystemTimeError,

    /// Internal processing error
    InternalError,

    /// Validation error from domain types
    ValidationError(ValidationError),

    /// Unsupported provider
    UnsupportedProvider(Provider),

    /// Provider-specific error
    ProviderError {
        /// Provider that generated the error
        provider: Provider,
        /// Error message
        message: String,
        /// HTTP status code if applicable
        status_code: Option<u16>,
    },
}

impl fmt::Display for CompletionResponseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IdTooLong(len) => write!(f, "Response ID too long: {} characters", len),
            Self::ModelNameTooLong(len) => write!(f, "Model name too long: {} characters", len),
            Self::TooManyChoices => write!(f, "Too many choices in response"),
            Self::TooManyToolCalls => write!(f, "Too many tool calls in choice"),
            Self::MissingField(field) => write!(f, "Missing required field: {}", field),
            Self::InvalidJson => write!(f, "Invalid JSON format"),
            Self::SystemTimeError => write!(f, "System time error"),
            Self::InternalError => write!(f, "Internal processing error"),
            Self::ValidationError(e) => write!(f, "Validation error: {}", e),
            Self::UnsupportedProvider(provider) => write!(f, "Unsupported provider: {}", provider),
            Self::ProviderError {
                provider,
                message,
                status_code,
            } => {
                if let Some(code) = status_code {
                    write!(f, "Provider {} error ({}): {}", provider, code, message)
                } else {
                    write!(f, "Provider {} error: {}", provider, message)
                }
            }
        }
    }
}

impl std::error::Error for CompletionResponseError {}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_completion_response_creation() {
        let usage = CommonUsage::new(10, 20);
        let choice = CompletionChoice::new(
            0,
            BaseMessage::assistant("Hello, world!"),
            Some(FinishReason::Stop),
        );

        let response = CompletionResponse::new("test-id", "gpt-4", vec![choice], usage)
            .expect("Should create response");

        assert_eq!(response.id.as_str(), "test-id");
        assert_eq!(response.model.as_str(), "gpt-4");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.usage.prompt_tokens, 10);
        assert_eq!(response.usage.completion_tokens, 20);
    }

    #[test]
    fn test_openai_response_parsing() {
        let json = json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        });

        let response = CompletionResponse::from_provider_json(Provider::OpenAI, &json)
            .expect("Should parse OpenAI response");

        assert_eq!(response.id.as_str(), "chatcmpl-123");
        assert_eq!(response.model.as_str(), "gpt-4");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(
            response.choices[0].text(),
            "Hello! How can I help you today?"
        );
        assert_eq!(response.usage.prompt_tokens, 9);
        assert_eq!(response.usage.completion_tokens, 12);
    }

    #[test]
    fn test_anthropic_response_parsing() {
        let json = json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-sonnet-20240229",
            "content": [{
                "type": "text",
                "text": "Hello! I'm Claude, an AI assistant created by Anthropic."
            }],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 25
            }
        });

        let response = CompletionResponse::from_provider_json(Provider::Anthropic, &json)
            .expect("Should parse Anthropic response");

        assert_eq!(response.id.as_str(), "msg_123");
        assert_eq!(response.model.as_str(), "claude-3-sonnet-20240229");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(
            response.choices[0].text(),
            "Hello! I'm Claude, an AI assistant created by Anthropic."
        );
        assert_eq!(response.usage.prompt_tokens, 10);
        assert_eq!(response.usage.completion_tokens, 25);
    }

    #[test]
    fn test_completion_chunk_creation() {
        let choice = ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: None,
                content: Some("Hello".to_string()),
                tool_calls: ArrayVec::new(),
            },
            finish_reason: None,
            logprobs: None,
        };

        let chunk =
            CompletionChunk::new("chunk-123", "gpt-4", vec![choice]).expect("Should create chunk");

        assert_eq!(chunk.id.as_str(), "chunk-123");
        assert_eq!(chunk.model.as_str(), "gpt-4");
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.text(), Some("Hello"));
        assert!(!chunk.is_final());
    }

    #[test]
    fn test_openai_chunk_parsing() {
        let data = br#"{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;

        let chunk = CompletionChunk::from_provider_chunk(Provider::OpenAI, data)
            .expect("Should parse OpenAI chunk");

        assert_eq!(chunk.id.as_str(), "chatcmpl-123");
        assert_eq!(chunk.model.as_str(), "gpt-4");
        assert_eq!(chunk.text(), Some("Hello"));
        assert!(!chunk.is_final());
    }

    #[test]
    fn test_anthropic_chunk_parsing() {
        let data = br#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;

        let chunk = CompletionChunk::from_provider_chunk(Provider::Anthropic, data)
            .expect("Should parse Anthropic chunk");

        assert_eq!(chunk.text(), Some("Hello"));
        assert!(!chunk.is_final());
    }

    #[test]
    fn test_streaming_response() {
        let mut stream = StreamingResponse::new(Provider::OpenAI, "gpt-4", "req-123")
            .expect("Should create streaming response");

        assert_eq!(stream.provider, Provider::OpenAI);
        assert_eq!(stream.model.as_str(), "gpt-4");
        assert_eq!(stream.request_id.as_str(), "req-123");
        assert!(!stream.completed);
        assert_eq!(stream.chunk_count, 0);

        let chunk = CompletionChunk::new("chunk-1", "gpt-4", vec![]).expect("Should create chunk");
        stream.record_chunk(&chunk);

        assert_eq!(stream.chunk_count, 1);
        assert!(!stream.completed);
    }

    #[test]
    fn test_provider_metadata() {
        let mut metadata = ProviderMetadata::new();
        assert!(metadata.is_empty());

        metadata.openai = Some(OpenAIMetadata {
            system_fingerprint: Some(ArrayString::from("fp_123").unwrap_or_default()),
            organization: None,
        });

        assert!(!metadata.is_empty());
    }

    #[test]
    fn test_tool_call_parsing() {
        let json = json!({
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\":\"San Francisco\"}"
            }
        });

        let tool_call = CompletionChoice::parse_openai_tool_call(&json)
            .expect("Should parse tool call")
            .expect("Should return Some");

        assert_eq!(tool_call.id().unwrap_or_default(), "call_123");
        assert_eq!(tool_call.function_name().unwrap_or_default(), "get_weather");
    }

    #[test]
    fn test_error_display() {
        let error = CompletionResponseError::IdTooLong(100);
        assert_eq!(format!("{}", error), "Response ID too long: 100 characters");

        let error = CompletionResponseError::MissingField("choices");
        assert_eq!(format!("{}", error), "Missing required field: choices");

        let error = CompletionResponseError::ProviderError {
            provider: Provider::OpenAI,
            message: "Rate limited".to_string(),
            status_code: Some(429),
        };
        assert_eq!(
            format!("{}", error),
            "Provider openai error (429): Rate limited"
        );
    }

    #[test]
    fn test_response_helpers() {
        let choice1 = CompletionChoice::new(
            0,
            BaseMessage::assistant("First response"),
            Some(FinishReason::Stop),
        );
        let choice2 = CompletionChoice::new(
            1,
            BaseMessage::assistant("Second response"),
            Some(FinishReason::Length),
        );

        let response = CompletionResponse::new(
            "test-id",
            "gpt-4",
            vec![choice1, choice2],
            CommonUsage::new(10, 20),
        )
        .expect("Should create response");

        assert_eq!(response.primary_text(), Some("First response"));
        assert_eq!(
            response.all_texts(),
            vec!["First response", "Second response"]
        );
        assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
        assert!(!response.has_tool_calls());
    }
}
