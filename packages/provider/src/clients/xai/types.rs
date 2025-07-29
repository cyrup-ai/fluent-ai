//! xAI (Grok) API request and response structures.
//! 
//! This module provides zero-allocation structures for interacting with xAI's
//! Grok models. xAI is largely OpenAI-compatible with some specific extensions.
//! All collections use ArrayVec for bounded, stack-allocated storage.

use serde::{Deserialize, Serialize};
use arrayvec::ArrayVec;
use crate::{MAX_MESSAGES, MAX_TOOLS, MAX_DOCUMENTS};

// ============================================================================
// Re-export OpenAI types for compatibility
// ============================================================================

pub use crate::clients::openai::{
    OpenAIMessage as XAIMessage,
    OpenAIContent as XAIContent,
    OpenAIContentPart as XAIContentPart,
    OpenAIImageUrl as XAIImageUrl,
    OpenAITool as XAITool,
    OpenAIFunction as XAIFunction,
    OpenAIToolCall as XAIToolCall,
    OpenAIFunctionCall as XAIFunctionCall,
    OpenAIToolChoice as XAIToolChoice,
    // OpenAIToolChoiceFunction as XAIToolChoiceFunction, // TODO: Add when OpenAIToolChoiceFunction is defined
    OpenAIResponseFormat as XAIResponseFormat,
    OpenAIResponseMessage as XAIResponseMessage,
    OpenAIResponseToolCall as XAIResponseToolCall,
    OpenAIResponseFunction as XAIResponseFunction,
    OpenAILogprobs as XAILogprobs,
};

// Additional XAI-specific types that are needed
pub use crate::clients::openai::{
    OpenAIMessage as XaiMessage,
    OpenAIContent as XaiContent,
    OpenAITool as XaiTool,
    OpenAIFunction as XaiFunction,
    OpenAIToolCall as XaiToolCall,
    OpenAIResponseMessage as XaiResponseMessage,
};

// XAI-specific request/response types
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct XaiChatRequest {
    pub model: String,
    pub messages: Vec<XaiMessage>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub tools: Option<Vec<XaiTool>>,
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct XaiChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<XaiChoice>,
    pub usage: Option<XaiUsage>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct XaiChoice {
    pub index: u32,
    pub message: XaiResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct XaiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct XaiStreamingChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<XaiChoice>,
    pub usage: Option<XaiUsage>,
}

// Type aliases for OpenAI compatibility
pub type XAIContentLogprob = OpenAIContentLogprob;
pub type XAITopLogprob = OpenAITopLogprob;
pub type XAIErrorResponse = OpenAIErrorResponse;
pub type XAIError = OpenAIError;
pub type XAIStreamingChunk = OpenAIStreamingChunk;
pub type XAIStreamingChoice = OpenAIStreamingChoice;
pub type XAIStreamingDelta = OpenAIStreamingDelta;
pub type XAIStreamingToolCall = OpenAIStreamingToolCall;
pub type XAIStreamingFunction = OpenAIStreamingFunction;

// ============================================================================
// Chat Completions API (OpenAI-compatible with xAI extensions)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XAIChatRequest<'a> {
    #[serde(borrow)]
    pub model: &'a str,
    pub messages: ArrayVec<XAIMessage, MAX_MESSAGES>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<ArrayVec<&'a str, 4>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ArrayVec<XAITool, MAX_TOOLS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<XAIToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<XAIResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<&'a str>,
    // xAI-specific extensions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grok_context: Option<XAIGrokContext<'a>>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XAIGrokContext<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub real_time_info: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_integration: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", borrow)]
    pub context_sources: Option<ArrayVec<&'a str, 8>>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XAIChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: ArrayVec<XAIChoice, 8>,
    pub usage: XAIUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    // xAI-specific response fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grok_metadata: Option<XAIGrokMetadata>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XAIChoice {
    pub index: u32,
    pub message: XAIResponseMessage,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<XAILogprobs>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XAIGrokMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_used: Option<ArrayVec<String, 8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub real_time_data: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_posts_referenced: Option<ArrayVec<String, 16>>}

// ============================================================================
// Models API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XAIModelsResponse {
    pub object: String,
    pub data: ArrayVec<XAIModel, 32>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XAIModel {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<ArrayVec<String, 8>>}

// ============================================================================
// Embeddings API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XAIEmbeddingRequest<'a> {
    #[serde(borrow)]
    pub model: &'a str,
    #[serde(borrow)]
    pub input: XAIEmbeddingInput<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<&'a str>}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum XAIEmbeddingInput<'a> {
    Single(&'a str),
    Multiple(ArrayVec<&'a str, MAX_DOCUMENTS>)}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XAIEmbeddingResponse {
    pub object: String,
    pub data: ArrayVec<XAIEmbeddingData, MAX_DOCUMENTS>,
    pub model: String,
    pub usage: XAIEmbeddingUsage}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XAIEmbeddingData {
    pub object: String,
    pub embedding: ArrayVec<f32, 1536>,
    pub index: u32}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XAIEmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32}

// ============================================================================
// Text Completions API (Legacy)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XAICompletionRequest<'a> {
    #[serde(borrow)]
    pub model: &'a str,
    #[serde(borrow)]
    pub prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub echo: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<ArrayVec<&'a str, 4>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_of: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<&'a str>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XAICompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: ArrayVec<XAICompletionChoice, 8>,
    pub usage: XAIUsage}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XAICompletionChoice {
    pub text: String,
    pub index: u32,
    pub logprobs: Option<XAICompletionLogprobs>,
    pub finish_reason: String}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XAICompletionLogprobs {
    pub tokens: ArrayVec<String, 1024>,
    pub token_logprobs: ArrayVec<f32, 1024>,
    pub top_logprobs: ArrayVec<serde_json::Value, 1024>,
    pub text_offset: ArrayVec<u32, 1024>}

// ============================================================================
// Common Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32}

// ============================================================================
// Streaming Support with xAI Extensions
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XAIStreamingChunkExtended {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: ArrayVec<XAIStreamingChoiceExtended, 8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grok_metadata: Option<XAIGrokMetadata>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XAIStreamingChoiceExtended {
    pub index: u32,
    pub delta: XAIStreamingDelta,
    pub finish_reason: Option<String>}

// ============================================================================
// Builder Patterns for Http3 Integration
// ============================================================================

impl<'a> XAIChatRequest<'a> {
    #[inline(always)]
    pub fn new(model: &'a str) -> Self {
        Self {
            model,
            messages: ArrayVec::new(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            user: None,
            system_prompt: None,
            grok_context: None}
    }

    #[inline(always)]
    pub fn add_message(mut self, role: &'a str, content: XAIContent) -> Self {
        if self.messages.len() < MAX_MESSAGES {
            let _ = self.messages.try_push(XAIMessage {
                role: role.to_string(),
                content: Some(content),
                name: None,
                tool_calls: None,
                tool_call_id: None});
        }
        self
    }

    #[inline(always)]
    pub fn add_text_message(self, role: &'a str, text: &'a str) -> Self {
        self.add_message(role, XAIContent::Text(text.to_string()))
    }

    #[inline(always)]
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    #[inline(always)]
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    #[inline(always)]
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    #[inline(always)]
    pub fn stream(mut self, streaming: bool) -> Self {
        self.stream = Some(streaming);
        self
    }

    #[inline(always)]
    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = Some(seed);
        self
    }

    #[inline(always)]
    pub fn with_tools(mut self, tools: ArrayVec<XAITool, MAX_TOOLS>) -> Self {
        self.tools = Some(tools);
        self
    }

    #[inline(always)]
    pub fn tool_choice_auto(mut self) -> Self {
        self.tool_choice = Some(XAIToolChoice::Auto("auto".to_string()));
        self
    }

    #[inline(always)]
    pub fn tool_choice_none(mut self) -> Self {
        self.tool_choice = Some(XAIToolChoice::Auto("none".to_string()));
        self
    }

    #[inline(always)]
    pub fn response_format_json(mut self) -> Self {
        self.response_format = Some(serde_json::json!({
            "type": "json_object"
        }));
        self
    }

    #[inline(always)]
    pub fn user(mut self, user: &'a str) -> Self {
        self.user = Some(user);
        self
    }

    #[inline(always)]
    pub fn stop_sequences(mut self, stop: ArrayVec<&'a str, 4>) -> Self {
        self.stop = Some(stop);
        self
    }

    #[inline(always)]
    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    #[inline(always)]
    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    // xAI-specific builder methods
    #[inline(always)]
    pub fn system_prompt(mut self, prompt: &'a str) -> Self {
        self.system_prompt = Some(prompt);
        self
    }

    #[inline(always)]
    pub fn enable_real_time_info(mut self) -> Self {
        let context = XAIGrokContext {
            real_time_info: Some(true),
            x_integration: None,
            context_sources: None};
        self.grok_context = Some(context);
        self
    }

    #[inline(always)]
    pub fn enable_x_integration(mut self) -> Self {
        let mut context = self.grok_context.unwrap_or_default();
        context.x_integration = Some(true);
        self.grok_context = Some(context);
        self
    }

    #[inline(always)]
    pub fn with_context_sources(mut self, sources: ArrayVec<&'a str, 8>) -> Self {
        let mut context = self.grok_context.unwrap_or_default();
        context.context_sources = Some(sources);
        self.grok_context = Some(context);
        self
    }
}

impl<'a> Default for XAIGrokContext<'a> {
    fn default() -> Self {
        Self {
            real_time_info: None,
            x_integration: None,
            context_sources: None}
    }
}

impl<'a> XAIEmbeddingRequest<'a> {
    pub fn new_single(model: &'a str, input: &'a str) -> Self {
        Self {
            model,
            input: XAIEmbeddingInput::Single(input),
            encoding_format: None,
            dimensions: None,
            user: None}
    }

    pub fn new_multiple(model: &'a str, inputs: ArrayVec<&'a str, MAX_DOCUMENTS>) -> Self {
        Self {
            model,
            input: XAIEmbeddingInput::Multiple(inputs),
            encoding_format: None,
            dimensions: None,
            user: None}
    }

    pub fn encoding_format(mut self, format: &'a str) -> Self {
        self.encoding_format = Some(format);
        self
    }

    pub fn dimensions(mut self, dims: u32) -> Self {
        self.dimensions = Some(dims);
        self
    }

    pub fn user(mut self, user: &'a str) -> Self {
        self.user = Some(user);
        self
    }
}

impl<'a> XAICompletionRequest<'a> {
    pub fn new(model: &'a str, prompt: &'a str) -> Self {
        Self {
            model,
            prompt,
            max_tokens: None,
            temperature: None,
            top_p: None,
            n: None,
            stream: None,
            logprobs: None,
            echo: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            best_of: None,
            logit_bias: None,
            user: None}
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn stream(mut self, streaming: bool) -> Self {
        self.stream = Some(streaming);
        self
    }

    pub fn logprobs(mut self, logprobs: u32) -> Self {
        self.logprobs = Some(logprobs);
        self
    }

    pub fn echo(mut self, echo: bool) -> Self {
        self.echo = Some(echo);
        self
    }

    pub fn user(mut self, user: &'a str) -> Self {
        self.user = Some(user);
        self
    }
}