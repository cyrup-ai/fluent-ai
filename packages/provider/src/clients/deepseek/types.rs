//! DeepSeek API request and response structures.
//!
//! This module provides zero-allocation structures for interacting with DeepSeek's
//! API. DeepSeek is largely OpenAI-compatible with some specific extensions.
//! All collections use ArrayVec for bounded, stack-allocated storage.

use serde::Serialize;
use arrayvec::ArrayVec;
use crate::{MAX_MESSAGES, MAX_TOOLS};

// ============================================================================
// Re-export OpenAI types for compatibility
// ============================================================================

pub use crate::openai::{
    OpenAIMessage as DeepSeekMessage,
    OpenAIContent as DeepSeekContent,
    OpenAIContentPart as DeepSeekContentPart,
    OpenAIImageUrl as DeepSeekImageUrl,
    OpenAITool as DeepSeekTool,
    OpenAIFunction as DeepSeekFunction,
    OpenAIToolCall as DeepSeekToolCall,
    OpenAIFunctionCall as DeepSeekFunctionCall,
    OpenAIToolChoice as DeepSeekToolChoice,
    // OpenAIToolChoiceFunction as DeepSeekToolChoiceFunction, // TODO: Add when OpenAIToolChoiceFunction is defined
    OpenAIResponseFormat as DeepSeekResponseFormat,
    OpenAIResponseMessage as DeepSeekResponseMessage,
    OpenAIResponseToolCall as DeepSeekResponseToolCall,
    OpenAIResponseFunction as DeepSeekResponseFunction,
    OpenAILogprobs as DeepSeekLogprobs,
    OpenAIContentLogprob as DeepSeekContentLogprob,
    OpenAITopLogprob as DeepSeekTopLogprob,
    OpenAIErrorResponse as DeepSeekErrorResponse,
    OpenAIError as DeepSeekError,
    OpenAIStreamingChunk as DeepSeekStreamingChunk,
    OpenAIStreamingChoice as DeepSeekStreamingChoice,
    OpenAIStreamingDelta as DeepSeekStreamingDelta,
    OpenAIStreamingToolCall as DeepSeekStreamingToolCall,
    OpenAIStreamingFunction as DeepSeekStreamingFunction};

// ============================================================================
// Chat Completions API (OpenAI-compatible with DeepSeek extensions)
// ============================================================================

/// DeepSeek Chat Completions Request
///
/// This structure is largely compatible with OpenAI's Chat Completions API,
/// with some DeepSeek-specific extensions like `reasoning_effort`.
#[derive(Debug, Serialize)]
pub struct DeepSeekChatRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<ArrayVec<DeepSeekMessage, MAX_MESSAGES>>,
    pub model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<ArrayVec<&'a str, 4>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<DeepSeekResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ArrayVec<DeepSeekTool, MAX_TOOLS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<DeepSeekToolChoice>,

    // DeepSeek-specific fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_reasoning_tokens: Option<u32>}

// ============================================================================
// Fine-tuning API
// ============================================================================

/// Request to create a fine-tuning job.
#[derive(Debug, Serialize)]
pub struct DeepSeekFineTuneRequest<'a> {
    pub model: &'a str,
    pub training_file: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_file: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperparameters: Option<DeepSeekHyperparameters<'a>>}

/// Hyperparameters for fine-tuning
#[derive(Debug, Serialize, Default)]
pub struct DeepSeekHyperparameters<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate_multiplier: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_epochs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<&'a str>}

// ============================================================================
// Reasoning API (DeepSeek-specific)
// ============================================================================

/// Request for a reasoning task.
#[derive(Debug, Serialize)]
pub struct DeepSeekReasoningRequest<'a> {
    pub model: &'a str,
    pub prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_reasoning_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>}

// ============================================================================
// Builder Implementations
// ============================================================================

impl<'a> DeepSeekChatRequest<'a> {
    #[inline(always)]
    pub fn new(model: &'a str) -> Self {
        Self {
            model,
            messages: Some(ArrayVec::new()),
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: None,
            temperature: None,
            top_p: None,
            user: None,
            max_tokens: None,
            logprobs: None,
            top_logprobs: None,
            response_format: None,
            seed: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            max_reasoning_tokens: None}
    }

    #[inline(always)]
    pub fn add_message(mut self, role: &'a str, content: DeepSeekContent) -> Self {
        if let Some(ref mut messages) = self.messages {
            messages.push(DeepSeekMessage { 
                role: role.to_string(), 
                content: Some(content),
                name: None,
                tool_calls: None,
                tool_call_id: None});
        }
        self
    }

    #[inline(always)]
    pub fn add_system_message(self, text: &'a str) -> Self {
        self.add_message("system", DeepSeekContent::Text(text.to_string()))
    }

    #[inline(always)]
    pub fn add_user_message(self, text: &'a str) -> Self {
        self.add_message("user", DeepSeekContent::Text(text.to_string()))
    }

    #[inline(always)]
    pub fn add_assistant_message(self, text: &'a str) -> Self {
        self.add_message("assistant", DeepSeekContent::Text(text.to_string()))
    }

    #[inline(always)]
    pub fn add_text_message(self, role: &'a str, text: &'a str) -> Self {
        self.add_message(role, DeepSeekContent::Text(text.to_string()))
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
    pub fn with_tools(mut self, tools: ArrayVec<DeepSeekTool, MAX_TOOLS>) -> Self {
        self.tools = Some(tools);
        self
    }

    #[inline(always)]
    pub fn tool_choice_auto(mut self) -> Self {
        self.tool_choice = Some(DeepSeekToolChoice::Auto("auto".to_string()));
        self
    }

    #[inline(always)]
    pub fn tool_choice_none(mut self) -> Self {
        self.tool_choice = Some(DeepSeekToolChoice::Auto("none".to_string()));
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
    pub fn logprobs(mut self, logprobs: bool) -> Self {
        self.logprobs = Some(logprobs);
        self
    }

    #[inline(always)]
    pub fn top_logprobs(mut self, top_logprobs: u32) -> Self {
        self.top_logprobs = Some(top_logprobs);
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

    // DeepSeek-specific builder methods
    #[inline(always)]
    pub fn reasoning_effort_low(mut self) -> Self {
        self.reasoning_effort = Some("low");
        self
    }

    #[inline(always)]
    pub fn reasoning_effort_medium(mut self) -> Self {
        self.reasoning_effort = Some("medium");
        self
    }

    #[inline(always)]
    pub fn reasoning_effort_high(mut self) -> Self {
        self.reasoning_effort = Some("high");
        self
    }

    #[inline(always)]
    pub fn max_reasoning_tokens(mut self, tokens: u32) -> Self {
        self.max_reasoning_tokens = Some(tokens);
        self
    }
}

impl<'a> DeepSeekFineTuneRequest<'a> {
    pub fn new(model: &'a str, training_file: &'a str) -> Self {
        Self {
            model,
            training_file,
            validation_file: None,
            hyperparameters: None}
    }

    pub fn validation_file(mut self, file: &'a str) -> Self {
        self.validation_file = Some(file);
        self
    }

    pub fn with_hyperparameters(mut self, hyperparams: DeepSeekHyperparameters<'a>) -> Self {
        self.hyperparameters = Some(hyperparams);
        self
    }
}

impl<'a> DeepSeekHyperparameters<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn batch_size(mut self, size: u32) -> Self {
        self.batch_size = Some(size);
        self
    }

    pub fn learning_rate_multiplier(mut self, multiplier: f32) -> Self {
        self.learning_rate_multiplier = Some(multiplier);
        self
    }

    pub fn n_epochs(mut self, epochs: u32) -> Self {
        self.n_epochs = Some(epochs);
        self
    }

    pub fn suffix(mut self, suffix: &'a str) -> Self {
        self.suffix = Some(suffix);
        self
    }
}

impl<'a> DeepSeekReasoningRequest<'a> {
    pub fn new(model: &'a str, prompt: &'a str) -> Self {
        Self {
            model,
            prompt,
            reasoning_effort: None,
            max_reasoning_tokens: None,
            temperature: None,
            stream: None}
    }

    pub fn reasoning_effort_high(mut self) -> Self {
        self.reasoning_effort = Some("high");
        self
    }

    pub fn max_reasoning_tokens(mut self, tokens: u32) -> Self {
        self.max_reasoning_tokens = Some(tokens);
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
}
