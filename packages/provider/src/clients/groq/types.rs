//! Groq API request and response structures.
//! 
//! This module provides zero-allocation structures for interacting with Groq's
//! ultra-fast inference API. Groq is largely OpenAI-compatible with some
//! specific differences for hardware acceleration.
//! All collections use ArrayVec for bounded, stack-allocated storage.

use serde::{Deserialize, Serialize};
use arrayvec::ArrayVec;
use crate::{MAX_MESSAGES, MAX_TOOLS, MAX_IMAGES};

// ============================================================================
// Chat Completions API (OpenAI-compatible with Groq extensions)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqChatRequest<'a> {
    #[serde(borrow)]
    pub model: &'a str,
    #[serde(borrow)]
    pub messages: ArrayVec<GroqMessage<'a>, MAX_MESSAGES>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<ArrayVec<&'a str, 4>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ArrayVec<GroqTool<'a>, MAX_TOOLS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<GroqToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<GroqResponseFormat<'a>>,
    // Groq-specific extensions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqMessage<'a> {
    #[serde(borrow)]
    pub role: &'a str,
    #[serde(borrow)]
    pub content: GroqContent<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ArrayVec<GroqToolCall<'a>, MAX_TOOLS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<&'a str>}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GroqContent<'a> {
    Text(&'a str),
    Array(ArrayVec<GroqContentPart<'a>, MAX_IMAGES>)}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GroqContentPart<'a> {
    #[serde(rename = "text")]
    Text {
        #[serde(borrow)]
        text: &'a str},
    #[serde(rename = "image_url")]
    ImageUrl {
        image_url: GroqImageUrl<'a>}}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqImageUrl<'a> {
    #[serde(borrow)]
    pub url: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<&'a str>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqTool<'a> {
    #[serde(rename = "type")]
    #[serde(borrow)]
    pub tool_type: &'a str,
    pub function: GroqFunction<'a>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqFunction<'a> {
    #[serde(borrow)]
    pub name: &'a str,
    #[serde(borrow)]
    pub description: &'a str,
    pub parameters: serde_json::Value}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqToolCall<'a> {
    #[serde(borrow)]
    pub id: &'a str,
    #[serde(rename = "type")]
    #[serde(borrow)]
    pub call_type: &'a str,
    pub function: GroqFunctionCall<'a>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqFunctionCall<'a> {
    #[serde(borrow)]
    pub name: &'a str,
    #[serde(borrow)]
    pub arguments: &'a str}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GroqToolChoice<'a> {
    Auto(&'a str), // "auto" or "none"
    Required {
        #[serde(rename = "type")]
        #[serde(borrow)]
        choice_type: &'a str,
        function: GroqToolChoiceFunction<'a>}}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqToolChoiceFunction<'a> {
    #[serde(borrow)]
    pub name: &'a str}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqResponseFormat<'a> {
    #[serde(rename = "type")]
    #[serde(borrow)]
    pub format_type: &'a str, // "text" or "json_object"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: ArrayVec<GroqChoice, 8>,
    pub usage: GroqUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    // Groq-specific response fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_groq: Option<GroqExtensions>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqChoice {
    pub index: u32,
    pub message: GroqResponseMessage,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<GroqLogprobs>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqResponseMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ArrayVec<GroqResponseToolCall, MAX_TOOLS>>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqResponseToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: GroqResponseFunction}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqResponseFunction {
    pub name: String,
    pub arguments: String}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqLogprobs {
    pub content: ArrayVec<GroqContentLogprob, 1024>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqContentLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<ArrayVec<u8, 4>>,
    pub top_logprobs: ArrayVec<GroqTopLogprob, 10>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqTopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<ArrayVec<u8, 4>>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqExtensions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<GroqUsageExtensions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqUsageExtensions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_time: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_time: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_time: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_time: Option<f32>}

// ============================================================================
// Models API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqModelsResponse {
    pub object: String,
    pub data: ArrayVec<GroqModel, 64>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqModel {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    pub active: bool,
    pub context_window: u32,
    pub public_apps: Option<ArrayVec<String, 16>>}

// ============================================================================
// Audio API (Whisper-compatible)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqAudioTranscriptionRequest<'a> {
    #[serde(borrow)]
    pub file: &'a [u8],
    #[serde(borrow)]
    pub model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_granularities: Option<ArrayVec<&'a str, 2>>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqAudioTranslationRequest<'a> {
    #[serde(borrow)]
    pub file: &'a [u8],
    #[serde(borrow)]
    pub model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqAudioResponse {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<ArrayVec<GroqWord, 2048>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<ArrayVec<GroqSegment, 256>>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqWord {
    pub word: String,
    pub start: f32,
    pub end: f32}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqSegment {
    pub id: u32,
    pub seek: u32,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub tokens: ArrayVec<u32, 512>,
    pub temperature: f32,
    pub avg_logprob: f32,
    pub compression_ratio: f32,
    pub no_speech_prob: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<ArrayVec<GroqWord, 256>>}

// ============================================================================
// Common Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    // Groq-specific timing information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_time: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_time: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_time: Option<f32>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqErrorResponse {
    pub error: GroqError}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>}

// ============================================================================
// Streaming Support
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqStreamingChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: ArrayVec<GroqStreamingChoice, 8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_groq: Option<GroqExtensions>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqStreamingChoice {
    pub index: u32,
    pub delta: GroqStreamingDelta,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<GroqLogprobs>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqStreamingDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ArrayVec<GroqStreamingToolCall, MAX_TOOLS>>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqStreamingToolCall {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<GroqStreamingFunction>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqStreamingFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>}

// ============================================================================
// Builder Patterns for Http3 Integration
// ============================================================================

impl<'a> GroqChatRequest<'a> {
    pub fn new(model: &'a str) -> Self {
        Self {
            model,
            messages: ArrayVec::new(),
            temperature: None,
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            max_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            seed: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            parallel_tool_calls: None,
            logprobs: None,
            top_logprobs: None}
    }

    pub fn add_message(mut self, role: &'a str, content: GroqContent<'a>) -> Self {
        if self.messages.len() < MAX_MESSAGES {
            let _ = self.messages.try_push(GroqMessage {
                role,
                content,
                name: None,
                tool_calls: None,
                tool_call_id: None});
        }
        self
    }

    pub fn add_text_message(self, role: &'a str, text: &'a str) -> Self {
        self.add_message(role, GroqContent::Text(text))
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    pub fn stream(mut self, streaming: bool) -> Self {
        self.stream = Some(streaming);
        self
    }

    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_tools(mut self, tools: ArrayVec<GroqTool<'a>, MAX_TOOLS>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn tool_choice_auto(mut self) -> Self {
        self.tool_choice = Some(GroqToolChoice::Auto("auto"));
        self
    }

    pub fn tool_choice_none(mut self) -> Self {
        self.tool_choice = Some(GroqToolChoice::Auto("none"));
        self
    }

    pub fn response_format_json(mut self) -> Self {
        self.response_format = Some(GroqResponseFormat {
            format_type: "json_object"});
        self
    }

    pub fn parallel_tool_calls(mut self, parallel: bool) -> Self {
        self.parallel_tool_calls = Some(parallel);
        self
    }

    pub fn logprobs(mut self, logprobs: bool) -> Self {
        self.logprobs = Some(logprobs);
        self
    }

    pub fn top_logprobs(mut self, top_logprobs: u32) -> Self {
        self.top_logprobs = Some(top_logprobs);
        self
    }

    pub fn user(mut self, user: &'a str) -> Self {
        self.user = Some(user);
        self
    }

    pub fn stop_sequences(mut self, stop: ArrayVec<&'a str, 4>) -> Self {
        self.stop = Some(stop);
        self
    }

    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }
}

impl<'a> GroqAudioTranscriptionRequest<'a> {
    pub fn new(file: &'a [u8], model: &'a str) -> Self {
        Self {
            file,
            model,
            language: None,
            prompt: None,
            response_format: None,
            temperature: None,
            timestamp_granularities: None}
    }

    pub fn language(mut self, language: &'a str) -> Self {
        self.language = Some(language);
        self
    }

    pub fn prompt(mut self, prompt: &'a str) -> Self {
        self.prompt = Some(prompt);
        self
    }

    pub fn response_format(mut self, format: &'a str) -> Self {
        self.response_format = Some(format);
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn timestamp_granularities(mut self, granularities: ArrayVec<&'a str, 2>) -> Self {
        self.timestamp_granularities = Some(granularities);
        self
    }
}

impl<'a> GroqAudioTranslationRequest<'a> {
    pub fn new(file: &'a [u8], model: &'a str) -> Self {
        Self {
            file,
            model,
            prompt: None,
            response_format: None,
            temperature: None}
    }

    pub fn prompt(mut self, prompt: &'a str) -> Self {
        self.prompt = Some(prompt);
        self
    }

    pub fn response_format(mut self, format: &'a str) -> Self {
        self.response_format = Some(format);
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }
}