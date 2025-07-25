//! Together AI API request and response structures.
//! 
//! This module provides zero-allocation structures for interacting with Together AI's
//! inference platform. Together AI is largely OpenAI-compatible with
//! specific model and fine-tuning extensions.
//! All collections use ArrayVec for bounded, stack-allocated storage.

use serde::{Deserialize, Serialize};
use arrayvec::ArrayVec;
use crate::{MAX_MESSAGES, MAX_TOOLS, MAX_DOCUMENTS};

// ============================================================================
// Re-export OpenAI types for compatibility
// ============================================================================

pub use crate::openai::{
    OpenAIMessage as TogetherMessage,
    OpenAIContent as TogetherContent,
    OpenAIContentPart as TogetherContentPart,
    OpenAIImageUrl as TogetherImageUrl,
    OpenAITool as TogetherTool,
    OpenAIFunction as TogetherFunction,
    OpenAIToolCall as TogetherToolCall,
    OpenAIFunctionCall as TogetherFunctionCall,
    OpenAIToolChoice as TogetherToolChoice,
    // OpenAIToolChoiceFunction as TogetherToolChoiceFunction, // TODO: Add when OpenAIToolChoiceFunction is defined
    OpenAIResponseFormat as TogetherResponseFormat,
    OpenAIResponseMessage as TogetherResponseMessage,
    OpenAIResponseToolCall as TogetherResponseToolCall,
    OpenAIResponseFunction as TogetherResponseFunction,
    OpenAILogprobs as TogetherLogprobs,
    OpenAIContentLogprob as TogetherContentLogprob,
    OpenAITopLogprob as TogetherTopLogprob,
    OpenAIErrorResponse as TogetherErrorResponse,
    OpenAIError as TogetherError,
    OpenAIStreamingChunk as TogetherStreamingChunk,
    OpenAIStreamingChoice as TogetherStreamingChoice,
    OpenAIStreamingDelta as TogetherStreamingDelta,
    OpenAIStreamingToolCall as TogetherStreamingToolCall,
    OpenAIStreamingFunction as TogetherStreamingFunction};

// ============================================================================
// Chat Completions API (OpenAI-compatible with Together extensions)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherChatRequest<'a> {
    #[serde(borrow)]
    pub model: &'a str,
    pub messages: ArrayVec<TogetherMessage, MAX_MESSAGES>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
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
    pub repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ArrayVec<TogetherTool, MAX_TOOLS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<TogetherToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<TogetherResponseFormat>,
    // Together AI-specific extensions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub echo: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_model: Option<&'a str>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: ArrayVec<TogetherChoice, 8>,
    pub usage: TogetherUsage}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherChoice {
    pub index: u32,
    pub message: TogetherResponseMessage,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<TogetherChoiceLogprobs>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherChoiceLogprobs {
    pub tokens: ArrayVec<String, 1024>,
    pub token_logprobs: ArrayVec<f32, 1024>,
    pub top_logprobs: ArrayVec<serde_json::Value, 1024>,
    pub text_offset: ArrayVec<u32, 1024>}

// ============================================================================
// Text Completions API (Legacy)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherCompletionRequest<'a> {
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
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<ArrayVec<&'a str, 4>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub echo: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_model: Option<&'a str>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: ArrayVec<TogetherCompletionChoice, 8>,
    pub usage: TogetherUsage}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherCompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<TogetherChoiceLogprobs>}

// ============================================================================
// Image Generation API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherImageRequest<'a> {
    #[serde(borrow)]
    pub model: &'a str,
    #[serde(borrow)]
    pub prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherImageResponse {
    pub created: u64,
    pub data: ArrayVec<TogetherImageData, 8>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherImageData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>}

// ============================================================================
// Embeddings API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherEmbeddingRequest<'a> {
    #[serde(borrow)]
    pub model: &'a str,
    #[serde(borrow)]
    pub input: TogetherEmbeddingInput<'a>}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TogetherEmbeddingInput<'a> {
    Single(&'a str),
    Multiple(ArrayVec<&'a str, MAX_DOCUMENTS>)}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherEmbeddingResponse {
    pub object: String,
    pub data: ArrayVec<TogetherEmbeddingData, MAX_DOCUMENTS>,
    pub model: String,
    pub usage: TogetherUsage}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherEmbeddingData {
    pub object: String,
    pub embedding: ArrayVec<f32, 1536>,
    pub index: u32}

// ============================================================================
// Models API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherModelsResponse {
    pub data: ArrayVec<TogetherModel, 256>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherModel {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    #[serde(rename = "type")]
    pub model_type: String,
    pub display_name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<TogetherModelConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing: Option<TogetherModelPricing>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherModelConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<ArrayVec<String, 8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherModelPricing {
    pub input: f32,
    pub output: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base: Option<f32>}

// ============================================================================
// Fine-tuning API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherFineTuneRequest<'a> {
    #[serde(borrow)]
    pub training_file: &'a str,
    #[serde(borrow)]
    pub model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_epochs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_evals: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wandb_api_key: Option<&'a str>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherFineTuneJob {
    pub id: String,
    pub object: String,
    pub model: String,
    pub created_at: u64,
    pub finished_at: Option<u64>,
    pub fine_tuned_model: Option<String>,
    pub status: String,
    pub trained_tokens: Option<u64>,
    pub training_file: String,
    pub validation_file: Option<String>,
    pub result_files: ArrayVec<String, 8>,
    pub events: ArrayVec<TogetherFineTuneEvent, 256>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherFineTuneEvent {
    pub object: String,
    pub created_at: u64,
    pub level: String,
    pub message: String}

// ============================================================================
// Files API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherFileUploadRequest<'a> {
    #[serde(borrow)]
    pub file: &'a [u8],
    #[serde(borrow)]
    pub purpose: &'a str}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherFile {
    pub id: String,
    pub object: String,
    pub bytes: u64,
    pub created_at: u64,
    pub filename: String,
    pub purpose: String}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherFileList {
    pub object: String,
    pub data: ArrayVec<TogetherFile, 256>}

// ============================================================================
// Common Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TogetherUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32}

// ============================================================================
// Builder Patterns for Http3 Integration
// ============================================================================

impl<'a> TogetherChatRequest<'a> {
    #[inline(always)]
    pub fn new(model: &'a str) -> Self {
        Self {
            model,
            messages: ArrayVec::new(),
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: None,
            min_p: None,
            seed: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            logprobs: None,
            echo: None,
            n: None,
            safety_model: None}
    }

    #[inline(always)]
    pub fn add_message(mut self, role: &'a str, content: TogetherContent) -> Self {
        if self.messages.len() < MAX_MESSAGES {
            let _ = self.messages.try_push(TogetherMessage {
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
        self.add_message(role, TogetherContent::Text(text.to_string()))
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
    pub fn top_k(mut self, k: u32) -> Self {
        self.top_k = Some(k);
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
    pub fn repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = Some(penalty);
        self
    }

    #[inline(always)]
    pub fn min_p(mut self, min_p: f32) -> Self {
        self.min_p = Some(min_p);
        self
    }

    #[inline(always)]
    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = Some(seed);
        self
    }

    #[inline(always)]
    pub fn with_tools(mut self, tools: ArrayVec<TogetherTool, MAX_TOOLS>) -> Self {
        self.tools = Some(tools);
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
    pub fn logprobs(mut self, logprobs: u32) -> Self {
        self.logprobs = Some(logprobs);
        self
    }

    #[inline(always)]
    pub fn echo(mut self, echo: bool) -> Self {
        self.echo = Some(echo);
        self
    }

    #[inline(always)]
    pub fn safety_model(mut self, model: &'a str) -> Self {
        self.safety_model = Some(model);
        self
    }
}

impl<'a> TogetherCompletionRequest<'a> {
    #[inline(always)]
    pub fn new(model: &'a str, prompt: &'a str) -> Self {
        Self {
            model,
            prompt,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            stream: None,
            stop: None,
            logprobs: None,
            echo: None,
            n: None,
            safety_model: None}
    }

    #[inline(always)]
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
}

impl<'a> TogetherImageRequest<'a> {
    pub fn new(model: &'a str, prompt: &'a str) -> Self {
        Self {
            model,
            prompt,
            width: None,
            height: None,
            steps: None,
            n: None,
            response_format: None,
            seed: None}
    }

    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    pub fn steps(mut self, steps: u32) -> Self {
        self.steps = Some(steps);
        self
    }

    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn response_format(mut self, format: &'a str) -> Self {
        self.response_format = Some(format);
        self
    }
}

impl<'a> TogetherEmbeddingRequest<'a> {
    pub fn new_single(model: &'a str, input: &'a str) -> Self {
        Self {
            model,
            input: TogetherEmbeddingInput::Single(input)}
    }

    pub fn new_multiple(model: &'a str, inputs: ArrayVec<&'a str, MAX_DOCUMENTS>) -> Self {
        Self {
            model,
            input: TogetherEmbeddingInput::Multiple(inputs)}
    }
}

impl<'a> TogetherFineTuneRequest<'a> {
    pub fn new(training_file: &'a str, model: &'a str) -> Self {
        Self {
            training_file,
            model,
            n_epochs: None,
            n_evals: None,
            batch_size: None,
            learning_rate: None,
            suffix: None,
            wandb_api_key: None}
    }

    pub fn n_epochs(mut self, epochs: u32) -> Self {
        self.n_epochs = Some(epochs);
        self
    }

    pub fn learning_rate(mut self, rate: f32) -> Self {
        self.learning_rate = Some(rate);
        self
    }

    pub fn batch_size(mut self, size: u32) -> Self {
        self.batch_size = Some(size);
        self
    }

    pub fn suffix(mut self, suffix: &'a str) -> Self {
        self.suffix = Some(suffix);
        self
    }
}