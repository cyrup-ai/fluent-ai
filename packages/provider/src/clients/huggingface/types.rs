/Volumes/samsung_t9/fluent-ai/packages/http-structs//! HuggingFace Inference API request and response structures.
//! 
//! This module provides zero-allocation structures for interacting with HuggingFace's
//! Inference API including text generation, embeddings, classification, and more.
//! All collections use ArrayVec for bounded, stack-allocated storage.

use serde::{Deserialize, Serialize};
use arrayvec::ArrayVec;
use crate::{MAX_MESSAGES, MAX_TOOLS, MAX_DOCUMENTS};

// ============================================================================
// Text Generation API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceTextGenerationRequest<'a> {
    #[serde(borrow)]
    pub inputs: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<HuggingFaceGenerationParameters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<HuggingFaceOptions>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceGenerationParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_of: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoder_input_details: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub do_sample: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_time: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_full_text: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<ArrayVec<String, 4>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typical_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub watermark: Option<bool>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_cache: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wait_for_model: Option<bool>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceTextGenerationResponse {
    pub generated_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<HuggingFaceGenerationDetails>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceGenerationDetails {
    pub finish_reason: String,
    pub generated_tokens: u32,
    pub seed: Option<u64>,
    pub prefill: ArrayVec<HuggingFaceToken, 2048>,
    pub tokens: ArrayVec<HuggingFaceToken, 2048>,
    pub best_of_sequences: Option<ArrayVec<HuggingFaceSequence, 8>>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceToken {
    pub id: u32,
    pub text: String,
    pub logprob: f32,
    pub special: bool}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceSequence {
    pub finish_reason: String,
    pub generated_text: String,
    pub generated_tokens: u32,
    pub prefill: ArrayVec<HuggingFaceToken, 2048>,
    pub tokens: ArrayVec<HuggingFaceToken, 2048>,
    pub seed: Option<u64>}

// ============================================================================
// Chat Completion API (Messages)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceChatRequest<'a> {
    #[serde(borrow)]
    pub messages: ArrayVec<HuggingFaceMessage<'a>, MAX_MESSAGES>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<ArrayVec<&'a str, 4>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<HuggingFaceResponseFormat<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ArrayVec<HuggingFaceTool<'a>, MAX_TOOLS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<&'a str>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceMessage<'a> {
    #[serde(borrow)]
    pub role: &'a str,
    #[serde(borrow)]
    pub content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ArrayVec<HuggingFaceToolCall<'a>, MAX_TOOLS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<&'a str>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceToolCall<'a> {
    #[serde(borrow)]
    pub id: &'a str,
    #[serde(rename = "type")]
    #[serde(borrow)]
    pub call_type: &'a str,
    pub function: HuggingFaceFunction<'a>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceFunction<'a> {
    #[serde(borrow)]
    pub name: &'a str,
    #[serde(borrow)]
    pub arguments: &'a str}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceTool<'a> {
    #[serde(rename = "type")]
    #[serde(borrow)]
    pub tool_type: &'a str,
    pub function: HuggingFaceToolFunction<'a>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceToolFunction<'a> {
    #[serde(borrow)]
    pub name: &'a str,
    #[serde(borrow)]
    pub description: &'a str,
    pub parameters: serde_json::Value}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceResponseFormat<'a> {
    #[serde(rename = "type")]
    #[serde(borrow)]
    pub format_type: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: ArrayVec<HuggingFaceChoice, 8>,
    pub usage: HuggingFaceUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceChoice {
    pub index: u32,
    pub message: HuggingFaceResponseMessage,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<HuggingFaceLogprobs>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceResponseMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ArrayVec<HuggingFaceResponseToolCall, MAX_TOOLS>>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceResponseToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: HuggingFaceResponseFunction}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceResponseFunction {
    pub name: String,
    pub arguments: String}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceLogprobs {
    pub content: ArrayVec<HuggingFaceContentLogprob, 1024>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceContentLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<ArrayVec<u8, 4>>,
    pub top_logprobs: ArrayVec<HuggingFaceTopLogprob, 10>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceTopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<ArrayVec<u8, 4>>}

// ============================================================================
// Embeddings API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceEmbeddingRequest<'a> {
    #[serde(borrow)]
    pub inputs: HuggingFaceEmbeddingInput<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<HuggingFaceOptions>}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HuggingFaceEmbeddingInput<'a> {
    Single(&'a str),
    Multiple(ArrayVec<&'a str, MAX_DOCUMENTS>)}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HuggingFaceEmbeddingResponse {
    Single(ArrayVec<f32, 1536>),
    Multiple(ArrayVec<ArrayVec<f32, 1536>, MAX_DOCUMENTS>)}

// ============================================================================
// Classification API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceClassificationRequest<'a> {
    #[serde(borrow)]
    pub inputs: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<HuggingFaceClassificationParameters<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<HuggingFaceOptions>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceClassificationParameters<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_labels: Option<ArrayVec<&'a str, 32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multi_label: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hypothesis_template: Option<&'a str>}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HuggingFaceClassificationResponse {
    ZeroShot(ArrayVec<HuggingFaceZeroShotResult, 32>),
    TextClassification(ArrayVec<HuggingFaceTextClassificationResult, 32>)}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceZeroShotResult {
    pub sequence: String,
    pub labels: ArrayVec<String, 32>,
    pub scores: ArrayVec<f32, 32>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceTextClassificationResult {
    pub label: String,
    pub score: f32}

// ============================================================================
// Question Answering API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceQuestionAnsweringRequest<'a> {
    #[serde(borrow)]
    pub inputs: HuggingFaceQAInput<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<HuggingFaceQAParameters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<HuggingFaceOptions>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceQAInput<'a> {
    #[serde(borrow)]
    pub question: &'a str,
    #[serde(borrow)]
    pub context: &'a str}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceQAParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_stride: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_answer_len: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_seq_len: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_question_len: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub handle_impossible_answer: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub align_to_words: Option<bool>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceQAResponse {
    pub score: f32,
    pub start: u32,
    pub end: u32,
    pub answer: String}

// ============================================================================
// Image-to-Text API  
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceImageToTextRequest<'a> {
    #[serde(borrow)]
    pub inputs: &'a str, // base64 encoded image
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<HuggingFaceImageToTextParameters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<HuggingFaceOptions>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceImageToTextParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_new_tokens: Option<u32>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceImageToTextResponse {
    pub generated_text: String}

// ============================================================================
// Text-to-Image API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceTextToImageRequest<'a> {
    #[serde(borrow)]
    pub inputs: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<HuggingFaceTextToImageParameters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<HuggingFaceOptions>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceTextToImageParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_inference_steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guidance_scale: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>}

// Response is binary image data, so no specific structure needed

// ============================================================================
// Common Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceErrorResponse {
    pub error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>}

// ============================================================================
// Streaming Support
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceStreamingChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: ArrayVec<HuggingFaceStreamingChoice, 8>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceStreamingChoice {
    pub index: u32,
    pub delta: HuggingFaceStreamingDelta,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<HuggingFaceLogprobs>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceStreamingDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ArrayVec<HuggingFaceResponseToolCall, MAX_TOOLS>>}

// ============================================================================
// Builder Patterns for Http3 Integration
// ============================================================================

impl<'a> HuggingFaceTextGenerationRequest<'a> {
    pub fn new(inputs: &'a str) -> Self {
        Self {
            inputs,
            parameters: None,
            options: None}
    }

    pub fn with_parameters(mut self, parameters: HuggingFaceGenerationParameters) -> Self {
        self.parameters = Some(parameters);
        self
    }

    pub fn with_options(mut self, options: HuggingFaceOptions) -> Self {
        self.options = Some(options);
        self
    }
}

impl HuggingFaceGenerationParameters {
    pub fn new() -> Self {
        Self {
            best_of: None,
            decoder_input_details: None,
            details: None,
            do_sample: None,
            max_new_tokens: None,
            max_time: None,
            repetition_penalty: None,
            return_full_text: None,
            seed: None,
            stop: None,
            temperature: None,
            top_k: None,
            top_p: None,
            truncate: None,
            typical_p: None,
            watermark: None}
    }

    pub fn max_new_tokens(mut self, tokens: u32) -> Self {
        self.max_new_tokens = Some(tokens);
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    pub fn top_k(mut self, k: u32) -> Self {
        self.top_k = Some(k);
        self
    }

    pub fn do_sample(mut self, sample: bool) -> Self {
        self.do_sample = Some(sample);
        self
    }

    pub fn return_full_text(mut self, full: bool) -> Self {
        self.return_full_text = Some(full);
        self
    }
}

impl<'a> HuggingFaceChatRequest<'a> {
    pub fn new() -> Self {
        Self {
            messages: ArrayVec::new(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            stream: None,
            seed: None,
            frequency_penalty: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            n: None,
            presence_penalty: None,
            response_format: None,
            tools: None,
            tool_choice: None}
    }

    pub fn add_message(mut self, role: &'a str, content: &'a str) -> Self {
        if self.messages.len() < MAX_MESSAGES {
            let _ = self.messages.try_push(HuggingFaceMessage {
                role,
                content,
                name: None,
                tool_calls: None,
                tool_call_id: None});
        }
        self
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

    pub fn with_tools(mut self, tools: ArrayVec<HuggingFaceTool<'a>, MAX_TOOLS>) -> Self {
        self.tools = Some(tools);
        self
    }
}

impl<'a> HuggingFaceEmbeddingRequest<'a> {
    pub fn new_single(input: &'a str) -> Self {
        Self {
            inputs: HuggingFaceEmbeddingInput::Single(input),
            options: None}
    }

    pub fn new_multiple(inputs: ArrayVec<&'a str, MAX_DOCUMENTS>) -> Self {
        Self {
            inputs: HuggingFaceEmbeddingInput::Multiple(inputs),
            options: None}
    }

    pub fn with_options(mut self, options: HuggingFaceOptions) -> Self {
        self.options = Some(options);
        self
    }
}

impl<'a> HuggingFaceClassificationRequest<'a> {
    pub fn new(inputs: &'a str) -> Self {
        Self {
            inputs,
            parameters: None,
            options: None}
    }

    pub fn with_candidate_labels(mut self, labels: ArrayVec<&'a str, 32>) -> Self {
        let mut params = self.parameters.unwrap_or_default();
        params.candidate_labels = Some(labels);
        self.parameters = Some(params);
        self
    }

    pub fn multi_label(mut self, multi: bool) -> Self {
        let mut params = self.parameters.unwrap_or_default();
        params.multi_label = Some(multi);
        self.parameters = Some(params);
        self
    }
}

impl<'a> Default for HuggingFaceClassificationParameters<'a> {
    fn default() -> Self {
        Self {
            candidate_labels: None,
            multi_label: None,
            hypothesis_template: None}
    }
}

impl<'a> HuggingFaceQuestionAnsweringRequest<'a> {
    pub fn new(question: &'a str, context: &'a str) -> Self {
        Self {
            inputs: HuggingFaceQAInput { question, context },
            parameters: None,
            options: None}
    }

    pub fn with_parameters(mut self, parameters: HuggingFaceQAParameters) -> Self {
        self.parameters = Some(parameters);
        self
    }
}

impl HuggingFaceOptions {
    pub fn new() -> Self {
        Self {
            use_cache: None,
            wait_for_model: None}
    }

    pub fn use_cache(mut self, cache: bool) -> Self {
        self.use_cache = Some(cache);
        self
    }

    pub fn wait_for_model(mut self, wait: bool) -> Self {
        self.wait_for_model = Some(wait);
        self
    }
}
