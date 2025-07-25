//! Mistral AI API request and response structures.
//! 
//! This module provides zero-allocation structures for interacting with Mistral AI's
//! API including chat completions, embeddings, fine-tuning, and model management.
//! All collections use ArrayVec for bounded, stack-allocated storage.

use serde::{Deserialize, Serialize};
use arrayvec::ArrayVec;
use crate::{MAX_MESSAGES, MAX_TOOLS, MAX_IMAGES, MAX_DOCUMENTS};

// ============================================================================
// Chat Completions API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralChatRequest<'a> {
    #[serde(borrow)]
    pub model: &'a str,
    #[serde(borrow)]
    pub messages: ArrayVec<MistralMessage<'a>, MAX_MESSAGES>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safe_prompt: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub random_seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ArrayVec<MistralTool<'a>, MAX_TOOLS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<MistralToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<MistralResponseFormat<'a>>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralMessage<'a> {
    #[serde(borrow)]
    pub role: &'a str,
    #[serde(borrow)]
    pub content: MistralContent<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ArrayVec<MistralToolCall<'a>, MAX_TOOLS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<&'a str>}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MistralContent<'a> {
    Text(&'a str),
    Array(ArrayVec<MistralContentPart<'a>, MAX_IMAGES>)}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MistralContentPart<'a> {
    #[serde(rename = "text")]
    Text {
        #[serde(borrow)]
        text: &'a str},
    #[serde(rename = "image_url")]
    ImageUrl {
        image_url: MistralImageUrl<'a>}}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralImageUrl<'a> {
    #[serde(borrow)]
    pub url: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<&'a str>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralTool<'a> {
    #[serde(rename = "type")]
    #[serde(borrow)]
    pub tool_type: &'a str,
    pub function: MistralFunction<'a>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralFunction<'a> {
    #[serde(borrow)]
    pub name: &'a str,
    #[serde(borrow)]
    pub description: &'a str,
    pub parameters: serde_json::Value}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralToolCall<'a> {
    #[serde(borrow)]
    pub id: &'a str,
    #[serde(rename = "type")]
    #[serde(borrow)]
    pub call_type: &'a str,
    pub function: MistralFunctionCall<'a>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralFunctionCall<'a> {
    #[serde(borrow)]
    pub name: &'a str,
    #[serde(borrow)]
    pub arguments: &'a str}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MistralToolChoice<'a> {
    Auto(&'a str), // "auto" or "none"
    Required {
        #[serde(rename = "type")]
        #[serde(borrow)]
        choice_type: &'a str,
        function: MistralToolChoiceFunction<'a>}}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralToolChoiceFunction<'a> {
    #[serde(borrow)]
    pub name: &'a str}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralResponseFormat<'a> {
    #[serde(rename = "type")]
    #[serde(borrow)]
    pub format_type: &'a str, // "text" or "json_object"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: ArrayVec<MistralChoice, 8>,
    pub usage: MistralUsage}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralChoice {
    pub index: u32,
    pub message: MistralResponseMessage,
    pub finish_reason: String}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralResponseMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ArrayVec<MistralResponseToolCall, MAX_TOOLS>>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralResponseToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: MistralResponseFunction}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralResponseFunction {
    pub name: String,
    pub arguments: String}

// ============================================================================
// Embeddings API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralEmbeddingRequest<'a> {
    #[serde(borrow)]
    pub model: &'a str,
    #[serde(borrow)]
    pub input: MistralEmbeddingInput<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<&'a str>}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MistralEmbeddingInput<'a> {
    Single(&'a str),
    Multiple(ArrayVec<&'a str, MAX_DOCUMENTS>)}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralEmbeddingResponse {
    pub id: String,
    pub object: String,
    pub data: ArrayVec<MistralEmbeddingData, MAX_DOCUMENTS>,
    pub model: String,
    pub usage: MistralEmbeddingUsage}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralEmbeddingData {
    pub object: String,
    pub embedding: ArrayVec<f32, 1024>,
    pub index: u32}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralEmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32}

// ============================================================================
// Models API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralModelsResponse {
    pub object: String,
    pub data: ArrayVec<MistralModel, 64>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralModel {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    pub root: Option<String>,
    pub parent: Option<String>,
    pub permission: ArrayVec<MistralPermission, 8>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralPermission {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub allow_create_engine: bool,
    pub allow_sampling: bool,
    pub allow_logprobs: bool,
    pub allow_search_indices: bool,
    pub allow_view: bool,
    pub allow_fine_tuning: bool,
    pub organization: String,
    pub group: Option<String>,
    pub is_blocking: bool}

// ============================================================================
// Fine-tuning API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralFineTuningRequest<'a> {
    #[serde(borrow)]
    pub model: &'a str,
    #[serde(borrow)]
    pub training_files: ArrayVec<&'a str, 16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_files: Option<ArrayVec<&'a str, 16>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperparameters: Option<MistralHyperparameters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub integrations: Option<ArrayVec<MistralIntegration<'a>, 8>>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralHyperparameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_decay: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup_fraction: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epochs: Option<f32>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralIntegration<'a> {
    #[serde(rename = "type")]
    #[serde(borrow)]
    pub integration_type: &'a str,
    pub project: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<&'a str>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralFineTuningJob {
    pub id: String,
    pub object: String,
    pub model: String,
    pub created_at: u64,
    pub finished_at: Option<u64>,
    pub fine_tuned_model: Option<String>,
    pub status: String,
    pub trained_tokens: Option<u64>,
    pub training_files: ArrayVec<String, 16>,
    pub validation_files: ArrayVec<String, 16>,
    pub result_files: ArrayVec<String, 16>,
    pub hyperparameters: MistralHyperparameters,
    pub integrations: ArrayVec<MistralResponseIntegration, 8>,
    pub events: ArrayVec<MistralFineTuningEvent, 256>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralResponseIntegration {
    #[serde(rename = "type")]
    pub integration_type: String,
    pub project: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralFineTuningEvent {
    pub id: String,
    pub object: String,
    pub created_at: u64,
    pub level: String,
    pub message: String,
    #[serde(rename = "type")]
    pub event_type: String,
    pub data: Option<serde_json::Value>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralFineTuningJobList {
    pub object: String,
    pub data: ArrayVec<MistralFineTuningJob, 64>,
    pub has_more: bool}

// ============================================================================
// Files API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralFileUploadRequest<'a> {
    #[serde(borrow)]
    pub file: &'a [u8],
    #[serde(borrow)]
    pub purpose: &'a str}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralFile {
    pub id: String,
    pub object: String,
    pub bytes: u64,
    pub created_at: u64,
    pub filename: String,
    pub purpose: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_lines: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralFileList {
    pub object: String,
    pub data: ArrayVec<MistralFile, 256>}

// ============================================================================
// Agents API (Experimental)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralAgentRequest<'a> {
    #[serde(borrow)]
    pub name: &'a str,
    #[serde(borrow)]
    pub description: &'a str,
    #[serde(borrow)]
    pub model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ArrayVec<MistralTool<'a>, MAX_TOOLS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralAgent {
    pub id: String,
    pub object: String,
    pub created_at: u64,
    pub name: String,
    pub description: String,
    pub model: String,
    pub instructions: Option<String>,
    pub tools: ArrayVec<MistralResponseTool, MAX_TOOLS>,
    pub metadata: Option<serde_json::Value>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralResponseTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: MistralResponseToolFunction}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralResponseToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralAgentList {
    pub object: String,
    pub data: ArrayVec<MistralAgent, 64>,
    pub first_id: String,
    pub last_id: String,
    pub has_more: bool}

// ============================================================================
// Common Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralErrorResponse {
    pub error: MistralError}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralError {
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
pub struct MistralStreamingChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: ArrayVec<MistralStreamingChoice, 8>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralStreamingChoice {
    pub index: u32,
    pub delta: MistralStreamingDelta,
    pub finish_reason: Option<String>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralStreamingDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ArrayVec<MistralStreamingToolCall, MAX_TOOLS>>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralStreamingToolCall {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<MistralStreamingFunction>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralStreamingFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>}

// ============================================================================
// Builder Patterns for Http3 Integration
// ============================================================================

impl<'a> MistralChatRequest<'a> {
    pub fn new(model: &'a str) -> Self {
        Self {
            model,
            messages: ArrayVec::new(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            safe_prompt: None,
            random_seed: None,
            tools: None,
            tool_choice: None,
            response_format: None}
    }

    pub fn add_message(mut self, role: &'a str, content: MistralContent<'a>) -> Self {
        if self.messages.len() < MAX_MESSAGES {
            let _ = self.messages.try_push(MistralMessage {
                role,
                content,
                name: None,
                tool_calls: None,
                tool_call_id: None});
        }
        self
    }

    pub fn add_text_message(self, role: &'a str, text: &'a str) -> Self {
        self.add_message(role, MistralContent::Text(text))
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

    pub fn safe_prompt(mut self, safe: bool) -> Self {
        self.safe_prompt = Some(safe);
        self
    }

    pub fn with_tools(mut self, tools: ArrayVec<MistralTool<'a>, MAX_TOOLS>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn tool_choice_auto(mut self) -> Self {
        self.tool_choice = Some(MistralToolChoice::Auto("auto"));
        self
    }

    pub fn tool_choice_none(mut self) -> Self {
        self.tool_choice = Some(MistralToolChoice::Auto("none"));
        self
    }

    pub fn response_format_json(mut self) -> Self {
        self.response_format = Some(MistralResponseFormat {
            format_type: "json_object"});
        self
    }
}

impl<'a> MistralEmbeddingRequest<'a> {
    pub fn new_single(model: &'a str, input: &'a str) -> Self {
        Self {
            model,
            input: MistralEmbeddingInput::Single(input),
            encoding_format: None}
    }

    pub fn new_multiple(model: &'a str, inputs: ArrayVec<&'a str, MAX_DOCUMENTS>) -> Self {
        Self {
            model,
            input: MistralEmbeddingInput::Multiple(inputs),
            encoding_format: None}
    }

    pub fn encoding_format(mut self, format: &'a str) -> Self {
        self.encoding_format = Some(format);
        self
    }
}

impl<'a> MistralFineTuningRequest<'a> {
    pub fn new(model: &'a str, training_files: ArrayVec<&'a str, 16>) -> Self {
        Self {
            model,
            training_files,
            validation_files: None,
            hyperparameters: None,
            suffix: None,
            integrations: None}
    }

    pub fn with_validation_files(mut self, files: ArrayVec<&'a str, 16>) -> Self {
        self.validation_files = Some(files);
        self
    }

    pub fn with_hyperparameters(mut self, hyperparams: MistralHyperparameters) -> Self {
        self.hyperparameters = Some(hyperparams);
        self
    }

    pub fn suffix(mut self, suffix: &'a str) -> Self {
        self.suffix = Some(suffix);
        self
    }

    pub fn with_integrations(mut self, integrations: ArrayVec<MistralIntegration<'a>, 8>) -> Self {
        self.integrations = Some(integrations);
        self
    }
}

impl MistralHyperparameters {
    pub fn new() -> Self {
        Self {
            training_steps: None,
            learning_rate: None,
            weight_decay: None,
            warmup_fraction: None,
            epochs: None}
    }

    pub fn training_steps(mut self, steps: u32) -> Self {
        self.training_steps = Some(steps);
        self
    }

    pub fn learning_rate(mut self, rate: f32) -> Self {
        self.learning_rate = Some(rate);
        self
    }

    pub fn weight_decay(mut self, decay: f32) -> Self {
        self.weight_decay = Some(decay);
        self
    }

    pub fn epochs(mut self, epochs: f32) -> Self {
        self.epochs = Some(epochs);
        self
    }
}

impl<'a> MistralAgentRequest<'a> {
    pub fn new(name: &'a str, description: &'a str, model: &'a str) -> Self {
        Self {
            name,
            description,
            model,
            instructions: None,
            tools: None,
            metadata: None}
    }

    pub fn instructions(mut self, instructions: &'a str) -> Self {
        self.instructions = Some(instructions);
        self
    }

    pub fn with_tools(mut self, tools: ArrayVec<MistralTool<'a>, MAX_TOOLS>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}