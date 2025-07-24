//! AWS Bedrock API request and response structures.
//! 
//! This module provides zero-allocation structures for interacting with AWS Bedrock
//! foundation models including Claude, Titan, Jurassic, Command, and Llama models.
//! All collections use ArrayVec for bounded, stack-allocated storage.

use serde::{Deserialize, Serialize};
use arrayvec::ArrayVec;
use crate::{MAX_MESSAGES, MAX_TOOLS, MAX_IMAGES};

// ============================================================================
// Claude Models (Anthropic on Bedrock)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockClaudeRequest<'a> {
    #[serde(borrow)]
    pub prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens_to_sample: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<ArrayVec<&'a str, 4>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anthropic_version: Option<&'a str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockClaudeResponse {
    pub completion: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

// ============================================================================
// Claude Messages API (newer format)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockClaudeMessagesRequest<'a> {
    #[serde(borrow)]
    pub messages: ArrayVec<BedrockClaudeMessage<'a>, MAX_MESSAGES>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<ArrayVec<&'a str, 4>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ArrayVec<BedrockTool<'a>, MAX_TOOLS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anthropic_version: Option<&'a str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockClaudeMessage<'a> {
    #[serde(borrow)]
    pub role: &'a str,
    pub content: BedrockClaudeContent<'a>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BedrockClaudeContent<'a> {
    Text(&'a str),
    Array(ArrayVec<BedrockClaudeContentBlock<'a>, MAX_IMAGES>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum BedrockClaudeContentBlock<'a> {
    #[serde(rename = "text")]
    Text { 
        #[serde(borrow)]
        text: &'a str 
    },
    #[serde(rename = "image")]
    Image { 
        source: BedrockImageSource<'a> 
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        #[serde(borrow)]
        id: &'a str,
        #[serde(borrow)]
        name: &'a str,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        #[serde(borrow)]
        tool_use_id: &'a str,
        #[serde(borrow)]
        content: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockImageSource<'a> {
    #[serde(rename = "type")]
    pub source_type: &'a str, // "base64"
    #[serde(borrow)]
    pub media_type: &'a str,
    #[serde(borrow)]
    pub data: &'a str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockTool<'a> {
    #[serde(borrow)]
    pub name: &'a str,
    #[serde(borrow)]
    pub description: &'a str,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockClaudeMessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: ArrayVec<BedrockClaudeResponseBlock, MAX_IMAGES>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: BedrockUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum BedrockClaudeResponseBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

// ============================================================================
// Titan Models (Amazon)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockTitanRequest<'a> {
    #[serde(borrow)]
    pub inputText: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub textGenerationConfig: Option<BedrockTitanConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockTitanConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maxTokenCount: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stopSequences: Option<ArrayVec<String, 4>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topP: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockTitanResponse {
    pub inputTextTokenCount: u32,
    pub results: ArrayVec<BedrockTitanResult, 8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockTitanResult {
    pub tokenCount: u32,
    pub outputText: String,
    pub completionReason: String,
}

// ============================================================================
// Titan Embeddings
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockTitanEmbeddingRequest<'a> {
    #[serde(borrow)]
    pub inputText: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalize: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockTitanEmbeddingResponse {
    pub embedding: ArrayVec<f32, 1536>,
    pub inputTextTokenCount: u32,
}

// ============================================================================
// Jurassic Models (AI21)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockJurassicRequest<'a> {
    #[serde(borrow)]
    pub prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maxTokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topP: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stopSequences: Option<ArrayVec<&'a str, 4>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub countPenalty: Option<BedrockPenalty>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presencePenalty: Option<BedrockPenalty>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequencyPenalty: Option<BedrockPenalty>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockPenalty {
    pub scale: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applyToWhitespaces: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applyToPunctuations: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applyToNumbers: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applyToStopwords: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applyToEmojis: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockJurassicResponse {
    pub id: String,
    pub prompt: BedrockPromptInfo,
    pub completions: ArrayVec<BedrockJurassicCompletion, 8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockPromptInfo {
    pub text: String,
    pub tokens: ArrayVec<BedrockToken, 2048>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockToken {
    pub generatedToken: BedrockGeneratedToken,
    pub topTokens: Option<ArrayVec<BedrockTopToken, 10>>,
    pub textRange: BedrockTextRange,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockGeneratedToken {
    pub token: String,
    pub logprob: f32,
    pub raw_logprob: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockTopToken {
    pub token: String,
    pub logprob: f32,
    pub raw_logprob: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockTextRange {
    pub start: u32,
    pub end: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockJurassicCompletion {
    pub data: BedrockCompletionData,
    pub finishReason: BedrockFinishReason,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockCompletionData {
    pub text: String,
    pub tokens: ArrayVec<BedrockToken, 2048>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockFinishReason {
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub length: Option<u32>,
}

// ============================================================================
// Command Models (Cohere on Bedrock)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockCommandRequest<'a> {
    #[serde(borrow)]
    pub prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<ArrayVec<&'a str, 4>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_likelihoods: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_generations: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<&'a str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockCommandResponse {
    pub generations: ArrayVec<BedrockGeneration, 8>,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockGeneration {
    pub finish_reason: String,
    pub id: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub likelihood: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_likelihoods: Option<ArrayVec<BedrockTokenLikelihood, 2048>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockTokenLikelihood {
    pub token: String,
    pub likelihood: f32,
}

// ============================================================================
// Llama Models (Meta on Bedrock)  
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockLlamaRequest<'a> {
    #[serde(borrow)]
    pub prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_gen_len: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockLlamaResponse {
    pub generation: String,
    pub prompt_token_count: u32,
    pub generation_token_count: u32,
    pub stop_reason: String,
}

// ============================================================================
// Common Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockErrorResponse {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
}

// ============================================================================
// Streaming Support
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum BedrockStreamingChunk {
    #[serde(rename = "message_start")]
    MessageStart {
        message: BedrockStreamingMessage,
    },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: u32,
        content_block: BedrockStreamingContentBlock,
    },
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: u32,
        delta: BedrockStreamingDelta,
    },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop {
        index: u32,
    },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: BedrockMessageDelta,
        usage: BedrockUsage,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockStreamingMessage {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub role: String,
    pub content: ArrayVec<serde_json::Value, MAX_IMAGES>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: BedrockUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum BedrockStreamingContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum BedrockStreamingDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockMessageDelta {
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
}

// ============================================================================
// Builder Patterns for Http3 Integration
// ============================================================================

impl<'a> BedrockClaudeRequest<'a> {
    pub fn new(prompt: &'a str) -> Self {
        Self {
            prompt,
            max_tokens_to_sample: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            anthropic_version: Some("bedrock-2023-05-31"),
        }
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens_to_sample = Some(tokens);
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
}

impl<'a> BedrockClaudeMessagesRequest<'a> {
    pub fn new() -> Self {
        Self {
            messages: ArrayVec::new(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system: None,
            tools: None,
            anthropic_version: Some("bedrock-2023-05-31"),
        }
    }

    pub fn add_message(mut self, role: &'a str, content: BedrockClaudeContent<'a>) -> Self {
        if self.messages.len() < MAX_MESSAGES {
            let _ = self.messages.try_push(BedrockClaudeMessage { role, content });
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

    pub fn system(mut self, system: &'a str) -> Self {
        self.system = Some(system);
        self
    }
}

impl<'a> BedrockTitanRequest<'a> {
    pub fn new(input_text: &'a str) -> Self {
        Self {
            inputText: input_text,
            textGenerationConfig: None,
        }
    }

    pub fn with_config(mut self, config: BedrockTitanConfig) -> Self {
        self.textGenerationConfig = Some(config);
        self
    }
}

impl BedrockTitanConfig {
    pub fn new() -> Self {
        Self {
            maxTokenCount: None,
            stopSequences: None,
            temperature: None,
            topP: None,
        }
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.maxTokenCount = Some(tokens);
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }
}

impl<'a> BedrockJurassicRequest<'a> {
    pub fn new(prompt: &'a str) -> Self {
        Self {
            prompt,
            maxTokens: None,
            temperature: None,
            topP: None,
            stopSequences: None,
            countPenalty: None,
            presencePenalty: None,
            frequencyPenalty: None,
        }
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.maxTokens = Some(tokens);
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }
}

impl<'a> BedrockCommandRequest<'a> {
    pub fn new(prompt: &'a str) -> Self {
        Self {
            prompt,
            max_tokens: None,
            temperature: None,
            p: None,
            k: None,
            stop_sequences: None,
            return_likelihoods: None,
            stream: None,
            num_generations: None,
            logit_bias: None,
            truncate: None,
        }
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
}

impl<'a> BedrockLlamaRequest<'a> {
    pub fn new(prompt: &'a str) -> Self {
        Self {
            prompt,
            max_gen_len: None,
            temperature: None,
            top_p: None,
        }
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_gen_len = Some(tokens);
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }
}