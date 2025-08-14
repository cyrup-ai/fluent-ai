//! OpenAI API request/response structures
//!
//! Comprehensive, zero-allocation structures for all OpenAI endpoints:
//! - Chat Completions (GPT-4, ChatGPT)
//! - Embeddings
//! - Moderation
//! - Audio (Whisper transcription, TTS)
//! - Vision (Image analysis)
//!
//! All structures are optimized for performance with ArrayVec for bounded collections,
//! proper lifetime annotations, and efficient serialization patterns.

use std::collections::HashMap;

use arrayvec::ArrayVec;
use serde::{Deserialize, Serialize};

use crate::{MAX_CHOICES, MAX_EMBEDDINGS, MAX_MESSAGES, MAX_TOOLS};
// =============================================================================
// Chat Completions API
// =============================================================================

/// OpenAI chat completion request with zero-allocation design
#[derive(Debug, Serialize)]
pub struct OpenAICompletionRequest {
    /// Model identifier (e.g., "gpt-4", "gpt-3.5-turbo")
    pub model: String,
    /// Array of messages in the conversation
    pub messages: ArrayVec<OpenAIMessage, MAX_MESSAGES>,
    /// Sampling temperature between 0 and 2
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Nucleus sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Frequency penalty for reducing repetition
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// Presence penalty for encouraging new topics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Available tools for function calling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ArrayVec<serde_json::Value, MAX_TOOLS>>,
    /// Tool choice strategy
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<OpenAIToolChoice>,
    /// Whether to stream responses
    pub stream: bool,
    /// Stream options configuration
    pub stream_options: serde_json::Value,
    /// Stop sequences to halt generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<ArrayVec<String, 4>>,
    /// Number of choices to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Unique identifier for the request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Logit bias for token probabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,
    /// Whether to return log probabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// Number of log probabilities to return
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
}

/// OpenAI chat completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompletionResponse {
    /// Unique identifier for the completion response
    pub id: String,
    /// Object type (always "chat.completion")
    pub object: String,
    /// Unix timestamp when the completion was created
    pub created: u64,
    /// Model that generated the completion
    pub model: String,
    /// Array of completion choices
    pub choices: ArrayVec<OpenAIChoice, MAX_CHOICES>,
    /// Usage statistics for the completion request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAIUsage>,
    /// System fingerprint for reproducibility
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// Tool choice configuration for function calling
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAIToolChoice {
    /// Automatic tool selection
    Auto(String), // "auto"
    /// No tool usage
    None(String), // "none"
    /// Specific function to call
    Function {
        /// Tool type (always "function")
        r#type: String,
        /// Function specification
        function: OpenAIFunctionChoice,
    },
}

/// Specific function choice for tool calling
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAIFunctionChoice {
    /// Name of the function to call
    pub name: String,
}

/// Message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIMessage {
    /// Role of the message sender
    pub role: String, // "system", "user", "assistant", "tool"
    /// Text content of the message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<OpenAIMessageContent>,
    /// Tool calls made by the assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ArrayVec<OpenAIToolCall, MAX_TOOLS>>,
    /// Tool call ID (for tool role messages)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Name of the function (for function role messages)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Message content (text or multimodal)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAIMessageContent {
    /// Simple text content
    Text(String),
    /// Multimodal content (text + images)
    Parts(ArrayVec<OpenAIContentPart, 16>),
}

/// Content part for multimodal messages
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OpenAIContentPart {
    /// Text content
    #[serde(rename = "text")]
    Text {
        /// Text content
        text: String,
    },
    /// Image content
    #[serde(rename = "image_url")]
    ImageUrl {
        /// Image URL configuration
        image_url: OpenAIImageUrl,
    },
}

/// Image URL configuration for vision
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAIImageUrl {
    /// Image URL (data: or https:)
    pub url: String,
    /// Detail level for image processing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>, // "low", "high", "auto"
}

/// Tool call made by the assistant
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    /// Unique identifier for the tool call
    pub id: String,
    /// Type of tool (always "function")
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function call details
    pub function: OpenAIFunction,
}

/// Function call details
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAIFunction {
    /// Name of the function
    pub name: String,
    /// Function arguments as JSON string
    pub arguments: String,
}

// =============================================================================
// Streaming Response Types
// =============================================================================

/// Streaming response chunk from OpenAI
#[derive(Debug, Deserialize)]
pub struct OpenAIStreamChunk {
    /// Unique identifier for the completion
    pub id: String,
    /// Object type (always "chat.completion.chunk")
    pub object: String,
    /// Unix timestamp of creation
    pub created: u64,
    /// Model used for the completion
    pub model: String,
    /// Array of choice chunks
    pub choices: ArrayVec<OpenAIChoice, MAX_CHOICES>,
    /// Token usage statistics (only in final chunk)
    #[serde(default)]
    pub usage: Option<OpenAIUsage>,
    /// System fingerprint for backend configuration
    #[serde(default)]
    pub system_fingerprint: Option<String>,
}

/// Choice in a streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIChoice {
    /// Index of this choice
    pub index: u32,
    /// Delta content for this chunk
    pub delta: OpenAIDelta,
    /// Log probabilities (if requested)
    #[serde(default)]
    pub logprobs: Option<OpenAILogprobs>,
    /// Reason for finishing (if completed)
    #[serde(default)]
    pub finish_reason: Option<String>, // "stop", "length", "tool_calls", "content_filter"
}

/// Delta content in a streaming chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIDelta {
    /// Role (only in first chunk)
    #[serde(default)]
    pub role: Option<String>,
    /// Incremental content
    #[serde(default)]
    pub content: Option<String>,
    /// Tool calls delta
    #[serde(default)]
    pub tool_calls: Option<ArrayVec<OpenAIToolCallDelta, MAX_TOOLS>>,
}

/// Tool call delta in streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCallDelta {
    /// Index of the tool call
    pub index: u32,
    /// Tool call ID (if starting)
    #[serde(default)]
    pub id: Option<String>,
    /// Tool type (if starting)
    #[serde(rename = "type", default)]
    pub tool_type: Option<String>,
    /// Function delta
    #[serde(default)]
    pub function: Option<OpenAIFunctionDelta>,
}

/// Function call delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunctionDelta {
    /// Function name (if starting)
    #[serde(default)]
    pub name: Option<String>,
    /// Incremental arguments
    #[serde(default)]
    pub arguments: Option<String>,
}

/// Log probabilities information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAILogprobs {
    /// Content log probabilities
    pub content: Option<ArrayVec<OpenAIContentLogprob, 256>>,
}

/// Content log probability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIContentLogprob {
    /// Token string
    pub token: String,
    /// Log probability
    pub logprob: f64,
    /// Raw bytes of the token
    pub bytes: Option<Vec<u8>>,
    /// Top alternative tokens with probabilities
    pub top_logprobs: ArrayVec<OpenAITopLogprob, 16>,
}

/// Top alternative token with probability
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAITopLogprob {
    /// Token string
    pub token: String,
    /// Log probability
    pub logprob: f64,
    /// Raw bytes of the token
    pub bytes: Option<Vec<u8>>,
}

/// Token usage statistics
#[derive(Debug, Deserialize)]
pub struct OpenAIUsage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,
    /// Number of tokens in the completion
    pub completion_tokens: u32,
    /// Total number of tokens used
    pub total_tokens: u32,
    /// Detailed token usage breakdown
    #[serde(default)]
    pub prompt_tokens_details: Option<OpenAIPromptTokensDetails>,
    /// Detailed completion token usage
    #[serde(default)]
    pub completion_tokens_details: Option<OpenAICompletionTokensDetails>,
}

/// Detailed prompt token usage
#[derive(Debug, Deserialize)]
pub struct OpenAIPromptTokensDetails {
    /// Cached tokens from previous requests
    pub cached_tokens: u32,
}

/// Detailed completion token usage
#[derive(Debug, Deserialize)]
pub struct OpenAICompletionTokensDetails {
    /// Tokens generated for reasoning
    pub reasoning_tokens: u32,
}

// =============================================================================
// Embeddings API
// =============================================================================

/// OpenAI embeddings request
#[derive(Debug, Serialize)]
pub struct OpenAIEmbeddingRequest {
    /// Model identifier (e.g., "text-embedding-ada-002")
    pub model: String,
    /// Input text(s) to embed
    pub input: OpenAIEmbeddingInput,
    /// Format of the embedding vectors
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>, // "float", "base64"
    /// Number of dimensions (for newer models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// Unique identifier for the request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// Input for embedding generation
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum OpenAIEmbeddingInput {
    /// Single text input
    Single(String),
    /// Multiple text inputs
    Multiple(ArrayVec<String, MAX_EMBEDDINGS>),
}

/// OpenAI embeddings response
#[derive(Debug, Deserialize)]
pub struct OpenAIEmbeddingResponse {
    /// Object type (always "list")
    pub object: String,
    /// Array of embedding data
    pub data: ArrayVec<OpenAIEmbeddingData, MAX_EMBEDDINGS>,
    /// Model used for embeddings
    pub model: String,
    /// Token usage statistics
    pub usage: OpenAIEmbeddingUsage,
}

/// Individual embedding data
#[derive(Debug, Deserialize)]
pub struct OpenAIEmbeddingData {
    /// Object type (always "embedding")
    pub object: String,
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Index in the input array
    pub index: usize,
}

/// Token usage for embeddings
#[derive(Debug, Deserialize)]
pub struct OpenAIEmbeddingUsage {
    /// Number of tokens in the input
    pub prompt_tokens: u32,
    /// Total tokens used (same as prompt_tokens)
    pub total_tokens: u32,
}

// =============================================================================
// Moderation API
// =============================================================================

/// OpenAI moderation request
#[derive(Debug, Serialize)]
pub struct OpenAIModerationRequest {
    /// Input text to moderate
    pub input: OpenAIModerationInput,
    /// Model to use for moderation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>, // "text-moderation-latest", "text-moderation-stable"
}

/// Input for content moderation
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum OpenAIModerationInput {
    /// Single text input
    Single(String),
    /// Multiple text inputs
    Multiple(Vec<String>),
}

/// OpenAI moderation response
#[derive(Debug, Deserialize)]
pub struct OpenAIModerationResponse {
    /// Unique identifier for the request
    pub id: String,
    /// Model used for moderation
    pub model: String,
    /// Array of moderation results
    pub results: Vec<OpenAIModerationResult>,
}

/// Individual moderation result
#[derive(Debug, Deserialize)]
pub struct OpenAIModerationResult {
    /// Whether the content is flagged
    pub flagged: bool,
    /// Category flags
    pub categories: OpenAIModerationCategories,
    /// Category confidence scores
    pub category_scores: OpenAIModerationScores,
}

/// Moderation category flags
#[derive(Debug, Deserialize)]
pub struct OpenAIModerationCategories {
    /// Sexual content
    pub sexual: bool,
    /// Hate speech
    pub hate: bool,
    /// Harassment
    pub harassment: bool,
    /// Self-harm content
    #[serde(rename = "self-harm")]
    pub self_harm: bool,
    /// Sexual content involving minors
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: bool,
    /// Threatening hate speech
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: bool,
    /// Graphic violence
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: bool,
    /// Self-harm intent
    #[serde(rename = "self-harm/intent")]
    pub self_harm_intent: bool,
    /// Self-harm instructions
    #[serde(rename = "self-harm/instructions")]
    pub self_harm_instructions: bool,
    /// Threatening harassment
    #[serde(rename = "harassment/threatening")]
    pub harassment_threatening: bool,
    /// Violence
    pub violence: bool,
}

/// Moderation confidence scores (0.0 to 1.0)
#[derive(Debug, Deserialize)]
pub struct OpenAIModerationScores {
    /// Sexual content score
    pub sexual: f64,
    /// Hate speech score
    pub hate: f64,
    /// Harassment score
    pub harassment: f64,
    /// Self-harm score
    #[serde(rename = "self-harm")]
    pub self_harm: f64,
    /// Sexual minors score
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: f64,
    /// Threatening hate score
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: f64,
    /// Graphic violence score
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: f64,
    /// Self-harm intent score
    #[serde(rename = "self-harm/intent")]
    pub self_harm_intent: f64,
    /// Self-harm instructions score
    #[serde(rename = "self-harm/instructions")]
    pub self_harm_instructions: f64,
    /// Threatening harassment score
    #[serde(rename = "harassment/threatening")]
    pub harassment_threatening: f64,
    /// Violence score
    pub violence: f64,
}

// =============================================================================
// Audio API (Transcription, Translation, TTS)
// =============================================================================

/// Audio transcription request (multipart form data)
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAITranscriptionRequest {
    /// Audio file bytes
    pub file: Vec<u8>,
    /// Filename for the audio file
    pub filename: String,
    /// Model to use for transcription
    pub model: String, // "whisper-1"
    /// Language of the input audio (ISO-639-1)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Optional prompt to guide the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Response format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>, // "json", "text", "srt", "verbose_json", "vtt"
    /// Sampling temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Timestamp granularities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_granularities: Option<Vec<String>>, // ["word"], ["segment"]
}

/// Audio transcription response
#[derive(Debug, Deserialize)]
pub struct OpenAITranscriptionResponse {
    /// Transcribed text
    pub text: String,
    /// Language detected (if not specified)
    #[serde(default)]
    pub language: Option<String>,
    /// Duration of the audio file
    #[serde(default)]
    pub duration: Option<f64>,
    /// Word-level timestamps (if requested)
    #[serde(default)]
    pub words: Option<Vec<OpenAITranscriptionWord>>,
    /// Segment-level information (if verbose format)
    #[serde(default)]
    pub segments: Option<Vec<OpenAITranscriptionSegment>>,
}

/// Word-level transcription data
#[derive(Debug, Deserialize)]
pub struct OpenAITranscriptionWord {
    /// Word text
    pub word: String,
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
}

/// Segment-level transcription data
#[derive(Debug, Deserialize)]
pub struct OpenAITranscriptionSegment {
    /// Segment ID
    pub id: u32,
    /// Seek offset
    pub seek: u32,
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
    /// Segment text
    pub text: String,
    /// Token IDs
    pub tokens: Vec<u32>,
    /// Temperature used
    pub temperature: f64,
    /// Average log probability
    pub avg_logprob: f64,
    /// Compression ratio
    pub compression_ratio: f64,
    /// No speech probability
    pub no_speech_prob: f64,
}

/// Audio translation request (same as transcription)
pub type OpenAITranslationRequest = OpenAITranscriptionRequest;

/// Audio translation response (same as transcription)
pub type OpenAITranslationResponse = OpenAITranscriptionResponse;

/// Text-to-speech request
#[derive(Debug, Serialize)]
pub struct OpenAITTSRequest {
    /// Model to use for TTS
    pub model: String, // "tts-1", "tts-1-hd"
    /// Text to convert to speech
    pub input: String,
    /// Voice to use
    pub voice: String, // "alloy", "echo", "fable", "onyx", "nova", "shimmer"
    /// Audio format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>, // "mp3", "opus", "aac", "flac", "wav", "pcm"
    /// Speech speed (0.25 to 4.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
}

// TTS response is raw audio bytes, no struct needed

// =============================================================================
// Utility Types and Implementations
// =============================================================================

impl Default for OpenAICompletionRequest {
    fn default() -> Self {
        Self {
            model: "gpt-3.5-turbo".to_string(),
            messages: ArrayVec::new(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            tools: None,
            tool_choice: None,
            stream: false,
            stream_options: serde_json::json!({"include_usage": true}),
            stop: None,
            n: None,
            user: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
        }
    }
}

impl OpenAICompletionRequest {
    /// Create a new completion request with the specified model
    #[inline]
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            ..Default::default()
        }
    }

    /// Add a message to the conversation
    pub fn add_message(&mut self, role: &str, content: &str) -> Result<(), &'static str> {
        let message = OpenAIMessage {
            role: role.to_string(),
            content: Some(OpenAIMessageContent::Text(content.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };

        self.messages
            .try_push(message)
            .map_err(|_| "Maximum messages exceeded")
    }

    /// Set streaming mode
    #[inline]
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Set temperature
    #[inline]
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set max tokens
    #[inline]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

impl Default for OpenAIEmbeddingRequest {
    fn default() -> Self {
        Self {
            model: "text-embedding-ada-002".to_string(),
            input: OpenAIEmbeddingInput::Single(String::new()),
            encoding_format: None,
            dimensions: None,
            user: None,
        }
    }
}

impl OpenAIEmbeddingRequest {
    /// Create a new embedding request with the specified model
    #[inline]
    pub fn new(model: String, input: String) -> Self {
        Self {
            model,
            input: OpenAIEmbeddingInput::Single(input),
            ..Default::default()
        }
    }

    /// Create a batch embedding request
    #[inline]
    pub fn batch(model: String, inputs: ArrayVec<String, MAX_EMBEDDINGS>) -> Self {
        Self {
            model,
            input: OpenAIEmbeddingInput::Multiple(inputs),
            ..Default::default()
        }
    }
}

impl Default for OpenAIModerationRequest {
    fn default() -> Self {
        Self {
            input: OpenAIModerationInput::Single(String::new()),
            model: Some("text-moderation-latest".to_string()),
        }
    }
}

impl OpenAIModerationRequest {
    /// Create a new moderation request
    #[inline]
    pub fn new(input: String) -> Self {
        Self {
            input: OpenAIModerationInput::Single(input),
            ..Default::default()
        }
    }
}

/// Helper functions for common operations
impl OpenAIStreamChunk {
    /// Check if this is the final chunk
    #[inline]
    pub fn is_done(&self) -> bool {
        self.choices
            .iter()
            .any(|choice| choice.finish_reason.is_some())
    }

    /// Get the text content from the first choice
    #[inline]
    pub fn text(&self) -> Option<&str> {
        self.choices.first()?.delta.content.as_deref()
    }

    /// Get the finish reason if available
    #[inline]
    pub fn finish_reason(&self) -> Option<&str> {
        self.choices.first()?.finish_reason.as_deref()
    }
}

// =============================================================================
// Type Aliases for Provider Compatibility
// =============================================================================

/// Alias for OpenAI message content (used by compatible providers)
pub type OpenAIContent = OpenAIMessageContent;

/// Alias for OpenAI function call (used by compatible providers)
pub type OpenAIFunctionCall = OpenAIFunction;

/// Alias for OpenAI tool choice function (used by compatible providers)
// TODO: Define OpenAIFunctionChoice or remove this alias
// pub type OpenAIToolChoiceFunction = OpenAIFunctionChoice;

/// Tool definition type (alias for json Value for now)
pub type OpenAITool = serde_json::Value;

/// Response format type (alias for json Value for now)
pub type OpenAIResponseFormat = serde_json::Value;

/// Response message type (alias for completion choice)
pub type OpenAIResponseMessage = OpenAIChoice;

/// Response tool call type (alias for tool call)
pub type OpenAIResponseToolCall = OpenAIToolCall;

/// Response function type (alias for function)
pub type OpenAIResponseFunction = OpenAIFunction;

/// Error response structure for API errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIErrorResponse {
    pub error: OpenAIError,
}

/// Error detail structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: Option<String>,
    pub param: Option<String>,
    pub code: Option<String>,
}

/// Streaming chunk type (alias for existing)
pub type OpenAIStreamingChunk = OpenAIStreamChunk;

/// Streaming choice type (alias for existing)
pub type OpenAIStreamingChoice = OpenAIChoice;

/// Streaming delta type (alias for existing)
pub type OpenAIStreamingDelta = OpenAIDelta;

/// Streaming tool call type (alias for existing)
pub type OpenAIStreamingToolCall = OpenAIToolCallDelta;

/// Streaming function type (alias for existing)
pub type OpenAIStreamingFunction = OpenAIFunctionDelta;
