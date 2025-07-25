//! Google AI API request/response structures
//!
//! Comprehensive, zero-allocation structures for Google AI APIs:
//! - Gemini API (Generate Content)
//! - Vertex AI (Generate Content)
//! - Embeddings API
//! - Function calling and tool use
//! - Content blocks (text, images, function calls/responses)
//! - Safety settings and content filtering
//! - Generation configuration
//!
//! All structures are optimized for performance with ArrayVec for bounded collections,
//! proper lifetime annotations, and efficient serialization patterns.

use crate::{MAX_MESSAGES, MAX_TOOLS, MAX_CHOICES};
use serde::{Deserialize, Serialize};
use arrayvec::ArrayVec;


// =============================================================================
// Generate Content API
// =============================================================================

/// Google Gemini generate content request
#[derive(Debug, Clone, Serialize)]
pub struct GeminiGenerateContentRequest {
    /// Array of content for the conversation
    pub contents: ArrayVec<GeminiContent, MAX_MESSAGES>,
    /// Available tools for function calling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ArrayVec<GeminiTool, MAX_TOOLS>>,
    /// Tool configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<GeminiToolConfig>,
    /// Safety settings for content filtering
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_settings: Option<ArrayVec<GeminiSafetySetting, 8>>,
    /// Generation configuration parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GeminiGenerationConfig>,
    /// System instruction for the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<GeminiContent>,
    /// Cached content reference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_content: Option<String>}

/// Content in a conversation turn
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiContent {
    /// Role of the content ("user" or "model")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Array of content parts
    pub parts: ArrayVec<GeminiPart, 16>}

/// Individual content part (text, image, function call, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GeminiPart {
    /// Text content
    Text {
        /// Text content
        text: String},
    /// Inline data (images, etc.)
    InlineData {
        /// Inline data blob
        inline_data: GeminiInlineData},
    /// File data reference
    FileData {
        /// File data reference
        file_data: GeminiFileData},
    /// Function call
    FunctionCall {
        /// Function call details
        function_call: GeminiFunctionCall},
    /// Function response
    FunctionResponse {
        /// Function response details
        function_response: GeminiFunctionResponse},
    /// Executable code
    ExecutableCode {
        /// Executable code details
        executable_code: GeminiExecutableCode},
    /// Code execution result
    CodeExecutionResult {
        /// Code execution result details
        code_execution_result: GeminiCodeExecutionResult}}

/// Inline data (base64 encoded)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiInlineData {
    /// MIME type of the data
    pub mime_type: String,
    /// Base64 encoded data
    pub data: String}

/// File data reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFileData {
    /// MIME type of the file
    pub mime_type: String,
    /// File URI
    pub file_uri: String}

/// Function call in content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFunctionCall {
    /// Name of the function
    pub name: String,
    /// Function arguments
    pub args: serde_json::Value}

/// Function response in content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFunctionResponse {
    /// Name of the function
    pub name: String,
    /// Function response data
    pub response: serde_json::Value}

/// Executable code part
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiExecutableCode {
    /// Programming language
    pub language: String, // "python"
    /// Code content
    pub code: String}

/// Code execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiCodeExecutionResult {
    /// Execution outcome
    pub outcome: String, // "ok", "error"
    /// Execution output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>}

/// Tool definition for function calling
#[derive(Debug, Clone, Serialize)]
pub struct GeminiTool {
    /// Function declarations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_declarations: Option<ArrayVec<GeminiFunctionDeclaration, MAX_TOOLS>>,
    /// Code execution tool
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_execution: Option<GeminiCodeExecution>,
    /// Google search tool
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google_search_retrieval: Option<GeminiGoogleSearchRetrieval>}

/// Function declaration for tools
#[derive(Debug, Clone, Serialize)]
pub struct GeminiFunctionDeclaration {
    /// Function name
    pub name: String,
    /// Function description
    pub description: String,
    /// Function parameters (JSON schema)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>}

/// Code execution tool configuration
#[derive(Debug, Clone, Serialize)]
pub struct GeminiCodeExecution {
    /// Supported languages
    #[serde(skip_serializing_if = "Option::is_none")]
    pub languages: Option<Vec<String>>}

/// Google search retrieval tool
#[derive(Debug, Clone, Serialize)]
pub struct GeminiGoogleSearchRetrieval {
    /// Dynamic retrieval config
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_retrieval_config: Option<GeminiDynamicRetrievalConfig>}

/// Dynamic retrieval configuration
#[derive(Debug, Clone, Serialize)]
pub struct GeminiDynamicRetrievalConfig {
    /// Mode for dynamic retrieval
    pub mode: String, // "mode_dynamic"
    /// Dynamic threshold
    pub dynamic_threshold: f32}

/// Tool configuration
#[derive(Debug, Clone, Serialize)]
pub struct GeminiToolConfig {
    /// Function calling configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_calling_config: Option<GeminiFunctionCallingConfig>}

/// Function calling configuration
#[derive(Debug, Clone, Serialize)]
pub struct GeminiFunctionCallingConfig {
    /// Mode for function calling
    pub mode: String, // "auto", "any", "none"
    /// Allowed function names (for "any" mode)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>}

/// Safety setting for content filtering
#[derive(Debug, Clone, Serialize)]
pub struct GeminiSafetySetting {
    /// Harm category
    pub category: String, // "HARM_CATEGORY_HARASSMENT", etc.
    /// Safety threshold
    pub threshold: String, // "BLOCK_MEDIUM_AND_ABOVE", etc.
    /// Method for applying the setting
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>, // "severity", "probability"
}

/// Generation configuration parameters
#[derive(Debug, Clone, Serialize)]
pub struct GeminiGenerationConfig {
    /// Stop sequences to halt generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Response MIME type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>, // "text/plain", "application/json"
    /// Response schema for structured output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<serde_json::Value>,
    /// Number of response candidates
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<u32>,
    /// Maximum output tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Sampling temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top-p sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k sampling parameter  
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Presence penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Frequency penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Response logprobs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_logprobs: Option<bool>,
    /// Logprobs count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>}

// =============================================================================
// Response Types
// =============================================================================

/// Generate content response
#[derive(Debug, Deserialize)]
pub struct GeminiGenerateContentResponse {
    /// Response candidates
    pub candidates: ArrayVec<GeminiCandidate, MAX_CHOICES>,
    /// Prompt feedback
    #[serde(default)]
    pub prompt_feedback: Option<GeminiPromptFeedback>,
    /// Usage metadata
    #[serde(default)]
    pub usage_metadata: Option<GeminiUsageMetadata>,
    /// Model version
    #[serde(default)]
    pub model_version: Option<String>}

/// Response candidate
#[derive(Debug, Deserialize)]
pub struct GeminiCandidate {
    /// Generated content
    pub content: GeminiContent,
    /// Finish reason
    #[serde(default)]
    pub finish_reason: Option<String>, // "stop", "max_tokens", "safety", "recitation", "language", "other"
    /// Safety ratings
    #[serde(default)]
    pub safety_ratings: Option<ArrayVec<GeminiSafetyRating, 8>>,
    /// Citation metadata
    #[serde(default)]
    pub citation_metadata: Option<GeminiCitationMetadata>,
    /// Token count
    #[serde(default)]
    pub token_count: Option<u32>,
    /// Grounding attributions
    #[serde(default)]
    pub grounding_attributions: Option<Vec<GeminiGroundingAttribution>>,
    /// Index of this candidate
    #[serde(default)]
    pub index: Option<u32>}

/// Safety rating for content
#[derive(Debug, Deserialize)]
pub struct GeminiSafetyRating {
    /// Harm category
    pub category: String,
    /// Probability of harm
    pub probability: String, // "negligible", "low", "medium", "high"
    /// Blocked status
    #[serde(default)]
    pub blocked: Option<bool>}

/// Citation metadata
#[derive(Debug, Deserialize)]
pub struct GeminiCitationMetadata {
    /// Citation sources
    pub citation_sources: Vec<GeminiCitationSource>}

/// Citation source
#[derive(Debug, Deserialize)]
pub struct GeminiCitationSource {
    /// Start index in the content
    #[serde(default)]
    pub start_index: Option<u32>,
    /// End index in the content
    #[serde(default)]
    pub end_index: Option<u32>,
    /// Source URI
    #[serde(default)]
    pub uri: Option<String>,
    /// License information
    #[serde(default)]
    pub license: Option<String>}

/// Grounding attribution
#[derive(Debug, Deserialize)]
pub struct GeminiGroundingAttribution {
    /// Attribution source
    pub source_id: Option<String>,
    /// Content that is attributed
    pub content: GeminiContent}

/// Prompt feedback
#[derive(Debug, Deserialize)]
pub struct GeminiPromptFeedback {
    /// Block reason
    #[serde(default)]
    pub block_reason: Option<String>, // "safety", "other"
    /// Safety ratings for the prompt
    #[serde(default)]
    pub safety_ratings: Option<ArrayVec<GeminiSafetyRating, 8>>}

/// Usage metadata
#[derive(Debug, Deserialize)]
pub struct GeminiUsageMetadata {
    /// Prompt token count
    pub prompt_token_count: u32,
    /// Candidates token count
    #[serde(default)]
    pub candidates_token_count: Option<u32>,
    /// Total token count
    pub total_token_count: u32,
    /// Cached content token count
    #[serde(default)]
    pub cached_content_token_count: Option<u32>}

// =============================================================================
// Streaming Response Types
// =============================================================================

/// Streaming generate content response
#[derive(Debug, Deserialize)]
pub struct GeminiStreamGenerateContentResponse {
    /// Response candidates
    pub candidates: ArrayVec<GeminiCandidate, MAX_CHOICES>,
    /// Prompt feedback (only in first chunk)
    #[serde(default)]
    pub prompt_feedback: Option<GeminiPromptFeedback>,
    /// Usage metadata (only in final chunk)
    #[serde(default)]
    pub usage_metadata: Option<GeminiUsageMetadata>,
    /// Model version
    #[serde(default)]
    pub model_version: Option<String>}

// =============================================================================
// Embeddings API
// =============================================================================

/// Embed content request
#[derive(Debug, Serialize)]
pub struct GeminiEmbedContentRequest {
    /// Content to embed
    pub content: GeminiContent,
    /// Task type for the embedding
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<String>, // "retrieval_query", "retrieval_document", "semantic_similarity", "classification", "clustering", "question_answering", "fact_verification"
    /// Title for the content (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Output dimensionality
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dimensionality: Option<u32>}

/// Embed content response
#[derive(Debug, Deserialize)]
pub struct GeminiEmbedContentResponse {
    /// Content embedding
    pub embedding: GeminiContentEmbedding}

/// Content embedding
#[derive(Debug, Deserialize)]
pub struct GeminiContentEmbedding {
    /// Embedding values
    pub values: Vec<f32>}

/// Batch embed contents request
#[derive(Debug, Serialize)]
pub struct GeminiBatchEmbedContentsRequest {
    /// Array of embed requests
    pub requests: Vec<GeminiEmbedContentRequest>}

/// Batch embed contents response
#[derive(Debug, Deserialize)]
pub struct GeminiBatchEmbedContentsResponse {
    /// Array of embeddings
    pub embeddings: Vec<GeminiContentEmbedding>}

// =============================================================================
// Count Tokens API
// =============================================================================

/// Count tokens request
#[derive(Debug, Serialize)]
pub struct GeminiCountTokensRequest {
    /// Contents to count tokens for
    pub contents: ArrayVec<GeminiContent, MAX_MESSAGES>,
    /// Generate content request (alternative to contents)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate_content_request: Option<GeminiGenerateContentRequest>}

/// Count tokens response
#[derive(Debug, Deserialize)]
pub struct GeminiCountTokensResponse {
    /// Total token count
    pub total_tokens: u32,
    /// Cached content token count
    #[serde(default)]
    pub cached_content_token_count: Option<u32>}

// =============================================================================
// Utility Types and Implementations
// =============================================================================

impl Default for GeminiGenerateContentRequest {
    fn default() -> Self {
        Self {
            contents: ArrayVec::new(),
            tools: None,
            tool_config: None,
            safety_settings: None,
            generation_config: None,
            system_instruction: None,
            cached_content: None}
    }
}

impl GeminiGenerateContentRequest {
    /// Create a new generate content request
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add user content to the request
    pub fn add_user_content(&mut self, text: String) -> Result<(), &'static str> {
        let mut parts = ArrayVec::new();
        parts.try_push(GeminiPart::Text { text })
            .map_err(|_| "Failed to add text part")?;
        
        let content = GeminiContent {
            role: Some("user".to_string()),
            parts};
        
        self.contents.try_push(content)
            .map_err(|_| "Maximum contents exceeded")
    }

    /// Add model content to the request
    pub fn add_model_content(&mut self, text: String) -> Result<(), &'static str> {
        let mut parts = ArrayVec::new();
        parts.try_push(GeminiPart::Text { text })
            .map_err(|_| "Failed to add text part")?;
        
        let content = GeminiContent {
            role: Some("model".to_string()),
            parts};
        
        self.contents.try_push(content)
            .map_err(|_| "Maximum contents exceeded")
    }

    /// Set system instruction
    pub fn with_system_instruction(mut self, text: String) -> Self {
        let mut parts = ArrayVec::new();
        let _ = parts.try_push(GeminiPart::Text { text });
        
        self.system_instruction = Some(GeminiContent {
            role: None,
            parts});
        self
    }

    /// Set generation configuration
    #[inline]
    pub fn with_generation_config(mut self, config: GeminiGenerationConfig) -> Self {
        self.generation_config = Some(config);
        self
    }

    /// Add a function declaration tool
    pub fn add_function_tool(&mut self, name: String, description: String, parameters: Option<serde_json::Value>) -> Result<(), &'static str> {
        let function_declaration = GeminiFunctionDeclaration {
            name,
            description,
            parameters};

        if self.tools.is_none() {
            self.tools = Some(ArrayVec::new());
        }

        if let Some(ref mut tools) = self.tools {
            if tools.is_empty() {
                let mut function_declarations = ArrayVec::new();
                function_declarations.try_push(function_declaration)
                    .map_err(|_| "Failed to add function declaration")?;
                
                let tool = GeminiTool {
                    function_declarations: Some(function_declarations),
                    code_execution: None,
                    google_search_retrieval: None};
                
                tools.try_push(tool)
                    .map_err(|_| "Maximum tools exceeded")
            } else if let Some(ref mut existing_tool) = tools.get_mut(0) {
                if let Some(ref mut declarations) = existing_tool.function_declarations {
                    declarations.try_push(function_declaration)
                        .map_err(|_| "Maximum function declarations exceeded")
                } else {
                    let mut function_declarations = ArrayVec::new();
                    function_declarations.try_push(function_declaration)
                        .map_err(|_| "Failed to add function declaration")?;
                    existing_tool.function_declarations = Some(function_declarations);
                    Ok(())
                }
            } else {
                Err("Failed to access existing tool")
            }
        } else {
            Err("Failed to initialize tools")
        }
    }
}

impl Default for GeminiGenerationConfig {
    fn default() -> Self {
        Self {
            stop_sequences: None,
            response_mime_type: None,
            response_schema: None,
            candidate_count: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            presence_penalty: None,
            frequency_penalty: None,
            response_logprobs: None,
            logprobs: None}
    }
}

impl GeminiGenerationConfig {
    /// Create a new generation config
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set temperature
    #[inline]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set max output tokens
    #[inline]
    pub fn with_max_output_tokens(mut self, max_tokens: u32) -> Self {
        self.max_output_tokens = Some(max_tokens);
        self
    }

    /// Set top-p
    #[inline]
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set top-k
    #[inline]
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set response format to JSON
    #[inline]
    pub fn with_json_response(mut self, schema: Option<serde_json::Value>) -> Self {
        self.response_mime_type = Some("application/json".to_string());
        self.response_schema = schema;
        self
    }
}

impl GeminiContent {
    /// Create user content with text
    #[inline]
    pub fn user_text(text: String) -> Self {
        let mut parts = ArrayVec::new();
        let _ = parts.try_push(GeminiPart::Text { text });
        
        Self {
            role: Some("user".to_string()),
            parts}
    }

    /// Create model content with text
    #[inline]
    pub fn model_text(text: String) -> Self {
        let mut parts = ArrayVec::new();
        let _ = parts.try_push(GeminiPart::Text { text });
        
        Self {
            role: Some("model".to_string()),
            parts}
    }

    /// Create content with text and image
    pub fn user_multimodal(text: String, image_data: String, mime_type: String) -> Self {
        let mut parts = ArrayVec::new();
        let _ = parts.try_push(GeminiPart::Text { text });
        let _ = parts.try_push(GeminiPart::InlineData {
            inline_data: GeminiInlineData {
                mime_type,
                data: image_data}});
        
        Self {
            role: Some("user".to_string()),
            parts}
    }

    /// Add a text part
    pub fn add_text(&mut self, text: String) -> Result<(), &'static str> {
        self.parts.try_push(GeminiPart::Text { text })
            .map_err(|_| "Maximum parts exceeded")
    }

    /// Add an image part
    pub fn add_image(&mut self, data: String, mime_type: String) -> Result<(), &'static str> {
        let inline_data = GeminiInlineData { mime_type, data };
        self.parts.try_push(GeminiPart::InlineData { inline_data })
            .map_err(|_| "Maximum parts exceeded")
    }

    /// Add a function call part
    pub fn add_function_call(&mut self, name: String, args: serde_json::Value) -> Result<(), &'static str> {
        let function_call = GeminiFunctionCall { name, args };
        self.parts.try_push(GeminiPart::FunctionCall { function_call })
            .map_err(|_| "Maximum parts exceeded")
    }

    /// Add a function response part
    pub fn add_function_response(&mut self, name: String, response: serde_json::Value) -> Result<(), &'static str> {
        let function_response = GeminiFunctionResponse { name, response };
        self.parts.try_push(GeminiPart::FunctionResponse { function_response })
            .map_err(|_| "Maximum parts exceeded")
    }
}

/// Helper functions for safety settings
impl GeminiSafetySetting {
    /// Create harassment safety setting
    #[inline]
    pub fn block_harassment_medium_and_above() -> Self {
        Self {
            category: "HARM_CATEGORY_HARASSMENT".to_string(),
            threshold: "BLOCK_MEDIUM_AND_ABOVE".to_string(),
            method: None}
    }

    /// Create hate speech safety setting
    #[inline]
    pub fn block_hate_speech_medium_and_above() -> Self {
        Self {
            category: "HARM_CATEGORY_HATE_SPEECH".to_string(),
            threshold: "BLOCK_MEDIUM_AND_ABOVE".to_string(),
            method: None}
    }

    /// Create sexually explicit safety setting
    #[inline]
    pub fn block_sexually_explicit_medium_and_above() -> Self {
        Self {
            category: "HARM_CATEGORY_SEXUALLY_EXPLICIT".to_string(),
            threshold: "BLOCK_MEDIUM_AND_ABOVE".to_string(),
            method: None}
    }

    /// Create dangerous content safety setting
    #[inline]
    pub fn block_dangerous_content_medium_and_above() -> Self {
        Self {
            category: "HARM_CATEGORY_DANGEROUS_CONTENT".to_string(),
            threshold: "BLOCK_MEDIUM_AND_ABOVE".to_string(),
            method: None}
    }
}

/// Helper functions for response processing
impl GeminiGenerateContentResponse {
    /// Get the first candidate's text content
    pub fn text(&self) -> Option<String> {
        self.candidates.first()?
            .content.parts.iter()
            .find_map(|part| {
                if let GeminiPart::Text { text } = part {
                    Some(text.clone())
                } else {
                    None
                }
            })
    }

    /// Get the first candidate's finish reason
    #[inline]
    pub fn finish_reason(&self) -> Option<&str> {
        self.candidates.first()?.finish_reason.as_deref()
    }

    /// Check if the response is blocked by safety filters
    #[inline]
    pub fn is_blocked(&self) -> bool {
        self.candidates.first()
            .map(|candidate| {
                candidate.finish_reason.as_deref() == Some("safety") ||
                candidate.safety_ratings.as_ref()
                    .map(|ratings| ratings.iter().any(|rating| rating.blocked.unwrap_or(false)))
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    }

    /// Get function calls from the first candidate
    pub fn function_calls(&self) -> Vec<&GeminiFunctionCall> {
        self.candidates.first()
            .map(|candidate| {
                candidate.content.parts.iter()
                    .filter_map(|part| {
                        if let GeminiPart::FunctionCall { function_call } = part {
                            Some(function_call)
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl GeminiUsageMetadata {
    /// Calculate total billable tokens
    #[inline]
    pub fn billable_tokens(&self) -> u32 {
        // Cached tokens are typically not billed
        self.total_token_count - self.cached_content_token_count.unwrap_or(0)
    }
}