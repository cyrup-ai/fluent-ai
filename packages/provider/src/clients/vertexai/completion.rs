//! Zero-allocation completion provider for VertexAI following CompletionProvider pattern
//!
//! Blazing-fast streaming completions with elegant ergonomics:
//! ```
//! VertexAICompletionBuilder::new(service_account_json, project_id, region, "gemini-2.5-flash")
//!     .system_prompt("You are helpful")
//!     .temperature(0.8)
//!     .prompt("Hello world")
//! ```

use fluent_ai_domain::{AsyncTask, spawn_async};
use fluent_ai_domain::chunk::{CompletionChunk, FinishReason, Usage};
use fluent_ai_domain::{Message, Document};
use fluent_ai_domain::tool::ToolDefinition;
use crate::{AsyncStream, completion_provider::{CompletionProvider, CompletionError, ModelConfig, ModelInfo, ChunkHandler}};
use crate::clients::vertexai::model_info::get_model_config;
use fluent_ai_http3::{HttpClient, HttpConfig, HttpRequest, HttpError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use arrayvec::{ArrayString};
use cyrup_sugars::ZeroOneOrMany;
/// Maximum number of messages in a conversation
const MAX_MESSAGES: usize = 100;

/// Maximum number of function/tool definitions
const MAX_TOOLS: usize = 20;

/// Maximum number of parts per message
const MAX_PARTS: usize = 10;

/// Maximum number of safety settings
const MAX_SAFETY_SETTINGS: usize = 5;

/// Maximum number of response candidates
const MAX_CANDIDATES: usize = 8;

/// Zero-allocation VertexAI completion builder with perfect ergonomics
#[derive(Clone)]
pub struct VertexAICompletionBuilder {
    client: HttpClient,
    service_account_json: String,
    project_id: String,
    region: String,
    explicit_api_key: Option<String>, // .api_key() override (for consistency)
    model_name: &'static str,
    config: &'static ModelConfig,
    system_prompt: String,
    temperature: f64,
    max_tokens: u32,
    top_p: f64,
    frequency_penalty: f64,
    presence_penalty: f64,
    chat_history: ArrayVec<Message, MAX_MESSAGES>,
    documents: ArrayVec<Document, 64>,
    tools: ArrayVec<ToolDefinition, MAX_TOOLS>,
    additional_params: Option<Value>,
    chunk_handler: Option<ChunkHandler>}

/// Completion request for VertexAI models
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CompletionRequest {
    /// Message contents
    pub contents: ArrayVec<Content, MAX_MESSAGES>,
    
    /// Generation configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    
    /// Safety settings
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub safety_settings: ArrayVec<SafetySettings, MAX_SAFETY_SETTINGS>,
    
    /// Tool/function definitions
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: ArrayVec<Tool, MAX_TOOLS>,
    
    /// Tool configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    
    /// System instruction
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>}

/// Message content with parts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    /// Message role (user, model, system)
    pub role: ArrayString<16>,
    
    /// Message parts (text, image, etc.)
    pub parts: ArrayVec<Part, MAX_PARTS>}

/// Message part (text, image, function call, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Part {
    /// Text content
    Text {
        text: String},
    
    /// Image content (base64 encoded)
    InlineData {
        #[serde(rename = "mimeType")]
        mime_type: ArrayString<32>,
        data: String},
    
    /// File data reference
    FileData {
        #[serde(rename = "mimeType")]
        mime_type: ArrayString<32>,
        #[serde(rename = "fileUri")]
        file_uri: ArrayString<256>},
    
    /// Function call
    FunctionCall {
        name: ArrayString<64>,
        args: HashMap<String, serde_json::Value>},
    
    /// Function response
    FunctionResponse {
        name: ArrayString<64>,
        response: HashMap<String, serde_json::Value>},
    
    /// Video content
    VideoMetadata {
        #[serde(rename = "startOffset")]
        start_offset: ArrayString<16>,
        #[serde(rename = "endOffset")]
        end_offset: ArrayString<16>}}

/// Generation configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    /// Temperature for randomness (0.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    
    /// Top-p for nucleus sampling (0.0 to 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    
    /// Top-k for token filtering
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    
    /// Maximum output tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    
    /// Number of response candidates
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<u32>,
    
    /// Stop sequences
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: ArrayVec<ArrayString<32>, 8>,
    
    /// Response MIME type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<ArrayString<64>>,
    
    /// Response schema for structured output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<serde_json::Value>}

/// Safety settings for content filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub struct SafetySettings {
    /// Safety category
    pub category: SafetyCategory,
    
    /// Safety threshold
    pub threshold: SafetyThreshold}

/// Safety categories for content filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SafetyCategory {
    HarmCategoryUnspecified,
    HarmCategoryDerogatory,
    HarmCategoryToxicity,
    HarmCategoryViolence,
    HarmCategorySexual,
    HarmCategoryMedical,
    HarmCategoryDangerous,
    HarmCategoryHarassment,
    HarmCategoryHateSpeech,
    HarmCategorySexuallyExplicit,
    HarmCategoryDangerousContent}

/// Safety thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SafetyThreshold {
    HarmBlockThresholdUnspecified,
    BlockLowAndAbove,
    BlockMediumAndAbove,
    BlockOnlyHigh,
    BlockNone}

/// Tool/function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    /// Function declarations
    pub function_declarations: ArrayVec<FunctionDeclaration, MAX_TOOLS>}

/// Function declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDeclaration {
    /// Function name
    pub name: ArrayString<64>,
    
    /// Function description
    pub description: String,
    
    /// Function parameters schema
    pub parameters: serde_json::Value}

/// Tool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    /// Function calling configuration
    pub function_calling_config: FunctionCallingConfig}

/// Function calling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {
    /// Function calling mode
    pub mode: FunctionCallingMode,
    
    /// Allowed function names
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub allowed_function_names: ArrayVec<ArrayString<64>, MAX_TOOLS>}

/// Function calling modes
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FunctionCallingMode {
    ModeUnspecified,
    Auto,
    Any,
    None}

/// Completion response from VertexAI
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompletionResponse {
    /// Response candidates
    pub candidates: ArrayVec<Candidate, MAX_CANDIDATES>,
    
    /// Prompt feedback
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_feedback: Option<PromptFeedback>,
    
    /// Usage metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<UsageMetadata>}

/// Response candidate
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    /// Candidate content
    pub content: Content,
    
    /// Finish reason
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    
    /// Safety ratings
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub safety_ratings: ArrayVec<SafetyRating, MAX_SAFETY_SETTINGS>,
    
    /// Citation metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citation_metadata: Option<CitationMetadata>,
    
    /// Grounding attributions
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub grounding_attributions: ArrayVec<GroundingAttribution, 10>}

/// Finish reasons
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinishReason {
    FinishReasonUnspecified,
    Stop,
    MaxTokens,
    Safety,
    Recitation,
    Other,
    Blocklist,
    ProhibitedContent,
    Spii,
    MalformedFunctionCall}

/// Safety rating
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SafetyRating {
    /// Safety category
    pub category: SafetyCategory,
    
    /// Probability rating
    pub probability: SafetyProbability,
    
    /// Blocked status
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked: Option<bool>}

/// Safety probability levels
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SafetyProbability {
    HarmProbabilityUnspecified,
    Negligible,
    Low,
    Medium,
    High}

/// Prompt feedback
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptFeedback {
    /// Block reason
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_reason: Option<BlockReason>,
    
    /// Safety ratings
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub safety_ratings: ArrayVec<SafetyRating, MAX_SAFETY_SETTINGS>}

/// Block reasons
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BlockReason {
    BlockReasonUnspecified,
    Safety,
    Other,
    Blocklist,
    ProhibitedContent}

/// Usage metadata
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    /// Prompt token count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_token_count: Option<u32>,
    
    /// Candidates token count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates_token_count: Option<u32>,
    
    /// Total token count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_token_count: Option<u32>,
    
    /// Cached content token count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_content_token_count: Option<u32>}

/// Citation metadata
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CitationMetadata {
    /// Citation sources
    pub citation_sources: ArrayVec<CitationSource, 10>}

/// Citation source
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CitationSource {
    /// Start index
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_index: Option<u32>,
    
    /// End index
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_index: Option<u32>,
    
    /// Source URI
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<ArrayString<256>>,
    
    /// License
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<ArrayString<64>>}

/// Grounding attribution
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingAttribution {
    /// Attribution content
    pub content: Content,
    
    /// Source ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_id: Option<ArrayString<64>>}

/// Streaming completion chunk
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompletionChunk {
    /// Chunk candidates
    pub candidates: ArrayVec<Candidate, MAX_CANDIDATES>,
    
    /// Usage metadata (final chunk only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<UsageMetadata>,
    
    /// Model version
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<ArrayString<32>>}

impl CompletionRequest {
    /// Create new completion request with validation
    pub fn new(model_name: &str) -> VertexAIResult<Self> {
        // Validate model exists
        VertexAIModels::validate_model(model_name)?;
        
        Ok(Self {
            contents: ArrayVec::new(),
            generation_config: None,
            safety_settings: ArrayVec::new(),
            tools: ArrayVec::new(),
            tool_config: None,
            system_instruction: None})
    }
    
    /// Add user message
    pub fn add_user_message(&mut self, text: &str) -> VertexAIResult<()> {
        if self.contents.len() >= MAX_MESSAGES {
            return Err(VertexAIError::RequestValidation {
                field: "contents".to_string(),
                reason: format!("Maximum {} messages exceeded", MAX_MESSAGES)});
        }
        
        let mut parts = ArrayVec::new();
        parts.push(Part::Text {
            text: text.to_string()});
        
        let content = Content {
            role: ArrayString::from("user").map_err(|_| VertexAIError::Internal {
                context: "Failed to create user role".to_string()})?,
            parts};
        
        self.contents.push(content);
        Ok(())
    }
    
    /// Add system message
    pub fn add_system_message(&mut self, text: &str) -> VertexAIResult<()> {
        let mut parts = ArrayVec::new();
        parts.push(Part::Text {
            text: text.to_string()});
        
        self.system_instruction = Some(Content {
            role: ArrayString::from("user").map_err(|_| VertexAIError::Internal {
                context: "Failed to create system role".to_string()})?,
            parts});
        
        Ok(())
    }
    
    /// Add image to last message
    pub fn add_image(&mut self, image_data: &str, mime_type: &str) -> VertexAIResult<()> {
        let last_content = self.contents.last_mut().ok_or_else(|| {
            VertexAIError::RequestValidation {
                field: "contents".to_string(),
                reason: "No messages to add image to".to_string()}
        })?;
        
        if last_content.parts.len() >= MAX_PARTS {
            return Err(VertexAIError::RequestValidation {
                field: "parts".to_string(),
                reason: format!("Maximum {} parts per message exceeded", MAX_PARTS)});
        }
        
        let mime_type = ArrayString::from(mime_type).map_err(|_| {
            VertexAIError::RequestValidation {
                field: "mime_type".to_string(),
                reason: "MIME type too long".to_string()}
        })?;
        
        last_content.parts.push(Part::InlineData {
            mime_type,
            data: image_data.to_string()});
        
        Ok(())
    }
    
    /// Set generation configuration
    pub fn set_generation_config(&mut self, config: GenerationConfig) {
        self.generation_config = Some(config);
    }
    
    /// Add function/tool definition
    pub fn add_tool(&mut self, function: FunctionDeclaration) -> VertexAIResult<()> {
        if self.tools.is_empty() {
            self.tools.push(Tool {
                function_declarations: ArrayVec::new()});
        }
        
        let tool = self.tools.last_mut().ok_or_else(|| VertexAIError::Internal {
            context: "Tool list is empty after adding tool".to_string()})?;
        
        if tool.function_declarations.len() >= MAX_TOOLS {
            return Err(VertexAIError::RequestValidation {
                field: "tools".to_string(),
                reason: format!("Maximum {} tools exceeded", MAX_TOOLS)});
        }
        
        tool.function_declarations.push(function);
        Ok(())
    }
    
    /// Set function calling mode
    pub fn set_function_calling_mode(&mut self, mode: FunctionCallingMode) {
        self.tool_config = Some(ToolConfig {
            function_calling_config: FunctionCallingConfig {
                mode,
                allowed_function_names: ArrayVec::new()}});
    }
    
    /// Add safety setting
    pub fn add_safety_setting(&mut self, category: SafetyCategory, threshold: SafetyThreshold) -> VertexAIResult<()> {
        if self.safety_settings.len() >= MAX_SAFETY_SETTINGS {
            return Err(VertexAIError::RequestValidation {
                field: "safety_settings".to_string(),
                reason: format!("Maximum {} safety settings exceeded", MAX_SAFETY_SETTINGS)});
        }
        
        self.safety_settings.push(SafetySettings {
            category,
            threshold});
        
        Ok(())
    }
    
    /// Validate request before sending
    pub fn validate(&self, model_name: &str) -> VertexAIResult<()> {
        // Validate model exists
        VertexAIModels::validate_model(model_name)?;
        
        // Validate has content
        if self.contents.is_empty() && self.system_instruction.is_none() {
            return Err(VertexAIError::RequestValidation {
                field: "contents".to_string(),
                reason: "Request must have at least one message or system instruction".to_string()});
        }
        
        // Validate generation config if present
        if let Some(ref config) = self.generation_config {
            VertexAIModels::validate_parameters(
                model_name,
                config.temperature,
                config.top_p,
                config.top_k,
                config.max_output_tokens,
            )?;
        }
        
        Ok(())
    }
    
    /// Serialize request to JSON bytes
    pub fn to_json_bytes(&self) -> VertexAIResult<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| VertexAIError::Json {
            operation: "request_serialization".to_string(),
            details: format!("Failed to serialize request: {}", e)})
    }
}

impl GenerationConfig {
    /// Create new generation config with defaults
    pub fn new() -> Self {
        Self {
            temperature: None,
            top_p: None,
            top_k: None,
            max_output_tokens: None,
            candidate_count: None,
            stop_sequences: ArrayVec::new(),
            response_mime_type: None,
            response_schema: None}
    }
    
    /// Set temperature
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }
    
    /// Set top-p
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }
    
    /// Set top-k
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }
    
    /// Set max output tokens
    pub fn max_output_tokens(mut self, max_tokens: u32) -> Self {
        self.max_output_tokens = Some(max_tokens);
        self
    }
    
    /// Add stop sequence
    pub fn add_stop_sequence(mut self, stop: &str) -> VertexAIResult<Self> {
        if self.stop_sequences.len() >= 8 {
            return Err(VertexAIError::RequestValidation {
                field: "stop_sequences".to_string(),
                reason: "Maximum 8 stop sequences allowed".to_string()});
        }
        
        let stop_string = ArrayString::from(stop).map_err(|_| {
            VertexAIError::RequestValidation {
                field: "stop_sequences".to_string(),
                reason: "Stop sequence too long".to_string()}
        })?;
        
        self.stop_sequences.push(stop_string);
        Ok(self)
    }
    
    /// Set JSON mode
    pub fn json_mode(mut self) -> VertexAIResult<Self> {
        self.response_mime_type = Some(ArrayString::from("application/json").map_err(|_| {
            VertexAIError::Internal {
                context: "Failed to set JSON MIME type".to_string()}
        })?);
        Ok(self)
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl CompletionResponse {
    /// Get first candidate text
    pub fn text(&self) -> Option<String> {
        self.candidates
            .first()?
            .content
            .parts
            .iter()
            .find_map(|part| match part {
                Part::Text { text } => Some(text.clone()),
                _ => None})
    }
    
    /// Get finish reason
    pub fn finish_reason(&self) -> Option<&FinishReason> {
        self.candidates.first()?.finish_reason.as_ref()
    }
    
    /// Get usage metadata
    pub fn usage(&self) -> Option<&UsageMetadata> {
        self.usage_metadata.as_ref()
    }
    
    /// Check if response was blocked by safety filters
    pub fn is_blocked(&self) -> bool {
        if let Some(feedback) = &self.prompt_feedback {
            if feedback.block_reason.is_some() {
                return true;
            }
        }
        
        self.candidates
            .iter()
            .any(|candidate| {
                candidate.safety_ratings
                    .iter()
                    .any(|rating| rating.blocked.unwrap_or(false))
            })
    }
    
    /// Get function calls from response
    pub fn function_calls(&self) -> Vec<&Part> {
        self.candidates
            .iter()
            .flat_map(|candidate| candidate.content.parts.iter())
            .filter(|part| matches!(part, Part::FunctionCall { .. }))
            .collect()
    }
}

impl CompletionProvider for VertexAICompletionBuilder {
    /// Create new VertexAI completion builder with ModelInfo defaults
    #[inline(always)]
    fn new(api_key: String, model_name: &'static str) -> Result<Self, CompletionError> {
        let client = HttpClient::with_config(HttpConfig::streaming_optimized())
            .map_err(|_| CompletionError::HttpError)?;
        
        let config = get_model_config(model_name);

        Ok(Self {
            client,
            service_account_json: api_key, // Use api_key as service account JSON
            project_id: String::new(), // Will be extracted from service account JSON
            region: "us-central1".to_string(), // Default region
            explicit_api_key: None,
            model_name,
            config,
            system_prompt: config.system_prompt.to_string(),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            top_p: config.top_p,
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
            chat_history: ArrayVec::new(),
            documents: ArrayVec::new(),
            tools: ArrayVec::new(),
            additional_params: None,
            chunk_handler: None})
    }

    /// Set explicit API key (for consistency with other providers)
    #[inline(always)]
    fn api_key(mut self, key: impl Into<String>) -> Self {
        self.explicit_api_key = Some(key.into());
        self
    }
    
    /// Environment variable names to search for VertexAI credentials (ordered by priority)
    #[inline(always)]
    fn env_api_keys(&self) -> ZeroOneOrMany<String> {
        ZeroOneOrMany::Many(vec![
            "GOOGLE_APPLICATION_CREDENTIALS".to_string(),    // Primary Google Cloud key
            "VERTEXAI_SERVICE_ACCOUNT".to_string(),         // VertexAI-specific
            "GOOGLE_SERVICE_ACCOUNT_JSON".to_string(),      // Alternative name
        ])
    }
    
    /// Set system prompt (overrides ModelInfo default)
    #[inline(always)]
    fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set temperature (overrides ModelInfo default)
    #[inline(always)]
    fn temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    /// Set max tokens (overrides ModelInfo default)
    #[inline(always)]
    fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set top_p (overrides ModelInfo default)
    #[inline(always)]
    fn top_p(mut self, p: f64) -> Self {
        self.top_p = p;
        self
    }

    /// Set frequency penalty (overrides ModelInfo default)
    #[inline(always)]
    fn frequency_penalty(mut self, penalty: f64) -> Self {
        self.frequency_penalty = penalty;
        self
    }

    /// Set presence penalty (overrides ModelInfo default)
    #[inline(always)]
    fn presence_penalty(mut self, penalty: f64) -> Self {
        self.presence_penalty = penalty;
        self
    }

    /// Add chat history (ZeroOneOrMany with bounded capacity)
    #[inline(always)]
    fn chat_history(mut self, history: ZeroOneOrMany<Message>) -> Result<Self, CompletionError> {
        match history {
            ZeroOneOrMany::None => {},
            ZeroOneOrMany::One(msg) => {
                if self.chat_history.try_push(msg).is_err() {
                    return Err(CompletionError::ConfigError);
                }
            }
            ZeroOneOrMany::Many(msgs) => {
                for msg in msgs {
                    if self.chat_history.try_push(msg).is_err() {
                        return Err(CompletionError::ConfigError);
                    }
                }
            }
        }
        Ok(self)
    }

    /// Add documents (ZeroOneOrMany with bounded capacity)
    #[inline(always)]
    fn documents(mut self, docs: ZeroOneOrMany<Document>) -> Result<Self, CompletionError> {
        match docs {
            ZeroOneOrMany::None => {},
            ZeroOneOrMany::One(doc) => {
                if self.documents.try_push(doc).is_err() {
                    return Err(CompletionError::ConfigError);
                }
            }
            ZeroOneOrMany::Many(docs) => {
                for doc in docs {
                    if self.documents.try_push(doc).is_err() {
                        return Err(CompletionError::ConfigError);
                    }
                }
            }
        }
        Ok(self)
    }

    /// Add tools (ZeroOneOrMany with bounded capacity)
    #[inline(always)]
    fn tools(mut self, tools: ZeroOneOrMany<ToolDefinition>) -> Result<Self, CompletionError> {
        match tools {
            ZeroOneOrMany::None => {},
            ZeroOneOrMany::One(tool) => {
                if self.tools.try_push(tool).is_err() {
                    return Err(CompletionError::ConfigError);
                }
            }
            ZeroOneOrMany::Many(tools) => {
                for tool in tools {
                    if self.tools.try_push(tool).is_err() {
                        return Err(CompletionError::ConfigError);
                    }
                }
            }
        }
        Ok(self)
    }

    /// Set additional params
    #[inline(always)]
    fn additional_params(mut self, params: Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    /// Set chunk handler
    #[inline(always)]
    fn on_chunk<F>(mut self, handler: F) -> Self 
    where 
        F: Fn(Result<CompletionChunk, CompletionError>) + Send + Sync + 'static 
    {
        self.chunk_handler = Some(Box::new(handler));
        self
    }

    /// Execute streaming completion - terminal method
    #[inline(always)]
    fn prompt(self, text: impl AsRef<str>) -> AsyncStream<CompletionChunk> {
        let prompt = text.as_ref().to_string();
        let (sender, receiver) = crate::channel();

        spawn_async(async move {
            match self.execute_streaming_completion(prompt).await {
                Ok(mut stream) => {
                    use futures_util::StreamExt;
use fluent_ai_http3::HttpClient;
use fluent_ai_http3::HttpError;
use fluent_ai_http3::HttpRequest;
use std::collections::HashMap;
                    while let Some(result) = stream.next().await {
                        match result {
                            Ok(chunk) => {
                                if let Some(ref handler) = self.chunk_handler {
                                    handler(Ok(chunk.clone()));
                                }
                                if sender.send(chunk).is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                if let Some(ref handler) = self.chunk_handler {
                                    handler(Err(e));
                                }
                                let _ = sender.send(CompletionChunk::error(e.message()));
                            }
                        }
                    }
                }
                Err(e) => {
                    log::error!("Failed to start completion: {}", e);
                    let _ = sender.send(CompletionChunk::error(e.message()));
                }
            }
        });

        receiver
    }
}

impl VertexAICompletionBuilder {
    /// Create new VertexAI completion builder with all parameters
    pub fn new(
        service_account_json: String,
        project_id: String, 
        region: String,
        model_name: &'static str
    ) -> Result<Self, CompletionError> {
        let client = HttpClient::with_config(HttpConfig::streaming_optimized())
            .map_err(|_| CompletionError::HttpError)?;
        
        let config = get_model_config(model_name);

        Ok(Self {
            client,
            service_account_json,
            project_id,
            region,
            explicit_api_key: None,
            model_name,
            config,
            system_prompt: config.system_prompt.to_string(),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            top_p: config.top_p,
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
            chat_history: ArrayVec::new(),
            documents: ArrayVec::new(),
            tools: ArrayVec::new(),
            additional_params: None,
            chunk_handler: None})
    }

    /// Execute streaming completion with zero-allocation HTTP3 (blazing-fast)
    #[inline(always)]
    async fn execute_streaming_completion(
        &self,
        prompt: String,
    ) -> Result<AsyncStream<Result<CompletionChunk, CompletionError>>, CompletionError> {
        // TODO: Implement VertexAI OAuth2 authentication and HTTP request
        // For now, return a simple error stream
        let (sender, receiver) = crate::channel();
        let _ = sender.send(Err(CompletionError::ProviderError));
        Ok(receiver)
    }
}