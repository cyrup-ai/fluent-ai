//! OpenRouter API request and response structures.
//! 
//! This module provides zero-allocation structures for interacting with OpenRouter's
//! multi-provider LLM gateway. OpenRouter is largely OpenAI-compatible with
//! provider selection and routing extensions.
//! All collections use ArrayVec for bounded, stack-allocated storage.

use serde::{Deserialize, Serialize};
use arrayvec::ArrayVec;
use crate::{MAX_MESSAGES, MAX_TOOLS};

// ============================================================================
// Re-export OpenAI types for compatibility
// ============================================================================

pub use crate::openai::{
    OpenAIMessage as OpenRouterMessage,
    OpenAIContent as OpenRouterContent,
    OpenAIContentPart as OpenRouterContentPart,
    OpenAIImageUrl as OpenRouterImageUrl,
    OpenAITool as OpenRouterTool,
    OpenAIFunctionCall as OpenRouterFunction,
    OpenAIToolCall as OpenRouterToolCall,
    OpenAIFunctionCall as OpenRouterFunctionCall,
    OpenAIToolChoice as OpenRouterToolChoice,
    // OpenAIToolChoiceFunction as OpenRouterToolChoiceFunction, // TODO: Add when OpenAIToolChoiceFunction is defined
    OpenAIResponseFormat as OpenRouterResponseFormat,
    OpenAIResponseMessage as OpenRouterResponseMessage,
    OpenAIResponseToolCall as OpenRouterResponseToolCall,
    OpenAIResponseFunction as OpenRouterResponseFunction,
    OpenAILogprobs as OpenRouterLogprobs,
    OpenAIContentLogprob as OpenRouterContentLogprob,
    OpenAITopLogprob as OpenRouterTopLogprob,
    OpenAIErrorResponse as OpenRouterErrorResponse,
    OpenAIError as OpenRouterError,
    OpenAIStreamingChunk as OpenRouterStreamingChunk,
    OpenAIStreamingChoice as OpenRouterStreamingChoice,
    OpenAIStreamingDelta as OpenRouterStreamingDelta,
    OpenAIStreamingToolCall as OpenRouterStreamingToolCall,
    OpenAIStreamingFunction as OpenRouterStreamingFunction};

// ============================================================================
// Chat Completions API (OpenAI-compatible with OpenRouter extensions)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
pub struct OpenRouterChatRequest<'a> {
    #[serde(borrow)]
    pub model: &'a str,
    pub messages: ArrayVec<OpenRouterMessage, MAX_MESSAGES>,
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
    pub tools: Option<ArrayVec<OpenRouterTool, MAX_TOOLS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<OpenRouterToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenRouterResponseFormat>,
    // OpenRouter-specific extensions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<OpenRouterProvider<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route: Option<&'a str>, // "fallback"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<ArrayVec<&'a str, 8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transforms: Option<ArrayVec<&'a str, 8>>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterProvider<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<ArrayVec<&'a str, 16>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<&'a str>, // "deny", "allow"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: ArrayVec<OpenRouterChoice, 8>,
    pub usage: OpenRouterUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    // OpenRouter-specific response fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterChoice {
    pub index: u32,
    pub message: OpenRouterResponseMessage,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<OpenRouterLogprobs>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_cost: Option<f32>}

// ============================================================================
// Models API with OpenRouter Extensions
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterModelsResponse {
    pub data: ArrayVec<OpenRouterModel, 256>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterModel {
    pub id: String,
    pub name: String,
    pub description: String,
    pub context_length: u32,
    pub architecture: OpenRouterArchitecture,
    pub pricing: OpenRouterPricing,
    pub top_provider: OpenRouterTopProvider,
    pub per_request_limits: Option<OpenRouterLimits>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterArchitecture {
    pub modality: String,
    pub tokenizer: String,
    pub instruct_type: Option<String>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterPricing {
    pub prompt: String,
    pub completion: String,
    pub image: Option<String>,
    pub request: Option<String>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterTopProvider {
    pub max_completion_tokens: Option<u32>,
    pub is_moderated: bool}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterLimits {
    pub prompt_tokens: Option<String>,
    pub completion_tokens: Option<String>}

// ============================================================================
// Generation Stats API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterGenerationRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_date: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_date: Option<&'a str>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterGenerationResponse {
    pub data: ArrayVec<OpenRouterGenerationStat, 365>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterGenerationStat {
    pub date: String,
    pub requests: u32,
    pub tokens: u64,
    pub cost: f32}

// ============================================================================
// Activity API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterActivityRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<u32>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterActivityResponse {
    pub data: ArrayVec<OpenRouterActivity, 100>,
    pub has_more: bool}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterActivity {
    pub id: String,
    pub app_id: Option<String>,
    pub cost: f32,
    pub created_at: String,
    pub model: String,
    pub origin: String,
    pub streamed: bool,
    pub total_tokens: u32,
    pub usage: OpenRouterActivityUsage}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterActivityUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32}

// ============================================================================
// Credit Balance API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterCreditResponse {
    pub data: OpenRouterCredit}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterCredit {
    pub balance: f32,
    pub usage: f32,
    pub limit: Option<f32>}

// ============================================================================
// Streaming Support with OpenRouter Extensions
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterStreamingChunkExtended {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: ArrayVec<OpenRouterStreamingChoiceExtended, 8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterStreamingChoiceExtended {
    pub index: u32,
    pub delta: OpenRouterStreamingDelta,
    pub finish_reason: Option<String>}

// ============================================================================
// Builder Patterns for Http3 Integration
// ============================================================================

impl<'a> OpenRouterChatRequest<'a> {
    #[inline(always)]
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
            provider: None,
            route: None,
            models: None,
            transforms: None}
    }

    #[inline(always)]
    pub fn add_message(mut self, role: &'a str, content: OpenRouterContent) -> Self {
        if self.messages.len() < MAX_MESSAGES {
            let _ = self.messages.try_push(OpenRouterMessage {
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
        self.add_message(role, OpenRouterContent::Text(text.to_string()))
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
    pub fn with_tools(mut self, tools: ArrayVec<OpenRouterTool, MAX_TOOLS>) -> Self {
        self.tools = Some(tools);
        self
    }

    #[inline(always)]
    pub fn tool_choice_auto(mut self) -> Self {
        self.tool_choice = Some(OpenRouterToolChoice::Auto("auto".to_string()));
        self
    }

    pub fn tool_choice_none(mut self) -> Self {
        self.tool_choice = Some(OpenRouterToolChoice::Auto("none".to_string()));
        self
    }

    pub fn response_format_json(mut self) -> Self {
        self.response_format = Some(serde_json::json!({
            "type": "json_object"
        }));
        self
    }

    // OpenRouter-specific builder methods
    pub fn with_provider_order(mut self, providers: ArrayVec<&'a str, 16>) -> Self {
        let provider = OpenRouterProvider {
            order: Some(providers),
            allow_fallbacks: None,
            require_parameters: None,
            data_collection: None};
        self.provider = Some(provider);
        self
    }

    pub fn allow_fallbacks(mut self, allow: bool) -> Self {
        let mut provider = self.provider.unwrap_or_default();
        provider.allow_fallbacks = Some(allow);
        self.provider = Some(provider);
        self
    }

    pub fn route_fallback(mut self) -> Self {
        self.route = Some("fallback");
        self
    }

    pub fn with_model_choices(mut self, models: ArrayVec<&'a str, 8>) -> Self {
        self.models = Some(models);
        self
    }

    pub fn data_collection_deny(mut self) -> Self {
        let mut provider = self.provider.unwrap_or_default();
        provider.data_collection = Some("deny");
        self.provider = Some(provider);
        self
    }

    pub fn data_collection_allow(mut self) -> Self {
        let mut provider = self.provider.unwrap_or_default();
        provider.data_collection = Some("allow");
        self.provider = Some(provider);
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

impl<'a> Default for OpenRouterProvider<'a> {
    fn default() -> Self {
        Self {
            order: None,
            allow_fallbacks: None,
            require_parameters: None,
            data_collection: None}
    }
}

impl<'a> OpenRouterGenerationRequest<'a> {
    pub fn new() -> Self {
        Self {
            start_date: None,
            end_date: None}
    }

    pub fn date_range(mut self, start: &'a str, end: &'a str) -> Self {
        self.start_date = Some(start);
        self.end_date = Some(end);
        self
    }
}

impl OpenRouterActivityRequest {
    pub fn new() -> Self {
        Self {
            limit: None,
            offset: None}
    }

    pub fn limit(mut self, limit: u32) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn offset(mut self, offset: u32) -> Self {
        self.offset = Some(offset);
        self
    }

    pub fn paginate(mut self, limit: u32, offset: u32) -> Self {
        self.limit = Some(limit);
        self.offset = Some(offset);
        self
    }
}