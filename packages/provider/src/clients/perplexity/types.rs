//! Perplexity API request and response structures.
//! 
//! This module provides zero-allocation structures for interacting with Perplexity's
//! online search-capable AI models. Perplexity is largely OpenAI-compatible with
//! search and citation extensions.
//! All collections use ArrayVec for bounded, stack-allocated storage.

use serde::{Deserialize, Serialize};
use arrayvec::ArrayVec;
use crate::{MAX_MESSAGES};

// ============================================================================
// Re-export OpenAI types for compatibility
// ============================================================================

pub use crate::openai::{
    OpenAIMessage as PerplexityMessage,
    OpenAIContent as PerplexityContent,
    OpenAIContentPart as PerplexityContentPart,
    OpenAIImageUrl as PerplexityImageUrl,
    OpenAITool as PerplexityTool,
    OpenAIFunction as PerplexityFunction,
    OpenAIToolCall as PerplexityToolCall,
    OpenAIFunctionCall as PerplexityFunctionCall,
    OpenAIToolChoice as PerplexityToolChoice,
    // OpenAIToolChoiceFunction as PerplexityToolChoiceFunction, // TODO: Add when OpenAIToolChoiceFunction is defined
    OpenAIResponseFormat as PerplexityResponseFormat,
    OpenAIResponseMessage as PerplexityResponseMessage,
    OpenAIResponseToolCall as PerplexityResponseToolCall,
    OpenAIResponseFunction as PerplexityResponseFunction,
    OpenAILogprobs as PerplexityLogprobs,
    OpenAIContentLogprob as PerplexityContentLogprob,
    OpenAITopLogprob as PerplexityTopLogprob,
    OpenAIErrorResponse as PerplexityErrorResponse,
    OpenAIError as PerplexityError,
    OpenAIStreamingChunk as PerplexityStreamingChunk,
    OpenAIStreamingChoice as PerplexityStreamingChoice,
    OpenAIStreamingDelta as PerplexityStreamingDelta,
    OpenAIStreamingToolCall as PerplexityStreamingToolCall,
    OpenAIStreamingFunction as PerplexityStreamingFunction};

// ============================================================================
// Chat Completions API (OpenAI-compatible with Perplexity extensions)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerplexityChatRequest<'a> {
    #[serde(borrow)]
    pub model: &'a str,
    pub messages: ArrayVec<PerplexityMessage, MAX_MESSAGES>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    // Perplexity-specific extensions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_citations: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_images: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_related_questions: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_domain_filter: Option<ArrayVec<&'a str, 16>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_recency_filter: Option<&'a str>, // "month", "week", "day", "hour"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerplexityChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: ArrayVec<PerplexityChoice, 8>,
    pub usage: PerplexityUsage,
    // Perplexity-specific response fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<ArrayVec<PerplexityCitation, 64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<ArrayVec<PerplexityImage, 16>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub related_questions: Option<ArrayVec<String, 8>>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerplexityChoice {
    pub index: u32,
    pub message: PerplexityResponseMessage,
    pub finish_reason: String,
    pub delta: Option<PerplexityStreamingDelta>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerplexityCitation {
    pub number: u32,
    pub url: String,
    pub title: String,
    pub snippet: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub published_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<String>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerplexityImage {
    pub url: String,
    pub alt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerplexityUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32}

// ============================================================================
// Models API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerplexityModelsResponse {
    pub object: String,
    pub data: ArrayVec<PerplexityModel, 32>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerplexityModel {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    pub display_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing: Option<PerplexityPricing>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerplexityPricing {
    pub prompt: String,
    pub completion: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<String>}

// ============================================================================
// Streaming Support with Perplexity Extensions
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerplexityStreamingChunkExtended {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: ArrayVec<PerplexityStreamingChoiceExtended, 8>,
    // Perplexity streaming extensions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<ArrayVec<PerplexityCitation, 64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub related_questions: Option<ArrayVec<String, 8>>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerplexityStreamingChoiceExtended {
    pub index: u32,
    pub delta: PerplexityStreamingDeltaExtended,
    pub finish_reason: Option<String>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerplexityStreamingDeltaExtended {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    // Perplexity streaming delta extensions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<ArrayVec<PerplexityCitation, 64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<ArrayVec<PerplexityImage, 16>>}

// ============================================================================
// Builder Patterns for Http3 Integration
// ============================================================================

impl<'a> PerplexityChatRequest<'a> {
    #[inline(always)]
    pub fn new(model: &'a str) -> Self {
        Self {
            model,
            messages: ArrayVec::new(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            presence_penalty: None,
            frequency_penalty: None,
            return_citations: None,
            return_images: None,
            return_related_questions: None,
            search_domain_filter: None,
            search_recency_filter: None}
    }

    #[inline(always)]
    pub fn add_message(mut self, role: &'a str, content: PerplexityContent) -> Self {
        if self.messages.len() < MAX_MESSAGES {
            let _ = self.messages.try_push(PerplexityMessage {
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
        self.add_message(role, PerplexityContent::Text(text.to_string()))
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
    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    #[inline(always)]
    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    // Perplexity-specific builder methods
    #[inline(always)]
    pub fn return_citations(mut self, citations: bool) -> Self {
        self.return_citations = Some(citations);
        self
    }

    #[inline(always)]
    pub fn return_images(mut self, images: bool) -> Self {
        self.return_images = Some(images);
        self
    }

    #[inline(always)]
    pub fn return_related_questions(mut self, questions: bool) -> Self {
        self.return_related_questions = Some(questions);
        self
    }

    #[inline(always)]
    pub fn search_domain_filter(mut self, domains: ArrayVec<&'a str, 16>) -> Self {
        self.search_domain_filter = Some(domains);
        self
    }

    #[inline(always)]
    pub fn search_recency_filter(mut self, recency: &'a str) -> Self {
        self.search_recency_filter = Some(recency);
        self
    }

    pub fn enable_search_features(mut self) -> Self {
        self.return_citations = Some(true);
        self.return_images = Some(true);
        self.return_related_questions = Some(true);
        self
    }
}