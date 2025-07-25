//! Cohere API request and response structures.
//! 
//! This module provides zero-allocation structures for interacting with Cohere's
//! API including chat, generate, embed, rerank, classify, and detect language.
//! All collections use ArrayVec for bounded, stack-allocated storage.

use serde::{Deserialize, Serialize};
use arrayvec::ArrayVec;
use crate::{MAX_MESSAGES, MAX_TOOLS, MAX_DOCUMENTS};

// ============================================================================
// Chat API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereChatRequest<'a> {
    #[serde(borrow)]
    pub message: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preamble: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_history: Option<ArrayVec<CohereChatMessage<'a>, MAX_MESSAGES>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_truncation: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connectors: Option<ArrayVec<CohereConnector<'a>, 8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_queries_only: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub documents: Option<ArrayVec<CohereDocument<'a>, MAX_DOCUMENTS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citation_quality: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<ArrayVec<&'a str, 4>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ArrayVec<CohereTool<'a>, MAX_TOOLS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_results: Option<ArrayVec<CohereToolResult<'a>, MAX_TOOLS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub force_single_step: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereChatMessage<'a> {
    #[serde(borrow)]
    pub role: &'a str,
    #[serde(borrow)]
    pub message: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ArrayVec<CohereToolCall<'a>, MAX_TOOLS>>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereConnector<'a> {
    #[serde(borrow)]
    pub id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_access_token: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub continue_on_failure: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<serde_json::Value>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereDocument<'a> {
    #[serde(borrow)]
    pub title: &'a str,
    #[serde(borrow)]
    pub snippet: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<&'a str>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereTool<'a> {
    #[serde(borrow)]
    pub name: &'a str,
    #[serde(borrow)]
    pub description: &'a str,
    pub parameter_definitions: serde_json::Value}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereToolCall<'a> {
    #[serde(borrow)]
    pub name: &'a str,
    pub parameters: serde_json::Value}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereToolResult<'a> {
    #[serde(borrow)]
    pub call: CohereToolCall<'a>,
    pub outputs: ArrayVec<serde_json::Value, 8>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereChatResponse {
    pub text: String,
    pub generation_id: String,
    pub conversation_id: String,
    pub prompt: String,
    pub chatlog: ArrayVec<CohereChatlogEntry, MAX_MESSAGES>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preamble: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_search_required: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_queries: Option<ArrayVec<CohereSearchQuery, 8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_results: Option<ArrayVec<CohereSearchResult, 32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub documents: Option<ArrayVec<CohereDocumentReference, MAX_DOCUMENTS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<ArrayVec<CohereCitation, 64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ArrayVec<CohereResponseToolCall, MAX_TOOLS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    pub meta: CohereMeta}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereChatlogEntry {
    pub role: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_name: Option<String>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereSearchQuery {
    pub text: String,
    pub generation_id: String}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereSearchResult {
    pub search_query: CohereSearchQuery,
    pub connector: CohereSearchConnector,
    pub document_ids: ArrayVec<String, 16>,
    pub error_message: Option<String>,
    pub continue_on_failure: bool}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereSearchConnector {
    pub id: String}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereDocumentReference {
    pub id: String,
    pub title: String,
    pub snippet: String,
    pub url: String}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereCitation {
    pub start: u32,
    pub end: u32,
    pub text: String,
    pub document_ids: ArrayVec<String, 8>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereResponseToolCall {
    pub name: String,
    pub parameters: serde_json::Value}

// ============================================================================
// Generate API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereGenerateRequest<'a> {
    #[serde(borrow)]
    pub prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_generations: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preset: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_sequences: Option<ArrayVec<&'a str, 4>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<ArrayVec<&'a str, 4>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_likelihoods: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<serde_json::Value>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereGenerateResponse {
    pub id: String,
    pub generations: ArrayVec<CohereGeneration, 8>,
    pub prompt: String,
    pub meta: CohereMeta}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereGeneration {
    pub id: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub likelihood: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_likelihoods: Option<ArrayVec<CohereTokenLikelihood, 2048>>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereTokenLikelihood {
    pub token: String,
    pub likelihood: f32}

// ============================================================================
// Embed API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereEmbedRequest<'a> {
    #[serde(borrow)]
    pub texts: ArrayVec<&'a str, MAX_DOCUMENTS>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_types: Option<ArrayVec<&'a str, 4>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<&'a str>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereEmbedResponse {
    pub id: String,
    pub embeddings: ArrayVec<ArrayVec<f32, 1024>, MAX_DOCUMENTS>,
    pub texts: ArrayVec<String, MAX_DOCUMENTS>,
    pub meta: CohereMeta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_type: Option<String>}

// ============================================================================
// Rerank API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereRerankRequest<'a> {
    #[serde(borrow)]
    pub query: &'a str,
    #[serde(borrow)]
    pub documents: ArrayVec<CohereRerankDocument<'a>, MAX_DOCUMENTS>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_documents: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_chunks_per_doc: Option<u32>}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CohereRerankDocument<'a> {
    Text(&'a str),
    Object {
        #[serde(borrow)]
        text: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<&'a str>}}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereRerankResponse {
    pub id: String,
    pub results: ArrayVec<CohereRerankResult, MAX_DOCUMENTS>,
    pub meta: CohereMeta}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereRerankResult {
    pub index: u32,
    pub relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<CohereRerankResultDocument>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereRerankResultDocument {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>}

// ============================================================================
// Classify API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereClassifyRequest<'a> {
    #[serde(borrow)]
    pub inputs: ArrayVec<&'a str, 96>,
    #[serde(borrow)]
    pub examples: ArrayVec<CohereClassifyExample<'a>, 96>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preset: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<&'a str>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereClassifyExample<'a> {
    #[serde(borrow)]
    pub text: &'a str,
    #[serde(borrow)]
    pub label: &'a str}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereClassifyResponse {
    pub id: String,
    pub classifications: ArrayVec<CohereClassification, 96>,
    pub meta: CohereMeta}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereClassification {
    pub id: String,
    pub input: String,
    pub prediction: String,
    pub confidence: f32,
    pub confidences: ArrayVec<CohereClassificationConfidence, 32>,
    pub labels: serde_json::Value,
    pub classification_type: String}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereClassificationConfidence {
    pub option: String,
    pub confidence: f32}

// ============================================================================
// Detect Language API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereDetectLanguageRequest<'a> {
    #[serde(borrow)]
    pub texts: ArrayVec<&'a str, 96>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereDetectLanguageResponse {
    pub id: String,
    pub results: ArrayVec<CohereLanguageDetection, 96>,
    pub meta: CohereMeta}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereLanguageDetection {
    pub language_code: String,
    pub language_name: String}

// ============================================================================
// Summarize API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereSummarizeRequest<'a> {
    #[serde(borrow)]
    pub text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub length: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extractiveness: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub additional_command: Option<&'a str>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereSummarizeResponse {
    pub id: String,
    pub summary: String,
    pub meta: CohereMeta}

// ============================================================================
// Common Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub billed_units: Option<CohereBilledUnits>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<CohereTokens>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<ArrayVec<String, 8>>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereBilledUnits {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_units: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classifications: Option<u32>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereTokens {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereErrorResponse {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>}

// ============================================================================
// Streaming Support
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type")]
pub enum CohereStreamingEvent {
    #[serde(rename = "stream-start")]
    StreamStart {
        is_finished: bool,
        generation_id: String},
    #[serde(rename = "text-generation")]
    TextGeneration {
        text: String,
        is_finished: bool},
    #[serde(rename = "citation-generation")]
    CitationGeneration {
        citations: ArrayVec<CohereCitation, 64>},
    #[serde(rename = "search-queries-generation")]
    SearchQueriesGeneration {
        search_queries: ArrayVec<CohereSearchQuery, 8>},
    #[serde(rename = "search-results")]
    SearchResults {
        search_results: ArrayVec<CohereSearchResult, 32>},
    #[serde(rename = "tool-calls-generation")]
    ToolCallsGeneration {
        tool_calls: ArrayVec<CohereResponseToolCall, MAX_TOOLS>},
    #[serde(rename = "stream-end")]
    StreamEnd {
        is_finished: bool,
        finish_reason: String,
        response: CohereChatResponse}}

// ============================================================================
// Builder Patterns for Http3 Integration
// ============================================================================

impl<'a> CohereChatRequest<'a> {
    pub fn new(message: &'a str) -> Self {
        Self {
            message,
            model: None,
            preamble: None,
            chat_history: None,
            conversation_id: None,
            prompt_truncation: None,
            connectors: None,
            search_queries_only: None,
            documents: None,
            citation_quality: None,
            temperature: None,
            max_tokens: None,
            max_input_tokens: None,
            k: None,
            p: None,
            seed: None,
            stop_sequences: None,
            frequency_penalty: None,
            presence_penalty: None,
            tools: None,
            tool_results: None,
            force_single_step: None,
            stream: None}
    }

    pub fn model(mut self, model: &'a str) -> Self {
        self.model = Some(model);
        self
    }

    pub fn preamble(mut self, preamble: &'a str) -> Self {
        self.preamble = Some(preamble);
        self
    }

    pub fn with_chat_history(mut self, history: ArrayVec<CohereChatMessage<'a>, MAX_MESSAGES>) -> Self {
        self.chat_history = Some(history);
        self
    }

    pub fn with_documents(mut self, documents: ArrayVec<CohereDocument<'a>, MAX_DOCUMENTS>) -> Self {
        self.documents = Some(documents);
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
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

    pub fn with_tools(mut self, tools: ArrayVec<CohereTool<'a>, MAX_TOOLS>) -> Self {
        self.tools = Some(tools);
        self
    }
}

impl<'a> CohereGenerateRequest<'a> {
    pub fn new(prompt: &'a str) -> Self {
        Self {
            prompt,
            model: None,
            num_generations: None,
            stream: None,
            max_tokens: None,
            truncate: None,
            temperature: None,
            seed: None,
            preset: None,
            end_sequences: None,
            stop_sequences: None,
            k: None,
            p: None,
            frequency_penalty: None,
            presence_penalty: None,
            return_likelihoods: None,
            logit_bias: None}
    }

    pub fn model(mut self, model: &'a str) -> Self {
        self.model = Some(model);
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

    pub fn num_generations(mut self, num: u32) -> Self {
        self.num_generations = Some(num);
        self
    }
}

impl<'a> CohereEmbedRequest<'a> {
    pub fn new(texts: ArrayVec<&'a str, MAX_DOCUMENTS>) -> Self {
        Self {
            texts,
            model: None,
            input_type: None,
            embedding_types: None,
            truncate: None}
    }

    pub fn model(mut self, model: &'a str) -> Self {
        self.model = Some(model);
        self
    }

    pub fn input_type(mut self, input_type: &'a str) -> Self {
        self.input_type = Some(input_type);
        self
    }

    pub fn embedding_types(mut self, types: ArrayVec<&'a str, 4>) -> Self {
        self.embedding_types = Some(types);
        self
    }
}

impl<'a> CohereRerankRequest<'a> {
    pub fn new(query: &'a str, documents: ArrayVec<CohereRerankDocument<'a>, MAX_DOCUMENTS>) -> Self {
        Self {
            query,
            documents,
            model: None,
            top_n: None,
            return_documents: None,
            max_chunks_per_doc: None}
    }

    pub fn model(mut self, model: &'a str) -> Self {
        self.model = Some(model);
        self
    }

    pub fn top_n(mut self, n: u32) -> Self {
        self.top_n = Some(n);
        self
    }

    pub fn return_documents(mut self, return_docs: bool) -> Self {
        self.return_documents = Some(return_docs);
        self
    }
}

impl<'a> CohereClassifyRequest<'a> {
    pub fn new(inputs: ArrayVec<&'a str, 96>, examples: ArrayVec<CohereClassifyExample<'a>, 96>) -> Self {
        Self {
            inputs,
            examples,
            model: None,
            preset: None,
            truncate: None}
    }

    pub fn model(mut self, model: &'a str) -> Self {
        self.model = Some(model);
        self
    }

    pub fn preset(mut self, preset: &'a str) -> Self {
        self.preset = Some(preset);
        self
    }
}

impl<'a> CohereSummarizeRequest<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            text,
            model: None,
            length: None,
            format: None,
            extractiveness: None,
            temperature: None,
            additional_command: None}
    }

    pub fn model(mut self, model: &'a str) -> Self {
        self.model = Some(model);
        self
    }

    pub fn length(mut self, length: &'a str) -> Self {
        self.length = Some(length);
        self
    }

    pub fn format(mut self, format: &'a str) -> Self {
        self.format = Some(format);
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }
}