//! AI21 Labs API request and response structures.
//! 
//! This module provides zero-allocation structures for interacting with AI21 Labs'
//! Jurassic models and Studio API. Unlike other providers, AI21 has its own
//! unique API structure and capabilities.
//! All collections use ArrayVec for bounded, stack-allocated storage.

use serde::{Deserialize, Serialize};
use arrayvec::ArrayVec;
use crate::{MAX_MESSAGES, MAX_TOOLS, MAX_DOCUMENTS};

// ============================================================================
// Completion API (Jurassic Models)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21CompletionRequest<'a> {
    #[serde(borrow)]
    pub prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub numResults: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maxTokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minTokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topP: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topKReturn: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stopSequences: Option<ArrayVec<&'a str, 16>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub countPenalty: Option<AI21Penalty>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presencePenalty: Option<AI21Penalty>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequencyPenalty: Option<AI21Penalty>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21Penalty {
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
pub struct AI21CompletionResponse {
    pub id: String,
    pub prompt: AI21PromptInfo,
    pub completions: ArrayVec<AI21Completion, 16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21PromptInfo {
    pub text: String,
    pub tokens: ArrayVec<AI21Token, 2048>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21Token {
    pub generatedToken: AI21GeneratedToken,
    pub topTokens: Option<ArrayVec<AI21TopToken, 100>>,
    pub textRange: AI21TextRange,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21GeneratedToken {
    pub token: String,
    pub logprob: f32,
    pub raw_logprob: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21TopToken {
    pub token: String,
    pub logprob: f32,
    pub raw_logprob: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21TextRange {
    pub start: u32,
    pub end: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21Completion {
    pub data: AI21CompletionData,
    pub finishReason: AI21FinishReason,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21CompletionData {
    pub text: String,
    pub tokens: ArrayVec<AI21Token, 2048>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21FinishReason {
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub length: Option<u32>,
}

// ============================================================================
// Chat API (Jamba Models)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21ChatRequest<'a> {
    #[serde(borrow)]
    pub model: &'a str,
    #[serde(borrow)]
    pub messages: ArrayVec<AI21ChatMessage<'a>, MAX_MESSAGES>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<ArrayVec<&'a str, 4>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ArrayVec<AI21Tool<'a>, MAX_TOOLS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<AI21ResponseFormat<'a>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21ChatMessage<'a> {
    #[serde(borrow)]
    pub role: &'a str,
    #[serde(borrow)]
    pub content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ArrayVec<AI21ToolCall<'a>, MAX_TOOLS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<&'a str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21Tool<'a> {
    #[serde(rename = "type")]
    #[serde(borrow)]
    pub tool_type: &'a str,
    pub function: AI21Function<'a>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21Function<'a> {
    #[serde(borrow)]
    pub name: &'a str,
    #[serde(borrow)]
    pub description: &'a str,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21ToolCall<'a> {
    #[serde(borrow)]
    pub id: &'a str,
    #[serde(rename = "type")]
    #[serde(borrow)]
    pub call_type: &'a str,
    pub function: AI21FunctionCall<'a>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21FunctionCall<'a> {
    #[serde(borrow)]
    pub name: &'a str,
    #[serde(borrow)]
    pub arguments: &'a str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21ResponseFormat<'a> {
    #[serde(rename = "type")]
    #[serde(borrow)]
    pub format_type: &'a str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: ArrayVec<AI21ChatChoice, 8>,
    pub usage: AI21Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21ChatChoice {
    pub index: u32,
    pub message: AI21ChatResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21ChatResponseMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ArrayVec<AI21ResponseToolCall, MAX_TOOLS>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21ResponseToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: AI21ResponseFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21ResponseFunction {
    pub name: String,
    pub arguments: String,
}

// ============================================================================
// Summarize API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21SummarizeRequest<'a> {
    #[serde(borrow)]
    pub source: &'a str,
    #[serde(borrow)]
    pub sourceType: &'a str, // "TEXT" or "URL"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub focus: Option<&'a str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21SummarizeResponse {
    pub id: String,
    pub summary: String,
}

// ============================================================================
// Paraphrase API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21ParaphraseRequest<'a> {
    #[serde(borrow)]
    pub text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<&'a str>, // "general", "formal", "casual"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub startIndex: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endIndex: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21ParaphraseResponse {
    pub id: String,
    pub suggestions: ArrayVec<AI21ParaphraseSuggestion, 16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21ParaphraseSuggestion {
    pub text: String,
    pub startIndex: u32,
    pub endIndex: u32,
}

// ============================================================================
// Grammar Correction API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21GrammarRequest<'a> {
    #[serde(borrow)]
    pub text: &'a str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21GrammarResponse {
    pub id: String,
    pub corrections: ArrayVec<AI21GrammarCorrection, 64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21GrammarCorrection {
    pub suggestion: String,
    pub startIndex: u32,
    pub endIndex: u32,
    pub originalText: String,
    pub correctionType: String,
}

// ============================================================================
// Text Improvements API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21ImprovementsRequest<'a> {
    #[serde(borrow)]
    pub text: &'a str,
    #[serde(borrow)]
    pub types: ArrayVec<&'a str, 8>, // "fluency", "specificity", "conciseness", "clarity"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21ImprovementsResponse {
    pub id: String,
    pub improvements: ArrayVec<AI21Improvement, 32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21Improvement {
    pub suggestion: String,
    pub startIndex: u32,
    pub endIndex: u32,
    pub originalText: String,
    pub improvementType: String,
}

// ============================================================================
// Contextual Answers API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21AnswerRequest<'a> {
    #[serde(borrow)]
    pub context: &'a str,
    #[serde(borrow)]
    pub question: &'a str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21AnswerResponse {
    pub id: String,
    pub answerInContext: bool,
    pub answer: Option<String>,
}

// ============================================================================
// Segmentation API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21SegmentationRequest<'a> {
    #[serde(borrow)]
    pub source: &'a str,
    #[serde(borrow)]
    pub sourceType: &'a str, // "TEXT" or "URL"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21SegmentationResponse {
    pub id: String,
    pub segments: ArrayVec<AI21Segment, 256>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21Segment {
    pub segmentText: String,
    pub segmentType: String,
    pub segmentHtml: Option<String>,
}

// ============================================================================
// Embed API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21EmbedRequest<'a> {
    #[serde(borrow)]
    pub texts: ArrayVec<&'a str, MAX_DOCUMENTS>,
    #[serde(rename = "type")]
    #[serde(borrow)]
    pub embed_type: &'a str, // "segment" or "query"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21EmbedResponse {
    pub id: String,
    pub results: ArrayVec<AI21EmbedResult, MAX_DOCUMENTS>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21EmbedResult {
    pub embedding: ArrayVec<f32, 768>,
}

// ============================================================================
// Library Management API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21LibraryCreateRequest<'a> {
    #[serde(borrow)]
    pub libraryName: &'a str,
    #[serde(borrow)]
    pub files: ArrayVec<AI21LibraryFile<'a>, 256>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<ArrayVec<&'a str, 32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21LibraryFile<'a> {
    #[serde(borrow)]
    pub fileName: &'a str,
    #[serde(borrow)]
    pub fileContent: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<ArrayVec<&'a str, 32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21LibraryResponse {
    pub libraryId: String,
    pub libraryName: String,
    pub creationDate: String,
    pub lastUpdated: String,
    pub publicUrl: Option<String>,
    pub labels: ArrayVec<String, 32>,
    pub fileCount: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21LibraryListResponse {
    pub libraries: ArrayVec<AI21LibraryResponse, 64>,
}

// ============================================================================
// Common Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21ErrorResponse {
    pub detail: ArrayVec<AI21ErrorDetail, 8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21ErrorDetail {
    #[serde(rename = "type")]
    pub error_type: String,
    pub msg: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loc: Option<ArrayVec<String, 4>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
}

// ============================================================================
// Streaming Support
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21StreamingChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: ArrayVec<AI21StreamingChoice, 8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21StreamingChoice {
    pub index: u32,
    pub delta: AI21StreamingDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI21StreamingDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ArrayVec<AI21ResponseToolCall, MAX_TOOLS>>,
}

// ============================================================================
// Builder Patterns for Http3 Integration
// ============================================================================

impl<'a> AI21CompletionRequest<'a> {
    pub fn new(prompt: &'a str) -> Self {
        Self {
            prompt,
            numResults: None,
            maxTokens: None,
            minTokens: None,
            temperature: None,
            topP: None,
            topKReturn: None,
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

    pub fn top_p(mut self, p: f32) -> Self {
        self.topP = Some(p);
        self
    }

    pub fn num_results(mut self, num: u32) -> Self {
        self.numResults = Some(num);
        self
    }

    pub fn stop_sequences(mut self, sequences: ArrayVec<&'a str, 16>) -> Self {
        self.stopSequences = Some(sequences);
        self
    }

    pub fn with_count_penalty(mut self, scale: f32) -> Self {
        self.countPenalty = Some(AI21Penalty {
            scale,
            applyToWhitespaces: None,
            applyToPunctuations: None,
            applyToNumbers: None,
            applyToStopwords: None,
            applyToEmojis: None,
        });
        self
    }

    pub fn with_presence_penalty(mut self, scale: f32) -> Self {
        self.presencePenalty = Some(AI21Penalty {
            scale,
            applyToWhitespaces: None,
            applyToPunctuations: None,
            applyToNumbers: None,
            applyToStopwords: None,
            applyToEmojis: None,
        });
        self
    }
}

impl<'a> AI21ChatRequest<'a> {
    pub fn new(model: &'a str) -> Self {
        Self {
            model,
            messages: ArrayVec::new(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            n: None,
            stream: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        }
    }

    pub fn add_message(mut self, role: &'a str, content: &'a str) -> Self {
        if self.messages.len() < MAX_MESSAGES {
            let _ = self.messages.try_push(AI21ChatMessage {
                role,
                content,
                tool_calls: None,
                tool_call_id: None,
            });
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

    pub fn with_tools(mut self, tools: ArrayVec<AI21Tool<'a>, MAX_TOOLS>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn response_format_json(mut self) -> Self {
        self.response_format = Some(AI21ResponseFormat {
            format_type: "json_object",
        });
        self
    }
}

impl<'a> AI21SummarizeRequest<'a> {
    pub fn new_text(text: &'a str) -> Self {
        Self {
            source: text,
            sourceType: "TEXT",
            focus: None,
        }
    }

    pub fn new_url(url: &'a str) -> Self {
        Self {
            source: url,
            sourceType: "URL",
            focus: None,
        }
    }

    pub fn focus(mut self, focus: &'a str) -> Self {
        self.focus = Some(focus);
        self
    }
}

impl<'a> AI21ParaphraseRequest<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            text,
            style: None,
            startIndex: None,
            endIndex: None,
        }
    }

    pub fn style_formal(mut self) -> Self {
        self.style = Some("formal");
        self
    }

    pub fn style_casual(mut self) -> Self {
        self.style = Some("casual");
        self
    }

    pub fn range(mut self, start: u32, end: u32) -> Self {
        self.startIndex = Some(start);
        self.endIndex = Some(end);
        self
    }
}

impl<'a> AI21EmbedRequest<'a> {
    pub fn new_segment(texts: ArrayVec<&'a str, MAX_DOCUMENTS>) -> Self {
        Self {
            texts,
            embed_type: "segment",
        }
    }

    pub fn new_query(texts: ArrayVec<&'a str, MAX_DOCUMENTS>) -> Self {
        Self {
            texts,
            embed_type: "query",
        }
    }
}

impl<'a> AI21LibraryCreateRequest<'a> {
    pub fn new(library_name: &'a str, files: ArrayVec<AI21LibraryFile<'a>, 256>) -> Self {
        Self {
            libraryName: library_name,
            files,
            labels: None,
        }
    }

    pub fn with_labels(mut self, labels: ArrayVec<&'a str, 32>) -> Self {
        self.labels = Some(labels);
        self
    }
}

impl<'a> AI21AnswerRequest<'a> {
    pub fn new(context: &'a str, question: &'a str) -> Self {
        Self { context, question }
    }
}

impl<'a> AI21SegmentationRequest<'a> {
    pub fn new_text(text: &'a str) -> Self {
        Self {
            source: text,
            sourceType: "TEXT",
        }
    }

    pub fn new_url(url: &'a str) -> Self {
        Self {
            source: url,
            sourceType: "URL",
        }
    }
}

impl<'a> AI21ImprovementsRequest<'a> {
    pub fn new(text: &'a str, types: ArrayVec<&'a str, 8>) -> Self {
        Self { text, types }
    }

    pub fn fluency_and_clarity(text: &'a str) -> Self {
        let mut types = ArrayVec::new();
        let _ = types.try_push("fluency");
        let _ = types.try_push("clarity");
        Self::new(text, types)
    }
}