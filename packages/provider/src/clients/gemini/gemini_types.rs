//! Gemini model constants and utility functions
//!
//! This module contains Gemini-specific model constants and conversion utilities.
//! All HTTP request/response types are now centralized in fluent_ai_http_structs::google.

use fluent_ai_domain::chunk::{CompletionChunk, FinishReason, Usage};
// Import centralized HTTP types - no more local definitions!
use fluent_ai_http_structs::google::{
    GeminiCandidate, GeminiFunctionCall, GeminiGenerateContentRequest,
    GeminiGenerateContentResponse, GeminiPart, GeminiStreamGenerateContentResponse,
    GeminiUsageMetadata,
};

use super::gemini_error::{GeminiError, GeminiResult};
use crate::{
    OneOrMany,
    completion::{self, CompletionError},
    message,
};

// =================================================================
// Gemini Model Constants (Compile-time optimized)
// =================================================================

/// `gemini-2.5-pro-preview-06-05` completion model
pub const GEMINI_2_5_PRO_PREVIEW_06_05: &str = "gemini-2.5-pro-preview-06-05";
/// `gemini-2.5-pro-preview-05-06` completion model
pub const GEMINI_2_5_PRO_PREVIEW_05_06: &str = "gemini-2.5-pro-preview-05-06";
/// `gemini-2.5-pro-preview-03-25` completion model
pub const GEMINI_2_5_PRO_PREVIEW_03_25: &str = "gemini-2.5-pro-preview-03-25";
/// `gemini-2.5-flash-preview-05-20` completion model
pub const GEMINI_2_5_FLASH_PREVIEW_05_20: &str = "gemini-2.5-flash-preview-05-20";
/// `gemini-2.5-flash-preview-04-17` completion model
pub const GEMINI_2_5_FLASH_PREVIEW_04_17: &str = "gemini-2.5-flash-preview-04-17";
/// `gemini-2.5-pro-exp-03-25` experimental completion model
pub const GEMINI_2_5_PRO_EXP_03_25: &str = "gemini-2.5-pro-exp-03-25";
/// `gemini-2.0-flash-lite` completion model
pub const GEMINI_2_0_FLASH_LITE: &str = "gemini-2.0-flash-lite";
/// `gemini-2.0-flash` completion model
pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";
/// `gemini-1.5-flash` completion model
pub const GEMINI_1_5_FLASH: &str = "gemini-1.5-flash";
/// `gemini-1.5-pro` completion model
pub const GEMINI_1_5_PRO: &str = "gemini-1.5-pro";
/// `gemini-1.5-pro-8b` completion model
pub const GEMINI_1_5_PRO_8B: &str = "gemini-1.5-pro-8b";
/// `gemini-1.0-pro` completion model
pub const GEMINI_1_0_PRO: &str = "gemini-1.0-pro";

/// Get available Gemini models (compile-time constant)
#[inline(always)]
pub const fn available_models() -> &'static [&'static str] {
    &[
        GEMINI_2_5_PRO_PREVIEW_06_05,
        GEMINI_2_5_FLASH_PREVIEW_05_20,
        GEMINI_2_0_FLASH,
        GEMINI_1_5_PRO,
        GEMINI_1_5_FLASH,
        GEMINI_1_5_PRO_8B,
        GEMINI_1_0_PRO,
    ]
}

// =================================================================
// Legacy type aliases for backward compatibility
// =================================================================

/// Legacy alias for centralized Gemini response type
pub type GenerateContentResponse = GeminiGenerateContentResponse;

/// Legacy alias for centralized Gemini candidate type
pub type ContentCandidate = GeminiCandidate;

/// Legacy alias for centralized content type (using the Google module types)
pub type Content = fluent_ai_http_structs::google::GeminiContent;

// Legacy conversion implementations removed - use centralized types instead

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Model,
}

/// Legacy alias for centralized part type
pub type Part = fluent_ai_http_structs::google::GeminiPart;

// Legacy conversion implementations removed - use centralized types

// Legacy type aliases for centralized Gemini types
pub type Blob = fluent_ai_http_structs::google::GeminiInlineData;
pub type FunctionCall = fluent_ai_http_structs::google::GeminiFunctionCall;
pub type FunctionResponse = fluent_ai_http_structs::google::GeminiFunctionResponse;
pub type FileData = fluent_ai_http_structs::google::GeminiFileData;
pub type ExecutableCode = fluent_ai_http_structs::google::GeminiExecutableCode;
pub type CodeExecutionResult = fluent_ai_http_structs::google::GeminiCodeExecutionResult;

// Legacy type aliases for centralized safety and usage types
pub type SafetyRating = fluent_ai_http_structs::google::GeminiSafetyRating;
pub type UsageMetadata = fluent_ai_http_structs::google::GeminiUsageMetadata;

// Implementation moved to centralized types

// Legacy type aliases for centralized Gemini types
pub type PromptFeedback = fluent_ai_http_structs::google::GeminiPromptFeedback;
pub type CitationMetadata = fluent_ai_http_structs::google::GeminiCitationMetadata;
pub type CitationSource = fluent_ai_http_structs::google::GeminiCitationSource;

// Legacy type alias for centralized generation config
pub type GenerationConfig = fluent_ai_http_structs::google::GeminiGenerationConfig;

// Schema type is defined as JSON Value in centralized types
pub type Schema = serde_json::Value;

// Schema conversion implementations removed - use centralized types

// Legacy type aliases for centralized request types
pub type GenerateContentRequest = fluent_ai_http_structs::google::GeminiGenerateContentRequest;
pub type Tool = fluent_ai_http_structs::google::GeminiTool;
pub type FunctionDeclaration = fluent_ai_http_structs::google::GeminiFunctionDeclaration;
pub type ToolConfig = fluent_ai_http_structs::google::GeminiToolConfig;
pub type CodeExecution = fluent_ai_http_structs::google::GeminiCodeExecution;
pub type SafetySetting = fluent_ai_http_structs::google::GeminiSafetySetting;

// =================================================================
// Conversion Implementations
// =================================================================

// Legacy conversion implementations removed - use centralized types

/// Convert Gemini finish reason to domain finish reason
impl From<GeminiFinishReason> for fluent_ai_domain::chunk::FinishReason {
    fn from(reason: GeminiFinishReason) -> Self {
        match reason {
            GeminiFinishReason::Stop => Self::Stop,
            GeminiFinishReason::MaxTokens => Self::Length,
            GeminiFinishReason::Safety
            | GeminiFinishReason::Blocklist
            | GeminiFinishReason::ProhibitedContent => Self::ContentFilter,
            _ => Self::Stop,
        }
    }
}

/// Convert usage metadata to domain usage
impl From<UsageMetadata> for Usage {
    fn from(usage: UsageMetadata) -> Self {
        Self {
            prompt_tokens: usage.prompt_token_count as u32,
            completion_tokens: usage.candidates_token_count as u32,
            total_tokens: usage.total_token_count as u32,
        }
    }
}

/// Parse Gemini SSE chunk to completion chunk (zero-copy where possible)
pub fn parse_gemini_chunk(data: &[u8]) -> GeminiResult<CompletionChunk> {
    // Fast JSON parsing from bytes using serde_json
    let response: GenerateContentResponse = serde_json::from_slice(data)
        .map_err(|e| GeminiError::parse_error(format!("Invalid JSON in SSE chunk: {}", e)))?;

    let candidate = response
        .candidates
        .first()
        .ok_or_else(|| GeminiError::invalid_response("No candidates in chunk"))?;

    let mut text_content = String::new();
    let mut has_tool_calls = false;

    for part in &candidate.content.parts {
        match part {
            Part::Text(text) => {
                text_content.push_str(text);
            }
            Part::FunctionCall(_) => {
                has_tool_calls = true;
            }
            _ => {}
        }
    }

    // Handle finish reason
    if let Some(ref finish_reason) = candidate.finish_reason {
        let reason: fluent_ai_domain::chunk::FinishReason = finish_reason.clone().into();

        let usage_info = response.usage_metadata.map(|u| u.into());

        return Ok(CompletionChunk::Complete {
            text: text_content,
            finish_reason: Some(reason),
            usage: usage_info,
        });
    }

    if has_tool_calls {
        Ok(CompletionChunk::tool_partial("", "", &text_content))
    } else {
        Ok(CompletionChunk::text(&text_content))
    }
}
