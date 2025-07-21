//! Google Gemini Completion Integration - Modular Implementation
//!
//! This module re-exports the new modular Gemini implementation for backward compatibility.
//! The implementation has been split into focused modules for better maintainability:
//!
//! - `gemini_error`: Comprehensive error handling
//! - `gemini_types`: Type definitions and API structures
//! - `gemini_streaming`: High-performance streaming implementation
//! - `gemini_client`: Core completion provider implementation

// Re-export all the constants from the new modular implementation
// Re-export the main client implementation
pub use super::gemini_client::{
    CompletionModel, GeminiCompletionBuilder, completion_builder, create_request_body,
};
// Re-export the error types
pub use super::gemini_error::{
    GeminiError, GeminiResult, parse_api_error_response, parse_http_status_error,
};
// Re-export the streaming functionality
pub use super::gemini_streaming::{
    GeminiStreamProcessor, StreamingConfig, StreamingMetrics, StreamingResponse,
    create_streaming_processor,
};
// Re-export the main types and structures
pub use super::gemini_types::{
    Content, ContentCandidate, FinishReason, FunctionCall, FunctionDeclaration, FunctionResponse,
    GenerateContentRequest, GenerateContentResponse, GenerationConfig, HarmCategory,
    HarmProbability, Part, Role, SafetyRating, Schema, Tool, UsageMetadata, parse_gemini_chunk,
};
pub use super::gemini_types::{
    GEMINI_1_0_PRO, GEMINI_1_5_FLASH, GEMINI_1_5_PRO, GEMINI_1_5_PRO_8B, GEMINI_2_0_FLASH,
    GEMINI_2_0_FLASH_LITE, GEMINI_2_5_FLASH_PREVIEW_04_17, GEMINI_2_5_FLASH_PREVIEW_05_20,
    GEMINI_2_5_PRO_EXP_03_25, GEMINI_2_5_PRO_PREVIEW_03_25, GEMINI_2_5_PRO_PREVIEW_05_06,
    GEMINI_2_5_PRO_PREVIEW_06_05, available_models,
};

// For maximum backward compatibility, also provide the compatibility imports
// This ensures existing code continues to work without modification
pub mod gemini_api_types {
    pub use super::super::gemini_types::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backward_compatibility_constants() {
        // Verify all model constants are available
        assert_eq!(GEMINI_1_5_PRO, "gemini-1.5-pro");
        assert_eq!(GEMINI_1_5_FLASH, "gemini-1.5-flash");
        assert_eq!(GEMINI_2_0_FLASH, "gemini-2.0-flash");
    }

    #[test]
    fn test_available_models() {
        let models = available_models();
        assert!(!models.is_empty());
        assert!(models.contains(&GEMINI_1_5_PRO));
        assert!(models.contains(&GEMINI_1_5_FLASH));
    }

    #[tokio::test]
    async fn test_completion_builder_interface() {
        // Test that the completion builder interface is preserved
        let result = completion_builder("test-key".to_string(), GEMINI_1_5_FLASH);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compatibility_gemini_api_types_import() {
        // Test that compatibility import path still works
        use gemini_api_types::GenerateContentResponse;

        // This should compile without errors, ensuring backward compatibility
        let _type_check: Option<GenerateContentResponse> = None;
    }
}
