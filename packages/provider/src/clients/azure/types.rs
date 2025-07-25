//! Azure OpenAI API request/response structures
//!
//! Azure OpenAI uses the same request/response structures as OpenAI
//! but with different authentication and endpoint patterns.
//!
//! Key differences:
//! - Uses `api-key` header instead of `Authorization: Bearer`
//! - Different base URL pattern: `https://{resource}.openai.azure.com/openai/deployments/{deployment}/`
//! - API version query parameter required: `?api-version=2024-02-15-preview`
//!
//! All request/response structures are re-exported from the OpenAI module
//! for consistency and zero-allocation performance.

// Re-export all OpenAI types for Azure OpenAI compatibility
pub use crate::openai::*;

/// Azure-specific authentication configuration
#[derive(Debug, Clone)]
pub struct AzureOpenAIConfig {
    /// Azure resource name
    pub resource: String,
    /// Deployment name
    pub deployment: String,
    /// API key for authentication
    pub api_key: String,
    /// API version (e.g., "2024-02-15-preview")
    pub api_version: String}

impl AzureOpenAIConfig {
    /// Create a new Azure OpenAI configuration
    #[inline]
    pub fn new(resource: String, deployment: String, api_key: String) -> Self {
        Self {
            resource,
            deployment,
            api_key,
            api_version: "2024-02-15-preview".to_string()}
    }

    /// Get the base URL for Azure OpenAI
    #[inline]
    pub fn base_url(&self) -> String {
        format!(
            "https://{}.openai.azure.com/openai/deployments/{}", 
            self.resource, 
            self.deployment
        )
    }

    /// Get the chat completions endpoint URL
    #[inline]
    pub fn chat_completions_url(&self) -> String {
        format!(
            "{}/chat/completions?api-version={}", 
            self.base_url(), 
            self.api_version
        )
    }

    /// Get the embeddings endpoint URL
    #[inline]
    pub fn embeddings_url(&self) -> String {
        format!(
            "{}/embeddings?api-version={}", 
            self.base_url(), 
            self.api_version
        )
    }

    /// Get the audio transcriptions endpoint URL
    #[inline]
    pub fn transcriptions_url(&self) -> String {
        format!(
            "{}/audio/transcriptions?api-version={}", 
            self.base_url(), 
            self.api_version
        )
    }

    /// Get the audio translations endpoint URL
    #[inline]
    pub fn translations_url(&self) -> String {
        format!(
            "{}/audio/translations?api-version={}", 
            self.base_url(), 
            self.api_version
        )
    }

    /// Get the moderations endpoint URL
    #[inline]
    pub fn moderations_url(&self) -> String {
        format!(
            "{}/moderations?api-version={}", 
            self.base_url(), 
            self.api_version
        )
    }
}

impl Default for AzureOpenAIConfig {
    fn default() -> Self {
        Self {
            resource: String::new(),
            deployment: String::new(),
            api_key: String::new(),
            api_version: "2024-02-15-preview".to_string()}
    }
}

/// Azure OpenAI error response (same format as OpenAI)
pub type AzureOpenAIErrorResponse = serde_json::Value;

/// Type aliases for consistency with Azure OpenAI documentation
pub type AzureChatCompletionRequest = OpenAICompletionRequest;
pub type AzureChatCompletionResponse = OpenAIStreamChunk;
pub type AzureEmbeddingRequest = OpenAIEmbeddingRequest;
pub type AzureEmbeddingResponse = OpenAIEmbeddingResponse;
pub type AzureModerationRequest = OpenAIModerationRequest;
pub type AzureModerationResponse = OpenAIModerationResponse;
pub type AzureTranscriptionRequest = OpenAITranscriptionRequest;
pub type AzureTranscriptionResponse = OpenAITranscriptionResponse;
pub type AzureTranslationRequest = OpenAITranslationRequest;
pub type AzureTranslationResponse = OpenAITranslationResponse;