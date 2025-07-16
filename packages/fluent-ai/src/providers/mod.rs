pub mod anthropic;
pub mod openai;
pub mod embedding;

// Remove ambiguous glob re-exports - use explicit re-exports below instead

// Re-export core provider types for convenience
pub use anthropic::{
    AnthropicProvider, AnthropicClient, AnthropicError, AnthropicResult,
    AnthropicCompletionRequest, AnthropicCompletionResponse,
};

pub use openai::{
    OpenAIProvider, OpenAIClient, OpenAIError, OpenAIResult,
    OpenAICompletionRequest, OpenAICompletionResponse,
};