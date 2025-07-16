pub mod anthropic;
pub mod azure;
pub mod candle;
pub mod deepseek;
pub mod gemini;
pub mod groq;
pub mod huggingface;
pub mod mistral;
pub mod ollama;
pub mod openai;
pub mod openrouter;
pub mod perplexity;
pub mod together;
pub mod xai;

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