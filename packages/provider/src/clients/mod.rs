pub mod anthropic;
pub mod candle;
pub mod huggingface;
pub mod mistral;
pub mod openai;
pub mod together;
pub mod xai;

// Remove ambiguous glob re-exports - use explicit re-exports below instead

// Re-export core provider types for convenience
pub use anthropic::{
    AnthropicClient, AnthropicCompletionRequest, AnthropicError, AnthropicProvider, AnthropicResult};
// Huggingface client
pub use huggingface::{
    Client as HuggingfaceClient, HuggingfaceCompletionBuilder, SubProvider};
// Mistral client
pub use mistral::{
    Client as MistralClient, MistralCompletionBuilder,
    NewMistralCompletionBuilder, mistral_completion_builder};

pub use openai::{
    OpenAIClient, OpenAICompletionRequest, OpenAICompletionResponse, OpenAIError, OpenAIProvider,
    OpenAIResult};
// OpenRouter client removed - not supported by model-info

// Together client
pub use together::{Client as TogetherClient, TogetherCompletionBuilder};
// xAI client
pub use xai::{Client as XAIClient, XAICompletionBuilder};
