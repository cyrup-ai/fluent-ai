/// Glue-crate convenience re-exports for the "Better RIG" fluent API.
/// 
/// Import everything you need to get started:
/// ```rust
/// use cyrup_agent::prelude::*;
/// 
/// // Semantic streaming completion - closure gets clean unwrapped text chunks
/// CompletionProviderFactory::openai()
///     .model("gpt-4")
///     .system_prompt("You are a helpful AI assistant")
///     .context(docs)
///     .tool_type::<Calc>()
///     .temperature(1.0)
///     .completion("Hello! How's the framework?", |text_chunk| {
///         print!("{}", text_chunk); // Pre-unwrapped String, no Result handling!
///     }).await?;
/// 
/// // Semantic streaming embeddings - closure gets clean (doc, embedding) pairs  
/// CompletionProviderFactory::openai()
///     .model("text-embedding-3-small")
///     .embeddings(documents, |doc, embedding| {
///         database.store(doc, embedding); // Pre-unwrapped! No error handling needed
///     }).await?;
/// 
/// // Semantic streaming transcription - closure gets clean text fragments
/// CompletionProviderFactory::openai()
///     .model("whisper-1")
///     .transcription(audio_data, |text_fragment| {
///         ui.update_transcript(text_fragment); // Pre-unwrapped String!
///     }).await?;
/// ```

// Core runtime exports
pub use crate::runtime::{spawn_async, AsyncTask, AsyncStream, ThreadPool};

// Provider builder - the main entry point
pub use crate::provider_builder::{
    CompletionProvider, 
    ProviderBuilder, 
    CompletionReadyBuilder,
    // Example tools
    Calc, 
    WebSearch, 
    FileSystem,
    // Provider markers
    OpenAiProvider,
    AnthropicProvider,
    OllamaProvider,
    GroqProvider,
    DeepSeekProvider,
    OpenRouterProvider,
};

// Agent builder for advanced use cases
pub use crate::agent::{
    Agent, 
    AgentBuilder, 
    CompletionBuilder as AgentCompletionBuilder,
    MissingSys, 
    MissingCtx, 
    Ready,
};

// Core types
pub use crate::completion::{
    CompletionModel,
    CompletionError,
    CompletionRequest,
    CompletionRequestBuilder,
    CompletionResponse,
    Document,
    ToolDefinition,
};

// Message types
pub use crate::message::{Message, Text, AssistantContent, ToolCall, ToolFunction};

// Tool system
pub use crate::tool::{Tool, ToolSet};

// Streaming
pub use crate::streaming::StreamingCompletionResponse;

// Vector store
pub use crate::vector_store::VectorStoreIndexDyn;
