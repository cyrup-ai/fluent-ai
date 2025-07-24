//! Re-exports of all Candle types - NO logic here, just imports/exports

pub mod candle_completion;
pub mod candle_chat;
pub mod candle_context;
pub mod candle_model;
pub mod candle_engine;
pub mod candle_utils;

// Re-export specific types to avoid ambiguous glob re-exports and ensure proper exports
// Chat types
pub use candle_chat::message::{CandleMessage, CandleMessageRole, SearchChatMessage as CandleSearchChatMessage};
pub use candle_chat::conversation::{CandleChat, Conversation, ConversationBuilder as CandleChatBuilder};

// Completion types  
pub use candle_completion::{
    CompletionParams, CandleCompletionRequest, CompletionRequestBuilder,
    CompletionResponse, CandleStreamingResponse, CandleStreamingChoice,
    CandleFinishReason, ToolDefinition, CandleCompletionError, 
    CandleCompletionResult, CandleExtractionError, CandleExtractionResult
};

// Additional type aliases for missing types
pub type CandleCompletionParams = CompletionParams;
pub type CandleCompletionChunk = CandleStreamingChoice;
pub type StreamingResponse = CandleStreamingResponse;
pub type FinishReason = CandleFinishReason;
pub type Stream<T> = fluent_ai_async::AsyncStream<T>;

// Context types
pub use candle_context::{Document as CandleDocument, CompletionChunk as CandleChunk};

// Model types
pub use candle_model::*;

// Additional model type aliases
pub type CandleModelInfo = candle_model::ModelInfo;
pub type CandleUsage = candle_model::Usage;
pub use candle_model::traits::Model as CandleModel;

// Engine types
pub use candle_engine::*;

// Utility types
pub use candle_utils::*;

// Generator types
pub use crate::generator::CandleTokenStream;

// Provider type alias - temporarily commented out due to import issues
// pub use candle_model::provider::Provider as CandleProvider;

// Type aliases for backward compatibility and clear naming
pub type CandleCompletionResponse<'a> = CompletionResponse<'a>;
pub type CandleToolDefinition = ToolDefinition;
pub type CompletionCoreResult<T> = CandleCompletionResult<T>;
// GenerationStats is now GenerationState - avoid recursive type alias
// pub type GenerationStats = crate::generator::GenerationState;

// Export all Candle-prefixed model traits from the canonical types module
pub use candle_model::traits::{
    CandleLoadableModel, CandleUsageTrackingModel, CandleCompletionModel, 
    CandleConfigurableModel, CandleTokenizerModel
};