//! Type re-exports and aliases for the Candle AI framework
//!
//! This module provides a centralized location for all type definitions,
//! re-exports, and aliases used throughout the Candle ecosystem. It serves
//! as the primary type interface for consumers of the library.
//!
//! # Organization
//!
//! - **Chat types**: Conversation, messages, and chat-related functionality
//! - **Completion types**: API completion requests, responses, and streaming
//! - **Context types**: Document processing and context management
//! - **Model types**: Model loading, configuration, and trait definitions
//! - **Engine types**: Core engine functionality and abstractions
//! - **Utility types**: Helper types and common utilities
//!
//! # Type Aliases
//!
//! Many types have aliases for backward compatibility and improved ergonomics.
//! Prefer the canonical names when possible, but aliases are provided for
//! migration and convenience.

pub mod candle_chat;
pub mod candle_completion;
pub mod candle_context;
pub mod candle_engine;
pub mod candle_model;
pub mod candle_utils;
pub mod extensions;

// Re-export specific types to avoid ambiguous glob re-exports and ensure proper exports

/// Chat and conversation types for interactive AI systems
///
/// These types provide the foundation for building conversational AI applications
/// with message handling, conversation state management, and search capabilities.
pub use candle_chat::conversation::{
    CandleChat, 
    Conversation, 
    ConversationBuilder as CandleChatBuilder
};

/// Message types for individual conversational exchanges
pub use candle_chat::message::{
    CandleMessage, 
    CandleMessageRole, 
    SearchChatMessage as CandleSearchChatMessage
};
/// Completion API types for text generation and AI responses
///
/// These types handle API requests, responses, streaming, and error management
/// for text completion operations across different AI providers.
pub use candle_completion::{
    CandleCompletionError, 
    CandleCompletionRequest, 
    CandleCompletionResult, 
    CandleExtractionError,
    CandleExtractionResult, 
    CandleFinishReason, 
    CandleStreamingChoice, 
    CandleStreamingDelta,
    CandleStreamingResponse, 
    CompletionParams, 
    CompletionRequestBuilder, 
    CompletionResponse,
    ToolDefinition
};

/// Type aliases for improved ergonomics and backward compatibility

/// Alias for CompletionParams with Candle prefix for clarity
pub type CandleCompletionParams = CompletionParams;

/// Alias for individual streaming completion chunks
pub type CandleCompletionChunk = CandleStreamingChoice;

/// Shorter alias for streaming response type
pub type StreamingResponse = CandleStreamingResponse;

/// Shorter alias for completion finish reason enumeration
pub type FinishReason = CandleFinishReason;

/// Generic async stream type used throughout the Candle ecosystem
/// 
/// This is the primary streaming primitive for all async operations,
/// providing zero-allocation, lock-free streaming with backpressure.
pub type Stream<T> = fluent_ai_async::AsyncStream<T>;

/// Context and document processing types
///
/// These types handle document processing, context extraction, and
/// content management for retrieval-augmented generation (RAG) systems.
pub use candle_context::{
    CompletionChunk as CandleChunk, 
    Document as CandleDocument
};

/// Model loading, configuration, and execution types
///
/// Comprehensive re-export of all model-related functionality including
/// loading, configuration, traits, and runtime management.
pub use candle_model::*;

/// Additional model type aliases for consistency

/// Model metadata and information container
pub type CandleModelInfo = candle_model::ModelInfo;

/// Token and resource usage tracking information
pub type CandleUsage = candle_model::Usage;

/// Core engine types and abstractions
///
/// Engine components provide the runtime foundation for model execution,
/// resource management, and system coordination.
pub use candle_engine::*;

/// Primary model trait for inference operations
pub use candle_model::traits::Model as CandleModel;

/// Utility types and helper functions
///
/// Common utilities, helpers, and convenience types used across
/// the Candle ecosystem.
pub use candle_utils::*;

// Generator types
// CandleTokenStream removed - converted to AsyncStream pattern

// Provider type alias - temporarily commented out due to import issues
// pub use candle_model::provider::Provider as CandleProvider;

/// Type aliases for backward compatibility and clear naming

/// Lifetime-parameterized completion response for zero-copy operations
pub type CandleCompletionResponse<'a> = CompletionResponse<'a>;

/// Tool definition alias with Candle prefix for consistency
pub type CandleToolDefinition = ToolDefinition;

/// Core result type alias for completion operations
pub type CompletionCoreResult<T> = CandleCompletionResult<T>;
// GenerationStats is now GenerationState - avoid recursive type alias
// pub type GenerationStats = crate::generator::GenerationState;

/// Model trait re-exports with Candle prefixes
///
/// These traits define the interface contracts for different model capabilities
/// and provide a consistent API across different model implementations.
pub use candle_model::traits::{
    CandleCompletionModel, 
    CandleConfigurableModel, 
    CandleLoadableModel, 
    CandleTokenizerModel,
    CandleUsageTrackingModel
};
