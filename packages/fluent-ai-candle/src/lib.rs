//! Fluent AI Candle Library
//!
//! This crate provides Candle ML framework integration for AI services.
//! All Candle-prefixed domain types, builders, and providers are defined here
//! to ensure complete independence from the main fluent-ai packages.

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![forbid(unsafe_code)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

// Candle-specific modules
/// Candle agent abstractions and management  
// pub mod agent; // Temporarily disabled for testing core functionality
/// Candle builders for zero-allocation construction patterns
pub mod builders;  
/// Candle chat functionality and conversation management
pub mod chat; // Chat functionality and conversation management
/// Candle concurrency primitives and async utilities
pub mod concurrency;
/// Candle context processing and document handling
pub mod context;
/// Candle core domain types and traits
pub mod core;
/// Candle domain types (replaces fluent_ai_domain dependency)
pub mod domain;
/// Candle embedding generation and similarity search
pub mod embedding;
/// Candle engine abstractions for AI services
pub mod engine;
/// Candle HTTP client and request/response types
pub mod http;
/// Candle image processing and generation
pub mod image;
/// Candle initialization and startup utilities
pub mod init;
/// Candle memory management and cognitive processing
pub mod memory;
/// Candle model configuration and management
pub mod model;
/// Candle prompt templates and processing
pub mod prompt;
/// Candle model providers for local inference
pub mod providers;
/// Candle tool calling and function execution
pub mod tool;
/// Candle utility functions and helpers
pub mod util;
/// Candle voice processing and transcription
pub mod voice;
/// Candle workflow orchestration and task management
pub mod workflow;

// Essential Candle re-exports for public API
pub use domain::{
    // Main Candle domain types
    CandleMessage, CandleMessageRole, CandleMessageChunk,
    CandleZeroOneOrMany, CandleAgent,
    
    // Core Candle types from context
    CandleDocument, CandleContext,
    
    // Candle Chat system
    CandleChatConfig, CandleCommandExecutor, CandleCommandRegistry, 
    CandleConversationImpl, CandleImmutableChatCommand,
    CandlePersonalityConfig,
    
    // Candle Completion system  
    CandleCompletionModel,
    
    // Candle Model system
    CandleModelInfo, CandleModelCapabilities, CandleModelPerformance,
    CandleCapability, CandleUsage, CandleUseCase,
    
    // Candle HTTP types
    // CandleProvider, // Temporarily removed - not found
    
    // Other domain types (AsyncStream imported separately below)
};

// Re-export main builders  
pub use builders::{CandleFluentAi, CandleAgentRoleBuilder};

// Re-export main providers
pub use providers::CandleKimiK2Provider;

// Streaming primitives from fluent-ai-async (kept as-is per requirements)
pub use fluent_ai_async::{AsyncStream, AsyncStreamSender, AsyncTask, spawn_task};

// Alias for backward compatibility - people expect async_task module
pub use fluent_ai_async as async_task;
pub use fluent_ai_async::spawn_task as spawn_async;
