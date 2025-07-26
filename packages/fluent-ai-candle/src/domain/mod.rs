//! Fluent AI Candle Domain Library
//!
//! This crate provides Candle-prefixed domain types and traits for AI services.
//! All domain logic, message types, and business objects are defined here with Candle prefixes
//! to ensure complete independence from the main fluent-ai domain package.

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![forbid(unsafe_code)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

// Core modules
/// Agent abstractions and management
pub mod agent;
/// Chat functionality and conversation management
pub mod chat;
/// Completion request/response types
pub mod completion;
/// Concurrency primitives and async utilities
pub mod concurrency;
/// Context processing and document handling
pub mod context;
/// Core domain types and traits
pub mod core;
/// Embedding generation and similarity search
pub mod embedding;
/// Engine abstractions for AI services
pub mod engine;
/// Error types and handling
pub mod error;
/// HTTP client and request/response types
pub mod http;
/// Image processing and generation
pub mod image;
/// Initialization and startup utilities
pub mod init;
/// Memory management and cognitive processing
pub mod memory;
/// Model configuration and management
pub mod model;
/// Prompt templates and processing
pub mod prompt;
/// Tool calling and function execution
pub mod tool;
/// Utility functions and helpers
pub mod util;
/// Voice processing and transcription
pub mod voice;
/// Workflow orchestration and task management
pub mod workflow;

// Additional module re-exports for provider compatibility
// Essential re-exports for Candle public API
// Candle-prefixed compatibility aliases
pub use core::{CandleChannelError, CandleChannelSender};

// Re-export HashMap from hashbrown for domain consistency
pub use hashbrown::HashMap;

pub use chat::message;
pub use context::chunk;
pub use model::usage;
// Alias for backward compatibility - people expect async_task module
pub use fluent_ai_async as async_task;
// Re-export from cyrup_sugars for convenience with Candle prefixes
pub use cyrup_sugars::{ByteSize, OneOrMany};
// Create Candle-specific enum that preserves ARCHITECTURE.md interface with ::None variant
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CandleZeroOneOrMany<T> {
    /// No items - preserves ARCHITECTURE.md interface
    None,
    /// Exactly one item
    One(T),
    /// Multiple items
    Many(Vec<T>),
}

impl<T> CandleZeroOneOrMany<T> {
    /// Create from zero items
    pub fn none() -> Self {
        Self::None
    }
    
    /// Create from one item
    pub fn one(item: T) -> Self {
        Self::One(item)
    }
    
    /// Create from many items
    pub fn many(items: Vec<T>) -> Self {
        if items.is_empty() {
            Self::None
        } else if items.len() == 1 {
            Self::One(items.into_iter().next().unwrap())
        } else {
            Self::Many(items)
        }
    }
    
    /// Check if is none
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
    
    /// Check if is one
    pub fn is_one(&self) -> bool {
        matches!(self, Self::One(_))
    }
    
    /// Check if is many
    pub fn is_many(&self) -> bool {
        matches!(self, Self::Many(_))
    }
}

impl<T> Default for CandleZeroOneOrMany<T> {
    fn default() -> Self {
        Self::None
    }
}

impl<T> From<Vec<T>> for CandleZeroOneOrMany<T> {
    fn from(items: Vec<T>) -> Self {
        Self::many(items)
    }
}

impl<T> From<Option<T>> for CandleZeroOneOrMany<T> {
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(item) => Self::One(item),
            None => Self::None,
        }
    }
}

impl<T> IntoIterator for CandleZeroOneOrMany<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    
    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::None => Vec::new().into_iter(),
            Self::One(item) => vec![item].into_iter(),
            Self::Many(items) => items.into_iter(),
        }
    }
}
pub use fluent_ai_async::spawn_task as spawn_async; // Alias for backward compatibility
pub use util::json_util;
pub use {
    // Candle Chat system
    chat::{
        CandleChatConfig, CandleCommandExecutor, CandleCommandRegistry, CandleConversation as CandleConversationTrait,
        CandleConversationImpl, CandleImmutableChatCommand, CandleMessage, CandleMessageChunk, CandleMessageRole,
        CandlePersonalityConfig},

    // Candle Completion system
    completion::{CandleCompletionModel},

    // Candle Concurrency primitives
    concurrency::{CandleChannel, CandleIntoTask, CandleOneshotChannel},

    // Candle Context system
    context::{CandleContentFormat, CandleDocument, CandleDocumentLoader, CandleDocumentMediaType},
    // Candle Core types
    core::{CandleDomainInitError, candle_execute_with_circuit_breaker},

    // Candle HTTP types
    http::{CandleProvider},

    // Streaming primitives from fluent-ai-async (kept as-is per requirements)
    fluent_ai_async::{AsyncStream, AsyncStreamSender, AsyncTask, NotResult, spawn_task},

    // Candle Core initialization and management
    init::{
        candle_get_default_memory_config, candle_get_from_pool, candle_initialize_domain, candle_initialize_domain_with_config,
        candle_pool_size, candle_return_to_pool},

    // Candle Model system
    model::{
        CandleCapability, CandleModel, CandleModelCapabilities, CandleModelInfo, CandleModelPerformance, CandleUsage, CandleUseCase,
        CandleValidationError, CandleValidationIssue, CandleValidationReport, CandleValidationResult, CandleValidationSeverity},

    // Candle Prompt system
    prompt::CandlePrompt,

    // Candle Voice system
    voice::{CandleAudio, CandleAudioMediaType, transcription::CandleTranscription},

    // Candle Workflow system
    workflow::{CandleStepType, CandleWorkflow, CandleWorkflowStep}};

/// Extension trait to add missing methods to CandleZeroOneOrMany
pub trait CandleZeroOneOrManyExt<T> {
    /// Convert to Vec, always returning a Vec regardless of variant
    fn to_vec(self) -> Vec<T>;

    /// Get the length/count of items
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> CandleZeroOneOrManyExt<T> for CandleZeroOneOrMany<T> {
    fn to_vec(self) -> Vec<T> {
        match self {
            CandleZeroOneOrMany::None => vec![],
            CandleZeroOneOrMany::One(item) => vec![item],
            CandleZeroOneOrMany::Many(items) => items,
        }
    }

    fn len(&self) -> usize {
        match self {
            CandleZeroOneOrMany::None => 0,
            CandleZeroOneOrMany::One(_) => 1,
            CandleZeroOneOrMany::Many(items) => items.len(),
        }
    }
}
