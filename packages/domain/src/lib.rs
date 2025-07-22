//! Fluent AI Domain Library
//!
//! This crate provides core domain types and traits for AI services.
//! All domain logic, message types, and business objects are defined here.

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![forbid(unsafe_code)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

// Core modules
pub mod agent;
pub mod chat;
pub mod completion;
pub mod concurrency;
pub mod context;
pub mod core;
pub mod embedding;
pub mod engine;
pub mod error;
pub mod image;
pub mod init;
pub mod memory;
pub mod model;
pub mod prompt;
pub mod tool;
pub mod util;
pub mod voice;
pub mod workflow;

// Essential re-exports for public API
// Backward compatibility aliases
pub use core::{ChannelError, ChannelSender};

// Re-export from cyrup_sugars for convenience
pub use cyrup_sugars::{ByteSize, OneOrMany, ZeroOneOrMany};
pub use fluent_ai_async::spawn_task as spawn_async; // Alias for backward compatibility
pub use {
    // Chat system
    chat::{
        ChatConfig, CommandExecutor, CommandRegistry, Conversation as ConversationTrait,
        ConversationImpl, ImmutableChatCommand, Message, MessageChunk, MessageRole,
        PersonalityConfig,
    },

    // Concurrency primitives
    concurrency::{Channel, IntoTask, OneshotChannel},

    // Context system
    context::{ContentFormat, Document, DocumentLoader, DocumentMediaType},
    // Core types
    core::{DomainInitError, HashMap, execute_with_circuit_breaker},

    // Streaming primitives from fluent-ai-async
    fluent_ai_async::{
        AsyncStream, AsyncStreamSender, AsyncTask, NotResult, spawn_stream, spawn_task,
    },

    // Core initialization and management
    init::{
        get_default_memory_config, get_from_pool, initialize_domain, initialize_domain_with_config,
        pool_size, return_to_pool,
    },

    // Model system
    model::{
        Capability, Model, ModelCapabilities, ModelInfo, ModelPerformance, Usage, UseCase,
        ValidationError, ValidationIssue, ValidationReport, ValidationResult, ValidationSeverity,
    },

    // Prompt system
    prompt::Prompt,

    // Voice system
    voice::{Audio, AudioMediaType, transcription::Transcription},

    // Workflow system
    workflow::{StepType, Workflow, WorkflowStep},
};

/// Extension trait to add missing methods to ZeroOneOrMany
pub trait ZeroOneOrManyExt<T> {
    /// Convert to Vec, always returning a Vec regardless of variant
    fn to_vec(self) -> Vec<T>;

    /// Get the length/count of items
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> ZeroOneOrManyExt<T> for ZeroOneOrMany<T> {
    fn to_vec(self) -> Vec<T> {
        match self {
            ZeroOneOrMany::None => vec![],
            ZeroOneOrMany::One(item) => vec![item],
            ZeroOneOrMany::Many(items) => items,
        }
    }

    fn len(&self) -> usize {
        match self {
            ZeroOneOrMany::None => 0,
            ZeroOneOrMany::One(_) => 1,
            ZeroOneOrMany::Many(items) => items.len(),
        }
    }
}
