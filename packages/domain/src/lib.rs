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
pub mod async_task;
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
pub use {
    // Core initialization and management
    init::{
        initialize_domain, initialize_domain_with_config, get_default_memory_config,
        get_from_pool, return_to_pool, pool_size,
    },
    
    // Core types  
    core::{
        DomainInitError, HashMap, execute_with_circuit_breaker,
    },
    
    // Concurrency primitives
    concurrency::{Channel, OneshotChannel, IntoTask},
    
    // Async task system
    async_task::{
        AsyncStream, AsyncStreamSender, AsyncTask, NotResult, spawn_task, 
        error_handlers, emitter_builder,
    },
    
    // Model system
    model::{
        Capability, Model, ModelCapabilities, ModelInfo, ModelPerformance, UseCase,
        ValidationError, ValidationIssue, ValidationReport, ValidationResult, 
        ValidationSeverity, Usage,
    },
    
    // Chat system
    chat::{
        Conversation as ConversationTrait, ConversationImpl, ChatCommand, 
        CommandExecutor, CommandRegistry, ChatConfig, PersonalityConfig,
        Message, MessageRole, MessageChunk,
    },
    
    // Voice system
    voice::{Audio, AudioMediaType, transcription::Transcription},
    
    // Workflow system
    workflow::{Workflow, WorkflowStep, StepType},
    
    // Prompt system
    prompt::Prompt,
    
    // Context system
    context::{Document, DocumentLoader, DocumentMediaType, ContentFormat},
    
};

// Backward compatibility aliases
pub use core::{ChannelSender, ChannelError};
pub use async_task::spawn_task as spawn_async; // Alias for backward compatibility

// Re-export from cyrup_sugars for convenience
pub use cyrup_sugars::{ByteSize, OneOrMany, ZeroOneOrMany};

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