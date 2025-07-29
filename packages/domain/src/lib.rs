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
// Essential re-exports for public API
// Backward compatibility aliases
pub use core::{ChannelError, ChannelSender};

// Re-export HashMap from hashbrown for domain consistency
pub use hashbrown::HashMap;

pub use chat::message;
pub use context::chunk;
pub use context::chunk::ChatMessageChunk;
pub use model::usage;
// Alias for backward compatibility - people expect async_task module
pub use fluent_ai_async as async_task;
// Re-export from cyrup_sugars for convenience
pub use cyrup_sugars::{ByteSize, OneOrMany, ZeroOneOrMany};
pub use fluent_ai_async::spawn_task as spawn_async; // Alias for backward compatibility
pub use util::json_util;
pub use {
    // Chat system
    chat::{
        ChatConfig, CommandExecutor, CommandRegistry, Conversation as ConversationTrait,
        ConversationImpl, ImmutableChatCommand, Message, MessageChunk, MessageRole,
        PersonalityConfig},

    // Completion system
    completion::{CompletionModel, CompletionBackend as CompletionProvider},

    // Concurrency primitives
    concurrency::{Channel, IntoTask, OneshotChannel},

    // Context system
    context::{ContentFormat, Document, DocumentLoader, DocumentMediaType,
              provider::{ImmutableFileContext as Context}},
    // Core types
    core::{DomainInitError, execute_with_circuit_breaker},

    // HTTP types
    http::{ToolCall, FunctionCall},

    // Streaming primitives from fluent-ai-async
    fluent_ai_async::{AsyncStream, AsyncStreamSender, AsyncTask, NotResult, spawn_task},

    // Memory system
    memory::{Library, VectorStoreIndex, VectorStoreIndexDyn},

    // Core initialization and management
    init::{
        get_default_memory_config, get_from_pool, initialize_domain, initialize_domain_with_config,
        pool_size, return_to_pool},

    // Model system - primarily from model-info package
    model::{
        // Domain-specific capabilities and validation
        Capability, DomainModelCapabilities, ModelPerformance, Usage, UseCase,
        ValidationError, ValidationIssue, ValidationReport, ValidationResult, ValidationSeverity,
        // Core model types from model-info (single source of truth)
        OpenAi, Mistral, Anthropic, Together, OpenRouter, HuggingFace, Xai, Model,
        ModelInfo, ModelInfoBuilder, ProviderTrait, ModelError, Result, Provider,

        UnifiedModelRegistry, RegistryStats,
        // Domain-specific model-info integration
        LegacyModelRegistry, ModelCache, ModelValidator, ModelFilter, ModelQueryResult,
        CacheStats, CacheConfig, BatchValidationResult
        },

    // Prompt system
    prompt::Prompt,

    // Voice system
    voice::{Audio, AudioMediaType, transcription::Transcription},

    // Workflow system
    workflow::{StepType, Workflow, WorkflowStep},

    // Agent system
    agent::{AgentConversation, AgentConversationMessage, AgentRole, AgentRoleAgent, AgentRoleImpl},

    // Tool system
    tool::{Tool, NamedTool, McpTool, McpClient as McpServer, McpToolData}};

// Additional legacy compatibility aliases
pub use context::provider::ImmutableFileContext as FileContext;
pub use memory::Library as Memory;
pub use util::json_util as AdditionalParams;
pub use util::json_util as Metadata;
pub use chat::ConversationImpl as Conversation;

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
            ZeroOneOrMany::Many(items) => items}
    }

    fn len(&self) -> usize {
        match self {
            ZeroOneOrMany::None => 0,
            ZeroOneOrMany::One(_) => 1,
            ZeroOneOrMany::Many(items) => items.len()}
    }
}
