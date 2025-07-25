//! Chat syntax features module
//!
//! This module provides comprehensive chat syntax features with zero-allocation patterns
//! and production-ready functionality. All submodules follow blazing-fast, lock-free,
//! and elegant ergonomic design principles.
//!
//! ## Features
//! - Rich message formatting with SIMD-optimized parsing
//! - Command system with slash commands and auto-completion
//! - Template and macro system for reusable patterns
//! - Advanced configuration with nested settings
//! - Real-time features with atomic state management
//! - Enhanced history management with full-text search
//! - External integration system with plugin architecture
//!
//! ## Architecture
//! All modules use zero-allocation patterns with Arc<str> for string sharing,
//! crossbeam-skiplist for lock-free data structures, and atomic operations
//! for thread-safe state management.

pub mod commands;
pub mod config;
pub mod conversation;
pub mod export;
pub mod formatting;
pub mod integrations;
pub mod macros;
pub mod message;
// pub mod realtime; // Removed - now using decomposed realtime module at ../realtime/
pub mod search;
pub mod templates;

// Re-export specific types to avoid ambiguous glob re-exports
pub use commands::{CommandExecutor, CommandRegistry, ImmutableChatCommand};
pub use config::{ChatConfig, PersonalityConfig};
pub use conversation::{Conversation, ConversationBuilder, ConversationImpl};
pub use export::{ExportData, ExportFormat};
pub use formatting::{FormatStyle, StreamingMessageFormatter};
pub use integrations::{IntegrationConfig, IntegrationManager};
pub use macros::{MacroAction, MacroSystem};
// Use message types from local message module
pub use message::{CandleMessage, CandleMessageRole, Message, MessageRole};
// pub use realtime::RealTimeSystem; // Removed - now using decomposed realtime module at ../realtime/
pub use search::{ChatSearchIndex, SearchQuery};
pub use templates::{ChatTemplate, TemplateManager};
