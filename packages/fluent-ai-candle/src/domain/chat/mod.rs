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
pub mod realtime;
pub mod search;
pub mod templates;

// Re-export types with corrected names to avoid ambiguous glob re-exports
pub use commands::{
    CommandExecutor as CandleCommandExecutor, 
    CommandRegistry as CandleCommandRegistry, 
    ImmutableChatCommand as CandleImmutableChatCommand
};
pub use config::{CandleChatConfig, CandlePersonalityConfig};
pub use conversation::{CandleConversationEvent as CandleConversation, ConversationImpl as CandleConversationImpl};
pub use export::{ExportData as CandleExportData, ExportFormat as CandleExportFormat};
pub use formatting::{FormatStyle as CandleFormatStyle, StreamingMessageFormatter as CandleStreamingMessageFormatter};
pub use integrations::{IntegrationConfig as CandleIntegrationConfig, IntegrationManager as CandleIntegrationManager};
pub use macros::{
    MacroAction as CandleMacroAction, 
    MacroSystem as CandleMacroSystem,
    ChatMacro as CandleChatMacro,
    MacroExecutionConfig as CandleMacroExecutionConfig,
    MacroMetadata as CandleMacroMetadata,
    MacroSystemError as CandleMacroSystemError
};
pub use message::message_processing::{
    process_message as candle_process_message, sanitize_content as candle_sanitize_content, 
    validate_message as candle_validate_message, validate_message_sync as candle_validate_message_sync};
pub use message::types::{CandleMessage, CandleMessageChunk, CandleMessageRole};
pub use realtime::{RealTimeSystem as CandleRealTimeSystem};
pub use search::{
    ChatSearchIndex as CandleChatSearchIndex, 
    SearchQuery as CandleSearchQuery,
    SearchStatistics as CandleSearchStatistics,
    ConversationTagger as CandleConversationTagger,
    HistoryExporter as CandleHistoryExporter,
    EnhancedHistoryManager as CandleEnhancedHistoryManager,
    HistoryManagerStatistics as CandleHistoryManagerStatistics
};
pub use templates::{ChatTemplate as CandleChatTemplate, TemplateManager as CandleTemplateManager, TemplateCategory as CandleTemplateCategory};
