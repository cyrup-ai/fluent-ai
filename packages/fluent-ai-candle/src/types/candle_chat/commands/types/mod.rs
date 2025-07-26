//! Command types module with logical decomposition
//!
//! This module provides a complete command system with streaming execution,
//! zero-allocation parsing, and comprehensive error handling.

pub mod core;
pub mod commands;
pub mod events;
pub mod executor;
pub mod parser;
pub mod context;
pub mod metrics;
pub mod handler;


// Re-export all public types for convenience
pub use core::{CommandError, CommandResult, ParameterType, ParameterInfo, CommandInfo, ResourceUsage};
pub use commands::ImmutableChatCommand;
pub use events::{
    CommandEvent, OutputType, SearchScope, TemplateAction, MacroAction, BranchAction,
    SessionAction, ToolAction, StatsType, ThemeAction, DebugAction, HistoryAction,
    ImportType, CommandExecutionResult
};
pub use executor::{StreamingCommandExecutor, CommandExecutorStats, SettingsCategory};
pub use parser::CommandParser;
pub use context::{CommandContext, CommandOutput};
pub use metrics::ExecutionMetrics;
pub use handler::{CommandHandler, CommandHandlerMetadata, DefaultCommandHandler};

// Legacy compatibility
#[deprecated(note = "Use ImmutableChatCommand instead for zero-allocation streaming")]
pub type ChatCommand = ImmutableChatCommand;