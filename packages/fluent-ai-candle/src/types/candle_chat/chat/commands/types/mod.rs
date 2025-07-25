//! Command types module with logical decomposition
//!
//! This module provides a complete command system with streaming execution,
//! zero-allocation parsing, and comprehensive error handling. The module is
//! organized into focused submodules for optimal maintainability.

pub mod actions;
pub mod command;
pub mod context;
pub mod core;
pub mod events;
pub mod executor;
pub mod handler;
pub mod metrics;
pub mod parser;

#[cfg(test)]
mod tests;

// Re-export all public types for convenience
pub use actions::*;
pub use command::{ChatCommand, ImmutableChatCommand};
pub use context::{CommandContext, CommandOutput};
pub use core::{
    CommandError, CommandInfo, CommandResult, ParameterInfo, ParameterType, ResourceUsage,
    SettingsCategory,
};
pub use events::{CommandEvent, CommandExecutionResult, OutputType};
pub use executor::{CommandExecutorStats, StreamingCommandExecutor};
pub use handler::{CommandHandler, CommandHandlerMetadata, DefaultCommandHandler};
pub use metrics::ExecutionMetrics;
pub use parser::CommandParser;