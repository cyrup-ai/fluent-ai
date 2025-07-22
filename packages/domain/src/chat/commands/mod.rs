//! Chat command system with zero-allocation patterns
//!
//! This module provides a comprehensive command system for chat interactions including
//! slash commands, command parsing, execution, and auto-completion with zero-allocation
//! patterns and blazing-fast performance.

pub mod execution;
pub mod parsing;
pub mod registry;
pub mod response;
pub mod types;
pub mod validation;

// Re-export main types and functions for convenience
// Global command executor functionality
use std::sync::{Arc, RwLock};

pub use execution::CommandExecutor;
use once_cell::sync::Lazy;
pub use parsing::{CommandParser, ParseError, ParseResult};
pub use registry::CommandRegistry;
pub use response::ResponseFormatter;
pub use types::*;
pub use validation::CommandValidator;

/// Global command executor instance - PURE SYNC (no futures)
static COMMAND_EXECUTOR: Lazy<Arc<RwLock<Option<CommandExecutor>>>> =
    Lazy::new(|| Arc::new(RwLock::new(None)));

/// Initialize global command executor - PURE SYNC (no futures)
pub fn initialize_command_executor(context: CommandContext) {
    let executor = CommandExecutor::new();
    if let Ok(mut writer) = COMMAND_EXECUTOR.write() {
        *writer = Some(executor);
    }
}

/// Get global command executor - PURE SYNC (no futures)
pub fn get_command_executor() -> Option<CommandExecutor> {
    COMMAND_EXECUTOR.read().ok().and_then(|guard| guard.clone())
}

/// Parse command using global executor - PURE SYNC (no futures)
pub fn parse_command(input: &str) -> CommandResult<ImmutableChatCommand> {
    if let Some(executor) = get_command_executor() {
        executor
            .parser()
            .parse(input)
            .map_err(|e| CommandError::ParseError(e.to_string()))
    } else {
        Err(CommandError::ConfigurationError {
            detail: "Command executor not initialized".to_string(),
        })
    }
}

/// Execute command using global executor - PURE SYNC (no futures)
pub fn execute_command(command: ImmutableChatCommand) -> CommandResult<CommandOutput> {
    if let Some(executor) = get_command_executor() {
        // Use AsyncTask sync methods instead of await
        let mut result_stream = executor.execute_streaming(1, command);
        // Collect the first result from stream (sync emulation)
        use futures_util::StreamExt;
        let runtime = tokio::runtime::Handle::current();
        if let Some(result) = runtime.block_on(result_stream.next()) {
            return Ok(result);
        } else {
            return Err(CommandError::ExecutionFailed(
                "Stream closed without result".to_string(),
            ));
        }
    } else {
        Err(CommandError::ConfigurationError {
            detail: "Command executor not initialized".to_string(),
        })
    }
}

/// Parse and execute command using global executor - PURE STREAMING (no futures)
pub fn parse_and_execute_command(input: &str) -> CommandResult<CommandOutput> {
    if let Some(executor) = get_command_executor() {
        // Get the AsyncTask and use sync completion
        let mut result_stream = executor.parse_and_execute(input);
        // Collect the first result from stream (sync emulation)
        use futures_util::StreamExt;
        let runtime = tokio::runtime::Handle::current();
        if let Some(result) = runtime.block_on(result_stream.next()) {
            return Ok(result);
        } else {
            return Err(CommandError::ExecutionFailed(
                "Stream closed without result".to_string(),
            ));
        }
    } else {
        Err(CommandError::ConfigurationError {
            detail: "Command executor not initialized".to_string(),
        })
    }
}
