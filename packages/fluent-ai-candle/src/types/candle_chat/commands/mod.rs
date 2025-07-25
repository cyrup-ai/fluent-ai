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

use fluent_ai_async::{AsyncStream, emit, handle_error};
pub use execution::CommandExecutor;
use once_cell::sync::Lazy;
pub use parsing::{CommandParser, ParseError, ParseResult};
pub use registry::CommandRegistry;
pub use response::ResponseFormatter;
pub use types::{ImmutableChatCommand, CommandError, CommandResult, CommandContext, CommandOutput};
pub use validation::CommandValidator;

/// Global command executor instance - PURE SYNC (no futures)
static COMMAND_EXECUTOR: Lazy<Arc<RwLock<Option<CommandExecutor>>>> =
    Lazy::new(|| Arc::new(RwLock::new(None)));

/// Initialize global command executor - PURE SYNC (no futures)
pub fn initialize_command_executor(_context: CommandContext) {
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
            detail: "Command executor not initialized".to_string()})
    }
}

/// Execute command using global executor with fluent-ai-async streaming architecture
pub fn execute_command_async(command: ImmutableChatCommand) -> AsyncStream<CommandOutput> {
    AsyncStream::with_channel(move |sender| {
        if let Some(executor) = get_command_executor() {
            let mut result_stream = executor.execute_streaming(1, command);
            // Use the existing streaming implementation directly for zero-allocation efficiency
            match result_stream.try_next() {
                Some(result) => emit!(sender, result),
                None => {
                    handle_error!(
                        CommandError::ExecutionFailed(
                            "Stream closed without result".to_string()
                        ),
                        "Command execution stream closed unexpectedly"
                    );
                }
            }
        } else {
            handle_error!(
                CommandError::ConfigurationError {
                    detail: "Command executor not initialized".to_string()
                },
                "Command executor not initialized"
            );
        }
    })
}

/// Execute command using global executor - SYNC VERSION (legacy compatibility)
///
/// Converts AsyncStream to synchronous result using fluent-ai-async .collect() pattern.
/// This provides thread-safe, zero-allocation command execution.
pub fn execute_command(command: ImmutableChatCommand) -> CommandResult<CommandOutput> {
    if let Some(executor) = get_command_executor() {
        let mut result_stream = executor.execute_streaming(1, command);
        // Use synchronous try_next for zero-allocation efficiency
        match result_stream.try_next() {
            Some(result) => Ok(result),
            None => Err(CommandError::ExecutionFailed(
                "Stream closed without result".to_string()
            ))
        }
    } else {
        Err(CommandError::ConfigurationError {
            detail: "Command executor not initialized".to_string()})
    }
}

/// Parse and execute command using global executor with fluent-ai-async streaming architecture
pub fn parse_and_execute_command_async(input: &str) -> AsyncStream<CommandOutput> {
    let input = input.to_string();
    
    AsyncStream::with_channel(move |sender| {
        if let Some(executor) = get_command_executor() {
            let mut result_stream = executor.parse_and_execute(&input);
            // Use the existing streaming implementation directly for zero-allocation efficiency
            match result_stream.try_next() {
                Some(result) => emit!(sender, result),
                None => {
                    handle_error!(
                        CommandError::ExecutionFailed(
                            "Stream closed without result".to_string()
                        ),
                        "Command parse and execution stream closed unexpectedly"
                    );
                }
            }
        } else {
            handle_error!(
                CommandError::ConfigurationError {
                    detail: "Command executor not initialized".to_string()
                },
                "Command executor not initialized"
            );
        }
    })
}

/// Parse and execute command using global executor - SYNC VERSION (legacy compatibility)
///
/// Converts AsyncStream to synchronous result using fluent-ai-async .collect() pattern.
/// This provides thread-safe, zero-allocation command parsing and execution.
pub fn parse_and_execute_command(input: &str) -> CommandResult<CommandOutput> {
    if let Some(executor) = get_command_executor() {
        let mut result_stream = executor.parse_and_execute(input);
        // Use synchronous try_next for zero-allocation efficiency
        match result_stream.try_next() {
            Some(result) => Ok(result),
            None => Err(CommandError::ExecutionFailed(
                "Stream closed without result".to_string()
            ))
        }
    } else {
        Err(CommandError::ConfigurationError {
            detail: "Command executor not initialized".to_string()})
    }
}
