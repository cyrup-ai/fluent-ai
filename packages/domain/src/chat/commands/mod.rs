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
    let executor = CommandExecutor::with_context(&context);
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

/// Execute command using global executor - ASYNC VERSION (recommended)
pub async fn execute_command_async(command: ImmutableChatCommand) -> CommandResult<CommandOutput> {
    if let Some(executor) = get_command_executor() {
        let mut result_stream = executor.execute_streaming(1, command);
        use futures_util::StreamExt;
        if let Some(result) = result_stream.next().await {
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

/// Execute command using global executor - SYNC VERSION (legacy compatibility)
///
/// WARNING: This function uses runtime.block_on() which can deadlock if called from async contexts.
/// Use execute_command_async() when possible.
pub fn execute_command(command: ImmutableChatCommand) -> CommandResult<CommandOutput> {
    if let Some(executor) = get_command_executor() {
        let mut result_stream = executor.execute_streaming(1, command);
        use futures_util::StreamExt;

        // Safe block_on approach: detect if we're in async context
        match tokio::runtime::Handle::try_current() {
            Ok(_handle) => {
                // We're in async context - spawn blocking task to avoid deadlock
                match std::thread::spawn(move || {
                    let rt = tokio::runtime::Runtime::new().map_err(|_| {
                        CommandError::ExecutionFailed("Runtime creation failed".to_string())
                    })?;
                    rt.block_on(async {
                        result_stream.next().await.ok_or_else(|| {
                            CommandError::ExecutionFailed(
                                "Stream closed without result".to_string(),
                            )
                        })
                    })
                })
                .join()
                {
                    Ok(result) => result,
                    Err(_) => Err(CommandError::ExecutionFailed("Thread panic".to_string())),
                }
            }
            Err(_) => {
                // We're not in async context - safe to use new runtime
                let rt = tokio::runtime::Runtime::new().map_err(|_| {
                    CommandError::ExecutionFailed("Runtime creation failed".to_string())
                })?;
                rt.block_on(async {
                    result_stream.next().await.ok_or_else(|| {
                        CommandError::ExecutionFailed("Stream closed without result".to_string())
                    })
                })
            }
        }
    } else {
        Err(CommandError::ConfigurationError {
            detail: "Command executor not initialized".to_string(),
        })
    }
}

/// Parse and execute command using global executor - ASYNC VERSION (recommended)
pub async fn parse_and_execute_command_async(input: &str) -> CommandResult<CommandOutput> {
    if let Some(executor) = get_command_executor() {
        let mut result_stream = executor.parse_and_execute(input);
        use futures_util::StreamExt;
        if let Some(result) = result_stream.next().await {
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

/// Parse and execute command using global executor - SYNC VERSION (legacy compatibility)
///
/// WARNING: This function uses runtime.block_on() which can deadlock if called from async contexts.
/// Use parse_and_execute_command_async() when possible.
pub fn parse_and_execute_command(input: &str) -> CommandResult<CommandOutput> {
    if let Some(executor) = get_command_executor() {
        let mut result_stream = executor.parse_and_execute(input);
        use futures_util::StreamExt;

        // Safe block_on approach: detect if we're in async context
        match tokio::runtime::Handle::try_current() {
            Ok(_handle) => {
                // We're in async context - spawn blocking task to avoid deadlock
                match std::thread::spawn(move || {
                    let rt = tokio::runtime::Runtime::new().map_err(|_| {
                        CommandError::ExecutionFailed("Runtime creation failed".to_string())
                    })?;
                    rt.block_on(async {
                        result_stream.next().await.ok_or_else(|| {
                            CommandError::ExecutionFailed(
                                "Stream closed without result".to_string(),
                            )
                        })
                    })
                })
                .join()
                {
                    Ok(result) => result,
                    Err(_) => Err(CommandError::ExecutionFailed("Thread panic".to_string())),
                }
            }
            Err(_) => {
                // We're not in async context - safe to use new runtime
                let rt = tokio::runtime::Runtime::new().map_err(|_| {
                    CommandError::ExecutionFailed("Runtime creation failed".to_string())
                })?;
                rt.block_on(async {
                    result_stream.next().await.ok_or_else(|| {
                        CommandError::ExecutionFailed("Stream closed without result".to_string())
                    })
                })
            }
        }
    } else {
        Err(CommandError::ConfigurationError {
            detail: "Command executor not initialized".to_string(),
        })
    }
}
