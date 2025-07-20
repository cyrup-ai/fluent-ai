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
use std::sync::Arc;

pub use execution::CommandExecutor;
use once_cell::sync::Lazy;
pub use parsing::{CommandParser, ParseError, ParseResult};
pub use registry::CommandRegistry;
pub use response::ResponseFormatter;
use tokio::sync::RwLock;
pub use types::*;
pub use validation::CommandValidator;

/// Global command executor instance
static COMMAND_EXECUTOR: Lazy<Arc<RwLock<Option<CommandExecutor>>>> =
    Lazy::new(|| Arc::new(RwLock::new(None)));

/// Initialize global command executor
pub async fn initialize_command_executor(context: CommandContext) {
    let executor = CommandExecutor::new(context);
    *COMMAND_EXECUTOR.write().await = Some(executor);
}

/// Get global command executor
pub async fn get_command_executor() -> Option<CommandExecutor> {
    COMMAND_EXECUTOR.read().await.clone()
}

/// Parse command using global executor
pub async fn parse_command(input: &str) -> CommandResult<ChatCommand> {
    if let Some(executor) = get_command_executor().await {
        executor
            .parser()
            .parse(input)
            .map_err(|e| CommandError::ParseError {
                detail: Arc::from(e.to_string()),
            })
    } else {
        Err(CommandError::ConfigurationError {
            detail: Arc::from("Command executor not initialized"),
        })
    }
}

/// Execute command using global executor
pub async fn execute_command(command: ChatCommand) -> CommandResult<CommandOutput> {
    if let Some(executor) = get_command_executor().await {
        executor.execute(command).await
    } else {
        Err(CommandError::ConfigurationError {
            detail: Arc::from("Command executor not initialized"),
        })
    }
}

/// Parse and execute command using global executor
pub async fn parse_and_execute_command(input: &str) -> CommandResult<CommandOutput> {
    if let Some(executor) = get_command_executor().await {
        executor.parse_and_execute(input).await.map_err(|e| CommandError::ExecutionFailed {
            reason: Arc::from(format!("Task join error: {}", e))
        })?
    } else {
        Err(CommandError::ConfigurationError {
            detail: Arc::from("Command executor not initialized"),
        })
    }
}
