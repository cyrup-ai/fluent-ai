//! Candle chat command system with zero-allocation patterns
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

// Re-export main Candle types and functions for convenience
// Global Candle command executor functionality
use std::sync::{Arc, RwLock};
use crate::AsyncStream;

pub use execution::CommandExecutor;
use once_cell::sync::Lazy;
pub use parsing::{CommandParser, ParseError, ParseResult};
pub use registry::CommandRegistry;
pub use types::*;
pub use response::ResponseFormatter;
pub use validation::CommandValidator;

/// Global Candle command executor instance - PURE SYNC (no futures)
static CANDLE_COMMAND_EXECUTOR: Lazy<Arc<RwLock<Option<CommandExecutor>>>> =
    Lazy::new(|| Arc::new(RwLock::new(None)));

/// Initialize global Candle command executor - PURE SYNC (no futures)
pub fn initialize_candle_command_executor(context: &CommandExecutionContext) {
    let executor = CommandExecutor::with_context(context);
    if let Ok(mut writer) = CANDLE_COMMAND_EXECUTOR.write() {
        *writer = Some(executor);
    }
}

/// Get global Candle command executor - PURE SYNC (no futures)
pub fn get_candle_command_executor() -> Option<CommandExecutor> {
    CANDLE_COMMAND_EXECUTOR.read().ok().and_then(|guard| guard.clone())
}

/// Parse Candle command using global executor - PURE SYNC (no futures)
pub fn parse_candle_command(input: &str) -> CommandResult<ImmutableChatCommand> {
    if let Some(executor) = get_candle_command_executor() {
        executor
            .parser()
            .parse(input)
            .map_err(|e| CandleCommandError::ParseError(e.to_string()))
    } else {
        Err(CandleCommandError::ConfigurationError {
            detail: "Candle command executor not initialized".to_string()})
    }
}

/// Execute Candle command using global executor - STREAMING VERSION (streams-only architecture)
pub fn execute_candle_command_async(command: ImmutableChatCommand) -> AsyncStream<CommandEvent> {
    AsyncStream::with_channel(move |sender| {
        if let Some(executor) = get_candle_command_executor() {
            let mut result_stream = executor.execute_streaming(1, command);
            // Forward all events from executor stream
            while let Some(event) = result_stream.try_next() {
                fluent_ai_async::emit!(sender, event);
            }
        } else {
            fluent_ai_async::emit!(sender, CommandEvent::Failed {
                execution_id: 0,
                error: "Candle command executor not initialized".to_string(),
                error_code: 1001,
                duration_us: 0,
                resource_usage: crate::domain::chat::commands::types::metadata::ResourceUsage::new_with_start_time(),
                timestamp_us: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64
            });
        }
    })
}

/// Execute Candle command using global executor - STREAMING VERSION (streams-only architecture)
/// Note: Sync version removed - use streaming architecture only
pub fn execute_candle_command(command: ImmutableChatCommand) -> AsyncStream<CommandEvent> {
    AsyncStream::with_channel(move |sender| {
        if let Some(executor) = get_candle_command_executor() {
            let mut result_stream = executor.execute_streaming(1, command);
            // Forward all events from executor stream
            while let Some(event) = result_stream.try_next() {
                fluent_ai_async::emit!(sender, event);
            }
        } else {
            fluent_ai_async::emit!(sender, CommandEvent::Failed {
                execution_id: 0,
                error: "Candle command executor not initialized".to_string(),
                error_code: 1001,
                duration_us: 0,
                resource_usage: crate::domain::chat::commands::types::metadata::ResourceUsage::new_with_start_time(),
                timestamp_us: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64
            });
        }
    })
}

/// Parse and execute Candle command using global executor - STREAMING VERSION (streams-only architecture)
pub fn parse_and_execute_candle_command_async(input: &str) -> AsyncStream<CommandEvent> {
    let input_str = input.to_string();
    AsyncStream::with_channel(move |sender| {
        if let Some(executor) = get_candle_command_executor() {
            let mut result_stream = executor.parse_and_execute(&input_str);
            // Forward all events from executor stream
            while let Some(event) = result_stream.try_next() {
                fluent_ai_async::emit!(sender, event);
            }
        } else {
            fluent_ai_async::emit!(sender, CommandEvent::Failed {
                execution_id: 0,
                error: "Candle command executor not initialized".to_string(),
                error_code: 1001,
                duration_us: 0,
                resource_usage: crate::domain::chat::commands::types::metadata::ResourceUsage::new_with_start_time(),
                timestamp_us: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64
            });
        }
    })
}

/// Parse and execute Candle command using global executor - STREAMING VERSION (streams-only architecture)
/// Note: Sync version removed - use streaming architecture only
pub fn parse_and_execute_candle_command(input: &str) -> AsyncStream<CommandEvent> {
    let input_str = input.to_string();
    AsyncStream::with_channel(move |sender| {
        if let Some(executor) = get_candle_command_executor() {
            let mut result_stream = executor.parse_and_execute(&input_str);
            // Forward all events from executor stream
            while let Some(event) = result_stream.try_next() {
                fluent_ai_async::emit!(sender, event);
            }
        } else {
            fluent_ai_async::emit!(sender, CommandEvent::Failed {
                execution_id: 0,
                error: "Candle command executor not initialized".to_string(),
                error_code: 1001,
                duration_us: 0,
                resource_usage: crate::domain::chat::commands::types::metadata::ResourceUsage::new_with_start_time(),
                timestamp_us: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64
            });
        }
    })
}
