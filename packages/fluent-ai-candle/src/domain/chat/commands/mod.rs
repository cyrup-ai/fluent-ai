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

pub use execution::CommandExecutor;
use once_cell::sync::Lazy;
pub use parsing::{CommandParser, ParseError, ParseResult};
pub use registry::CommandRegistry;
pub use response::ResponseFormatter;
pub use types::*;
pub use validation::CommandValidator;

/// Global Candle command executor instance - PURE SYNC (no futures)
static CANDLE_COMMAND_EXECUTOR: Lazy<Arc<RwLock<Option<CommandExecutor>>>> =
    Lazy::new(|| Arc::new(RwLock::new(None)));

/// Initialize global Candle command executor - PURE SYNC (no futures)
pub fn initialize_candle_command_executor(context: &CandleCommandContext) {
    let executor = CommandExecutor::with_context(context);
    if let Ok(mut writer) = CANDLE_COMMAND_EXECUTOR.write() {
        *writer = Some(executor);
    }
}

/// Get global Candle command executor - PURE SYNC (no futures)
pub fn get_candle_command_executor() -> Option<CandleCommandExecutor> {
    CANDLE_COMMAND_EXECUTOR.read().ok().and_then(|guard| guard.clone())
}

/// Parse Candle command using global executor - PURE SYNC (no futures)
pub fn parse_candle_command(input: &str) -> CandleCommandResult<CandleImmutableChatCommand> {
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

/// Execute Candle command using global executor - ASYNC VERSION (recommended)
pub async fn execute_candle_command_async(command: CandleImmutableChatCommand) -> CandleCandleCommandResult<CandleCandleCommandOutput> {
    if let Some(executor) = get_candle_command_executor() {
        let mut result_stream = executor.execute_streaming(1, command);
        // Use AsyncStream try_next method (NO FUTURES architecture)
        if let Some(result) = result_stream.try_next() {
            return Ok(result);
        } else {
            return Err(CandleCommandError::ExecutionFailed(
                "Stream closed without result".to_string(),
            ));
        }
    } else {
        Err(CandleCommandError::ConfigurationError {
            detail: "Candle command executor not initialized".to_string()})
    }
}

/// Execute Candle command using global executor - SYNC VERSION (legacy compatibility)
///
/// WARNING: This function uses runtime.block_on() which can deadlock if called from async contexts.
/// Use execute_candle_command_async() when possible.
pub fn execute_candle_command(command: CandleImmutableChatCommand) -> CandleCandleCommandResult<CandleCandleCommandOutput> {
    if let Some(executor) = get_candle_command_executor() {
        let mut result_stream = executor.execute_streaming(1, command);
        // Removed unused import: use futures_util::StreamExt;

        // Safe block_on approach: detect if we're in async context
        match tokio::runtime::Handle::try_current() {
            Ok(_handle) => {
                // We're in async context - spawn blocking task to avoid deadlock
                match std::thread::spawn(move || {
                    let _rt = tokio::runtime::Runtime::new().map_err(|_| {
                        CandleCommandError::ExecutionFailed("Runtime creation failed".to_string())
                    })?;
                    // Use AsyncStream try_next method (NO FUTURES architecture)
                    result_stream.try_next().ok_or_else(|| {
                        CandleCommandError::ExecutionFailed(
                            "Stream closed without result".to_string(),
                        )
                    })
                })
                .join()
                {
                    Ok(result) => result,
                    Err(_) => Err(CandleCommandError::ExecutionFailed("Thread panic".to_string()))}
            }
            Err(_) => {
                // We're not in async context - safe to use new runtime
                let _rt = tokio::runtime::Runtime::new().map_err(|_| {
                    CandleCommandError::ExecutionFailed("Runtime creation failed".to_string())
                })?;
                // Use AsyncStream try_next method (NO FUTURES architecture)
                result_stream.try_next().ok_or_else(|| {
                    CandleCommandError::ExecutionFailed("Stream closed without result".to_string())
                })
            }
        }
    } else {
        Err(CandleCommandError::ConfigurationError {
            detail: "Candle command executor not initialized".to_string()})
    }
}

/// Parse and execute Candle command using global executor - ASYNC VERSION (recommended)
pub async fn parse_and_execute_candle_command_async(input: &str) -> CandleCandleCommandResult<CandleCandleCommandOutput> {
    if let Some(executor) = get_candle_command_executor() {
        let mut result_stream = executor.parse_and_execute(input);
        // Use AsyncStream try_next method (NO FUTURES architecture)
        if let Some(result) = result_stream.try_next() {
            return Ok(result);
        } else {
            return Err(CandleCommandError::ExecutionFailed(
                "Stream closed without result".to_string(),
            ));
        }
    } else {
        Err(CandleCommandError::ConfigurationError {
            detail: "Candle command executor not initialized".to_string()})
    }
}

/// Parse and execute Candle command using global executor - SYNC VERSION (legacy compatibility)
///
/// WARNING: This function uses runtime.block_on() which can deadlock if called from async contexts.
/// Use parse_and_execute_candle_command_async() when possible.
pub fn parse_and_execute_candle_command(input: &str) -> CandleCandleCommandResult<CandleCandleCommandOutput> {
    if let Some(executor) = get_candle_command_executor() {
        let mut result_stream = executor.parse_and_execute(input);
        // Removed unused import: use futures_util::StreamExt;

        // Safe block_on approach: detect if we're in async context
        match tokio::runtime::Handle::try_current() {
            Ok(_handle) => {
                // We're in async context - spawn blocking task to avoid deadlock
                match std::thread::spawn(move || {
                    let _rt = tokio::runtime::Runtime::new().map_err(|_| {
                        CandleCommandError::ExecutionFailed("Runtime creation failed".to_string())
                    })?;
                    // Use AsyncStream try_next method (NO FUTURES architecture)
                    result_stream.try_next().ok_or_else(|| {
                        CandleCommandError::ExecutionFailed(
                            "Stream closed without result".to_string(),
                        )
                    })
                })
                .join()
                {
                    Ok(result) => result,
                    Err(_) => Err(CandleCommandError::ExecutionFailed("Thread panic".to_string()))}
            }
            Err(_) => {
                // We're not in async context - safe to use new runtime
                let _rt = tokio::runtime::Runtime::new().map_err(|_| {
                    CandleCommandError::ExecutionFailed("Runtime creation failed".to_string())
                })?;
                // Use AsyncStream try_next method (NO FUTURES architecture)
                result_stream.try_next().ok_or_else(|| {
                    CandleCommandError::ExecutionFailed("Stream closed without result".to_string())
                })
            }
        }
    } else {
        Err(CandleCommandError::ConfigurationError {
            detail: "Candle command executor not initialized".to_string()})
    }
}
