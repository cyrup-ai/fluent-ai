//! Error types and results for the macro system
//!
//! This module defines all error types and result enums used throughout
//! the macro system for comprehensive error handling.

use std::sync::Arc;
use std::time::Duration;

/// Result of macro action execution
#[derive(Debug)]
pub enum ActionExecutionResult {
    Success,
    Wait(Duration),
    SkipToAction(usize),
    Error(Arc<str>),
}

/// Result of macro playback operation
#[derive(Debug)]
pub enum MacroPlaybackResult {
    ActionExecuted,
    Completed,
    Failed,
    SessionNotActive,
}

/// Macro system errors
#[derive(Debug, thiserror::Error)]
pub enum MacroSystemError {
    #[error("Recording session not found")]
    SessionNotFound,
    #[error("Recording not active")]
    RecordingNotActive,
    #[error("Macro not found")]
    MacroNotFound,
    #[error("System time error")]
    SystemTimeError,
    #[error("Lock error")]
    LockError,
    #[error("Execution error: {0}")]
    ExecutionError(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Action execution failed: {0}")]
    ActionExecutionFailed(String),
    #[error("Feature not implemented")]
    NotImplemented,
}