//! Command execution events and output types
//!
//! Provides event types for streaming command execution with zero-allocation patterns
//! and comprehensive execution result tracking.

use super::command::ImmutableChatCommand;
use super::core::CommandError;

/// Command execution event for streaming
#[derive(Debug, Clone)]
pub enum CommandEvent {
    /// Command started executing
    Started {
        command: ImmutableChatCommand,
        execution_id: u64,
        timestamp_nanos: u64},
    /// Command execution progress
    Progress {
        execution_id: u64,
        progress_percent: f32,
        message: Option<String>},
    /// Command produced output
    Output {
        execution_id: u64,
        output: String,
        output_type: OutputType},
    /// Command completed successfully
    Completed {
        execution_id: u64,
        result: CommandExecutionResult,
        duration_nanos: u64},
    /// Command failed
    Failed {
        execution_id: u64,
        error: CommandError,
        duration_nanos: u64},
    /// Command was cancelled
    Cancelled { execution_id: u64, reason: String }}

/// Command output type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputType {
    Text,
    Json,
    Html,
    Markdown,
    Binary}

/// Command execution result
#[derive(Debug, Clone)]
pub enum CommandExecutionResult {
    /// Simple success message
    Success(String),
    /// Data result with structured output
    Data(serde_json::Value),
    /// File result with path and metadata
    File {
        path: String,
        size_bytes: u64,
        mime_type: String},
    /// Multiple results
    Multiple(Vec<CommandExecutionResult>)}