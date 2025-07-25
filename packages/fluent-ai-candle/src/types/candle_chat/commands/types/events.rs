//! Command events and action enums
//!
//! Event types for streaming command execution and all action enums
//! used by different command types.

use serde::{Deserialize, Serialize};

use super::commands::ImmutableChatCommand;
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

/// Search-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchScope {
    All,
    Current,
    Recent,
    Bookmarked}

/// Template-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemplateAction {
    List,
    Create,
    Delete,
    Edit,
    Use}

/// Macro-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MacroAction {
    List,
    Create,
    Delete,
    Edit,
    Execute}

/// Branch-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BranchAction {
    List,
    Create,
    Switch,
    Merge,
    Delete}

/// Session-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionAction {
    List,
    New,
    Switch,
    Delete,
    Export,
    Import}

/// Tool-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolAction {
    List,
    Install,
    Remove,
    Configure,
    Update,
    Execute}

/// Stats-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatsType {
    Usage,
    Performance,
    History,
    Tokens,
    Costs,
    Errors}

/// Theme-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThemeAction {
    Set,
    List,
    Create,
    Export,
    Import,
    Edit}

/// Debug-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DebugAction {
    Info,
    Logs,
    Performance,
    Memory,
    Network,
    Cache}

/// History-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HistoryAction {
    Show,
    Search,
    Clear,
    Export,
    Import,
    Backup}

/// Import-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImportType {
    Chat,
    Config,
    Templates,
    Macros}

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