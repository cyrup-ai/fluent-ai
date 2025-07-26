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
        /// The command being executed
        command: ImmutableChatCommand,
        /// Unique execution identifier
        execution_id: u64,
        /// Start timestamp in nanoseconds
        timestamp_nanos: u64
    },
    /// Command execution progress
    Progress {
        /// Execution identifier for tracking
        execution_id: u64,
        /// Progress completion percentage (0.0-100.0)
        progress_percent: f32,
        /// Optional progress message
        message: Option<String>
    },
    /// Command produced output
    Output {
        /// Execution identifier for tracking
        execution_id: u64,
        /// Output content produced by command
        output: String,
        /// Type classification of the output
        output_type: OutputType
    },
    /// Command completed successfully
    Completed {
        /// Execution identifier for tracking
        execution_id: u64,
        /// Final result of command execution
        result: CommandExecutionResult,
        /// Total execution duration in nanoseconds
        duration_nanos: u64
    },
    /// Command failed
    Failed {
        /// Execution identifier for tracking
        execution_id: u64,
        /// Error that caused the failure
        error: CommandError,
        /// Duration before failure in nanoseconds
        duration_nanos: u64
    },
    /// Command was cancelled
    Cancelled {
        /// Execution identifier for tracking
        execution_id: u64,
        /// Reason for cancellation
        reason: String
    }
}

/// Command output type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputType {
    /// Plain text output
    Text,
    /// JSON formatted output
    Json,
    /// HTML formatted output
    Html,
    /// Markdown formatted output
    Markdown,
    /// Binary data output
    Binary
}

/// Search scope for search commands
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchScope {
    /// Search all available content
    All,
    /// Search only current session
    Current,
    /// Search recent conversations
    Recent,
    /// Search bookmarked items only
    Bookmarked
}

/// Template management actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemplateAction {
    /// List available templates
    List,
    /// Create a new template
    Create,
    /// Delete an existing template
    Delete,
    /// Edit an existing template
    Edit,
    /// Use/apply a template
    Use
}

/// Macro management actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MacroAction {
    /// List available macros
    List,
    /// Create a new macro
    Create,
    /// Delete an existing macro
    Delete,
    /// Edit an existing macro
    Edit,
    /// Execute a macro
    Execute
}

/// Branch management actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BranchAction {
    /// List available branches
    List,
    /// Create a new branch
    Create,
    /// Switch to a different branch
    Switch,
    /// Merge branches together
    Merge,
    /// Delete an existing branch
    Delete
}

/// Session management actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionAction {
    /// List available sessions
    List,
    /// Create a new session
    New,
    /// Switch to a different session
    Switch,
    /// Delete an existing session
    Delete,
    /// Export session data
    Export,
    /// Import session data
    Import
}

/// Tool management actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolAction {
    /// List available tools
    List,
    /// Install a new tool
    Install,
    /// Remove an existing tool
    Remove,
    /// Configure tool settings
    Configure,
    /// Update tool to latest version
    Update,
    /// Execute a tool command
    Execute
}

/// Statistics type for stats commands
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatsType {
    /// Usage statistics
    Usage,
    /// Performance metrics
    Performance,
    /// Historical data
    History,
    /// Token usage statistics
    Tokens,
    /// Cost and billing information
    Costs,
    /// Error statistics
    Errors
}

/// Theme management actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThemeAction {
    /// Set active theme
    Set,
    /// List available themes
    List,
    /// Create a new theme
    Create,
    /// Export theme configuration
    Export,
    /// Import theme configuration
    Import,
    /// Edit existing theme
    Edit
}

/// Debug information actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DebugAction {
    /// Show system information
    Info,
    /// Display log files
    Logs,
    /// Show performance metrics
    Performance,
    /// Display memory usage
    Memory,
    /// Show network statistics
    Network,
    /// Display cache information
    Cache
}

/// History management actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HistoryAction {
    /// Show conversation history
    Show,
    /// Search through history
    Search,
    /// Clear history data
    Clear,
    /// Export history to file
    Export,
    /// Import history from file
    Import,
    /// Create history backup
    Backup
}

/// Import data type for import commands
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImportType {
    /// Chat conversation data
    Chat,
    /// Configuration settings
    Config,
    /// Template definitions
    Templates,
    /// Macro definitions
    Macros
}

/// Command execution result
#[derive(Debug, Clone)]
pub enum CommandExecutionResult {
    /// Simple success message
    Success(
        /// Success message text
        String
    ),
    /// Data result with structured output
    Data(
        /// Structured JSON data result
        serde_json::Value
    ),
    /// File result with path and metadata
    File {
        /// File system path to the result file
        path: String,
        /// Size of the file in bytes
        size_bytes: u64,
        /// MIME type of the file content
        mime_type: String
    },
    /// Multiple results
    Multiple(
        /// Vector of multiple execution results
        Vec<CommandExecutionResult>
    )
}