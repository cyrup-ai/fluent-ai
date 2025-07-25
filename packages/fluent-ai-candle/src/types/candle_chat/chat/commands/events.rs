//! Command execution events and action enums
//!
//! Provides streaming events for command execution with zero-allocation patterns
//! and comprehensive action type definitions for all command categories.

use serde::{Deserialize, Serialize};

use super::{
    error::CommandError,
    command::ImmutableChatCommand};

/// Command execution event for streaming
///
/// Represents the lifecycle of command execution with detailed progress tracking
/// and comprehensive result information for streaming command processors.
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
    Cancelled { 
        execution_id: u64, 
        reason: String 
    }}

impl CommandEvent {
    /// Get the execution ID for this event
    #[inline(always)]
    pub fn execution_id(&self) -> u64 {
        match self {
            Self::Started { execution_id, .. }
            | Self::Progress { execution_id, .. }
            | Self::Output { execution_id, .. }
            | Self::Completed { execution_id, .. }
            | Self::Failed { execution_id, .. }
            | Self::Cancelled { execution_id, .. } => *execution_id}
    }

    /// Check if this is a terminal event (completion, failure, or cancellation)
    #[inline(always)]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed { .. } | Self::Failed { .. } | Self::Cancelled { .. })
    }

    /// Get event timestamp if available
    #[inline(always)]
    pub fn timestamp_nanos(&self) -> Option<u64> {
        match self {
            Self::Started { timestamp_nanos, .. } => Some(*timestamp_nanos),
            _ => None}
    }

    /// Get duration if this is a terminal event
    #[inline(always)]
    pub fn duration_nanos(&self) -> Option<u64> {
        match self {
            Self::Completed { duration_nanos, .. } | Self::Failed { duration_nanos, .. } => Some(*duration_nanos),
            _ => None}
    }
}

/// Command output type
///
/// Specifies the format and interpretation of command output data
/// for proper rendering and processing by output handlers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputType {
    /// Plain text output
    Text,
    /// JSON structured data
    Json,
    /// HTML markup
    Html,
    /// Markdown formatted text
    Markdown,
    /// Binary data
    Binary}

impl OutputType {
    /// Get MIME type for this output type
    #[inline(always)]
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Text => "text/plain",
            Self::Json => "application/json",
            Self::Html => "text/html",
            Self::Markdown => "text/markdown",
            Self::Binary => "application/octet-stream"}
    }

    /// Get file extension for this output type
    #[inline(always)]
    pub fn file_extension(&self) -> &'static str {
        match self {
            Self::Text => "txt",
            Self::Json => "json",
            Self::Html => "html",
            Self::Markdown => "md",
            Self::Binary => "bin"}
    }

    /// Check if output type is structured data
    #[inline(always)]
    pub fn is_structured(&self) -> bool {
        matches!(self, Self::Json)
    }

    /// Check if output type is text-based
    #[inline(always)]
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text | Self::Html | Self::Markdown)
    }
}

impl std::fmt::Display for OutputType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Text => "text",
            Self::Json => "json",
            Self::Html => "html",
            Self::Markdown => "markdown",
            Self::Binary => "binary"};
        write!(f, "{}", name)
    }
}

/// Command execution result
///
/// Represents the various types of results that can be produced by command execution
/// with support for simple messages, structured data, files, and composite results.
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

impl CommandExecutionResult {
    /// Create a success result
    #[inline(always)]
    pub fn success(message: impl Into<String>) -> Self {
        Self::Success(message.into())
    }

    /// Create a data result
    #[inline(always)]
    pub fn data(value: serde_json::Value) -> Self {
        Self::Data(value)
    }

    /// Create a file result
    #[inline(always)]
    pub fn file(path: impl Into<String>, size_bytes: u64, mime_type: impl Into<String>) -> Self {
        Self::File {
            path: path.into(),
            size_bytes,
            mime_type: mime_type.into()}
    }

    /// Create a multiple result
    #[inline(always)]
    pub fn multiple(results: Vec<CommandExecutionResult>) -> Self {
        Self::Multiple(results)
    }

    /// Check if result indicates success
    #[inline(always)]
    pub fn is_success(&self) -> bool {
        true // All results are considered successful; errors are handled separately
    }

    /// Get result summary as string
    pub fn summary(&self) -> String {
        match self {
            Self::Success(msg) => msg.clone(),
            Self::Data(value) => format!("Data result: {} bytes", value.to_string().len()),
            Self::File { path, size_bytes, .. } => format!("File: {} ({} bytes)", path, size_bytes),
            Self::Multiple(results) => format!("Multiple results: {} items", results.len())}
    }
}

/// Search scope enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SearchScope {
    /// Search all available content
    All,
    /// Search current session only
    Current,
    /// Search recent items only
    Recent,
    /// Search bookmarked items only
    Bookmarked}

impl Default for SearchScope {
    fn default() -> Self {
        Self::All
    }
}

/// Template action enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    Use}

/// Macro action enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    Execute}

/// Branch action enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BranchAction {
    /// List available branches
    List,
    /// Create a new branch
    Create,
    /// Switch to a different branch
    Switch,
    /// Merge branches
    Merge,
    /// Delete a branch
    Delete}

/// Session action enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SessionAction {
    /// List available sessions
    List,
    /// Create a new session
    New,
    /// Switch to a different session
    Switch,
    /// Delete a session
    Delete,
    /// Export session data
    Export,
    /// Import session data
    Import}

/// Tool action enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToolAction {
    /// List available tools
    List,
    /// Install a new tool
    Install,
    /// Remove an existing tool
    Remove,
    /// Configure a tool
    Configure,
    /// Update tool to latest version
    Update,
    /// Execute a tool
    Execute}

/// Statistics type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StatsType {
    /// Usage statistics
    Usage,
    /// Performance metrics
    Performance,
    /// History statistics
    History,
    /// Token usage statistics
    Tokens,
    /// Cost analysis
    Costs,
    /// Error statistics
    Errors}

/// Theme action enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    Edit}

/// Debug action enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DebugAction {
    /// Show system information
    Info,
    /// Display logs
    Logs,
    /// Show performance metrics
    Performance,
    /// Display memory usage
    Memory,
    /// Show network activity
    Network,
    /// Display cache information
    Cache}

/// History action enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HistoryAction {
    /// Show history
    Show,
    /// Search history
    Search,
    /// Clear history
    Clear,
    /// Export history
    Export,
    /// Import history
    Import,
    /// Backup history
    Backup}

/// Import type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImportType {
    /// Import chat conversations
    Chat,
    /// Import configuration
    Config,
    /// Import template definitions
    Templates,
    /// Import macro definitions
    Macros}

/// Settings category enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SettingsCategory {
    /// Appearance and visual settings
    Appearance,
    /// Behavior and interaction settings
    Behavior,
    /// Security and privacy settings
    Security,
    /// Performance and optimization settings
    Performance,
    /// Integration and plugin settings
    Integration,
    /// Advanced and developer settings
    Advanced}

impl Default for SettingsCategory {
    fn default() -> Self {
        Self::Behavior
    }
}

// Implementation of Display traits for better ergonomics
impl std::fmt::Display for SearchScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::All => "all",
            Self::Current => "current",
            Self::Recent => "recent",
            Self::Bookmarked => "bookmarked"};
        write!(f, "{}", name)
    }
}

impl std::fmt::Display for TemplateAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::List => "list",
            Self::Create => "create",
            Self::Delete => "delete",
            Self::Edit => "edit",
            Self::Use => "use"};
        write!(f, "{}", name)
    }
}

// Add Display implementations for other action enums following the same pattern
macro_rules! impl_display {
    ($enum_type:ty, $($variant:ident => $str:literal),+) => {
        impl std::fmt::Display for $enum_type {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let name = match self {
                    $(Self::$variant => $str,)+
                };
                write!(f, "{}", name)
            }
        }
    };
}

impl_display!(MacroAction,
    List => "list",
    Create => "create", 
    Delete => "delete",
    Edit => "edit",
    Execute => "execute"
);

impl_display!(BranchAction,
    List => "list",
    Create => "create",
    Switch => "switch", 
    Merge => "merge",
    Delete => "delete"
);

impl_display!(SessionAction,
    List => "list",
    New => "new",
    Switch => "switch",
    Delete => "delete",
    Export => "export",
    Import => "import"
);

impl_display!(ToolAction,
    List => "list",
    Install => "install",
    Remove => "remove",
    Configure => "configure",
    Update => "update",
    Execute => "execute"
);

impl_display!(StatsType,
    Usage => "usage",
    Performance => "performance",
    History => "history",
    Tokens => "tokens",
    Costs => "costs",
    Errors => "errors"
);

impl_display!(ThemeAction,
    Set => "set",
    List => "list",
    Create => "create",
    Export => "export",
    Import => "import",
    Edit => "edit"
);

impl_display!(DebugAction,
    Info => "info",
    Logs => "logs",
    Performance => "performance",
    Memory => "memory",
    Network => "network",
    Cache => "cache"
);

impl_display!(HistoryAction,
    Show => "show",
    Search => "search",
    Clear => "clear",
    Export => "export",
    Import => "import",
    Backup => "backup"
);

impl_display!(ImportType,
    Chat => "chat",
    Config => "config",
    Templates => "templates",
    Macros => "macros"
);

impl_display!(SettingsCategory,
    Appearance => "appearance",
    Behavior => "behavior",
    Security => "security",
    Performance => "performance",
    Integration => "integration",
    Advanced => "advanced"
);