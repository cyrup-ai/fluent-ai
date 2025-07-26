//! Action enums for different command types
//!
//! Provides action enums for various command categories with proper
//! serialization support and zero-allocation patterns.

use serde::{Deserialize, Serialize};

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