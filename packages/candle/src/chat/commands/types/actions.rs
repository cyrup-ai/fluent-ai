//! Action type definitions for command variants
//!
//! This module defines all the action enums used by different command types
//! to specify the specific operation to perform.

use serde::{Deserialize, Serialize};

/// Template operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TemplateAction {
    /// Create a new template
    Create,
    /// Update existing template
    Update,
    /// Delete template
    Delete,
    /// List available templates
    List,
    /// Show template details
    Show,
    /// Apply template
    Apply,
}

/// Macro operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MacroAction {
    /// Create a new macro
    Create,
    /// Update existing macro
    Update,
    /// Delete macro
    Delete,
    /// List available macros
    List,
    /// Execute macro
    Execute,
    /// Show macro details
    Show,
}

/// Search scope for chat history
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SearchScope {
    /// Search all messages
    All,
    /// Search current session only
    Session,
    /// Search user messages only
    User,
    /// Search assistant messages only
    Assistant,
    /// Search system messages only
    System,
    /// Search current conversation only
    Current,
    /// Search recent messages only
    Recent,
    /// Search bookmarked messages only
    Bookmarked,
}

/// Branch operations for conversation management
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BranchAction {
    /// Create new branch
    Create,
    /// Switch to branch
    Switch,
    /// Merge branches
    Merge,
    /// Delete branch
    Delete,
    /// List branches
    List,
}

/// Session management operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SessionAction {
    /// Create new session
    Create,
    /// Switch to session
    Switch,
    /// Delete session
    Delete,
    /// List sessions
    List,
    /// Save session
    Save,
    /// Load session
    Load,
}

/// Tool integration operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolAction {
    /// List available tools
    List,
    /// Execute tool
    Execute,
    /// Configure tool
    Configure,
    /// Install tool
    Install,
    /// Uninstall tool
    Uninstall,
}

/// Statistics types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StatsType {
    /// Usage statistics
    Usage,
    /// Performance metrics
    Performance,
    /// Command history
    Commands,
    /// Memory usage
    Memory,
    /// Token usage
    Tokens,
}

/// Theme operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ThemeAction {
    /// Set theme
    Set,
    /// List available themes
    List,
    /// Create custom theme
    Create,
    /// Delete theme
    Delete,
    /// Reset to default
    Reset,
}

/// Debug operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DebugAction {
    /// Show debug information
    Info,
    /// Enable debug mode
    Enable,
    /// Disable debug mode
    Disable,
    /// Show logs
    Logs,
    /// Clear logs
    Clear,
}

/// History operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HistoryAction {
    /// Show history
    Show,
    /// Clear history
    Clear,
    /// Export history
    Export,
    /// Search history
    Search,
}

/// Import types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImportType {
    /// Import configuration
    Config,
    /// Import conversation
    Conversation,
    /// Import templates
    Templates,
    /// Import macros
    Macros,
    /// Import themes
    Themes,
}

/// Settings categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SettingsCategory {
    /// General settings
    General,
    /// Display settings
    Display,
    /// Behavior settings
    Behavior,
    /// Security settings
    Security,
    /// Integration settings
    Integration,
    /// Advanced settings
    Advanced,
}