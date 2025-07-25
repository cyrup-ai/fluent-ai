//! Action enums for different command types
//!
//! Provides action enums for various command categories with proper
//! serialization support and zero-allocation patterns.

use serde::{Deserialize, Serialize};

/// Search-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchScope {
    All,
    Current,
    Recent,
    Bookmarked,
}

/// Template-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemplateAction {
    List,
    Create,
    Delete,
    Edit,
    Use,
}

/// Macro-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MacroAction {
    List,
    Create,
    Delete,
    Edit,
    Execute,
}

/// Branch-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BranchAction {
    List,
    Create,
    Switch,
    Merge,
    Delete,
}

/// Session-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionAction {
    List,
    New,
    Switch,
    Delete,
    Export,
    Import,
}

/// Tool-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolAction {
    List,
    Install,
    Remove,
    Configure,
    Update,
    Execute,
}

/// Stats-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatsType {
    Usage,
    Performance,
    History,
    Tokens,
    Costs,
    Errors,
}

/// Theme-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThemeAction {
    Set,
    List,
    Create,
    Export,
    Import,
    Edit,
}

/// Debug-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DebugAction {
    Info,
    Logs,
    Performance,
    Memory,
    Network,
    Cache,
}

/// History-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HistoryAction {
    Show,
    Search,
    Clear,
    Export,
    Import,
    Backup,
}

/// Import-related enums
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImportType {
    Chat,
    Config,
    Templates,
    Macros,
}