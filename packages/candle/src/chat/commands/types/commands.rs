//! Chat command definitions
//!
//! This module contains the main ImmutableChatCommand enum with all supported
//! command variants and their associated data.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use super::actions::*;

/// Immutable chat command with owned strings (allocated once)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImmutableChatCommand {
    /// Show help information
    Help {
        /// Optional command to get help for
        command: Option<String>,
        /// Show extended help
        extended: bool,
    },
    /// Clear chat history
    Clear {
        /// Confirm the action
        confirm: bool,
        /// Keep last N messages
        keep_last: Option<usize>,
    },
    /// Export conversation
    Export {
        /// Export format (json, markdown, pdf, html)
        format: String,
        /// Output file path
        output: Option<String>,
        /// Include metadata
        include_metadata: bool,
    },
    /// Modify configuration
    Config {
        /// Configuration key
        key: Option<String>,
        /// Configuration value
        value: Option<String>,
        /// Show current configuration
        show: bool,
        /// Reset to defaults
        reset: bool,
    },
    /// Template operations
    Template {
        /// Template action
        action: TemplateAction,
        /// Template name
        name: Option<String>,
        /// Template content
        content: Option<String>,
        /// Template variables
        variables: HashMap<String, String>,
    },
    /// Macro operations
    Macro {
        /// Macro action
        action: MacroAction,
        /// Macro name
        name: Option<String>,
        /// Auto-execute macro
        auto_execute: bool,
        /// Commands to execute in macro
        commands: Vec<String>,
    },
    /// Search chat history
    Search {
        /// Search query
        query: String,
        /// Search scope
        scope: SearchScope,
        /// Maximum results
        limit: Option<usize>,
        /// Include context
        include_context: bool,
    },
    /// Branch conversation
    Branch {
        /// Branch action
        action: BranchAction,
        /// Branch name
        name: Option<String>,
        /// Source branch for merging
        source: Option<String>,
    },
    /// Session management
    Session {
        /// Session action
        action: SessionAction,
        /// Session name
        name: Option<String>,
        /// Include configuration
        include_config: bool,
    },
    /// Tool integration
    Tool {
        /// Tool action
        action: ToolAction,
        /// Tool name
        name: Option<String>,
        /// Tool arguments
        args: HashMap<String, String>,
    },
    /// Statistics and analytics
    Stats {
        /// Statistics type
        stat_type: StatsType,
        /// Time period
        period: Option<String>,
        /// Show detailed breakdown
        detailed: bool,
    },
    /// Theme and appearance
    Theme {
        /// Theme action
        action: ThemeAction,
        /// Theme name
        name: Option<String>,
        /// Theme properties
        properties: HashMap<String, String>,
    },
    /// Debugging and diagnostics
    Debug {
        /// Debug action
        action: DebugAction,
        /// Debug level
        level: Option<String>,
        /// Show system information
        system_info: bool,
    },
    /// Chat history operations
    History {
        /// History action
        action: HistoryAction,
        /// Number of messages to show
        limit: Option<usize>,
        /// Filter criteria
        filter: Option<String>,
    },
    /// Save conversation state
    Save {
        /// Save name
        name: Option<String>,
        /// Include configuration
        include_config: bool,
        /// Save location
        location: Option<String>,
    },
    /// Load conversation state
    Load {
        /// Load name
        name: String,
        /// Merge with current session
        merge: bool,
        /// Load location
        location: Option<String>,
    },
    /// Import data or configuration
    Import {
        /// Import type
        import_type: ImportType,
        /// Source file or URL
        source: String,
        /// Import options
        options: HashMap<String, String>,
    },
    /// Application settings
    Settings {
        /// Setting category
        category: SettingsCategory,
        /// Setting key
        key: Option<String>,
        /// Setting value
        value: Option<String>,
        /// Show current settings
        show: bool,
        /// Reset to defaults
        reset: bool,
    },
    /// Custom command
    Custom {
        /// Command name
        name: String,
        /// Command arguments
        args: HashMap<String, String>,
        /// Command metadata
        metadata: Option<serde_json::Value>,
    },
}

impl ImmutableChatCommand {
    /// Get command name as borrowed string (zero allocation)
    pub fn name(&self) -> &str {
        match self {
            ImmutableChatCommand::Help { .. } => "help",
            ImmutableChatCommand::Clear { .. } => "clear",
            ImmutableChatCommand::Export { .. } => "export",
            ImmutableChatCommand::Config { .. } => "config",
            ImmutableChatCommand::Template { .. } => "template",
            ImmutableChatCommand::Macro { .. } => "macro",
            ImmutableChatCommand::Search { .. } => "search",
            ImmutableChatCommand::Branch { .. } => "branch",
            ImmutableChatCommand::Session { .. } => "session",
            ImmutableChatCommand::Tool { .. } => "tool",
            ImmutableChatCommand::Stats { .. } => "stats",
            ImmutableChatCommand::Theme { .. } => "theme",
            ImmutableChatCommand::Debug { .. } => "debug",
            ImmutableChatCommand::History { .. } => "history",
            ImmutableChatCommand::Save { .. } => "save",
            ImmutableChatCommand::Load { .. } => "load",
            ImmutableChatCommand::Import { .. } => "import",
            ImmutableChatCommand::Settings { .. } => "settings",
            ImmutableChatCommand::Custom { name, .. } => name,
        }
    }

    /// Check if command requires confirmation
    pub fn requires_confirmation(&self) -> bool {
        matches!(
            self,
            ImmutableChatCommand::Clear { confirm: false, .. }
                | ImmutableChatCommand::Branch { action: BranchAction::Delete, .. }
                | ImmutableChatCommand::Session { action: SessionAction::Delete, .. }
                | ImmutableChatCommand::History { action: HistoryAction::Clear, .. }
        )
    }

    /// Get command category for help organization
    pub fn category(&self) -> &'static str {
        match self {
            ImmutableChatCommand::Help { .. } => "Help",
            ImmutableChatCommand::Clear { .. } | ImmutableChatCommand::History { .. } => "History",
            ImmutableChatCommand::Export { .. } | ImmutableChatCommand::Import { .. } => "Import/Export",
            ImmutableChatCommand::Config { .. } | ImmutableChatCommand::Settings { .. } => "Configuration",
            ImmutableChatCommand::Template { .. } | ImmutableChatCommand::Macro { .. } => "Automation",
            ImmutableChatCommand::Search { .. } => "Search",
            ImmutableChatCommand::Branch { .. } | ImmutableChatCommand::Session { .. } => "Session Management",
            ImmutableChatCommand::Tool { .. } => "Tools",
            ImmutableChatCommand::Stats { .. } | ImmutableChatCommand::Debug { .. } => "Diagnostics",
            ImmutableChatCommand::Theme { .. } => "Appearance",
            ImmutableChatCommand::Save { .. } | ImmutableChatCommand::Load { .. } => "State Management",
            ImmutableChatCommand::Custom { .. } => "Custom",
        }
    }

    /// Check if command modifies state
    pub fn modifies_state(&self) -> bool {
        match self {
            ImmutableChatCommand::Help { .. }
            | ImmutableChatCommand::Search { .. }
            | ImmutableChatCommand::Stats { .. }
            | ImmutableChatCommand::History { action: HistoryAction::Show, .. }
            | ImmutableChatCommand::Config { show: true, .. }
            | ImmutableChatCommand::Settings { show: true, .. } => false,
            _ => true,
        }
    }
}