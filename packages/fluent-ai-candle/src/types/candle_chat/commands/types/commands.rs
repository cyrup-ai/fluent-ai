//! Immutable command definitions
//!
//! Core command enum with all supported chat commands using owned strings
//! for zero-allocation performance after initial construction.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use super::core::{CommandError, CommandResult};
use super::events::{
    SearchScope, TemplateAction, MacroAction, BranchAction, SessionAction,
    ToolAction, StatsType, ThemeAction, DebugAction, HistoryAction, ImportType
};
use super::executor::SettingsCategory;

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
    #[inline]
    pub fn command_name(&self) -> &'static str {
        match self {
            Self::Help { .. } => "help",
            Self::Clear { .. } => "clear",
            Self::Export { .. } => "export",
            Self::Config { .. } => "config",
            Self::Template { .. } => "template",
            Self::Macro { .. } => "macro",
            Self::Search { .. } => "search",
            Self::Branch { .. } => "branch",
            Self::Session { .. } => "session",
            Self::Tool { .. } => "tool",
            Self::Stats { .. } => "stats",
            Self::Theme { .. } => "theme",
            Self::Debug { .. } => "debug",
            Self::History { .. } => "history",
            Self::Save { .. } => "save",
            Self::Load { .. } => "load",
            Self::Import { .. } => "import",
            Self::Settings { .. } => "settings",
            Self::Custom { .. } => "custom",
        }
    }

    /// Check if command requires confirmation
    #[inline]
    pub fn requires_confirmation(&self) -> bool {
        matches!(
            self,
            Self::Clear { .. } | Self::Load { .. } | Self::Import { .. }
        )
    }

    /// Check if command modifies state
    #[inline]
    pub fn is_mutating(&self) -> bool {
        matches!(
            self,
            Self::Clear { .. }
                | Self::Config { .. }
                | Self::Template { .. }
                | Self::Macro { .. }
                | Self::Branch { .. }
                | Self::Session { .. }
                | Self::Save { .. }
                | Self::Load { .. }
                | Self::Import { .. }
                | Self::Settings { .. }
        )
    }

    /// Validate command arguments
    #[inline]
    pub fn validate(&self) -> CommandResult<()> {
        match self {
            Self::Export { format, .. } => {
                if !matches!(format.as_str(), "json" | "markdown" | "pdf" | "html") {
                    return Err(CommandError::InvalidArguments(
                        "Invalid export format".to_string(),
                    ));
                }
            }
            Self::Search { query, .. } => {
                if query.is_empty() {
                    return Err(CommandError::InvalidArguments(
                        "Search query cannot be empty".to_string(),
                    ));
                }
            }
            Self::Load { name, .. } => {
                if name.is_empty() {
                    return Err(CommandError::InvalidArguments(
                        "Load name cannot be empty".to_string(),
                    ));
                }
            }
            Self::Import { source, .. } => {
                if source.is_empty() {
                    return Err(CommandError::InvalidArguments(
                        "Import source cannot be empty".to_string(),
                    ));
                }
            }
            Self::Custom { name, .. } => {
                if name.is_empty() {
                    return Err(CommandError::InvalidArguments(
                        "Custom command name cannot be empty".to_string(),
                    ));
                }
            }
            _ => {}
        }
        Ok(())
    }
}