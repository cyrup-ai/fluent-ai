//! Main ImmutableChatCommand enum with zero allocation patterns
//!
//! Provides blazing-fast command enumeration with owned strings allocated once
//! for maximum performance. No Arc usage, no locking, comprehensive validation.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::actions::*;
use super::errors::{CandleCommandError, CommandResult};

/// Settings category enumeration for zero allocation dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SettingsCategory {
    /// General application settings
    General,
    /// User interface settings
    Interface,
    /// Performance and optimization settings
    Performance,
    /// Security and privacy settings
    Security,
    /// Network and connectivity settings
    Network,
    /// Storage and persistence settings
    Storage,
    /// Logging and debugging settings
    Logging,
    /// Integration and plugin settings
    Integrations,
    /// Appearance and theming settings
    Appearance,
    /// Accessibility settings
    Accessibility,
}

impl SettingsCategory {
    /// Get category name as static string for zero allocation
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::General => "general",
            Self::Interface => "interface",
            Self::Performance => "performance",
            Self::Security => "security",
            Self::Network => "network",
            Self::Storage => "storage",
            Self::Logging => "logging",
            Self::Integrations => "integrations",
            Self::Appearance => "appearance",
            Self::Accessibility => "accessibility",
        }
    }
}

/// Output type enumeration for zero allocation dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputType {
    /// Plain text output
    Text,
    /// JSON structured data
    Json,
    /// Markdown formatted text
    Markdown,
    /// HTML formatted output
    Html,
    /// CSV tabular data
    Csv,
    /// XML structured data
    Xml,
    /// YAML data format
    Yaml,
    /// Binary data
    Binary,
    /// Image data
    Image,
    /// Audio data
    Audio,
    /// Video data
    Video,
    /// Log entries
    Log,
    /// Error messages
    Error,
}

impl OutputType {
    /// Get type name as static string for zero allocation
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Json => "json",
            Self::Markdown => "markdown",
            Self::Html => "html",
            Self::Csv => "csv",
            Self::Xml => "xml",
            Self::Yaml => "yaml",
            Self::Binary => "binary",
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::Log => "log",
            Self::Error => "error",
        }
    }

    /// Get MIME type for HTTP responses
    #[inline]
    pub const fn mime_type(&self) -> &'static str {
        match self {
            Self::Text => "text/plain",
            Self::Json => "application/json",
            Self::Markdown => "text/markdown",
            Self::Html => "text/html",
            Self::Csv => "text/csv",
            Self::Xml => "application/xml",
            Self::Yaml => "application/yaml",
            Self::Binary => "application/octet-stream",
            Self::Image => "image/png",
            Self::Audio => "audio/mpeg",
            Self::Video => "video/mp4",
            Self::Log => "text/plain",
            Self::Error => "text/plain",
        }
    }
}

/// Command execution result with zero allocation patterns where possible
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandExecutionResult {
    /// Simple success message (owned string allocated once)
    Success(String),
    /// Data result with structured output
    Data(serde_json::Value),
    /// File result with path and metadata (owned strings allocated once)
    File {
        /// File path
        path: String,
        /// File size in bytes
        size_bytes: u64,
        /// MIME type of the file
        mime_type: String,
    },
    /// Multiple results (owned collection allocated once)
    Multiple(Vec<CommandExecutionResult>),
    /// Stream result for continuous output
    Stream {
        /// Stream identifier
        stream_id: String,
        /// Stream type
        stream_type: OutputType,
        /// Initial data if available
        initial_data: Option<String>,
    },
    /// Error result (owned string allocated once)
    Error(String),
}

impl CommandExecutionResult {
    /// Create success result with zero allocation constructor
    #[inline]
    pub fn success(message: impl Into<String>) -> Self {
        Self::Success(message.into())
    }

    /// Create data result with JSON value
    #[inline]
    pub fn data(value: serde_json::Value) -> Self {
        Self::Data(value)
    }

    /// Create file result with zero allocation constructor
    #[inline]
    pub fn file(path: impl Into<String>, size_bytes: u64, mime_type: impl Into<String>) -> Self {
        Self::File {
            path: path.into(),
            size_bytes,
            mime_type: mime_type.into(),
        }
    }

    /// Create multiple results
    #[inline]
    pub fn multiple(results: Vec<CommandExecutionResult>) -> Self {
        Self::Multiple(results)
    }

    /// Create stream result with zero allocation constructor
    #[inline]
    pub fn stream(
        stream_id: impl Into<String>,
        stream_type: OutputType,
        initial_data: Option<String>,
    ) -> Self {
        Self::Stream {
            stream_id: stream_id.into(),
            stream_type,
            initial_data,
        }
    }

    /// Create error result with zero allocation constructor
    #[inline]
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error(message.into())
    }

    /// Check if result indicates success
    #[inline]
    pub fn is_success(&self) -> bool {
        matches!(
            self,
            Self::Success(_)
                | Self::Data(_)
                | Self::File { .. }
                | Self::Multiple(_)
                | Self::Stream { .. }
        )
    }

    /// Check if result indicates error
    #[inline]
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }
}

/// Immutable chat command with owned strings allocated once for blazing-fast performance
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImmutableChatCommand {
    /// Show help information
    Help {
        /// Optional command to get help for (owned string allocated once)
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
        /// Export format (json, markdown, pdf, html) - owned string allocated once
        format: String,
        /// Output file path - owned string allocated once
        output: Option<String>,
        /// Include metadata
        include_metadata: bool,
    },
    /// Modify configuration
    Config {
        /// Configuration key - owned string allocated once
        key: Option<String>,
        /// Configuration value - owned string allocated once
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
        /// Template name - owned string allocated once
        name: Option<String>,
        /// Template content - owned string allocated once
        content: Option<String>,
        /// Template variables - owned map allocated once
        variables: HashMap<String, String>,
    },
    /// Macro operations
    Macro {
        /// Macro action
        action: MacroAction,
        /// Macro name - owned string allocated once
        name: Option<String>,
        /// Auto-execute macro
        auto_execute: bool,
        /// Commands to execute in macro - owned strings allocated once
        commands: Vec<String>,
    },
    /// Search chat history
    Search {
        /// Search query - owned string allocated once
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
        /// Branch name - owned string allocated once
        name: Option<String>,
        /// Source branch for merging - owned string allocated once
        source: Option<String>,
    },
    /// Session management
    Session {
        /// Session action
        action: SessionAction,
        /// Session name - owned string allocated once
        name: Option<String>,
        /// Include configuration
        include_config: bool,
    },
    /// Tool integration
    Tool {
        /// Tool action
        action: ToolAction,
        /// Tool name - owned string allocated once
        name: Option<String>,
        /// Tool arguments - owned map allocated once
        args: HashMap<String, String>,
    },
    /// Statistics and analytics
    Stats {
        /// Statistics type
        stat_type: StatsType,
        /// Time period - owned string allocated once
        period: Option<String>,
        /// Show detailed breakdown
        detailed: bool,
    },
    /// Theme and appearance
    Theme {
        /// Theme action
        action: ThemeAction,
        /// Theme name - owned string allocated once
        name: Option<String>,
        /// Theme properties - owned map allocated once
        properties: HashMap<String, String>,
    },
    /// Debugging and diagnostics
    Debug {
        /// Debug action
        action: DebugAction,
        /// Debug level - owned string allocated once
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
        /// Filter criteria - owned string allocated once
        filter: Option<String>,
    },
    /// Save conversation state
    Save {
        /// Save name - owned string allocated once
        name: Option<String>,
        /// Include configuration
        include_config: bool,
        /// Save location - owned string allocated once
        location: Option<String>,
    },
    /// Load conversation state
    Load {
        /// Load name - owned string allocated once
        name: String,
        /// Merge with current session
        merge: bool,
        /// Load location - owned string allocated once
        location: Option<String>,
    },
    /// Import data or configuration
    Import {
        /// Import type
        import_type: ImportType,
        /// Source file or URL - owned string allocated once
        source: String,
        /// Import options - owned map allocated once
        options: HashMap<String, String>,
    },
    /// Application settings
    Settings {
        /// Setting category
        category: SettingsCategory,
        /// Setting key - owned string allocated once
        key: Option<String>,
        /// Setting value - owned string allocated once
        value: Option<String>,
        /// Show current settings
        show: bool,
        /// Reset to defaults
        reset: bool,
    },
    /// Custom command
    Custom {
        /// Command name - owned string allocated once
        name: String,
        /// Command arguments - owned map allocated once
        args: HashMap<String, String>,
        /// Command metadata
        metadata: Option<serde_json::Value>,
    },
    /// Copy message or content
    Copy {
        /// Message ID to copy - owned string allocated once
        message_id: Option<String>,
        /// Content to copy - owned string allocated once
        content: Option<String>,
        /// Copy format
        format: OutputType,
    },
    /// Retry last command or message
    Retry {
        /// Command to retry - owned string allocated once
        command: Option<String>,
        /// Number of retry attempts
        attempts: Option<u32>,
        /// Retry with modifications
        modify: bool,
    },
    /// Undo last action
    Undo {
        /// Number of actions to undo
        count: Option<usize>,
        /// Confirm undo action
        confirm: bool,
        /// Show undo preview
        preview: bool,
    },
    /// Direct chat message
    Chat {
        /// Message content - owned string allocated once
        message: String,
        /// Message context - owned string allocated once
        context: Option<String>,
        /// Message priority
        priority: u8,
    },
}

impl ImmutableChatCommand {
    /// Get command name as static string for zero allocation dispatch
    #[inline]
    pub const fn command_name(&self) -> &'static str {
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
            Self::Copy { .. } => "copy",
            Self::Retry { .. } => "retry",
            Self::Undo { .. } => "undo",
            Self::Chat { .. } => "chat",
        }
    }

    /// Check if command requires confirmation - zero allocation check
    #[inline]
    pub const fn requires_confirmation(&self) -> bool {
        matches!(
            self,
            Self::Clear { .. }
                | Self::Load { .. }
                | Self::Import { .. }
                | Self::Undo { .. }
                | Self::Settings { reset: true, .. }
        )
    }

    /// Check if command modifies state - zero allocation check
    #[inline]
    pub const fn is_mutating(&self) -> bool {
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
                | Self::Undo { .. }
        )
    }

    /// Check if command requires network access - zero allocation check
    #[inline]
    pub const fn requires_network(&self) -> bool {
        matches!(
            self,
            Self::Import { .. } | Self::Tool { .. } | Self::Stats { .. }
        )
    }

    /// Check if command can be executed offline - zero allocation check
    #[inline]
    pub const fn is_offline_capable(&self) -> bool {
        !self.requires_network()
    }

    /// Get command priority for execution scheduling - zero allocation
    #[inline]
    pub const fn priority(&self) -> u8 {
        match self {
            Self::Chat { priority, .. } => *priority,
            Self::Help { .. } => 1, // High priority
            Self::Clear { .. } => 2,
            Self::Undo { .. } => 2,
            Self::Config { .. } => 3,
            Self::Settings { .. } => 3,
            Self::Debug { .. } => 4,
            Self::Stats { .. } => 5,
            _ => 10, // Standard priority
        }
    }

    /// Validate command arguments with comprehensive error checking
    #[inline]
    pub fn validate(&self) -> CommandResult<()> {
        match self {
            Self::Export { format, .. } => {
                if !matches!(
                    format.as_str(),
                    "json" | "markdown" | "pdf" | "html" | "csv" | "xml" | "yaml"
                ) {
                    return Err(CandleCommandError::invalid_arguments(
                        format!("Invalid export format '{}'. Supported: json, markdown, pdf, html, csv, xml, yaml", format)
                    ));
                }
            }
            Self::Search { query, .. } => {
                if query.is_empty() {
                    return Err(CandleCommandError::invalid_arguments(
                        "Search query cannot be empty",
                    ));
                }
                if query.len() > 1000 {
                    return Err(CandleCommandError::invalid_arguments(
                        "Search query too long (max 1000 characters)",
                    ));
                }
            }
            Self::Load { name, .. } => {
                if name.is_empty() {
                    return Err(CandleCommandError::invalid_arguments(
                        "Load name cannot be empty",
                    ));
                }
                if name.len() > 255 {
                    return Err(CandleCommandError::invalid_arguments(
                        "Load name too long (max 255 characters)",
                    ));
                }
            }
            Self::Import { source, .. } => {
                if source.is_empty() {
                    return Err(CandleCommandError::invalid_arguments(
                        "Import source cannot be empty",
                    ));
                }
                // Basic URL/path validation
                if !source.starts_with("http://")
                    && !source.starts_with("https://")
                    && !source.starts_with("file://")
                    && !source.starts_with("/")
                    && !source.starts_with("./")
                    && !source.starts_with("../")
                {
                    return Err(CandleCommandError::invalid_arguments(
                        "Import source must be a valid URL or file path",
                    ));
                }
            }
            Self::Custom { name, .. } => {
                if name.is_empty() {
                    return Err(CandleCommandError::invalid_arguments(
                        "Custom command name cannot be empty",
                    ));
                }
                if name.len() > 100 {
                    return Err(CandleCommandError::invalid_arguments(
                        "Custom command name too long (max 100 characters)",
                    ));
                }
                // Check for valid command name characters
                if !name
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
                {
                    return Err(CandleCommandError::invalid_arguments(
                        "Custom command name can only contain alphanumeric characters, underscores, and hyphens"
                    ));
                }
            }
            Self::Chat { message, .. } => {
                if message.is_empty() {
                    return Err(CandleCommandError::invalid_arguments(
                        "Chat message cannot be empty",
                    ));
                }
                if message.len() > 100_000 {
                    return Err(CandleCommandError::invalid_arguments(
                        "Chat message too long (max 100,000 characters)",
                    ));
                }
            }
            Self::Template {
                name: Some(name), ..
            }
            | Self::Macro {
                name: Some(name), ..
            }
            | Self::Branch {
                name: Some(name), ..
            }
            | Self::Session {
                name: Some(name), ..
            }
            | Self::Tool {
                name: Some(name), ..
            }
            | Self::Theme {
                name: Some(name), ..
            } => {
                if name.is_empty() {
                    return Err(CandleCommandError::invalid_arguments(format!(
                        "{} name cannot be empty",
                        self.command_name()
                    )));
                }
                if name.len() > 100 {
                    return Err(CandleCommandError::invalid_arguments(format!(
                        "{} name too long (max 100 characters)",
                        self.command_name()
                    )));
                }
            }
            Self::History {
                limit: Some(limit), ..
            } => {
                if *limit == 0 {
                    return Err(CandleCommandError::invalid_arguments(
                        "History limit must be greater than 0",
                    ));
                }
                if *limit > 10_000 {
                    return Err(CandleCommandError::invalid_arguments(
                        "History limit too large (max 10,000)",
                    ));
                }
            }
            Self::Undo {
                count: Some(count), ..
            } => {
                if *count == 0 {
                    return Err(CandleCommandError::invalid_arguments(
                        "Undo count must be greater than 0",
                    ));
                }
                if *count > 100 {
                    return Err(CandleCommandError::invalid_arguments(
                        "Undo count too large (max 100)",
                    ));
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Get estimated execution time in milliseconds for scheduling
    #[inline]
    pub const fn estimated_duration_ms(&self) -> u64 {
        match self {
            Self::Help { .. } => 100,
            Self::Clear { .. } => 50,
            Self::Export { .. } => 5000,
            Self::Search { .. } => 2000,
            Self::Import { .. } => 10000,
            Self::Stats { .. } => 3000,
            Self::Debug { .. } => 1000,
            Self::History { .. } => 1000,
            Self::Save { .. } => 2000,
            Self::Load { .. } => 3000,
            Self::Chat { .. } => 5000,
            _ => 1000,
        }
    }

    /// Get memory requirements in bytes for resource planning
    #[inline]
    pub const fn memory_requirement_bytes(&self) -> u64 {
        match self {
            Self::Export { .. } => 50 * 1024 * 1024, // 50MB for large exports
            Self::Import { .. } => 100 * 1024 * 1024, // 100MB for imports
            Self::Stats { .. } => 10 * 1024 * 1024,  // 10MB for statistics
            Self::History { .. } => 20 * 1024 * 1024, // 20MB for large history
            Self::Search { .. } => 5 * 1024 * 1024,  // 5MB for search operations
            _ => 1024 * 1024,                        // 1MB for standard operations
        }
    }

    /// Check if command can be parallelized with other commands
    #[inline]
    pub const fn is_parallelizable(&self) -> bool {
        matches!(
            self,
            Self::Help { .. }
                | Self::Stats { .. }
                | Self::Debug { .. }
                | Self::History { .. }
                | Self::Search { .. }
        )
    }
}
