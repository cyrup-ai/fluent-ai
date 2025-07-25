//! Core command types and definitions
//!
//! Provides immutable command structures with zero-allocation patterns
//! and efficient command representation for the streaming command system.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{
    error::{CommandError, CommandResult},
    parameter::ParameterInfo};

// Import action enums that will be defined in events.rs
use super::events::{
    TemplateAction, MacroAction, BranchAction, SessionAction, ToolAction, 
    StatsType, ThemeAction, DebugAction, HistoryAction, ImportType, SearchScope,
    SettingsCategory};

/// Command information for command registry with owned strings
///
/// Contains all metadata needed to register, document, and execute commands
/// with efficient owned string storage and comprehensive parameter definitions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CommandInfo {
    /// Command name
    pub name: String,
    /// Command description for help text
    pub description: String,
    /// Usage string showing syntax
    pub usage: String,
    /// Command parameters
    pub parameters: Vec<ParameterInfo>,
    /// Command aliases
    pub aliases: Vec<String>,
    /// Command category for organization
    pub category: String,
    /// Usage examples
    pub examples: Vec<String>}

impl CommandInfo {
    /// Create a new command info
    #[inline(always)]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            usage: String::new(),
            parameters: Vec::new(),
            aliases: Vec::new(),
            category: "general".to_string(),
            examples: Vec::new()}
    }

    /// Set usage string
    #[inline(always)]
    pub fn with_usage(mut self, usage: impl Into<String>) -> Self {
        self.usage = usage.into();
        self
    }

    /// Add parameter
    #[inline(always)]
    pub fn with_parameter(mut self, parameter: ParameterInfo) -> Self {
        self.parameters.push(parameter);
        self
    }

    /// Add alias
    #[inline(always)]
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.aliases.push(alias.into());
        self
    }

    /// Set category
    #[inline(always)]
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = category.into();
        self
    }

    /// Add example
    #[inline(always)]
    pub fn with_example(mut self, example: impl Into<String>) -> Self {
        self.examples.push(example.into());
        self
    }

    /// Generate usage string from parameters
    pub fn generate_usage(&mut self) {
        let mut usage_parts = vec![self.name.clone()];
        
        for param in &self.parameters {
            usage_parts.push(param.usage_string());
        }
        
        self.usage = usage_parts.join(" ");
    }

    /// Check if command matches name or alias
    #[inline(always)]
    pub fn matches(&self, name: &str) -> bool {
        self.name == name || self.aliases.contains(&name.to_string())
    }
}

/// Resource usage tracking for command execution
///
/// Tracks computational resources used during command execution
/// for performance monitoring and optimization.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ResourceUsage {
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// CPU time in microseconds
    pub cpu_time_us: u64,
    /// Number of network requests made
    pub network_requests: u32,
    /// Number of disk operations performed
    pub disk_operations: u32}

impl ResourceUsage {
    /// Create empty resource usage
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add memory usage
    #[inline(always)]
    pub fn add_memory(&mut self, bytes: u64) {
        self.memory_bytes = self.memory_bytes.saturating_add(bytes);
    }

    /// Add CPU time
    #[inline(always)]
    pub fn add_cpu_time(&mut self, microseconds: u64) {
        self.cpu_time_us = self.cpu_time_us.saturating_add(microseconds);
    }

    /// Increment network requests
    #[inline(always)]
    pub fn add_network_request(&mut self) {
        self.network_requests = self.network_requests.saturating_add(1);
    }

    /// Increment disk operations
    #[inline(always)]
    pub fn add_disk_operation(&mut self) {
        self.disk_operations = self.disk_operations.saturating_add(1);
    }

    /// Combine with another resource usage
    #[inline(always)]
    pub fn combine(&mut self, other: &ResourceUsage) {
        self.memory_bytes = self.memory_bytes.saturating_add(other.memory_bytes);
        self.cpu_time_us = self.cpu_time_us.saturating_add(other.cpu_time_us);
        self.network_requests = self.network_requests.saturating_add(other.network_requests);
        self.disk_operations = self.disk_operations.saturating_add(other.disk_operations);
    }

    /// Check if usage is significant
    #[inline(always)]
    pub fn is_significant(&self) -> bool {
        self.memory_bytes > 1024 || self.cpu_time_us > 1000 || 
        self.network_requests > 0 || self.disk_operations > 0
    }
}

/// Immutable chat command with owned strings (allocated once)
///
/// Represents all possible chat commands with their parameters in an immutable
/// structure. Uses owned strings to avoid lifetime issues while maintaining
/// efficient memory usage through strategic allocation patterns.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImmutableChatCommand {
    /// Show help information
    Help {
        /// Optional command to get help for
        command: Option<String>,
        /// Show extended help
        extended: bool},
    
    /// Clear chat history
    Clear {
        /// Confirm the action
        confirm: bool,
        /// Keep last N messages
        keep_last: Option<usize>},
    
    /// Export conversation
    Export {
        /// Export format (json, markdown, pdf, html)
        format: String,
        /// Output file path
        output: Option<String>,
        /// Include metadata
        include_metadata: bool},
    
    /// Modify configuration
    Config {
        /// Configuration key
        key: Option<String>,
        /// Configuration value
        value: Option<String>,
        /// Show current configuration
        show: bool,
        /// Reset to defaults
        reset: bool},
    
    /// Template operations
    Template {
        /// Template action
        action: TemplateAction,
        /// Template name
        name: Option<String>,
        /// Template content
        content: Option<String>,
        /// Template variables
        variables: HashMap<String, String>},
    
    /// Macro operations
    Macro {
        /// Macro action
        action: MacroAction,
        /// Macro name
        name: Option<String>,
        /// Auto-execute macro
        auto_execute: bool,
        /// Commands to execute in macro
        commands: Vec<String>},
    
    /// Search chat history
    Search {
        /// Search query
        query: String,
        /// Search scope
        scope: SearchScope,
        /// Maximum results
        limit: Option<usize>,
        /// Include context
        include_context: bool},
    
    /// Branch conversation
    Branch {
        /// Branch action
        action: BranchAction,
        /// Branch name
        name: Option<String>,
        /// Source branch for merging
        source: Option<String>},
    
    /// Session management
    Session {
        /// Session action
        action: SessionAction,
        /// Session name
        name: Option<String>,
        /// Include configuration
        include_config: bool},
    
    /// Tool integration
    Tool {
        /// Tool action
        action: ToolAction,
        /// Tool name
        name: Option<String>,
        /// Tool arguments
        args: HashMap<String, String>},
    
    /// Statistics and analytics
    Stats {
        /// Statistics type
        stat_type: StatsType,
        /// Time period
        period: Option<String>,
        /// Show detailed breakdown
        detailed: bool},
    
    /// Theme and appearance
    Theme {
        /// Theme action
        action: ThemeAction,
        /// Theme name
        name: Option<String>,
        /// Theme properties
        properties: HashMap<String, String>},
    
    /// Debugging and diagnostics
    Debug {
        /// Debug action
        action: DebugAction,
        /// Debug level
        level: Option<String>,
        /// Show system information
        system_info: bool},
    
    /// Chat history operations
    History {
        /// History action
        action: HistoryAction,
        /// Number of messages to show
        limit: Option<usize>,
        /// Filter criteria
        filter: Option<String>},
    
    /// Save conversation state
    Save {
        /// Save name
        name: Option<String>,
        /// Include configuration
        include_config: bool,
        /// Save location
        location: Option<String>},
    
    /// Load conversation state
    Load {
        /// Load name
        name: String,
        /// Merge with current session
        merge: bool,
        /// Load location
        location: Option<String>},
    
    /// Import data or configuration
    Import {
        /// Import type
        import_type: ImportType,
        /// Source file or URL
        source: String,
        /// Import options
        options: HashMap<String, String>},
    
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
        reset: bool},
    
    /// Custom command
    Custom {
        /// Command name
        name: String,
        /// Command arguments
        args: HashMap<String, String>,
        /// Command metadata
        metadata: Option<serde_json::Value>}}

impl ImmutableChatCommand {
    /// Get command name as borrowed string (zero allocation)
    #[inline(always)]
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
            Self::Custom { .. } => "custom"}
    }

    /// Check if command requires confirmation
    #[inline(always)]
    pub fn requires_confirmation(&self) -> bool {
        matches!(
            self,
            Self::Clear { .. } | Self::Load { .. } | Self::Import { .. }
        )
    }

    /// Check if command modifies state
    #[inline(always)]
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

    /// Check if command is read-only
    #[inline(always)]
    pub fn is_read_only(&self) -> bool {
        !self.is_mutating()
    }

    /// Get command priority for execution ordering
    #[inline(always)]
    pub fn priority(&self) -> u8 {
        match self {
            Self::Help { .. } => 1,
            Self::Debug { .. } => 2,
            Self::Config { .. } | Self::Settings { .. } => 3,
            Self::Clear { .. } => 10, // High priority for destructive actions
            _ => 5, // Normal priority
        }
    }

    /// Validate command arguments
    pub fn validate(&self) -> CommandResult<()> {
        match self {
            Self::Export { format, .. } => {
                if !matches!(format.as_str(), "json" | "markdown" | "pdf" | "html") {
                    return Err(CommandError::invalid_arguments(
                        format!("Invalid export format '{}'. Supported: json, markdown, pdf, html", format)
                    ));
                }
            }
            
            Self::Search { query, .. } => {
                if query.trim().is_empty() {
                    return Err(CommandError::invalid_arguments(
                        "Search query cannot be empty"
                    ));
                }
            }
            
            Self::Load { name, .. } => {
                if name.trim().is_empty() {
                    return Err(CommandError::invalid_arguments(
                        "Load name cannot be empty"
                    ));
                }
            }
            
            Self::Import { source, .. } => {
                if source.trim().is_empty() {
                    return Err(CommandError::invalid_arguments(
                        "Import source cannot be empty"
                    ));
                }
            }
            
            Self::Custom { name, .. } => {
                if name.trim().is_empty() {
                    return Err(CommandError::invalid_arguments(
                        "Custom command name cannot be empty"
                    ));
                }
            }
            
            _ => {}
        }
        Ok(())
    }

    /// Get estimated resource usage for the command
    pub fn estimated_resource_usage(&self) -> ResourceUsage {
        let mut usage = ResourceUsage::new();
        
        match self {
            Self::Export { .. } => {
                usage.add_memory(1024 * 1024); // 1MB for export operations
                usage.add_disk_operation();
            }
            
            Self::Search { .. } => {
                usage.add_memory(512 * 1024); // 512KB for search
            }
            
            Self::Import { .. } => {
                usage.add_memory(2 * 1024 * 1024); // 2MB for import
                usage.add_network_request();
                usage.add_disk_operation();
            }
            
            Self::Load { .. } | Self::Save { .. } => {
                usage.add_memory(256 * 1024); // 256KB for save/load
                usage.add_disk_operation();
            }
            
            _ => {
                usage.add_memory(64 * 1024); // 64KB for basic commands
            }
        }
        
        usage
    }
}