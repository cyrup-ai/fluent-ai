//! Command types and enums
//!
//! Provides comprehensive type definitions for the command system with zero-allocation patterns
//! and blazing-fast serialization/deserialization.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Chat command errors
#[derive(Error, Debug, Clone)]
pub enum CommandError {
    #[error("Unknown command: {command}")]
    UnknownCommand { command: Arc<str> },
    #[error("Invalid arguments: {detail}")]
    InvalidArguments { detail: Arc<str> },
    #[error("Execution failed: {reason}")]
    ExecutionFailed { reason: Arc<str> },
    #[error("Permission denied: {command}")]
    PermissionDenied { command: Arc<str> },
    #[error("Parse error: {detail}")]
    ParseError { detail: Arc<str> },
    #[error("Configuration error: {detail}")]
    ConfigurationError { detail: Arc<str> },
    #[error("IO error: {detail}")]
    IoError { detail: Arc<str> },
    #[error("Network error: {detail}")]
    NetworkError { detail: Arc<str> },
}

/// Result type for command operations
pub type CommandResult<T> = Result<T, CommandError>;

/// Chat command types with zero-allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChatCommand {
    /// Show help information
    Help {
        /// Optional command to get help for
        command: Option<Arc<str>>,
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
        format: Arc<str>,
        /// Output file path
        output: Option<Arc<str>>,
        /// Include metadata
        include_metadata: bool,
    },
    /// Modify configuration
    Config {
        /// Configuration key
        key: Option<Arc<str>>,
        /// Configuration value
        value: Option<Arc<str>>,
        /// Show current configuration
        show: bool,
        /// Reset to defaults
        reset: bool,
    },
    /// Template operations
    Template {
        /// Template action (create, use, list, delete)
        action: TemplateAction,
        /// Template name
        name: Option<Arc<str>>,
        /// Template content
        content: Option<Arc<str>>,
        /// Template variables
        variables: HashMap<Arc<str>, Arc<str>>,
    },
    /// Macro operations
    Macro {
        /// Macro action (record, play, list, delete)
        action: MacroAction,
        /// Macro name
        name: Option<Arc<str>>,
        /// Auto-execute macro
        auto_execute: bool,
    },
    /// Search chat history
    Search {
        /// Search query
        query: Arc<str>,
        /// Search scope (all, current, recent)
        scope: SearchScope,
        /// Maximum results
        limit: Option<usize>,
        /// Include context
        include_context: bool,
    },
    /// Branch conversation
    Branch {
        /// Branch action (create, switch, merge, delete)
        action: BranchAction,
        /// Branch name
        name: Option<Arc<str>>,
        /// Source branch for merging
        source: Option<Arc<str>>,
    },
    /// Session management
    Session {
        /// Session action (save, load, list, delete)
        action: SessionAction,
        /// Session name
        name: Option<Arc<str>>,
        /// Include configuration
        include_config: bool,
    },
    /// Tool integration
    Tool {
        /// Tool action (list, install, remove, execute)
        action: ToolAction,
        /// Tool name
        name: Option<Arc<str>>,
        /// Tool arguments
        args: HashMap<Arc<str>, Arc<str>>,
    },
    /// Statistics and analytics
    Stats {
        /// Statistics type (usage, performance, history, tokens, costs, errors)
        stat_type: StatsType,
        /// Time period (day, week, month, all)
        period: Option<Arc<str>>,
        /// Show detailed breakdown
        detailed: bool,
    },
    /// Theme and appearance
    Theme {
        /// Theme action (set, list, create, export)
        action: ThemeAction,
        /// Theme name
        name: Option<Arc<str>>,
        /// Theme properties
        properties: HashMap<Arc<str>, Arc<str>>,
    },
    /// Debugging and diagnostics
    Debug {
        /// Debug action (info, logs, performance, memory)
        action: DebugAction,
        /// Debug level
        level: Option<Arc<str>>,
        /// Show system information
        system_info: bool,
    },
}

/// Template actions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TemplateAction {
    Create,
    Use,
    List,
    Delete,
    Edit,
    Share,
    Import,
    Export,
}

/// Macro actions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MacroAction {
    Record,
    Play,
    List,
    Delete,
    Edit,
    Share,
}

/// Search scope options
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SearchScope {
    All,
    Current,
    Recent,
    Bookmarked,
}

/// Branch actions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BranchAction {
    Create,
    Switch,
    Merge,
    Delete,
    List,
    Rename,
}

/// Session actions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SessionAction {
    Save,
    Load,
    List,
    Delete,
    Rename,
    Export,
    Import,
}

/// Tool actions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolAction {
    List,
    Install,
    Remove,
    Execute,
    Configure,
    Update,
}

/// Statistics types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StatsType {
    Usage,
    Performance,
    History,
    Tokens,
    Costs,
    Errors,
}

/// Theme actions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ThemeAction {
    Set,
    List,
    Create,
    Export,
    Import,
    Edit,
}

/// Debug actions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DebugAction {
    Info,
    Logs,
    Performance,
    Memory,
    Network,
    Cache,
}

/// Command execution context
#[derive(Debug, Clone)]
pub struct CommandContext {
    /// Current user ID
    pub user_id: Arc<str>,
    /// Current session ID
    pub session_id: Arc<str>,
    /// Current working directory
    pub working_directory: Arc<str>,
    /// Environment variables
    pub environment: HashMap<Arc<str>, Arc<str>>,
    /// User permissions
    pub permissions: Vec<Arc<str>>,
    /// Configuration settings
    pub config: HashMap<Arc<str>, Arc<str>>,
}

/// Command execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandOutput {
    /// Whether the command succeeded
    pub success: bool,
    /// Output message
    pub message: Arc<str>,
    /// Optional structured data
    pub data: Option<serde_json::Value>,
    /// Execution time in microseconds
    pub execution_time: u64,
    /// Resource usage statistics
    pub resource_usage: ResourceUsage,
}

/// Resource usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// CPU time in microseconds
    pub cpu_time_us: u64,
    /// Number of network requests
    pub network_requests: u32,
    /// Number of disk operations
    pub disk_operations: u32,
}

/// Command information for registry
#[derive(Debug, Clone)]
pub struct CommandInfo {
    /// Command name
    pub name: Arc<str>,
    /// Command description
    pub description: Arc<str>,
    /// Usage string
    pub usage: Arc<str>,
    /// Parameter information
    pub parameters: Vec<ParameterInfo>,
    /// Command aliases
    pub aliases: Vec<Arc<str>>,
    /// Command category
    pub category: Arc<str>,
    /// Usage examples
    pub examples: Vec<Arc<str>>,
}

/// Parameter information
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    /// Parameter name
    pub name: Arc<str>,
    /// Parameter description
    pub description: Arc<str>,
    /// Parameter type
    pub parameter_type: ParameterType,
    /// Whether parameter is required
    pub required: bool,
    /// Default value if any
    pub default_value: Option<Arc<str>>,
}

/// Parameter types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParameterType {
    String,
    Integer,
    Float,
    Boolean,
    Enum,
    Array,
    Object,
    Path,
    Url,
}

/// Execution metrics for performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    /// Total commands executed
    pub total_commands: u64,
    /// Successful commands
    pub successful_commands: u64,
    /// Failed commands
    pub failed_commands: u64,
    /// Average execution time in microseconds
    pub average_execution_time: u64,
    /// Total execution time in microseconds
    pub total_execution_time: u64,
    /// Memory usage statistics
    pub memory_usage: ResourceUsage,
    /// Most used commands
    pub popular_commands: HashMap<Arc<str>, u64>,
    /// Error statistics
    pub error_counts: HashMap<Arc<str>, u64>,
}

impl Default for CommandContext {
    fn default() -> Self {
        Self {
            user_id: Arc::from("default"),
            session_id: Arc::from("default"),
            working_directory: Arc::from("."),
            environment: HashMap::new(),
            permissions: vec![Arc::from("read"), Arc::from("write")],
            config: HashMap::new(),
        }
    }
}

impl Default for ExecutionMetrics {
    fn default() -> Self {
        Self {
            total_commands: 0,
            successful_commands: 0,
            failed_commands: 0,
            average_execution_time: 0,
            total_execution_time: 0,
            memory_usage: ResourceUsage {
                memory_bytes: 0,
                cpu_time_us: 0,
                network_requests: 0,
                disk_operations: 0,
            },
            popular_commands: HashMap::new(),
            error_counts: HashMap::new(),
        }
    }
}

impl CommandOutput {
    /// Create a successful command output
    pub fn success(message: impl Into<Arc<str>>) -> Self {
        Self {
            success: true,
            message: message.into(),
            data: None,
            execution_time: 0,
            resource_usage: ResourceUsage {
                memory_bytes: 0,
                cpu_time_us: 0,
                network_requests: 0,
                disk_operations: 0,
            },
        }
    }

    /// Create a failed command output
    pub fn error(message: impl Into<Arc<str>>) -> Self {
        Self {
            success: false,
            message: message.into(),
            data: None,
            execution_time: 0,
            resource_usage: ResourceUsage {
                memory_bytes: 0,
                cpu_time_us: 0,
                network_requests: 0,
                disk_operations: 0,
            },
        }
    }

    /// Add structured data to the output
    pub fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = Some(data);
        self
    }

    /// Set execution time
    pub fn with_execution_time(mut self, time_us: u64) -> Self {
        self.execution_time = time_us;
        self
    }

    /// Set resource usage
    pub fn with_resource_usage(mut self, usage: ResourceUsage) -> Self {
        self.resource_usage = usage;
        self
    }
}
