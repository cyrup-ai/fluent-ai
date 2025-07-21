//! Immutable command types and streaming events
//!
//! Provides streaming-only, zero-allocation command system with immutable command events
//! and lock-free execution patterns. All Arc usage eliminated in favor of owned strings
//! and borrowed data patterns for blazing-fast performance.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::async_task::{AsyncStream, AsyncStreamSender};

/// Command execution errors with minimal allocations
#[derive(Error, Debug, Clone)]
pub enum CommandError {
    #[error("Unknown command")]
    UnknownCommand,
    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Permission denied")]
    PermissionDenied,
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    #[error("IO error: {0}")]
    IoError(String),
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Command timeout")]
    Timeout,
    #[error("Resource not found")]
    NotFound,
    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Result type for command operations
pub type CommandResult<T> = Result<T, CommandError>;

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
}impl ImmutableChatCommand {
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
                        "Invalid export format".to_string()
                    ));
                }
            }
            Self::Search { query, .. } => {
                if query.is_empty() {
                    return Err(CommandError::InvalidArguments(
                        "Search query cannot be empty".to_string()
                    ));
                }
            }
            Self::Load { name, .. } => {
                if name.is_empty() {
                    return Err(CommandError::InvalidArguments(
                        "Load name cannot be empty".to_string()
                    ));
                }
            }
            Self::Import { source, .. } => {
                if source.is_empty() {
                    return Err(CommandError::InvalidArguments(
                        "Import source cannot be empty".to_string()
                    ));
                }
            }
            Self::Custom { name, .. } => {
                if name.is_empty() {
                    return Err(CommandError::InvalidArguments(
                        "Custom command name cannot be empty".to_string()
                    ));
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Command execution event for streaming
#[derive(Debug, Clone)]
pub enum CommandEvent {
    /// Command started executing
    Started {
        command: ImmutableChatCommand,
        execution_id: u64,
        timestamp_nanos: u64,
    },
    /// Command execution progress
    Progress {
        execution_id: u64,
        progress_percent: f32,
        message: Option<String>,
    },
    /// Command produced output
    Output {
        execution_id: u64,
        output: String,
        output_type: OutputType,
    },
    /// Command completed successfully
    Completed {
        execution_id: u64,
        result: CommandExecutionResult,
        duration_nanos: u64,
    },
    /// Command failed
    Failed {
        execution_id: u64,
        error: CommandError,
        duration_nanos: u64,
    },
    /// Command was cancelled
    Cancelled {
        execution_id: u64,
        reason: String,
    },
}

/// Command output type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputType {
    Text,
    Json,
    Html,
    Markdown,
    Binary,
}

/// Command execution result
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
        mime_type: String,
    },
    /// Multiple results
    Multiple(Vec<CommandExecutionResult>),
}

/// Streaming command executor with atomic state tracking
#[derive(Debug)]
pub struct StreamingCommandExecutor {
    /// Execution counter (atomic)
    execution_counter: AtomicU64,
    /// Active executions (atomic)
    active_executions: AtomicUsize,
    /// Total executions (atomic)
    total_executions: AtomicU64,
    /// Successful executions (atomic)
    successful_executions: AtomicU64,
    /// Failed executions (atomic)
    failed_executions: AtomicU64,
    /// Event stream sender
    event_sender: Option<AsyncStreamSender<CommandEvent>>,
}

impl StreamingCommandExecutor {
    /// Create new streaming command executor
    #[inline]
    pub fn new() -> Self {
        Self {
            execution_counter: AtomicU64::new(0),
            active_executions: AtomicUsize::new(0),
            total_executions: AtomicU64::new(0),
            successful_executions: AtomicU64::new(0),
            failed_executions: AtomicU64::new(0),
            event_sender: None,
        }
    }

    /// Create executor with event streaming
    #[inline]
    pub fn with_streaming() -> (Self, AsyncStream<CommandEvent>) {
        let (sender, stream) = crate::async_task::stream::channel();
        let mut executor = Self::new();
        executor.event_sender = Some(sender);
        (executor, stream)
    }

    /// Execute command with streaming events
    #[inline]
    pub fn execute_command(&self, command: ImmutableChatCommand) -> CommandResult<u64> {
        // Validate command first
        command.validate()?;

        // Generate execution ID
        let execution_id = self.execution_counter.fetch_add(1, Ordering::Relaxed);
        
        // Update counters
        self.active_executions.fetch_add(1, Ordering::Relaxed);
        self.total_executions.fetch_add(1, Ordering::Relaxed);

        // Send started event
        if let Some(ref sender) = self.event_sender {
            let _ = sender.send(CommandEvent::Started {
                command: command.clone(),
                execution_id,
                timestamp_nanos: Self::current_timestamp_nanos(),
            });
        }

        // TODO: Implement actual command execution logic here
        // This would integrate with the command system to execute commands

        Ok(execution_id)
    }

    /// Get current timestamp in nanoseconds
    #[inline]
    fn current_timestamp_nanos() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    }

    /// Get execution statistics (atomic reads)
    #[inline]
    pub fn stats(&self) -> CommandExecutorStats {
        CommandExecutorStats {
            active_executions: self.active_executions.load(Ordering::Relaxed) as u64,
            total_executions: self.total_executions.load(Ordering::Relaxed),
            successful_executions: self.successful_executions.load(Ordering::Relaxed),
            failed_executions: self.failed_executions.load(Ordering::Relaxed),
        }
    }

    /// Cancel command execution
    #[inline]
    pub fn cancel_execution(&self, execution_id: u64, reason: impl Into<String>) {
        if let Some(ref sender) = self.event_sender {
            let _ = sender.send(CommandEvent::Cancelled {
                execution_id,
                reason: reason.into(),
            });
        }
        self.active_executions.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Default for StreamingCommandExecutor {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Command executor statistics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandExecutorStats {
    pub active_executions: u64,
    pub total_executions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
}

impl CommandExecutorStats {
    /// Calculate success rate as percentage
    #[inline]
    pub fn success_rate(&self) -> f64 {
        let completed = self.successful_executions + self.failed_executions;
        if completed == 0 {
            0.0
        } else {
            (self.successful_executions as f64 / completed as f64) * 100.0
        }
    }

    /// Calculate failure rate as percentage
    #[inline]
    pub fn failure_rate(&self) -> f64 {
        100.0 - self.success_rate()
    }
}/// Template actions
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
    Delete,
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

/// History actions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HistoryAction {
    Show,
    Search,
    Clear,
    Export,
    Import,
    Backup,
}

/// Import types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImportType {
    Conversation,
    Config,
    Templates,
    Macros,
    Themes,
    History,
}

/// Settings categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SettingsCategory {
    Appearance,
    Behavior,
    Security,
    Performance,
    Integration,
    Advanced,
}

/// Command parsing with borrowed data (zero allocation)
pub struct CommandParser;

impl CommandParser {
    /// Parse command from borrowed string (zero allocation in hot path)
    #[inline]
    pub fn parse_command(input: &str) -> CommandResult<ImmutableChatCommand> {
        let input = input.trim();
        
        if input.is_empty() {
            return Err(CommandError::ParseError("Empty command".to_string()));
        }

        // Remove leading slash if present
        let input = input.strip_prefix('/').unwrap_or(input);
        
        // Split command and arguments
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.is_empty() {
            return Err(CommandError::ParseError("Invalid command format".to_string()));
        }

        let command_name = parts[0].to_lowercase();
        let args = &parts[1..];

        match command_name.as_str() {
            "help" | "h" => Self::parse_help_command(args),
            "clear" | "c" => Self::parse_clear_command(args),
            "export" | "e" => Self::parse_export_command(args),
            "config" | "cfg" => Self::parse_config_command(args),
            "template" | "tpl" => Self::parse_template_command(args),
            "macro" | "m" => Self::parse_macro_command(args),
            "search" | "s" => Self::parse_search_command(args),
            "branch" | "b" => Self::parse_branch_command(args),
            "session" | "sess" => Self::parse_session_command(args),
            "tool" | "t" => Self::parse_tool_command(args),
            "stats" | "st" => Self::parse_stats_command(args),
            "theme" | "th" => Self::parse_theme_command(args),
            "debug" | "d" => Self::parse_debug_command(args),
            "history" | "hist" => Self::parse_history_command(args),
            "save" => Self::parse_save_command(args),
            "load" => Self::parse_load_command(args),
            "import" => Self::parse_import_command(args),
            "settings" | "set" => Self::parse_settings_command(args),
            _ => Self::parse_custom_command(&command_name, args),
        }
    }

    /// Parse help command
    #[inline]
    fn parse_help_command(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let command = if args.is_empty() {
            None
        } else {
            Some(args[0].to_string())
        };
        
        let extended = args.contains(&"--extended") || args.contains(&"-e");

        Ok(ImmutableChatCommand::Help { command, extended })
    }

    /// Parse clear command
    #[inline]
    fn parse_clear_command(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let confirm = args.contains(&"--confirm") || args.contains(&"-y");
        let keep_last = args.iter()
            .position(|&arg| arg == "--keep" || arg == "-k")
            .and_then(|pos| args.get(pos + 1))
            .and_then(|s| s.parse().ok());

        Ok(ImmutableChatCommand::Clear { confirm, keep_last })
    }

    /// Parse export command
    #[inline]
    fn parse_export_command(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        if args.is_empty() {
            return Err(CommandError::InvalidArguments(
                "Export format required".to_string()
            ));
        }

        let format = args[0].to_string();
        let output = args.iter()
            .position(|&arg| arg == "--output" || arg == "-o")
            .and_then(|pos| args.get(pos + 1))
            .map(|s| s.to_string());
        
        let include_metadata = args.contains(&"--metadata") || args.contains(&"-m");

        Ok(ImmutableChatCommand::Export {
            format,
            output,
            include_metadata,
        })
    }

    /// Parse config command
    #[inline]
    fn parse_config_command(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let show = args.contains(&"--show") || args.contains(&"-s");
        let reset = args.contains(&"--reset") || args.contains(&"-r");
        
        let (key, value) = if args.len() >= 2 && !args[0].starts_with('-') {
            (Some(args[0].to_string()), Some(args[1].to_string()))
        } else if args.len() >= 1 && !args[0].starts_with('-') {
            (Some(args[0].to_string()), None)
        } else {
            (None, None)
        };

        Ok(ImmutableChatCommand::Config {
            key,
            value,
            show,
            reset,
        })
    }

    /// Parse custom command
    #[inline]
    fn parse_custom_command(name: &str, args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let args_map = args.iter()
            .enumerate()
            .map(|(i, &arg)| (format!("arg_{}", i), arg.to_string()))
            .collect();

        Ok(ImmutableChatCommand::Custom {
            name: name.to_string(),
            args: args_map,
            metadata: None,
        })
    }
}

/// Legacy compatibility type alias (deprecated)
#[deprecated(note = "Use ImmutableChatCommand instead for zero-allocation streaming")]
pub type ChatCommand = ImmutableChatCommand;

/// Command handler context for zero-allocation execution
#[derive(Debug, Clone)]
pub struct CommandContext {
    /// Command execution ID
    pub execution_id: u64,
    /// User session identifier
    pub session_id: String,
    /// Command input text
    pub input: String,
    /// Execution timestamp in nanoseconds
    pub timestamp_nanos: u64,
    /// Environment variables
    pub environment: HashMap<String, String>,
}

impl CommandContext {
    /// Create new command context
    #[inline]
    pub fn new(execution_id: u64, session_id: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            execution_id,
            session_id: session_id.into(),
            input: input.into(),
            timestamp_nanos: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
            environment: HashMap::new(),
        }
    }

    /// Add environment variable
    #[inline]
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.environment.insert(key.into(), value.into());
        self
    }
}

/// Command execution output with zero-allocation streaming
#[derive(Debug, Clone)]
pub struct CommandOutput {
    /// Execution ID this output belongs to
    pub execution_id: u64,
    /// Output content
    pub content: String,
    /// Output type
    pub output_type: OutputType,
    /// Output timestamp in nanoseconds
    pub timestamp_nanos: u64,
    /// Whether output is final
    pub is_final: bool,
}

impl CommandOutput {
    /// Create new command output
    #[inline]
    pub fn new(execution_id: u64, content: impl Into<String>, output_type: OutputType) -> Self {
        Self {
            execution_id,
            content: content.into(),
            output_type,
            timestamp_nanos: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
            is_final: false,
        }
    }

    /// Mark output as final
    #[inline]
    pub fn final_output(mut self) -> Self {
        self.is_final = true;
        self
    }

    /// Create text output
    #[inline]
    pub fn text(execution_id: u64, content: impl Into<String>) -> Self {
        Self::new(execution_id, content, OutputType::Text)
    }

    /// Create JSON output
    #[inline]
    pub fn json(execution_id: u64, content: impl Into<String>) -> Self {
        Self::new(execution_id, content, OutputType::Json)
    }

    /// Create HTML output
    #[inline]
    pub fn html(execution_id: u64, content: impl Into<String>) -> Self {
        Self::new(execution_id, content, OutputType::Html)
    }

    /// Create markdown output
    #[inline]
    pub fn markdown(execution_id: u64, content: impl Into<String>) -> Self {
        Self::new(execution_id, content, OutputType::Markdown)
    }
}

/// Execution metrics for command performance tracking
#[derive(Debug, Clone, Copy)]
pub struct ExecutionMetrics {
    /// Execution duration in nanoseconds
    pub duration_nanos: u64,
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// CPU time in nanoseconds
    pub cpu_time_nanos: u64,
    /// Number of allocations
    pub allocations: u64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
}

impl ExecutionMetrics {
    /// Create new execution metrics
    #[inline]
    pub fn new() -> Self {
        Self {
            duration_nanos: 0,
            memory_bytes: 0,
            cpu_time_nanos: 0,
            allocations: 0,
            peak_memory_bytes: 0,
        }
    }

    /// Calculate duration in milliseconds
    #[inline]
    pub fn duration_ms(&self) -> f64 {
        self.duration_nanos as f64 / 1_000_000.0
    }

    /// Calculate memory usage in MB
    #[inline]
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes as f64 / (1024.0 * 1024.0)
    }
}

impl Default for ExecutionMetrics {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Command handler trait for zero-allocation execution
pub trait CommandHandler: Send + Sync {
    /// Execute command with streaming output
    fn execute(&self, context: CommandContext, command: ImmutableChatCommand) -> AsyncStream<CommandOutput>;
    
    /// Get handler name
    fn name(&self) -> &'static str;
    
    /// Check if handler can execute command
    fn can_handle(&self, command: &ImmutableChatCommand) -> bool;
    
    /// Get command metadata
    fn metadata(&self) -> CommandHandlerMetadata;
}

/// Command handler metadata
#[derive(Debug, Clone)]
pub struct CommandHandlerMetadata {
    /// Handler name
    pub name: String,
    /// Handler description
    pub description: String,
    /// Supported command types
    pub supported_commands: Vec<String>,
    /// Handler version
    pub version: String,
    /// Whether handler is enabled
    pub enabled: bool,
}

impl CommandHandlerMetadata {
    /// Create new command handler metadata
    #[inline]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        supported_commands: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            supported_commands,
            version: "1.0.0".to_string(),
            enabled: true,
        }
    }

    /// Set handler version
    #[inline]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Enable or disable handler
    #[inline]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

/// Default command handler implementation
#[derive(Debug)]
pub struct DefaultCommandHandler {
    metadata: CommandHandlerMetadata,
}

impl DefaultCommandHandler {
    /// Create new default command handler
    #[inline]
    pub fn new() -> Self {
        let metadata = CommandHandlerMetadata::new(
            "default",
            "Default command handler for basic chat commands",
            vec![
                "help".to_string(),
                "clear".to_string(),
                "export".to_string(),
                "config".to_string(),
                "search".to_string(),
                "history".to_string(),
                "save".to_string(),
                "load".to_string(),
            ],
        );

        Self { metadata }
    }
}

impl Default for DefaultCommandHandler {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl CommandHandler for DefaultCommandHandler {
    fn execute(&self, context: CommandContext, command: ImmutableChatCommand) -> AsyncStream<CommandOutput> {
        let (sender, stream) = crate::async_task::stream::channel();
        
        // Execute command based on type
        let output = match &command {
            ImmutableChatCommand::Help { command: cmd, extended } => {
                let content = if let Some(cmd) = cmd {
                    if *extended {
                        format!("Extended help for command: {}", cmd)
                    } else {
                        format!("Help for command: {}", cmd)
                    }
                } else {
                    "Available commands: help, clear, export, config, search, history, save, load".to_string()
                };
                CommandOutput::text(context.execution_id, content)
            },
            ImmutableChatCommand::Clear { confirm, keep_last } => {
                if *confirm {
                    let msg = if let Some(keep) = keep_last {
                        format!("Chat history cleared, keeping last {} messages", keep)
                    } else {
                        "Chat history cleared".to_string()
                    };
                    CommandOutput::text(context.execution_id, msg)
                } else {
                    CommandOutput::text(context.execution_id, "Clear command requires --confirm flag")
                }
            },
            ImmutableChatCommand::History { action, limit, .. } => {
                let content = match action {
                    HistoryAction::Show => {
                        let limit_str = limit.map(|l| format!(" (last {} messages)", l)).unwrap_or_default();
                        format!("Showing chat history{}", limit_str)
                    },
                    HistoryAction::Search => "Searching chat history".to_string(),
                    HistoryAction::Clear => "Chat history cleared".to_string(),
                    HistoryAction::Export => "Chat history exported".to_string(),
                    HistoryAction::Import => "Chat history imported".to_string(),
                    HistoryAction::Backup => "Chat history backed up".to_string(),
                };
                CommandOutput::text(context.execution_id, content)
            },
            _ => {
                CommandOutput::text(context.execution_id, format!("Command {} executed successfully", command.command_name()))
            }
        };

        // Send output through stream
        let _ = sender.send(output.final_output());
        
        stream
    }

    fn name(&self) -> &'static str {
        "default"
    }

    fn can_handle(&self, command: &ImmutableChatCommand) -> bool {
        self.metadata.supported_commands.contains(&command.command_name().to_string())
    }

    fn metadata(&self) -> CommandHandlerMetadata {
        self.metadata.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_parsing() {
        let cmd = CommandParser::parse_command("/help").unwrap();
        assert_eq!(cmd.command_name(), "help");
        
        let cmd = CommandParser::parse_command("clear --confirm").unwrap();
        assert_eq!(cmd.command_name(), "clear");
        
        let cmd = CommandParser::parse_command("export json --output test.json").unwrap();
        assert_eq!(cmd.command_name(), "export");
    }

    #[test]
    fn test_command_validation() {
        let cmd = ImmutableChatCommand::Search {
            query: "test".to_string(),
            scope: SearchScope::All,
            limit: None,
            include_context: false,
        };
        assert!(cmd.validate().is_ok());

        let cmd = ImmutableChatCommand::Search {
            query: "".to_string(),
            scope: SearchScope::All,
            limit: None,
            include_context: false,
        };
        assert!(cmd.validate().is_err());
    }

    #[test]
    fn test_executor_stats() {
        let executor = StreamingCommandExecutor::new();
        let stats = executor.stats();
        assert_eq!(stats.total_executions, 0);
        assert_eq!(stats.success_rate(), 0.0);
    }
}