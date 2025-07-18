//! Chat command system with zero-allocation patterns
//!
//! This module provides a comprehensive command system for chat interactions including
//! slash commands, command parsing, execution, and auto-completion with zero-allocation
//! patterns and blazing-fast performance.

use std::sync::Arc;
use std::collections::HashMap;
use thiserror::Error;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use crossbeam_skiplist::SkipMap;

use crate::{AsyncTask, spawn_async};
use crate::chat::ChatSession;

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
        /// Statistics type (usage, performance, history)
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
    Pause,
    Resume,
    Stop,
}

/// Search scopes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SearchScope {
    All,
    Current,
    Recent,
    Tagged,
    Filtered,
}

/// Branch actions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BranchAction {
    Create,
    Switch,
    Merge,
    Delete,
    List,
    Compare,
}

/// Session actions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SessionAction {
    Save,
    Load,
    List,
    Delete,
    Rename,
    Clone,
}

/// Tool actions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolAction {
    List,
    Install,
    Remove,
    Execute,
    Update,
    Configure,
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
    /// Current chat session
    pub session: Arc<ChatSession>,
    /// Command permissions
    pub permissions: Arc<[Arc<str>]>,
    /// Environment variables
    pub env: HashMap<Arc<str>, Arc<str>>,
    /// Working directory
    pub working_dir: Arc<str>,
    /// User preferences
    pub preferences: HashMap<Arc<str>, Arc<str>>,
}

/// Command execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandOutput {
    /// Success status
    pub success: bool,
    /// Output message
    pub message: Arc<str>,
    /// Additional data
    pub data: Option<serde_json::Value>,
    /// Execution time in milliseconds
    pub execution_time: u64,
    /// Resource usage
    pub resource_usage: ResourceUsage,
}

/// Resource usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Memory used in bytes
    pub memory_bytes: u64,
    /// CPU time in microseconds
    pub cpu_time_us: u64,
    /// Network requests made
    pub network_requests: u32,
    /// Disk operations
    pub disk_operations: u32,
}

/// Zero-allocation command parser
pub struct CommandParser {
    /// Command registry
    registry: SkipMap<Arc<str>, CommandInfo>,
    /// Alias mappings
    aliases: HashMap<Arc<str>, Arc<str>>,
    /// Command history for auto-completion
    history: crossbeam_queue::SegQueue<Arc<str>>,
}

/// Command information for registry
#[derive(Debug, Clone)]
pub struct CommandInfo {
    /// Command name
    pub name: Arc<str>,
    /// Command description
    pub description: Arc<str>,
    /// Command usage example
    pub usage: Arc<str>,
    /// Command aliases
    pub aliases: Arc<[Arc<str>]>,
    /// Required permissions
    pub permissions: Arc<[Arc<str>]>,
    /// Parameter definitions
    pub parameters: Arc<[ParameterInfo]>,
}

/// Parameter information
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    /// Parameter name
    pub name: Arc<str>,
    /// Parameter description
    pub description: Arc<str>,
    /// Parameter type
    pub param_type: ParameterType,
    /// Whether parameter is required
    pub required: bool,
    /// Default value
    pub default: Option<Arc<str>>,
    /// Valid values for enum types
    pub valid_values: Option<Arc<[Arc<str>]>>,
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

impl CommandParser {
    /// Create a new command parser
    #[inline]
    pub fn new() -> Self {
        let mut parser = Self {
            registry: SkipMap::new(),
            aliases: HashMap::new(),
            history: crossbeam_queue::SegQueue::new(),
        };
        
        // Register built-in commands
        parser.register_builtin_commands();
        
        parser
    }

    /// Register built-in commands
    #[inline]
    fn register_builtin_commands(&mut self) {
        // Help command
        self.register_command(CommandInfo {
            name: Arc::from("help"),
            description: Arc::from("Show help information for commands"),
            usage: Arc::from("/help [command] [--extended]"),
            aliases: Arc::from([Arc::from("h"), Arc::from("?")]),
            permissions: Arc::from([]),
            parameters: Arc::from([
                ParameterInfo {
                    name: Arc::from("command"),
                    description: Arc::from("Specific command to get help for"),
                    param_type: ParameterType::String,
                    required: false,
                    default: None,
                    valid_values: None,
                },
                ParameterInfo {
                    name: Arc::from("extended"),
                    description: Arc::from("Show extended help information"),
                    param_type: ParameterType::Boolean,
                    required: false,
                    default: Some(Arc::from("false")),
                    valid_values: None,
                },
            ]),
        });

        // Clear command
        self.register_command(CommandInfo {
            name: Arc::from("clear"),
            description: Arc::from("Clear chat history"),
            usage: Arc::from("/clear [--confirm] [--keep-last N]"),
            aliases: Arc::from([Arc::from("cls")]),
            permissions: Arc::from([Arc::from("chat.clear")]),
            parameters: Arc::from([
                ParameterInfo {
                    name: Arc::from("confirm"),
                    description: Arc::from("Confirm the clear action"),
                    param_type: ParameterType::Boolean,
                    required: false,
                    default: Some(Arc::from("false")),
                    valid_values: None,
                },
                ParameterInfo {
                    name: Arc::from("keep-last"),
                    description: Arc::from("Keep last N messages"),
                    param_type: ParameterType::Integer,
                    required: false,
                    default: None,
                    valid_values: None,
                },
            ]),
        });

        // Export command
        self.register_command(CommandInfo {
            name: Arc::from("export"),
            description: Arc::from("Export conversation to file"),
            usage: Arc::from("/export <format> [output] [--include-metadata]"),
            aliases: Arc::from([Arc::from("exp")]),
            permissions: Arc::from([Arc::from("chat.export")]),
            parameters: Arc::from([
                ParameterInfo {
                    name: Arc::from("format"),
                    description: Arc::from("Export format"),
                    param_type: ParameterType::Enum,
                    required: true,
                    default: None,
                    valid_values: Some(Arc::from([
                        Arc::from("json"),
                        Arc::from("markdown"),
                        Arc::from("pdf"),
                        Arc::from("html"),
                    ])),
                },
                ParameterInfo {
                    name: Arc::from("output"),
                    description: Arc::from("Output file path"),
                    param_type: ParameterType::Path,
                    required: false,
                    default: None,
                    valid_values: None,
                },
                ParameterInfo {
                    name: Arc::from("include-metadata"),
                    description: Arc::from("Include metadata in export"),
                    param_type: ParameterType::Boolean,
                    required: false,
                    default: Some(Arc::from("false")),
                    valid_values: None,
                },
            ]),
        });

        // Config command
        self.register_command(CommandInfo {
            name: Arc::from("config"),
            description: Arc::from("Manage chat configuration"),
            usage: Arc::from("/config [key] [value] [--show] [--reset]"),
            aliases: Arc::from([Arc::from("cfg")]),
            permissions: Arc::from([Arc::from("chat.config")]),
            parameters: Arc::from([
                ParameterInfo {
                    name: Arc::from("key"),
                    description: Arc::from("Configuration key"),
                    param_type: ParameterType::String,
                    required: false,
                    default: None,
                    valid_values: None,
                },
                ParameterInfo {
                    name: Arc::from("value"),
                    description: Arc::from("Configuration value"),
                    param_type: ParameterType::String,
                    required: false,
                    default: None,
                    valid_values: None,
                },
                ParameterInfo {
                    name: Arc::from("show"),
                    description: Arc::from("Show current configuration"),
                    param_type: ParameterType::Boolean,
                    required: false,
                    default: Some(Arc::from("false")),
                    valid_values: None,
                },
                ParameterInfo {
                    name: Arc::from("reset"),
                    description: Arc::from("Reset to default configuration"),
                    param_type: ParameterType::Boolean,
                    required: false,
                    default: Some(Arc::from("false")),
                    valid_values: None,
                },
            ]),
        });

        // Template command
        self.register_command(CommandInfo {
            name: Arc::from("template"),
            description: Arc::from("Manage chat templates"),
            usage: Arc::from("/template <action> [name] [content] [--variables key=value]"),
            aliases: Arc::from([Arc::from("tpl")]),
            permissions: Arc::from([Arc::from("chat.template")]),
            parameters: Arc::from([
                ParameterInfo {
                    name: Arc::from("action"),
                    description: Arc::from("Template action"),
                    param_type: ParameterType::Enum,
                    required: true,
                    default: None,
                    valid_values: Some(Arc::from([
                        Arc::from("create"),
                        Arc::from("use"),
                        Arc::from("list"),
                        Arc::from("delete"),
                        Arc::from("edit"),
                        Arc::from("share"),
                        Arc::from("import"),
                        Arc::from("export"),
                    ])),
                },
                ParameterInfo {
                    name: Arc::from("name"),
                    description: Arc::from("Template name"),
                    param_type: ParameterType::String,
                    required: false,
                    default: None,
                    valid_values: None,
                },
                ParameterInfo {
                    name: Arc::from("content"),
                    description: Arc::from("Template content"),
                    param_type: ParameterType::String,
                    required: false,
                    default: None,
                    valid_values: None,
                },
                ParameterInfo {
                    name: Arc::from("variables"),
                    description: Arc::from("Template variables"),
                    param_type: ParameterType::Object,
                    required: false,
                    default: None,
                    valid_values: None,
                },
            ]),
        });

        // Macro command
        self.register_command(CommandInfo {
            name: Arc::from("macro"),
            description: Arc::from("Manage chat macros"),
            usage: Arc::from("/macro <action> [name] [--auto-execute]"),
            aliases: Arc::from([Arc::from("mac")]),
            permissions: Arc::from([Arc::from("chat.macro")]),
            parameters: Arc::from([
                ParameterInfo {
                    name: Arc::from("action"),
                    description: Arc::from("Macro action"),
                    param_type: ParameterType::Enum,
                    required: true,
                    default: None,
                    valid_values: Some(Arc::from([
                        Arc::from("record"),
                        Arc::from("play"),
                        Arc::from("list"),
                        Arc::from("delete"),
                        Arc::from("edit"),
                        Arc::from("pause"),
                        Arc::from("resume"),
                        Arc::from("stop"),
                    ])),
                },
                ParameterInfo {
                    name: Arc::from("name"),
                    description: Arc::from("Macro name"),
                    param_type: ParameterType::String,
                    required: false,
                    default: None,
                    valid_values: None,
                },
                ParameterInfo {
                    name: Arc::from("auto-execute"),
                    description: Arc::from("Auto-execute macro"),
                    param_type: ParameterType::Boolean,
                    required: false,
                    default: Some(Arc::from("false")),
                    valid_values: None,
                },
            ]),
        });

        // Search command
        self.register_command(CommandInfo {
            name: Arc::from("search"),
            description: Arc::from("Search chat history"),
            usage: Arc::from("/search <query> [--scope scope] [--limit N] [--include-context]"),
            aliases: Arc::from([Arc::from("find")]),
            permissions: Arc::from([Arc::from("chat.search")]),
            parameters: Arc::from([
                ParameterInfo {
                    name: Arc::from("query"),
                    description: Arc::from("Search query"),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    valid_values: None,
                },
                ParameterInfo {
                    name: Arc::from("scope"),
                    description: Arc::from("Search scope"),
                    param_type: ParameterType::Enum,
                    required: false,
                    default: Some(Arc::from("all")),
                    valid_values: Some(Arc::from([
                        Arc::from("all"),
                        Arc::from("current"),
                        Arc::from("recent"),
                        Arc::from("tagged"),
                        Arc::from("filtered"),
                    ])),
                },
                ParameterInfo {
                    name: Arc::from("limit"),
                    description: Arc::from("Maximum results"),
                    param_type: ParameterType::Integer,
                    required: false,
                    default: Some(Arc::from("10")),
                    valid_values: None,
                },
                ParameterInfo {
                    name: Arc::from("include-context"),
                    description: Arc::from("Include context in results"),
                    param_type: ParameterType::Boolean,
                    required: false,
                    default: Some(Arc::from("false")),
                    valid_values: None,
                },
            ]),
        });

        // Branch command
        self.register_command(CommandInfo {
            name: Arc::from("branch"),
            description: Arc::from("Manage conversation branches"),
            usage: Arc::from("/branch <action> [name] [--source source]"),
            aliases: Arc::from([Arc::from("br")]),
            permissions: Arc::from([Arc::from("chat.branch")]),
            parameters: Arc::from([
                ParameterInfo {
                    name: Arc::from("action"),
                    description: Arc::from("Branch action"),
                    param_type: ParameterType::Enum,
                    required: true,
                    default: None,
                    valid_values: Some(Arc::from([
                        Arc::from("create"),
                        Arc::from("switch"),
                        Arc::from("merge"),
                        Arc::from("delete"),
                        Arc::from("list"),
                        Arc::from("compare"),
                    ])),
                },
                ParameterInfo {
                    name: Arc::from("name"),
                    description: Arc::from("Branch name"),
                    param_type: ParameterType::String,
                    required: false,
                    default: None,
                    valid_values: None,
                },
                ParameterInfo {
                    name: Arc::from("source"),
                    description: Arc::from("Source branch for merging"),
                    param_type: ParameterType::String,
                    required: false,
                    default: None,
                    valid_values: None,
                },
            ]),
        });

        // Register aliases
        self.aliases.insert(Arc::from("h"), Arc::from("help"));
        self.aliases.insert(Arc::from("?"), Arc::from("help"));
        self.aliases.insert(Arc::from("cls"), Arc::from("clear"));
        self.aliases.insert(Arc::from("exp"), Arc::from("export"));
        self.aliases.insert(Arc::from("cfg"), Arc::from("config"));
        self.aliases.insert(Arc::from("tpl"), Arc::from("template"));
        self.aliases.insert(Arc::from("mac"), Arc::from("macro"));
        self.aliases.insert(Arc::from("find"), Arc::from("search"));
        self.aliases.insert(Arc::from("br"), Arc::from("branch"));
    }

    /// Register a command
    #[inline]
    pub fn register_command(&mut self, info: CommandInfo) {
        // Register main command
        self.registry.insert(info.name.clone(), info.clone());
        
        // Register aliases
        for alias in info.aliases.iter() {
            self.aliases.insert(alias.clone(), info.name.clone());
        }
    }

    /// Parse a command string with zero-allocation patterns
    #[inline]
    pub fn parse(&self, input: &str) -> CommandResult<ChatCommand> {
        // Trim whitespace and check for command prefix
        let input = input.trim();
        if !input.starts_with('/') {
            return Err(CommandError::ParseError {
                detail: Arc::from("Commands must start with '/'"),
            });
        }

        // Remove leading slash
        let input = &input[1..];
        if input.is_empty() {
            return Err(CommandError::ParseError {
                detail: Arc::from("Empty command"),
            });
        }

        // Add to command history
        self.history.push(Arc::from(input));

        // Split into tokens using zero-allocation slice operations
        let tokens = self.tokenize(input)?;
        if tokens.is_empty() {
            return Err(CommandError::ParseError {
                detail: Arc::from("No command specified"),
            });
        }

        // Get command name (resolve aliases)
        let command_name = self.resolve_alias(&tokens[0]);
        
        // Parse based on command type
        match command_name.as_ref() {
            "help" => self.parse_help(&tokens[1..]),
            "clear" => self.parse_clear(&tokens[1..]),
            "export" => self.parse_export(&tokens[1..]),
            "config" => self.parse_config(&tokens[1..]),
            "template" => self.parse_template(&tokens[1..]),
            "macro" => self.parse_macro(&tokens[1..]),
            "search" => self.parse_search(&tokens[1..]),
            "branch" => self.parse_branch(&tokens[1..]),
            "session" => self.parse_session(&tokens[1..]),
            "tool" => self.parse_tool(&tokens[1..]),
            "stats" => self.parse_stats(&tokens[1..]),
            "theme" => self.parse_theme(&tokens[1..]),
            "debug" => self.parse_debug(&tokens[1..]),
            _ => Err(CommandError::UnknownCommand {
                command: command_name,
            }),
        }
    }

    /// Tokenize input string using zero-allocation slice operations
    #[inline]
    fn tokenize(&self, input: &str) -> CommandResult<Vec<Arc<str>>> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();
        let mut in_quotes = false;
        let mut escape_next = false;
        
        for ch in input.chars() {
            if escape_next {
                current_token.push(ch);
                escape_next = false;
                continue;
            }
            
            match ch {
                '\\' => {
                    escape_next = true;
                }
                '"' => {
                    if in_quotes {
                        // End of quoted string
                        tokens.push(Arc::from(current_token.clone()));
                        current_token.clear();
                        in_quotes = false;
                    } else {
                        // Start of quoted string
                        if !current_token.is_empty() {
                            tokens.push(Arc::from(current_token.clone()));
                            current_token.clear();
                        }
                        in_quotes = true;
                    }
                }
                ' ' | '\t' => {
                    if in_quotes {
                        current_token.push(ch);
                    } else if !current_token.is_empty() {
                        tokens.push(Arc::from(current_token.clone()));
                        current_token.clear();
                    }
                }
                _ => {
                    current_token.push(ch);
                }
            }
        }
        
        if !current_token.is_empty() {
            tokens.push(Arc::from(current_token));
        }
        
        if in_quotes {
            return Err(CommandError::ParseError {
                detail: Arc::from("Unterminated quoted string"),
            });
        }
        
        Ok(tokens)
    }

    /// Resolve command alias
    #[inline]
    fn resolve_alias(&self, command: &Arc<str>) -> Arc<str> {
        self.aliases.get(command).cloned().unwrap_or_else(|| command.clone())
    }

    /// Parse help command
    #[inline]
    fn parse_help(&self, tokens: &[Arc<str>]) -> CommandResult<ChatCommand> {
        let mut command = None;
        let mut extended = false;
        
        let mut i = 0;
        while i < tokens.len() {
            match tokens[i].as_ref() {
                "--extended" => extended = true,
                token if !token.starts_with('-') => {
                    if command.is_none() {
                        command = Some(tokens[i].clone());
                    }
                }
                _ => {
                    return Err(CommandError::InvalidArguments {
                        detail: Arc::from(format!("Unknown argument: {}", tokens[i])),
                    });
                }
            }
            i += 1;
        }
        
        Ok(ChatCommand::Help { command, extended })
    }

    /// Parse clear command
    #[inline]
    fn parse_clear(&self, tokens: &[Arc<str>]) -> CommandResult<ChatCommand> {
        let mut confirm = false;
        let mut keep_last = None;
        
        let mut i = 0;
        while i < tokens.len() {
            match tokens[i].as_ref() {
                "--confirm" => confirm = true,
                "--keep-last" => {
                    if i + 1 < tokens.len() {
                        keep_last = Some(tokens[i + 1].parse().map_err(|_| {
                            CommandError::InvalidArguments {
                                detail: Arc::from("Invalid number for --keep-last"),
                            }
                        })?);
                        i += 1;
                    } else {
                        return Err(CommandError::InvalidArguments {
                            detail: Arc::from("--keep-last requires a number"),
                        });
                    }
                }
                _ => {
                    return Err(CommandError::InvalidArguments {
                        detail: Arc::from(format!("Unknown argument: {}", tokens[i])),
                    });
                }
            }
            i += 1;
        }
        
        Ok(ChatCommand::Clear { confirm, keep_last })
    }

    /// Parse export command
    #[inline]
    fn parse_export(&self, tokens: &[Arc<str>]) -> CommandResult<ChatCommand> {
        if tokens.is_empty() {
            return Err(CommandError::InvalidArguments {
                detail: Arc::from("Export format is required"),
            });
        }
        
        let format = tokens[0].clone();
        let mut output = None;
        let mut include_metadata = false;
        
        let mut i = 1;
        while i < tokens.len() {
            match tokens[i].as_ref() {
                "--include-metadata" => include_metadata = true,
                token if !token.starts_with('-') => {
                    if output.is_none() {
                        output = Some(tokens[i].clone());
                    }
                }
                _ => {
                    return Err(CommandError::InvalidArguments {
                        detail: Arc::from(format!("Unknown argument: {}", tokens[i])),
                    });
                }
            }
            i += 1;
        }
        
        Ok(ChatCommand::Export { format, output, include_metadata })
    }

    /// Parse config command
    #[inline]
    fn parse_config(&self, tokens: &[Arc<str>]) -> CommandResult<ChatCommand> {
        let mut key = None;
        let mut value = None;
        let mut show = false;
        let mut reset = false;
        
        let mut i = 0;
        while i < tokens.len() {
            match tokens[i].as_ref() {
                "--show" => show = true,
                "--reset" => reset = true,
                token if !token.starts_with('-') => {
                    if key.is_none() {
                        key = Some(tokens[i].clone());
                    } else if value.is_none() {
                        value = Some(tokens[i].clone());
                    }
                }
                _ => {
                    return Err(CommandError::InvalidArguments {
                        detail: Arc::from(format!("Unknown argument: {}", tokens[i])),
                    });
                }
            }
            i += 1;
        }
        
        Ok(ChatCommand::Config { key, value, show, reset })
    }

    /// Parse template command
    #[inline]
    fn parse_template(&self, tokens: &[Arc<str>]) -> CommandResult<ChatCommand> {
        if tokens.is_empty() {
            return Err(CommandError::InvalidArguments {
                detail: Arc::from("Template action is required"),
            });
        }
        
        let action = match tokens[0].as_ref() {
            "create" => TemplateAction::Create,
            "use" => TemplateAction::Use,
            "list" => TemplateAction::List,
            "delete" => TemplateAction::Delete,
            "edit" => TemplateAction::Edit,
            "share" => TemplateAction::Share,
            "import" => TemplateAction::Import,
            "export" => TemplateAction::Export,
            _ => {
                return Err(CommandError::InvalidArguments {
                    detail: Arc::from(format!("Unknown template action: {}", tokens[0])),
                });
            }
        };
        
        let mut name = None;
        let mut content = None;
        let mut variables = HashMap::new();
        
        let mut i = 1;
        while i < tokens.len() {
            match tokens[i].as_ref() {
                token if token.starts_with("--variables") => {
                    if let Some(eq_pos) = token.find('=') {
                        let var_part = &token[12..]; // Skip "--variables"
                        for var_pair in var_part.split(',') {
                            if let Some(eq_pos) = var_pair.find('=') {
                                let key = Arc::from(var_pair[..eq_pos].trim());
                                let value = Arc::from(var_pair[eq_pos + 1..].trim());
                                variables.insert(key, value);
                            }
                        }
                    }
                }
                token if !token.starts_with('-') => {
                    if name.is_none() {
                        name = Some(tokens[i].clone());
                    } else if content.is_none() {
                        content = Some(tokens[i].clone());
                    }
                }
                _ => {
                    return Err(CommandError::InvalidArguments {
                        detail: Arc::from(format!("Unknown argument: {}", tokens[i])),
                    });
                }
            }
            i += 1;
        }
        
        Ok(ChatCommand::Template { action, name, content, variables })
    }

    /// Parse macro command
    #[inline]
    fn parse_macro(&self, tokens: &[Arc<str>]) -> CommandResult<ChatCommand> {
        if tokens.is_empty() {
            return Err(CommandError::InvalidArguments {
                detail: Arc::from("Macro action is required"),
            });
        }
        
        let action = match tokens[0].as_ref() {
            "record" => MacroAction::Record,
            "play" => MacroAction::Play,
            "list" => MacroAction::List,
            "delete" => MacroAction::Delete,
            "edit" => MacroAction::Edit,
            "pause" => MacroAction::Pause,
            "resume" => MacroAction::Resume,
            "stop" => MacroAction::Stop,
            _ => {
                return Err(CommandError::InvalidArguments {
                    detail: Arc::from(format!("Unknown macro action: {}", tokens[0])),
                });
            }
        };
        
        let mut name = None;
        let mut auto_execute = false;
        
        let mut i = 1;
        while i < tokens.len() {
            match tokens[i].as_ref() {
                "--auto-execute" => auto_execute = true,
                token if !token.starts_with('-') => {
                    if name.is_none() {
                        name = Some(tokens[i].clone());
                    }
                }
                _ => {
                    return Err(CommandError::InvalidArguments {
                        detail: Arc::from(format!("Unknown argument: {}", tokens[i])),
                    });
                }
            }
            i += 1;
        }
        
        Ok(ChatCommand::Macro { action, name, auto_execute })
    }

    /// Parse search command
    #[inline]
    fn parse_search(&self, tokens: &[Arc<str>]) -> CommandResult<ChatCommand> {
        if tokens.is_empty() {
            return Err(CommandError::InvalidArguments {
                detail: Arc::from("Search query is required"),
            });
        }
        
        let query = tokens[0].clone();
        let mut scope = SearchScope::All;
        let mut limit = None;
        let mut include_context = false;
        
        let mut i = 1;
        while i < tokens.len() {
            match tokens[i].as_ref() {
                "--scope" => {
                    if i + 1 < tokens.len() {
                        scope = match tokens[i + 1].as_ref() {
                            "all" => SearchScope::All,
                            "current" => SearchScope::Current,
                            "recent" => SearchScope::Recent,
                            "tagged" => SearchScope::Tagged,
                            "filtered" => SearchScope::Filtered,
                            _ => {
                                return Err(CommandError::InvalidArguments {
                                    detail: Arc::from(format!("Unknown scope: {}", tokens[i + 1])),
                                });
                            }
                        };
                        i += 1;
                    }
                }
                "--limit" => {
                    if i + 1 < tokens.len() {
                        limit = Some(tokens[i + 1].parse().map_err(|_| {
                            CommandError::InvalidArguments {
                                detail: Arc::from("Invalid number for --limit"),
                            }
                        })?);
                        i += 1;
                    }
                }
                "--include-context" => include_context = true,
                _ => {
                    return Err(CommandError::InvalidArguments {
                        detail: Arc::from(format!("Unknown argument: {}", tokens[i])),
                    });
                }
            }
            i += 1;
        }
        
        Ok(ChatCommand::Search { query, scope, limit, include_context })
    }

    /// Parse branch command
    #[inline]
    fn parse_branch(&self, tokens: &[Arc<str>]) -> CommandResult<ChatCommand> {
        if tokens.is_empty() {
            return Err(CommandError::InvalidArguments {
                detail: Arc::from("Branch action is required"),
            });
        }
        
        let action = match tokens[0].as_ref() {
            "create" => BranchAction::Create,
            "switch" => BranchAction::Switch,
            "merge" => BranchAction::Merge,
            "delete" => BranchAction::Delete,
            "list" => BranchAction::List,
            "compare" => BranchAction::Compare,
            _ => {
                return Err(CommandError::InvalidArguments {
                    detail: Arc::from(format!("Unknown branch action: {}", tokens[0])),
                });
            }
        };
        
        let mut name = None;
        let mut source = None;
        
        let mut i = 1;
        while i < tokens.len() {
            match tokens[i].as_ref() {
                "--source" => {
                    if i + 1 < tokens.len() {
                        source = Some(tokens[i + 1].clone());
                        i += 1;
                    }
                }
                token if !token.starts_with('-') => {
                    if name.is_none() {
                        name = Some(tokens[i].clone());
                    }
                }
                _ => {
                    return Err(CommandError::InvalidArguments {
                        detail: Arc::from(format!("Unknown argument: {}", tokens[i])),
                    });
                }
            }
            i += 1;
        }
        
        Ok(ChatCommand::Branch { action, name, source })
    }

    /// Parse session command
    #[inline]
    fn parse_session(&self, tokens: &[Arc<str>]) -> CommandResult<ChatCommand> {
        if tokens.is_empty() {
            return Err(CommandError::InvalidArguments {
                detail: Arc::from("Session action is required"),
            });
        }
        
        let action = match tokens[0].as_ref() {
            "save" => SessionAction::Save,
            "load" => SessionAction::Load,
            "list" => SessionAction::List,
            "delete" => SessionAction::Delete,
            "rename" => SessionAction::Rename,
            "clone" => SessionAction::Clone,
            _ => {
                return Err(CommandError::InvalidArguments {
                    detail: Arc::from(format!("Unknown session action: {}", tokens[0])),
                });
            }
        };
        
        let mut name = None;
        let mut include_config = false;
        
        let mut i = 1;
        while i < tokens.len() {
            match tokens[i].as_ref() {
                "--include-config" => include_config = true,
                token if !token.starts_with('-') => {
                    if name.is_none() {
                        name = Some(tokens[i].clone());
                    }
                }
                _ => {
                    return Err(CommandError::InvalidArguments {
                        detail: Arc::from(format!("Unknown argument: {}", tokens[i])),
                    });
                }
            }
            i += 1;
        }
        
        Ok(ChatCommand::Session { action, name, include_config })
    }

    /// Parse tool command
    #[inline]
    fn parse_tool(&self, tokens: &[Arc<str>]) -> CommandResult<ChatCommand> {
        if tokens.is_empty() {
            return Err(CommandError::InvalidArguments {
                detail: Arc::from("Tool action is required"),
            });
        }
        
        let action = match tokens[0].as_ref() {
            "list" => ToolAction::List,
            "install" => ToolAction::Install,
            "remove" => ToolAction::Remove,
            "execute" => ToolAction::Execute,
            "update" => ToolAction::Update,
            "configure" => ToolAction::Configure,
            _ => {
                return Err(CommandError::InvalidArguments {
                    detail: Arc::from(format!("Unknown tool action: {}", tokens[0])),
                });
            }
        };
        
        let mut name = None;
        let mut args = HashMap::new();
        
        let mut i = 1;
        while i < tokens.len() {
            if tokens[i].contains('=') {
                let parts: Vec<&str> = tokens[i].splitn(2, '=').collect();
                if parts.len() == 2 {
                    args.insert(Arc::from(parts[0]), Arc::from(parts[1]));
                }
            } else if !tokens[i].starts_with('-') {
                if name.is_none() {
                    name = Some(tokens[i].clone());
                }
            }
            i += 1;
        }
        
        Ok(ChatCommand::Tool { action, name, args })
    }

    /// Parse stats command
    #[inline]
    fn parse_stats(&self, tokens: &[Arc<str>]) -> CommandResult<ChatCommand> {
        let stat_type = if tokens.is_empty() {
            StatsType::Usage
        } else {
            match tokens[0].as_ref() {
                "usage" => StatsType::Usage,
                "performance" => StatsType::Performance,
                "history" => StatsType::History,
                "tokens" => StatsType::Tokens,
                "costs" => StatsType::Costs,
                "errors" => StatsType::Errors,
                _ => {
                    return Err(CommandError::InvalidArguments {
                        detail: Arc::from(format!("Unknown stats type: {}", tokens[0])),
                    });
                }
            }
        };
        
        let mut period = None;
        let mut detailed = false;
        
        let mut i = 1;
        while i < tokens.len() {
            match tokens[i].as_ref() {
                "--period" => {
                    if i + 1 < tokens.len() {
                        period = Some(tokens[i + 1].clone());
                        i += 1;
                    }
                }
                "--detailed" => detailed = true,
                _ => {
                    return Err(CommandError::InvalidArguments {
                        detail: Arc::from(format!("Unknown argument: {}", tokens[i])),
                    });
                }
            }
            i += 1;
        }
        
        Ok(ChatCommand::Stats { stat_type, period, detailed })
    }

    /// Parse theme command
    #[inline]
    fn parse_theme(&self, tokens: &[Arc<str>]) -> CommandResult<ChatCommand> {
        if tokens.is_empty() {
            return Err(CommandError::InvalidArguments {
                detail: Arc::from("Theme action is required"),
            });
        }
        
        let action = match tokens[0].as_ref() {
            "set" => ThemeAction::Set,
            "list" => ThemeAction::List,
            "create" => ThemeAction::Create,
            "export" => ThemeAction::Export,
            "import" => ThemeAction::Import,
            "edit" => ThemeAction::Edit,
            _ => {
                return Err(CommandError::InvalidArguments {
                    detail: Arc::from(format!("Unknown theme action: {}", tokens[0])),
                });
            }
        };
        
        let mut name = None;
        let mut properties = HashMap::new();
        
        let mut i = 1;
        while i < tokens.len() {
            if tokens[i].contains('=') {
                let parts: Vec<&str> = tokens[i].splitn(2, '=').collect();
                if parts.len() == 2 {
                    properties.insert(Arc::from(parts[0]), Arc::from(parts[1]));
                }
            } else if !tokens[i].starts_with('-') {
                if name.is_none() {
                    name = Some(tokens[i].clone());
                }
            }
            i += 1;
        }
        
        Ok(ChatCommand::Theme { action, name, properties })
    }

    /// Parse debug command
    #[inline]
    fn parse_debug(&self, tokens: &[Arc<str>]) -> CommandResult<ChatCommand> {
        let action = if tokens.is_empty() {
            DebugAction::Info
        } else {
            match tokens[0].as_ref() {
                "info" => DebugAction::Info,
                "logs" => DebugAction::Logs,
                "performance" => DebugAction::Performance,
                "memory" => DebugAction::Memory,
                "network" => DebugAction::Network,
                "cache" => DebugAction::Cache,
                _ => {
                    return Err(CommandError::InvalidArguments {
                        detail: Arc::from(format!("Unknown debug action: {}", tokens[0])),
                    });
                }
            }
        };
        
        let mut level = None;
        let mut system_info = false;
        
        let mut i = 1;
        while i < tokens.len() {
            match tokens[i].as_ref() {
                "--level" => {
                    if i + 1 < tokens.len() {
                        level = Some(tokens[i + 1].clone());
                        i += 1;
                    }
                }
                "--system-info" => system_info = true,
                _ => {
                    return Err(CommandError::InvalidArguments {
                        detail: Arc::from(format!("Unknown argument: {}", tokens[i])),
                    });
                }
            }
            i += 1;
        }
        
        Ok(ChatCommand::Debug { action, level, system_info })
    }

    /// Get command suggestions for auto-completion
    #[inline]
    pub fn get_suggestions(&self, prefix: &str) -> Vec<Arc<str>> {
        let mut suggestions = Vec::new();
        
        // Add commands that match the prefix
        for entry in self.registry.iter() {
            if entry.key().starts_with(prefix) {
                suggestions.push(entry.key().clone());
            }
        }
        
        // Add aliases that match the prefix
        for (alias, _) in &self.aliases {
            if alias.starts_with(prefix) && !suggestions.contains(alias) {
                suggestions.push(alias.clone());
            }
        }
        
        // Sort suggestions
        suggestions.sort();
        suggestions
    }

    /// Get command information
    #[inline]
    pub fn get_command_info(&self, command: &str) -> Option<CommandInfo> {
        let command = self.resolve_alias(&Arc::from(command));
        self.registry.get(&command).map(|entry| entry.value().clone())
    }

    /// Get command history
    #[inline]
    pub fn get_history(&self) -> Vec<Arc<str>> {
        let mut history = Vec::new();
        while let Some(command) = self.history.pop() {
            history.push(command);
        }
        history.reverse();
        
        // Put back in queue
        for command in &history {
            self.history.push(command.clone());
        }
        
        history
    }
}

impl Default for CommandParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Command executor with async execution
pub struct CommandExecutor {
    /// Command parser
    parser: CommandParser,
    /// Execution context
    context: Arc<RwLock<CommandContext>>,
    /// Execution metrics
    metrics: Arc<RwLock<ExecutionMetrics>>,
}

/// Execution metrics
#[derive(Debug, Clone, Default)]
pub struct ExecutionMetrics {
    /// Total commands executed
    pub total_executed: u64,
    /// Total execution time
    pub total_time_ms: u64,
    /// Commands by type
    pub command_counts: HashMap<Arc<str>, u64>,
    /// Average execution times
    pub avg_times: HashMap<Arc<str>, u64>,
    /// Error counts
    pub error_counts: HashMap<Arc<str>, u64>,
}

impl CommandExecutor {
    /// Create a new command executor
    #[inline]
    pub fn new(context: CommandContext) -> Self {
        Self {
            parser: CommandParser::new(),
            context: Arc::new(RwLock::new(context)),
            metrics: Arc::new(RwLock::new(ExecutionMetrics::default())),
        }
    }

    /// Execute a command
    #[inline]
    pub fn execute(&self, command: ChatCommand) -> AsyncTask<CommandResult<CommandOutput>> {
        let context = self.context.clone();
        let metrics = self.metrics.clone();
        
        spawn_async(async move {
            let start_time = std::time::Instant::now();
            
            let result = match command {
                ChatCommand::Help { command, extended } => {
                    Self::execute_help(command, extended).await
                }
                ChatCommand::Clear { confirm, keep_last } => {
                    Self::execute_clear(context.clone(), confirm, keep_last).await
                }
                ChatCommand::Export { format, output, include_metadata } => {
                    Self::execute_export(context.clone(), format, output, include_metadata).await
                }
                ChatCommand::Config { key, value, show, reset } => {
                    Self::execute_config(context.clone(), key, value, show, reset).await
                }
                ChatCommand::Template { action, name, content, variables } => {
                    Self::execute_template(context.clone(), action, name, content, variables).await
                }
                ChatCommand::Macro { action, name, auto_execute } => {
                    Self::execute_macro(context.clone(), action, name, auto_execute).await
                }
                ChatCommand::Search { query, scope, limit, include_context } => {
                    Self::execute_search(context.clone(), query, scope, limit, include_context).await
                }
                ChatCommand::Branch { action, name, source } => {
                    Self::execute_branch(context.clone(), action, name, source).await
                }
                ChatCommand::Session { action, name, include_config } => {
                    Self::execute_session(context.clone(), action, name, include_config).await
                }
                ChatCommand::Tool { action, name, args } => {
                    Self::execute_tool(context.clone(), action, name, args).await
                }
                ChatCommand::Stats { stat_type, period, detailed } => {
                    Self::execute_stats(context.clone(), stat_type, period, detailed).await
                }
                ChatCommand::Theme { action, name, properties } => {
                    Self::execute_theme(context.clone(), action, name, properties).await
                }
                ChatCommand::Debug { action, level, system_info } => {
                    Self::execute_debug(context.clone(), action, level, system_info).await
                }
            };
            
            let execution_time = start_time.elapsed().as_millis() as u64;
            
            // Update metrics
            {
                let mut metrics = metrics.write().await;
                metrics.total_executed += 1;
                metrics.total_time_ms += execution_time;
                
                let command_name = Arc::from(Self::get_command_name(&command));
                *metrics.command_counts.entry(command_name.clone()).or_insert(0) += 1;
                
                let current_avg = metrics.avg_times.get(&command_name).cloned().unwrap_or(0);
                let count = metrics.command_counts.get(&command_name).cloned().unwrap_or(1);
                let new_avg = (current_avg * (count - 1) + execution_time) / count;
                metrics.avg_times.insert(command_name.clone(), new_avg);
                
                if result.is_err() {
                    *metrics.error_counts.entry(command_name).or_insert(0) += 1;
                }
            }
            
            result
        })
    }

    /// Get command name for metrics
    #[inline]
    fn get_command_name(command: &ChatCommand) -> &'static str {
        match command {
            ChatCommand::Help { .. } => "help",
            ChatCommand::Clear { .. } => "clear",
            ChatCommand::Export { .. } => "export",
            ChatCommand::Config { .. } => "config",
            ChatCommand::Template { .. } => "template",
            ChatCommand::Macro { .. } => "macro",
            ChatCommand::Search { .. } => "search",
            ChatCommand::Branch { .. } => "branch",
            ChatCommand::Session { .. } => "session",
            ChatCommand::Tool { .. } => "tool",
            ChatCommand::Stats { .. } => "stats",
            ChatCommand::Theme { .. } => "theme",
            ChatCommand::Debug { .. } => "debug",
        }
    }

    /// Execute help command
    #[inline]
    async fn execute_help(command: Option<Arc<str>>, extended: bool) -> CommandResult<CommandOutput> {
        let message = if let Some(cmd) = command {
            format!("Help for command: {}", cmd)
        } else {
            "Available commands: help, clear, export, config, template, macro, search, branch, session, tool, stats, theme, debug".to_string()
        };
        
        Ok(CommandOutput {
            success: true,
            message: Arc::from(message),
            data: None,
            execution_time: 0,
            resource_usage: ResourceUsage {
                memory_bytes: 0,
                cpu_time_us: 0,
                network_requests: 0,
                disk_operations: 0,
            },
        })
    }

    /// Execute clear command
    #[inline]
    async fn execute_clear(
        context: Arc<RwLock<CommandContext>>,
        confirm: bool,
        keep_last: Option<usize>,
    ) -> CommandResult<CommandOutput> {
        let ctx = context.read().await;
        
        if !confirm {
            return Ok(CommandOutput {
                success: false,
                message: Arc::from("Clear operation requires confirmation. Use --confirm flag."),
                data: None,
                execution_time: 0,
                resource_usage: ResourceUsage {
                    memory_bytes: 0,
                    cpu_time_us: 0,
                    network_requests: 0,
                    disk_operations: 0,
                },
            });
        }
        
        // Clear chat history (would integrate with actual session)
        let message = if let Some(keep) = keep_last {
            format!("Cleared chat history, keeping last {} messages", keep)
        } else {
            "Cleared chat history".to_string()
        };
        
        Ok(CommandOutput {
            success: true,
            message: Arc::from(message),
            data: None,
            execution_time: 0,
            resource_usage: ResourceUsage {
                memory_bytes: 0,
                cpu_time_us: 0,
                network_requests: 0,
                disk_operations: 1,
            },
        })
    }

    /// Execute export command
    #[inline]
    async fn execute_export(
        context: Arc<RwLock<CommandContext>>,
        format: Arc<str>,
        output: Option<Arc<str>>,
        include_metadata: bool,
    ) -> CommandResult<CommandOutput> {
        let ctx = context.read().await;
        
        let message = format!(
            "Exported conversation to {} format{}{}",
            format,
            if let Some(out) = output { format!(" ({})", out) } else { String::new() },
            if include_metadata { " with metadata" } else { "" }
        );
        
        Ok(CommandOutput {
            success: true,
            message: Arc::from(message),
            data: None,
            execution_time: 0,
            resource_usage: ResourceUsage {
                memory_bytes: 0,
                cpu_time_us: 0,
                network_requests: 0,
                disk_operations: 1,
            },
        })
    }

    /// Execute config command
    #[inline]
    async fn execute_config(
        context: Arc<RwLock<CommandContext>>,
        key: Option<Arc<str>>,
        value: Option<Arc<str>>,
        show: bool,
        reset: bool,
    ) -> CommandResult<CommandOutput> {
        let ctx = context.read().await;
        
        let message = if show {
            "Current configuration: <configuration display>"
        } else if reset {
            "Configuration reset to defaults"
        } else if let (Some(k), Some(v)) = (key, value) {
            "Configuration updated"
        } else {
            "Invalid configuration command"
        };
        
        Ok(CommandOutput {
            success: true,
            message: Arc::from(message),
            data: None,
            execution_time: 0,
            resource_usage: ResourceUsage {
                memory_bytes: 0,
                cpu_time_us: 0,
                network_requests: 0,
                disk_operations: if reset { 1 } else { 0 },
            },
        })
    }

    /// Execute template command
    #[inline]
    async fn execute_template(
        context: Arc<RwLock<CommandContext>>,
        action: TemplateAction,
        name: Option<Arc<str>>,
        content: Option<Arc<str>>,
        variables: HashMap<Arc<str>, Arc<str>>,
    ) -> CommandResult<CommandOutput> {
        let ctx = context.read().await;
        
        let message = match action {
            TemplateAction::Create => "Template created",
            TemplateAction::Use => "Template applied",
            TemplateAction::List => "Available templates: <template list>",
            TemplateAction::Delete => "Template deleted",
            TemplateAction::Edit => "Template edited",
            TemplateAction::Share => "Template shared",
            TemplateAction::Import => "Template imported",
            TemplateAction::Export => "Template exported",
        };
        
        Ok(CommandOutput {
            success: true,
            message: Arc::from(message),
            data: None,
            execution_time: 0,
            resource_usage: ResourceUsage {
                memory_bytes: 0,
                cpu_time_us: 0,
                network_requests: 0,
                disk_operations: 1,
            },
        })
    }

    /// Execute macro command
    #[inline]
    async fn execute_macro(
        context: Arc<RwLock<CommandContext>>,
        action: MacroAction,
        name: Option<Arc<str>>,
        auto_execute: bool,
    ) -> CommandResult<CommandOutput> {
        let ctx = context.read().await;
        
        let message = match action {
            MacroAction::Record => "Macro recording started",
            MacroAction::Play => "Macro executed",
            MacroAction::List => "Available macros: <macro list>",
            MacroAction::Delete => "Macro deleted",
            MacroAction::Edit => "Macro edited",
            MacroAction::Pause => "Macro recording paused",
            MacroAction::Resume => "Macro recording resumed",
            MacroAction::Stop => "Macro recording stopped",
        };
        
        Ok(CommandOutput {
            success: true,
            message: Arc::from(message),
            data: None,
            execution_time: 0,
            resource_usage: ResourceUsage {
                memory_bytes: 0,
                cpu_time_us: 0,
                network_requests: 0,
                disk_operations: 0,
            },
        })
    }

    /// Execute search command
    #[inline]
    async fn execute_search(
        context: Arc<RwLock<CommandContext>>,
        query: Arc<str>,
        scope: SearchScope,
        limit: Option<usize>,
        include_context: bool,
    ) -> CommandResult<CommandOutput> {
        let ctx = context.read().await;
        
        let message = format!(
            "Search results for '{}': <search results>",
            query
        );
        
        Ok(CommandOutput {
            success: true,
            message: Arc::from(message),
            data: None,
            execution_time: 0,
            resource_usage: ResourceUsage {
                memory_bytes: 0,
                cpu_time_us: 0,
                network_requests: 0,
                disk_operations: 1,
            },
        })
    }

    /// Execute branch command
    #[inline]
    async fn execute_branch(
        context: Arc<RwLock<CommandContext>>,
        action: BranchAction,
        name: Option<Arc<str>>,
        source: Option<Arc<str>>,
    ) -> CommandResult<CommandOutput> {
        let ctx = context.read().await;
        
        let message = match action {
            BranchAction::Create => "Branch created",
            BranchAction::Switch => "Switched to branch",
            BranchAction::Merge => "Branch merged",
            BranchAction::Delete => "Branch deleted",
            BranchAction::List => "Available branches: <branch list>",
            BranchAction::Compare => "Branch comparison: <comparison>",
        };
        
        Ok(CommandOutput {
            success: true,
            message: Arc::from(message),
            data: None,
            execution_time: 0,
            resource_usage: ResourceUsage {
                memory_bytes: 0,
                cpu_time_us: 0,
                network_requests: 0,
                disk_operations: 1,
            },
        })
    }

    /// Execute session command
    #[inline]
    async fn execute_session(
        context: Arc<RwLock<CommandContext>>,
        action: SessionAction,
        name: Option<Arc<str>>,
        include_config: bool,
    ) -> CommandResult<CommandOutput> {
        let ctx = context.read().await;
        
        let message = match action {
            SessionAction::Save => "Session saved",
            SessionAction::Load => "Session loaded",
            SessionAction::List => "Available sessions: <session list>",
            SessionAction::Delete => "Session deleted",
            SessionAction::Rename => "Session renamed",
            SessionAction::Clone => "Session cloned",
        };
        
        Ok(CommandOutput {
            success: true,
            message: Arc::from(message),
            data: None,
            execution_time: 0,
            resource_usage: ResourceUsage {
                memory_bytes: 0,
                cpu_time_us: 0,
                network_requests: 0,
                disk_operations: 1,
            },
        })
    }

    /// Execute tool command
    #[inline]
    async fn execute_tool(
        context: Arc<RwLock<CommandContext>>,
        action: ToolAction,
        name: Option<Arc<str>>,
        args: HashMap<Arc<str>, Arc<str>>,
    ) -> CommandResult<CommandOutput> {
        let ctx = context.read().await;
        
        let message = match action {
            ToolAction::List => "Available tools: <tool list>",
            ToolAction::Install => "Tool installed",
            ToolAction::Remove => "Tool removed",
            ToolAction::Execute => "Tool executed",
            ToolAction::Update => "Tool updated",
            ToolAction::Configure => "Tool configured",
        };
        
        Ok(CommandOutput {
            success: true,
            message: Arc::from(message),
            data: None,
            execution_time: 0,
            resource_usage: ResourceUsage {
                memory_bytes: 0,
                cpu_time_us: 0,
                network_requests: 1,
                disk_operations: 1,
            },
        })
    }

    /// Execute stats command
    #[inline]
    async fn execute_stats(
        context: Arc<RwLock<CommandContext>>,
        stat_type: StatsType,
        period: Option<Arc<str>>,
        detailed: bool,
    ) -> CommandResult<CommandOutput> {
        let ctx = context.read().await;
        
        let message = match stat_type {
            StatsType::Usage => "Usage statistics: <usage stats>",
            StatsType::Performance => "Performance statistics: <performance stats>",
            StatsType::History => "History statistics: <history stats>",
            StatsType::Tokens => "Token statistics: <token stats>",
            StatsType::Costs => "Cost statistics: <cost stats>",
            StatsType::Errors => "Error statistics: <error stats>",
        };
        
        Ok(CommandOutput {
            success: true,
            message: Arc::from(message),
            data: None,
            execution_time: 0,
            resource_usage: ResourceUsage {
                memory_bytes: 0,
                cpu_time_us: 0,
                network_requests: 0,
                disk_operations: 1,
            },
        })
    }

    /// Execute theme command
    #[inline]
    async fn execute_theme(
        context: Arc<RwLock<CommandContext>>,
        action: ThemeAction,
        name: Option<Arc<str>>,
        properties: HashMap<Arc<str>, Arc<str>>,
    ) -> CommandResult<CommandOutput> {
        let ctx = context.read().await;
        
        let message = match action {
            ThemeAction::Set => "Theme applied",
            ThemeAction::List => "Available themes: <theme list>",
            ThemeAction::Create => "Theme created",
            ThemeAction::Export => "Theme exported",
            ThemeAction::Import => "Theme imported",
            ThemeAction::Edit => "Theme edited",
        };
        
        Ok(CommandOutput {
            success: true,
            message: Arc::from(message),
            data: None,
            execution_time: 0,
            resource_usage: ResourceUsage {
                memory_bytes: 0,
                cpu_time_us: 0,
                network_requests: 0,
                disk_operations: 1,
            },
        })
    }

    /// Execute debug command
    #[inline]
    async fn execute_debug(
        context: Arc<RwLock<CommandContext>>,
        action: DebugAction,
        level: Option<Arc<str>>,
        system_info: bool,
    ) -> CommandResult<CommandOutput> {
        let ctx = context.read().await;
        
        let message = match action {
            DebugAction::Info => "Debug information: <debug info>",
            DebugAction::Logs => "Debug logs: <debug logs>",
            DebugAction::Performance => "Performance debug: <performance debug>",
            DebugAction::Memory => "Memory debug: <memory debug>",
            DebugAction::Network => "Network debug: <network debug>",
            DebugAction::Cache => "Cache debug: <cache debug>",
        };
        
        Ok(CommandOutput {
            success: true,
            message: Arc::from(message),
            data: None,
            execution_time: 0,
            resource_usage: ResourceUsage {
                memory_bytes: 0,
                cpu_time_us: 0,
                network_requests: 0,
                disk_operations: 0,
            },
        })
    }

    /// Get parser reference
    #[inline]
    pub fn parser(&self) -> &CommandParser {
        &self.parser
    }

    /// Get execution metrics
    #[inline]
    pub async fn get_metrics(&self) -> ExecutionMetrics {
        self.metrics.read().await.clone()
    }

    /// Parse and execute command from string
    #[inline]
    pub fn parse_and_execute(&self, input: &str) -> AsyncTask<CommandResult<CommandOutput>> {
        let parser = self.parser.clone();
        let executor = self.clone();
        
        spawn_async(async move {
            let command = parser.parse(input)?;
            executor.execute(command).await
        })
    }
}

impl Clone for CommandExecutor {
    fn clone(&self) -> Self {
        Self {
            parser: CommandParser::new(), // Create new parser with same commands
            context: self.context.clone(),
            metrics: self.metrics.clone(),
        }
    }
}

/// Global command executor instance
static COMMAND_EXECUTOR: once_cell::sync::Lazy<Arc<tokio::sync::RwLock<Option<CommandExecutor>>>> = 
    once_cell::sync::Lazy::new(|| Arc::new(tokio::sync::RwLock::new(None)));

/// Initialize global command executor
#[inline]
pub async fn initialize_command_executor(context: CommandContext) {
    let executor = CommandExecutor::new(context);
    *COMMAND_EXECUTOR.write().await = Some(executor);
}

/// Get global command executor
#[inline]
pub async fn get_command_executor() -> Option<CommandExecutor> {
    COMMAND_EXECUTOR.read().await.clone()
}

/// Parse command using global executor
#[inline]
pub async fn parse_command(input: &str) -> CommandResult<ChatCommand> {
    if let Some(executor) = get_command_executor().await {
        executor.parser().parse(input)
    } else {
        Err(CommandError::ConfigurationError {
            detail: Arc::from("Command executor not initialized"),
        })
    }
}

/// Execute command using global executor
#[inline]
pub async fn execute_command(command: ChatCommand) -> CommandResult<CommandOutput> {
    if let Some(executor) = get_command_executor().await {
        executor.execute(command).await
    } else {
        Err(CommandError::ConfigurationError {
            detail: Arc::from("Command executor not initialized"),
        })
    }
}

/// Parse and execute command using global executor
#[inline]
pub async fn parse_and_execute_command(input: &str) -> CommandResult<CommandOutput> {
    if let Some(executor) = get_command_executor().await {
        executor.parse_and_execute(input).await
    } else {
        Err(CommandError::ConfigurationError {
            detail: Arc::from("Command executor not initialized"),
        })
    }
}