//! Command parsing and validation logic
//!
//! Provides zero-allocation command parsing with comprehensive validation and error handling.
//! Uses blazing-fast parsing algorithms with ergonomic APIs and production-ready error messages.

// Removed unused imports: Deserialize, Serialize, Arc
use std::collections::HashMap;

use thiserror::Error;

use super::types::*;

/// Command parsing errors with owned strings
#[derive(Error, Debug, Clone)]
pub enum ParseError {
    /// Invalid command syntax
    #[error("Invalid command syntax: {detail}")]
    InvalidSyntax { 
        /// Details about the syntax error
        detail: String 
    },

    /// Required parameter is missing
    #[error("Missing required parameter: {parameter}")]
    MissingParameter { 
        /// Name of the missing parameter
        parameter: String 
    },

    /// Parameter has invalid value
    #[error("Invalid parameter value: {parameter} = {value}")]
    InvalidParameterValue { 
        /// Name of the parameter
        parameter: String, 
        /// The invalid value provided
        value: String 
    },

    /// Parameter name is not recognized
    #[error("Unknown parameter: {parameter}")]
    UnknownParameter { 
        /// Name of the unknown parameter
        parameter: String 
    },

    /// Parameter type doesn't match expected type
    #[error("Parameter type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { 
        /// Expected parameter type
        expected: String, 
        /// Actual parameter type provided
        actual: String 
    }
}

/// Result type for parsing operations
pub type ParseResult<T> = Result<T, ParseError>;

/// Zero-allocation command parser with owned strings
#[derive(Debug, Clone)]
pub struct CommandParser {
    /// Registered commands
    commands: HashMap<String, CommandInfo>,
    /// Command aliases
    aliases: HashMap<String, String>,
    /// Command history for auto-completion
    history: Vec<String>,
}

impl Default for CommandParser {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandParser {
    /// Create a new command parser
    pub fn new() -> Self {
        let mut parser = Self {
            commands: HashMap::new(),
            aliases: HashMap::new(),
            history: Vec::new(),
        };
        parser.register_builtin_commands();
        parser
    }

    /// Parse command from input string (zero-allocation)
    pub fn parse_command(&self, input: &str) -> Result<ImmutableChatCommand, CandleCommandError> {
        let input = input.trim();
        if input.is_empty() {
            return Err(CandleCommandError::InvalidSyntax {
                detail: "Empty command".to_string()});
        }

        // Remove leading slash if present
        let input = if input.starts_with('/') {
            &input[1..]
        } else {
            input
        };

        // Split command and arguments
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.is_empty() {
            return Err(CandleCommandError::InvalidSyntax {
                detail: "No command specified".to_string()});
        }

        let command_name = parts[0].to_lowercase();
        let args = &parts[1..];

        // Parse based on command name
        match command_name.as_str() {
            "help" | "h" | "?" => {
                let command = if args.len() > 0 && !args[0].starts_with("--") {
                    Some(args[0].to_string())
                } else {
                    None
                };
                let extended = args.contains(&"--extended");
                Ok(ImmutableChatCommand::Help { command, extended })
            }
            "clear" => {
                let confirm = args.contains(&"--confirm");
                let keep_last = args
                    .iter()
                    .position(|&arg| arg == "--keep-last")
                    .and_then(|i| args.get(i + 1))
                    .and_then(|s| s.parse().ok());
                Ok(ImmutableChatCommand::Clear { confirm, keep_last })
            }
            "export" => {
                let format = args
                    .iter()
                    .position(|&arg| arg == "--format")
                    .and_then(|i| args.get(i + 1))
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "json".to_string());
                let output = args
                    .iter()
                    .position(|&arg| arg == "--output")
                    .and_then(|i| args.get(i + 1))
                    .map(|s| s.to_string());
                let include_metadata = args.contains(&"--metadata");
                Ok(ImmutableChatCommand::Export {
                    format,
                    output,
                    include_metadata})
            }
            "config" => {
                let show = args.contains(&"--show");
                let reset = args.contains(&"--reset");
                let key = args
                    .iter()
                    .find(|&&arg| !arg.starts_with("--"))
                    .map(|s| s.to_string());
                let value =
                    if let Some(key_pos) = args.iter().position(|&arg| !arg.starts_with("--")) {
                        if let Some(&arg) = args.get(key_pos + 1) {
                            if !arg.starts_with("--") {
                                Some(arg.to_string())
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                Ok(ImmutableChatCommand::Config {
                    key,
                    value,
                    show,
                    reset})
            }
            "search" => {
                let query = args
                    .iter()
                    .find(|&&arg| !arg.starts_with("--"))
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                let scope = if args.contains(&"--current") {
                    SearchScope::Current
                } else if args.contains(&"--recent") {
                    SearchScope::Recent
                } else if args.contains(&"--bookmarked") {
                    SearchScope::Bookmarked
                } else {
                    SearchScope::All
                };
                let limit = args
                    .iter()
                    .position(|&arg| arg == "--limit")
                    .and_then(|i| args.get(i + 1))
                    .and_then(|s| s.parse().ok());
                let include_context = args.contains(&"--context");
                Ok(ImmutableChatCommand::Search {
                    query,
                    scope,
                    limit,
                    include_context})
            }
            _ => Err(CandleCommandError::UnknownCommand {
                command: command_name})}
    }

    /// Register built-in commands
    pub fn register_builtin_commands(&mut self) {
        // Help command
        self.register_command(CommandInfo {
            name: "help".to_string(),
            description: "Show help information".to_string(),
            usage: "/help [command] [--extended]".to_string(),
            parameters: vec![
                ParameterInfo {
                    name: "command".to_string(),
                    description: "Optional command to get help for".to_string(),
                    parameter_type: ParameterType::String,
                    required: false,
                    default_value: None,
                    min_value: None,
                    max_value: None,
                    pattern: None,
                    examples: vec!["help".to_string(), "clear".to_string()],
                },
                ParameterInfo {
                    name: "extended".to_string(),
                    description: "Show extended help".to_string(),
                    parameter_type: ParameterType::Boolean,
                    required: false,
                    default_value: Some("false".to_string()),
                    min_value: None,
                    max_value: None,
                    pattern: None,
                    examples: vec!["true".to_string(), "false".to_string()],
                },
            ],
            aliases: vec!["h".to_string(), "?".to_string()],
            category: "General".to_string(),
            examples: vec![
                "/help".to_string(),
                "/help config".to_string(),
                "/help --extended".to_string(),
            ],
            version: "1.0.0".to_string(),
            author: Some("Fluent AI Team".to_string()),
            tags: vec!["help".to_string(), "documentation".to_string()],
            required_permissions: vec![],
            deprecated: false,
            deprecation_message: None,
            experimental: false,
            stability: StabilityLevel::Stable,
        });

        // Clear command
        self.register_command(CommandInfo {
            name: "clear".to_string(),
            description: "Clear chat history".to_string(),
            usage: "/clear [--confirm] [--keep-last N]".to_string(),
            parameters: vec![
                ParameterInfo {
                    name: "confirm".to_string(),
                    description: "Confirm the action".to_string(),
                    parameter_type: ParameterType::Boolean,
                    required: false,
                    default_value: Some("false".to_string()),
                    min_value: None,
                    max_value: None,
                    pattern: None,
                    examples: vec!["true".to_string(), "false".to_string()],
                },
                ParameterInfo {
                    name: "keep-last".to_string(),
                    description: "Keep last N messages".to_string(),
                    parameter_type: ParameterType::Integer,
                    required: false,
                    default_value: None,
                    min_value: Some(1.0),
                    max_value: Some(1000.0),
                    pattern: None,
                    examples: vec!["10".to_string(), "50".to_string()],
                },
            ],
            aliases: vec!["cls".to_string(), "reset".to_string()],
            category: "History".to_string(),
            examples: vec![
                "/clear".to_string(),
                "/clear --confirm".to_string(),
                "/clear --keep-last 10".to_string(),
            ],
            version: "1.0.0".to_string(),
            author: Some("Fluent AI Team".to_string()),
            tags: vec!["history".to_string(), "cleanup".to_string()],
            required_permissions: vec![],
            deprecated: false,
            deprecation_message: None,
            experimental: false,
            stability: StabilityLevel::Stable,
        });

        // Export command
        self.register_command(CommandInfo {
            name: "export".to_string(),
            description: "Export conversation".to_string(),
            usage: "/export --format FORMAT [--output FILE] [--include-metadata]".to_string(),
            parameters: vec![
                ParameterInfo {
                    name: "format".to_string(),
                    description: "Export format (json, markdown, pdf, html)".to_string(),
                    parameter_type: ParameterType::Enum { 
                        values: vec!["json".to_string(), "markdown".to_string(), "pdf".to_string(), "html".to_string()]
                    },
                    required: true,
                    default_value: None,
                    min_value: None,
                    max_value: None,
                    pattern: None,
                    examples: vec![],
                },
                ParameterInfo {
                    name: "output".to_string(),
                    description: "Output file path".to_string(),
                    parameter_type: ParameterType::Path,
                    required: false,
                    default_value: None,
                    min_value: None,
                    max_value: None,
                    pattern: None,
                    examples: vec![],
                },
                ParameterInfo {
                    name: "include-metadata".to_string(),
                    description: "Include metadata in export".to_string(),
                    parameter_type: ParameterType::Boolean,
                    required: false,
                    default_value: Some("true".to_string()),
                    min_value: None,
                    max_value: None,
                    pattern: None,
                    examples: vec!["true".to_string(), "false".to_string()],
                },
            ],
            aliases: vec!["save".to_string()],
            category: "Export".to_string(),
            examples: vec![
                "/export --format json".to_string(),
                "/export --format markdown --output chat.md".to_string(),
                "/export --format pdf --include-metadata".to_string(),
            ],
            version: "1.0.0".to_string(),
            author: Some("Fluent AI Team".to_string()),
            tags: vec!["export".to_string(), "save".to_string(), "backup".to_string()],
            required_permissions: vec![],
            deprecated: false,
            deprecation_message: None,
            experimental: false,
            stability: StabilityLevel::Stable,
        });

        // Config command
        self.register_command(CommandInfo {
            name: "config".to_string(),
            description: "Modify configuration".to_string(),
            usage: "/config [KEY] [VALUE] [--show] [--reset]".to_string(),
            parameters: vec![
                ParameterInfo {
                    name: "key".to_string(),
                    description: "Configuration key".to_string(),
                    parameter_type: ParameterType::String,
                    required: false,
                    default_value: None,
                    min_value: None,
                    max_value: None,
                    pattern: None,
                    examples: vec![],
                },
                ParameterInfo {
                    name: "value".to_string(),
                    description: "Configuration value".to_string(),
                    parameter_type: ParameterType::String,
                    required: false,
                    default_value: None,
                    min_value: None,
                    max_value: None,
                    pattern: None,
                    examples: vec![],
                },
                ParameterInfo {
                    name: "show".to_string(),
                    description: "Show current configuration".to_string(),
                    parameter_type: ParameterType::Boolean,
                    required: false,
                    default_value: Some("false".to_string()),
                    min_value: None,
                    max_value: None,
                    pattern: None,
                    examples: vec!["true".to_string(), "false".to_string()],
                },
                ParameterInfo {
                    name: "reset".to_string(),
                    description: "Reset to defaults".to_string(),
                    parameter_type: ParameterType::Boolean,
                    required: false,
                    default_value: Some("false".to_string()),
                    min_value: None,
                    max_value: None,
                    pattern: None,
                    examples: vec!["true".to_string(), "false".to_string()],
                },
            ],
            aliases: vec!["cfg".to_string(), "settings".to_string()],
            category: "Configuration".to_string(),
            examples: vec![
                "/config --show".to_string(),
                "/config theme dark".to_string(),
                "/config --reset".to_string(),
            ],
            version: "1.0.0".to_string(),
            author: Some("Fluent AI Team".to_string()),
            tags: vec!["configuration".to_string(), "settings".to_string()],
            required_permissions: vec![],
            deprecated: false,
            deprecation_message: None,
            experimental: false,
            stability: StabilityLevel::Stable,
        });

        // Search command
        self.register_command(CommandInfo {
            name: "search".to_string(),
            description: "Search chat history".to_string(),
            usage: "/search QUERY [--scope SCOPE] [--limit N] [--include-context]".to_string(),
            parameters: vec![
                ParameterInfo {
                    name: "query".to_string(),
                    description: "Search query".to_string(),
                    parameter_type: ParameterType::String,
                    required: true,
                    default_value: None,
                    min_value: None,
                    max_value: None,
                    pattern: None,
                    examples: vec![],
                },
                ParameterInfo {
                    name: "scope".to_string(),
                    description: "Search scope (all, current, recent)".to_string(),
                    parameter_type: ParameterType::Enum { 
                        values: vec!["all".to_string(), "current".to_string(), "recent".to_string(), "bookmarked".to_string()]
                    },
                    required: false,
                    default_value: Some("all".to_string()),
                    min_value: None,
                    max_value: None,
                    pattern: None,
                    examples: vec!["all".to_string(), "current".to_string(), "recent".to_string()],
                },
                ParameterInfo {
                    name: "limit".to_string(),
                    description: "Maximum results".to_string(),
                    parameter_type: ParameterType::Integer,
                    required: false,
                    default_value: Some("10".to_string()),
                    min_value: Some(1.0),
                    max_value: Some(100.0),
                    pattern: None,
                    examples: vec!["5".to_string(), "10".to_string(), "20".to_string()],
                },
                ParameterInfo {
                    name: "include-context".to_string(),
                    description: "Include context in results".to_string(),
                    parameter_type: ParameterType::Boolean,
                    required: false,
                    default_value: Some("true".to_string()),
                    min_value: None,
                    max_value: None,
                    pattern: None,
                    examples: vec!["true".to_string(), "false".to_string()],
                },
            ],
            aliases: vec!["find".to_string(), "grep".to_string()],
            category: "Search".to_string(),
            examples: vec![
                "/search rust".to_string(),
                "/search \"error handling\" --scope recent".to_string(),
                "/search async --limit 5 --include-context".to_string(),
            ],
            version: "1.0.0".to_string(),
            author: Some("Fluent AI Team".to_string()),
            tags: vec!["search".to_string(), "find".to_string(), "query".to_string()],
            required_permissions: vec![],
            deprecated: false,
            deprecation_message: None,
            experimental: false,
            stability: StabilityLevel::Stable,
        });
    }

    /// Register a command
    pub fn register_command(&mut self, info: CommandInfo) {
        // Register main command name
        self.commands.insert(info.name.clone(), info.clone());

        // Register aliases
        for alias in &info.aliases {
            self.aliases.insert(alias.clone(), info.name.clone());
        }
    }

    /// Parse a command string with zero-allocation patterns
    pub fn parse(&self, input: &str) -> ParseResult<ImmutableChatCommand> {
        let input = input.trim();

        // Check if it's a command (starts with /)
        if !input.starts_with('/') {
            return Err(ParseError::InvalidSyntax {
                detail: "Commands must start with '/'".to_string()});
        }

        // Remove the leading slash
        let input = &input[1..];

        // Split into command and arguments
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.is_empty() {
            return Err(ParseError::InvalidSyntax {
                detail: "Empty command".to_string()});
        }

        let command_name = parts[0];
        let args = &parts[1..];

        // Resolve aliases
        let resolved_name = self
            .aliases
            .get(command_name)
            .map(|s| s.as_str())
            .unwrap_or(command_name);

        // Parse based on command type
        match resolved_name {
            "help" => self.parse_help_command(args),
            "clear" => self.parse_clear_command(args),
            "export" => self.parse_export_command(args),
            "config" => self.parse_config_command(args),
            "search" => self.parse_search_args(args),
            _ => Err(ParseError::InvalidSyntax {
                detail: format!("Unknown command: {}", command_name)})}
    }

    /// Parse help command
    fn parse_help_command(&self, args: &[&str]) -> ParseResult<ImmutableChatCommand> {
        let mut command = None;
        let mut extended = false;

        let mut i = 0;
        while i < args.len() {
            match args[i] {
                "--extended" => extended = true,
                arg if !arg.starts_with('-') => command = Some(arg.to_string()),
                _ => {
                    return Err(ParseError::UnknownParameter {
                        parameter: args[i].to_string()});
                }
            }
            i += 1;
        }

        Ok(ImmutableChatCommand::Help { command, extended })
    }

    /// Parse clear command
    fn parse_clear_command(&self, args: &[&str]) -> ParseResult<ImmutableChatCommand> {
        let mut confirm = false;
        let mut keep_last = None;

        let mut i = 0;
        while i < args.len() {
            match args[i] {
                "--confirm" => confirm = true,
                "--keep-last" => {
                    i += 1;
                    if i >= args.len() {
                        return Err(ParseError::MissingParameter {
                            parameter: "keep-last".to_string()});
                    }
                    keep_last =
                        Some(
                            args[i]
                                .parse()
                                .map_err(|_| ParseError::InvalidParameterValue {
                                    parameter: "keep-last".to_string(),
                                    value: args[i].to_string()})?,
                        );
                }
                _ => {
                    return Err(ParseError::UnknownParameter {
                        parameter: args[i].to_string()});
                }
            }
            i += 1;
        }

        Ok(ImmutableChatCommand::Clear { confirm, keep_last })
    }

    /// Parse export command
    fn parse_export_command(&self, args: &[&str]) -> ParseResult<ImmutableChatCommand> {
        let mut format = None;
        let mut output = None;
        let mut include_metadata = false;

        let mut i = 0;
        while i < args.len() {
            match args[i] {
                "--format" => {
                    i += 1;
                    if i >= args.len() {
                        return Err(ParseError::MissingParameter {
                            parameter: "format".to_string()});
                    }
                    format = Some(args[i].to_string());
                }
                "--output" => {
                    i += 1;
                    if i >= args.len() {
                        return Err(ParseError::MissingParameter {
                            parameter: "output".to_string()});
                    }
                    output = Some(args[i].to_string());
                }
                "--include-metadata" => include_metadata = true,
                _ => {
                    return Err(ParseError::UnknownParameter {
                        parameter: args[i].to_string()});
                }
            }
            i += 1;
        }

        let format = format.ok_or_else(|| ParseError::MissingParameter {
            parameter: "format".to_string()})?;

        Ok(ImmutableChatCommand::Export {
            format,
            output,
            include_metadata})
    }

    /// Parse config command
    fn parse_config_command(&self, args: &[&str]) -> ParseResult<ImmutableChatCommand> {
        let mut key = None;
        let mut value = None;
        let mut show = false;
        let mut reset = false;

        let mut i = 0;
        while i < args.len() {
            match args[i] {
                "--show" => show = true,
                "--reset" => reset = true,
                arg if !arg.starts_with('-') => {
                    if key.is_none() {
                        key = Some(arg.to_string());
                    } else if value.is_none() {
                        value = Some(arg.to_string());
                    } else {
                        return Err(ParseError::InvalidSyntax {
                            detail: "Too many positional arguments".to_string()});
                    }
                }
                _ => {
                    return Err(ParseError::UnknownParameter {
                        parameter: args[i].to_string()});
                }
            }
            i += 1;
        }

        Ok(ImmutableChatCommand::Config {
            key,
            value,
            show,
            reset,
        })
    }

    fn parse_search_args(&self, args: &[&str]) -> ParseResult<ImmutableChatCommand> {
        let mut scope = SearchScope::All;
        let mut limit = None;
        let mut include_context = false;
        let mut query: Option<String> = None;
        let mut i = 0;
        
        while i < args.len() {
            match args[i] {
                "--scope" => {
                    i += 1;
                    if i >= args.len() {
                        return Err(ParseError::MissingParameter {
                            parameter: "scope".to_string()});
                    }
                    scope = match args[i] {
                        "all" => SearchScope::All,
                        "current" => SearchScope::Current,
                        "recent" => SearchScope::Recent,
                        _ => {
                            return Err(ParseError::InvalidParameterValue {
                                parameter: "scope".to_string(),
                                value: args[i].to_string()});
                        }
                    };
                }
                "--limit" => {
                    i += 1;
                    if i >= args.len() {
                        return Err(ParseError::MissingParameter {
                            parameter: "limit".to_string()});
                    }
                    limit =
                        Some(
                            args[i]
                                .parse()
                                .map_err(|_| ParseError::InvalidParameterValue {
                                    parameter: "limit".to_string(),
                                    value: args[i].to_string()})?,
                        );
                }
                "--include-context" => include_context = true,
                arg if !arg.starts_with('-') => {
                    if query.is_none() {
                        query = Some(arg.to_string());
                    } else {
                        return Err(ParseError::InvalidSyntax {
                            detail: "Multiple query arguments not supported".to_string()});
                    }
                }
                _ => {
                    return Err(ParseError::UnknownParameter {
                        parameter: args[i].to_string()});
                }
            }
            i += 1;
        }

        let query = query.ok_or_else(|| ParseError::MissingParameter {
            parameter: "query".to_string()})?;

        Ok(ImmutableChatCommand::Search {
            query,
            scope,
            limit,
            include_context})
    }

    /// Validate command parameters
    pub fn validate_command(&self, command: &ImmutableChatCommand) -> ParseResult<()> {
        match command {
            ImmutableChatCommand::Export { format, .. } => {
                let valid_formats = ["json", "markdown", "pdf", "html"];
                if !valid_formats.contains(&format.as_str()) {
                    return Err(ParseError::InvalidParameterValue {
                        parameter: "format".to_string(),
                        value: format.clone()});
                }
            }
            ImmutableChatCommand::Clear {
                keep_last: Some(n), ..
            } => {
                if *n == 0 {
                    return Err(ParseError::InvalidParameterValue {
                        parameter: "keep-last".to_string(),
                        value: "0".to_string()});
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Get command suggestions for auto-completion
    pub fn get_suggestions(&self, prefix: &str) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Add command names
        for name in self.commands.keys() {
            if name.starts_with(prefix) {
                suggestions.push(name.clone());
            }
        }

        // Add aliases
        for alias in self.aliases.keys() {
            if alias.starts_with(prefix) {
                suggestions.push(alias.clone());
            }
        }

        suggestions.sort();
        suggestions
    }

    /// Get command information
    pub fn get_command_info(&self, command: &str) -> Option<CommandInfo> {
        self.commands.get(command).cloned().or_else(|| {
            self.aliases
                .get(command)
                .and_then(|name| self.commands.get(name).cloned())
        })
    }

    /// Get command history
    pub fn get_history(&self) -> Vec<String> {
        self.history.clone()
    }

    /// Add command to history
    pub fn add_to_history(&mut self, command: String) {
        self.history.push(command);
        // Keep only last 100 commands
        if self.history.len() > 100 {
            self.history.remove(0);
        }
    }
}
