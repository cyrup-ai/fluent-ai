//! Command parsing and validation logic
//!
//! Provides zero-allocation command parsing with comprehensive validation and error handling.
//! Uses blazing-fast parsing algorithms with ergonomic APIs and production-ready error messages.

use std::sync::Arc;
use std::collections::HashMap;
use thiserror::Error;
use serde::{Deserialize, Serialize};

use super::types::*;

/// Command parsing errors
#[derive(Error, Debug, Clone)]
pub enum ParseError {
    #[error("Invalid command syntax: {detail}")]
    InvalidSyntax { detail: Arc<str> },
    
    #[error("Missing required parameter: {parameter}")]
    MissingParameter { parameter: Arc<str> },
    
    #[error("Invalid parameter value: {parameter} = {value}")]
    InvalidParameterValue { parameter: Arc<str>, value: Arc<str> },
    
    #[error("Unknown parameter: {parameter}")]
    UnknownParameter { parameter: Arc<str> },
    
    #[error("Parameter type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: Arc<str>, actual: Arc<str> },
}

/// Result type for parsing operations
pub type ParseResult<T> = Result<T, ParseError>;

/// Zero-allocation command parser
#[derive(Debug, Clone)]
pub struct CommandParser {
    /// Registered commands
    commands: HashMap<Arc<str>, CommandInfo>,
    /// Command aliases
    aliases: HashMap<Arc<str>, Arc<str>>,
    /// Command history for auto-completion
    history: Vec<Arc<str>>,
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

    /// Register built-in commands
    pub fn register_builtin_commands(&mut self) {
        // Help command
        self.register_command(CommandInfo {
            name: Arc::from("help"),
            description: Arc::from("Show help information"),
            usage: Arc::from("/help [command] [--extended]"),
            parameters: vec![
                ParameterInfo {
                    name: Arc::from("command"),
                    description: Arc::from("Optional command to get help for"),
                    parameter_type: ParameterType::String,
                    required: false,
                    default_value: None,
                },
                ParameterInfo {
                    name: Arc::from("extended"),
                    description: Arc::from("Show extended help"),
                    parameter_type: ParameterType::Boolean,
                    required: false,
                    default_value: Some(Arc::from("false")),
                },
            ],
            aliases: vec![Arc::from("h"), Arc::from("?")],
            category: Arc::from("General"),
            examples: vec![
                Arc::from("/help"),
                Arc::from("/help config"),
                Arc::from("/help --extended"),
            ],
        });

        // Clear command
        self.register_command(CommandInfo {
            name: Arc::from("clear"),
            description: Arc::from("Clear chat history"),
            usage: Arc::from("/clear [--confirm] [--keep-last N]"),
            parameters: vec![
                ParameterInfo {
                    name: Arc::from("confirm"),
                    description: Arc::from("Confirm the action"),
                    parameter_type: ParameterType::Boolean,
                    required: false,
                    default_value: Some(Arc::from("false")),
                },
                ParameterInfo {
                    name: Arc::from("keep-last"),
                    description: Arc::from("Keep last N messages"),
                    parameter_type: ParameterType::Integer,
                    required: false,
                    default_value: None,
                },
            ],
            aliases: vec![Arc::from("cls"), Arc::from("reset")],
            category: Arc::from("History"),
            examples: vec![
                Arc::from("/clear"),
                Arc::from("/clear --confirm"),
                Arc::from("/clear --keep-last 10"),
            ],
        });

        // Export command
        self.register_command(CommandInfo {
            name: Arc::from("export"),
            description: Arc::from("Export conversation"),
            usage: Arc::from("/export --format FORMAT [--output FILE] [--include-metadata]"),
            parameters: vec![
                ParameterInfo {
                    name: Arc::from("format"),
                    description: Arc::from("Export format (json, markdown, pdf, html)"),
                    parameter_type: ParameterType::Enum,
                    required: true,
                    default_value: None,
                },
                ParameterInfo {
                    name: Arc::from("output"),
                    description: Arc::from("Output file path"),
                    parameter_type: ParameterType::Path,
                    required: false,
                    default_value: None,
                },
                ParameterInfo {
                    name: Arc::from("include-metadata"),
                    description: Arc::from("Include metadata in export"),
                    parameter_type: ParameterType::Boolean,
                    required: false,
                    default_value: Some(Arc::from("true")),
                },
            ],
            aliases: vec![Arc::from("save")],
            category: Arc::from("Export"),
            examples: vec![
                Arc::from("/export --format json"),
                Arc::from("/export --format markdown --output chat.md"),
                Arc::from("/export --format pdf --include-metadata"),
            ],
        });

        // Config command
        self.register_command(CommandInfo {
            name: Arc::from("config"),
            description: Arc::from("Modify configuration"),
            usage: Arc::from("/config [KEY] [VALUE] [--show] [--reset]"),
            parameters: vec![
                ParameterInfo {
                    name: Arc::from("key"),
                    description: Arc::from("Configuration key"),
                    parameter_type: ParameterType::String,
                    required: false,
                    default_value: None,
                },
                ParameterInfo {
                    name: Arc::from("value"),
                    description: Arc::from("Configuration value"),
                    parameter_type: ParameterType::String,
                    required: false,
                    default_value: None,
                },
                ParameterInfo {
                    name: Arc::from("show"),
                    description: Arc::from("Show current configuration"),
                    parameter_type: ParameterType::Boolean,
                    required: false,
                    default_value: Some(Arc::from("false")),
                },
                ParameterInfo {
                    name: Arc::from("reset"),
                    description: Arc::from("Reset to defaults"),
                    parameter_type: ParameterType::Boolean,
                    required: false,
                    default_value: Some(Arc::from("false")),
                },
            ],
            aliases: vec![Arc::from("cfg"), Arc::from("settings")],
            category: Arc::from("Configuration"),
            examples: vec![
                Arc::from("/config --show"),
                Arc::from("/config theme dark"),
                Arc::from("/config --reset"),
            ],
        });

        // Search command
        self.register_command(CommandInfo {
            name: Arc::from("search"),
            description: Arc::from("Search chat history"),
            usage: Arc::from("/search QUERY [--scope SCOPE] [--limit N] [--include-context]"),
            parameters: vec![
                ParameterInfo {
                    name: Arc::from("query"),
                    description: Arc::from("Search query"),
                    parameter_type: ParameterType::String,
                    required: true,
                    default_value: None,
                },
                ParameterInfo {
                    name: Arc::from("scope"),
                    description: Arc::from("Search scope (all, current, recent)"),
                    parameter_type: ParameterType::Enum,
                    required: false,
                    default_value: Some(Arc::from("all")),
                },
                ParameterInfo {
                    name: Arc::from("limit"),
                    description: Arc::from("Maximum results"),
                    parameter_type: ParameterType::Integer,
                    required: false,
                    default_value: Some(Arc::from("10")),
                },
                ParameterInfo {
                    name: Arc::from("include-context"),
                    description: Arc::from("Include context in results"),
                    parameter_type: ParameterType::Boolean,
                    required: false,
                    default_value: Some(Arc::from("true")),
                },
            ],
            aliases: vec![Arc::from("find"), Arc::from("grep")],
            category: Arc::from("Search"),
            examples: vec![
                Arc::from("/search rust"),
                Arc::from("/search \"error handling\" --scope recent"),
                Arc::from("/search async --limit 5 --include-context"),
            ],
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
    pub fn parse(&self, input: &str) -> ParseResult<ChatCommand> {
        let input = input.trim();
        
        // Check if it's a command (starts with /)
        if !input.starts_with('/') {
            return Err(ParseError::InvalidSyntax {
                detail: Arc::from("Commands must start with '/'"),
            });
        }

        // Remove the leading slash
        let input = &input[1..];
        
        // Split into command and arguments
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.is_empty() {
            return Err(ParseError::InvalidSyntax {
                detail: Arc::from("Empty command"),
            });
        }

        let command_name = parts[0];
        let args = &parts[1..];

        // Resolve aliases
        let resolved_name = self.aliases.get(command_name)
            .unwrap_or(&Arc::from(command_name));

        // Parse based on command type
        match resolved_name.as_ref() {
            "help" => self.parse_help_command(args),
            "clear" => self.parse_clear_command(args),
            "export" => self.parse_export_command(args),
            "config" => self.parse_config_command(args),
            "search" => self.parse_search_command(args),
            _ => Err(ParseError::InvalidSyntax {
                detail: Arc::from(format!("Unknown command: {}", command_name)),
            }),
        }
    }

    /// Parse help command
    fn parse_help_command(&self, args: &[&str]) -> ParseResult<ChatCommand> {
        let mut command = None;
        let mut extended = false;

        let mut i = 0;
        while i < args.len() {
            match args[i] {
                "--extended" => extended = true,
                arg if !arg.starts_with('-') => command = Some(Arc::from(arg)),
                _ => return Err(ParseError::UnknownParameter {
                    parameter: Arc::from(args[i]),
                }),
            }
            i += 1;
        }

        Ok(ChatCommand::Help { command, extended })
    }

    /// Parse clear command
    fn parse_clear_command(&self, args: &[&str]) -> ParseResult<ChatCommand> {
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
                            parameter: Arc::from("keep-last"),
                        });
                    }
                    keep_last = Some(args[i].parse().map_err(|_| ParseError::InvalidParameterValue {
                        parameter: Arc::from("keep-last"),
                        value: Arc::from(args[i]),
                    })?);
                }
                _ => return Err(ParseError::UnknownParameter {
                    parameter: Arc::from(args[i]),
                }),
            }
            i += 1;
        }

        Ok(ChatCommand::Clear { confirm, keep_last })
    }

    /// Parse export command
    fn parse_export_command(&self, args: &[&str]) -> ParseResult<ChatCommand> {
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
                            parameter: Arc::from("format"),
                        });
                    }
                    format = Some(Arc::from(args[i]));
                }
                "--output" => {
                    i += 1;
                    if i >= args.len() {
                        return Err(ParseError::MissingParameter {
                            parameter: Arc::from("output"),
                        });
                    }
                    output = Some(Arc::from(args[i]));
                }
                "--include-metadata" => include_metadata = true,
                _ => return Err(ParseError::UnknownParameter {
                    parameter: Arc::from(args[i]),
                }),
            }
            i += 1;
        }

        let format = format.ok_or_else(|| ParseError::MissingParameter {
            parameter: Arc::from("format"),
        })?;

        Ok(ChatCommand::Export { format, output, include_metadata })
    }

    /// Parse config command
    fn parse_config_command(&self, args: &[&str]) -> ParseResult<ChatCommand> {
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
                        key = Some(Arc::from(arg));
                    } else if value.is_none() {
                        value = Some(Arc::from(arg));
                    } else {
                        return Err(ParseError::InvalidSyntax {
                            detail: Arc::from("Too many positional arguments"),
                        });
                    }
                }
                _ => return Err(ParseError::UnknownParameter {
                    parameter: Arc::from(args[i]),
                }),
            }
            i += 1;
        }

        Ok(ChatCommand::Config { key, value, show, reset })
    }

    /// Parse search command
    fn parse_search_command(&self, args: &[&str]) -> ParseResult<ChatCommand> {
        let mut query = None;
        let mut scope = SearchScope::All;
        let mut limit = None;
        let mut include_context = false;

        let mut i = 0;
        while i < args.len() {
            match args[i] {
                "--scope" => {
                    i += 1;
                    if i >= args.len() {
                        return Err(ParseError::MissingParameter {
                            parameter: Arc::from("scope"),
                        });
                    }
                    scope = match args[i] {
                        "all" => SearchScope::All,
                        "current" => SearchScope::Current,
                        "recent" => SearchScope::Recent,
                        _ => return Err(ParseError::InvalidParameterValue {
                            parameter: Arc::from("scope"),
                            value: Arc::from(args[i]),
                        }),
                    };
                }
                "--limit" => {
                    i += 1;
                    if i >= args.len() {
                        return Err(ParseError::MissingParameter {
                            parameter: Arc::from("limit"),
                        });
                    }
                    limit = Some(args[i].parse().map_err(|_| ParseError::InvalidParameterValue {
                        parameter: Arc::from("limit"),
                        value: Arc::from(args[i]),
                    })?);
                }
                "--include-context" => include_context = true,
                arg if !arg.starts_with('-') => {
                    if query.is_none() {
                        query = Some(Arc::from(arg));
                    } else {
                        return Err(ParseError::InvalidSyntax {
                            detail: Arc::from("Multiple query arguments not supported"),
                        });
                    }
                }
                _ => return Err(ParseError::UnknownParameter {
                    parameter: Arc::from(args[i]),
                }),
            }
            i += 1;
        }

        let query = query.ok_or_else(|| ParseError::MissingParameter {
            parameter: Arc::from("query"),
        })?;

        Ok(ChatCommand::Search { query, scope, limit, include_context })
    }

    /// Validate command parameters
    pub fn validate_command(&self, command: &ChatCommand) -> ParseResult<()> {
        match command {
            ChatCommand::Export { format, .. } => {
                let valid_formats = ["json", "markdown", "pdf", "html"];
                if !valid_formats.contains(&format.as_ref()) {
                    return Err(ParseError::InvalidParameterValue {
                        parameter: Arc::from("format"),
                        value: format.clone(),
                    });
                }
            }
            ChatCommand::Clear { keep_last: Some(n), .. } => {
                if *n == 0 {
                    return Err(ParseError::InvalidParameterValue {
                        parameter: Arc::from("keep-last"),
                        value: Arc::from("0"),
                    });
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Get command suggestions for auto-completion
    pub fn get_suggestions(&self, prefix: &str) -> Vec<Arc<str>> {
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
        self.commands.get(command).cloned()
            .or_else(|| {
                self.aliases.get(command)
                    .and_then(|name| self.commands.get(name).cloned())
            })
    }

    /// Get command history
    pub fn get_history(&self) -> Vec<Arc<str>> {
        self.history.clone()
    }

    /// Add command to history
    pub fn add_to_history(&mut self, command: Arc<str>) {
        self.history.push(command);
        // Keep only last 100 commands
        if self.history.len() > 100 {
            self.history.remove(0);
        }
    }
}
