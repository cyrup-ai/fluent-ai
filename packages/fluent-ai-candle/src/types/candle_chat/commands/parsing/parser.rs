//! Main command parser implementation
//!
//! Contains the CommandParser struct and core parsing functionality.

use std::collections::HashMap;
use super::errors::{ParseError, ParseResult};
use super::registry::CommandInfo;
use super::super::types::{ImmutableChatCommand, CommandError, SearchScope};

/// Zero-allocation command parser with owned strings
#[derive(Debug, Clone)]
pub struct CommandParser {
    /// Registered commands
    pub(super) commands: HashMap<String, CommandInfo>,
    /// Command aliases
    pub(super) aliases: HashMap<String, String>,
    /// Command history for auto-completion
    pub(super) history: Vec<String>}

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
            history: Vec::new()};
        parser.register_builtin_commands();
        parser
    }

    /// Parse command from input string (zero-allocation)
    pub fn parse_command(&self, input: &str) -> Result<ImmutableChatCommand, CommandError> {
        let input = input.trim();
        if input.is_empty() {
            return Err(CommandError::InvalidSyntax {
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
            return Err(CommandError::InvalidSyntax {
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
            _ => Err(CommandError::UnknownCommand {
                command: command_name})}
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
            "search" => self.parse_search_command(args),
            _ => Err(ParseError::InvalidSyntax {
                detail: format!("Unknown command: {}", command_name)})}
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