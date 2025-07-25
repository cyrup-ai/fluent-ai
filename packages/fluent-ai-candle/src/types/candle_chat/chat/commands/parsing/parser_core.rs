//! Core command parser implementation with zero-allocation patterns
//!
//! Provides the main CommandParser struct with comprehensive command registration,
//! parsing, and management functionality using blazing-fast algorithms and
//! production-ready error handling.

use std::collections::HashMap;

use super::{
    builtin_commands::BuiltinCommands,
    error_handling::{ParseError, ParseResult},
    lexer::CommandLexer,
    validation::CommandValidator};
use crate::types::candle_chat::chat::commands::types::{
    CommandError, CommandInfo, ImmutableChatCommand};

/// Zero-allocation command parser with owned strings
#[derive(Debug, Clone)]
pub struct CommandParser {
    /// Lexer for tokenization
    lexer: CommandLexer,
    /// Validator for parameter validation
    validator: CommandValidator,
    /// Registered commands
    commands: HashMap<String, CommandInfo>,
    /// Command aliases
    aliases: HashMap<String, String>,
    /// Command history for auto-completion
    history: Vec<String>}

impl Default for CommandParser {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandParser {
    /// Create a new command parser
    pub fn new() -> Self {
        let mut parser = Self {
            lexer: CommandLexer::new(),
            validator: CommandValidator::new(),
            commands: HashMap::new(),
            aliases: HashMap::new(),
            history: Vec::new()};
        parser.register_builtin_commands();
        parser
    }

    /// Parse command from input string (zero-allocation)
    pub fn parse_command(&self, input: &str) -> Result<ImmutableChatCommand, CommandError> {
        self.lexer.parse_command(input)
    }

    /// Register built-in commands
    pub fn register_builtin_commands(&mut self) {
        for command_info in BuiltinCommands::get_all() {
            self.register_command(command_info);
        }
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

        // Use lexer for parsing
        let command = self.lexer.parse_command(input)
            .map_err(|e| ParseError::InvalidSyntax { detail: e.to_string() })?;

        // Validate the parsed command
        self.validator.validate_command(&command)
            .map_err(|e| ParseError::InvalidSyntax { detail: e.to_string() })?;

        Ok(command)
    }

    /// Validate command parameters
    pub fn validate_command(&self, command: &ImmutableChatCommand) -> ParseResult<()> {
        self.validator.validate_command(command)
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
        self.commands.get(command).cloned()
            .or_else(|| {
                self.aliases.get(command)
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
        
        // Keep history limited to prevent memory growth
        if self.history.len() > 100 {
            self.history.remove(0);
        }
    }

    /// Clear command history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Get all registered commands
    pub fn get_all_commands(&self) -> Vec<CommandInfo> {
        self.commands.values().cloned().collect()
    }

    /// Get commands by category
    pub fn get_commands_by_category(&self, category: &str) -> Vec<CommandInfo> {
        self.commands
            .values()
            .filter(|info| info.category == category)
            .cloned()
            .collect()
    }

    /// Get all command categories
    pub fn get_categories(&self) -> Vec<String> {
        let mut categories: Vec<String> = self.commands
            .values()
            .map(|info| info.category.clone())
            .collect();
        categories.sort();
        categories.dedup();
        categories
    }

    /// Check if command exists
    pub fn command_exists(&self, command: &str) -> bool {
        self.commands.contains_key(command) || self.aliases.contains_key(command)
    }

    /// Get command name from alias
    pub fn resolve_alias(&self, alias: &str) -> Option<String> {
        self.aliases.get(alias).cloned()
    }
}