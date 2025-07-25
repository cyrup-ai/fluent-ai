//! Core command parser with routing logic
//!
//! Main parser entry point that routes commands to specialized parsers
//! based on command type for better organization and maintainability.

use super::super::commands::ImmutableChatCommand;
use super::super::core::{CommandError, CommandResult};
use super::{basic::BasicCommandParser, advanced::AdvancedCommandParser};

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
            return Err(CommandError::ParseError(
                "Invalid command format".to_string(),
            ));
        }

        let command_name = parts[0].to_lowercase();
        let args = &parts[1..];

        // Route to appropriate parser based on command type
        match command_name.as_str() {
            // Basic commands
            "help" | "h" | "clear" | "c" | "export" | "e" | "config" | "cfg" 
            | "settings" | "set" | "history" | "hist" | "save" | "load" | "import" => {
                BasicCommandParser::parse_command(&command_name, args)
            }
            // Advanced commands - use custom command parsing for now
            "search" | "s" | "template" | "tpl" | "macro" | "m" | "branch" | "b" | "session"
            | "sess" | "tool" | "t" | "stats" | "st" | "theme" | "th" | "debug" | "d" => {
                AdvancedCommandParser::parse_command(&command_name, args)
            }
            // Unknown commands
            _ => AdvancedCommandParser::parse_custom_command(&command_name, args),
        }
    }
}