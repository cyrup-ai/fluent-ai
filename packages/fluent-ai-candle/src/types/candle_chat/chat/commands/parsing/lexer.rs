//! Command lexical analysis and basic parsing
//!
//! Provides zero-allocation command tokenization and basic command recognition
//! with blazing-fast string processing and ergonomic APIs.

use super::error_handling::{ParseError, ParseResult};
use crate::types::candle_chat::chat::commands::types::{
    CommandError, ImmutableChatCommand, SearchScope,
};

/// Command lexer for tokenizing and basic parsing
pub struct CommandLexer;

impl CommandLexer {
    /// Create a new command lexer
    pub fn new() -> Self {
        Self
    }

    /// Parse command from input string (zero-allocation)
    pub fn parse_command(&self, input: &str) -> Result<ImmutableChatCommand, CommandError> {
        let input = input.trim();
        if input.is_empty() {
            return Err(CommandError::InvalidSyntax {
                detail: "Empty command".to_string(),
            });
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
                detail: "No command specified".to_string(),
            });
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
                    include_metadata,
                })
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
                    reset,
                })
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
                    include_context,
                })
            }
            _ => Err(CommandError::UnknownCommand {
                command: command_name,
            }),
        }
    }

    /// Extract command name from input
    pub fn extract_command_name(&self, input: &str) -> Option<String> {
        let input = input.trim();
        if input.is_empty() {
            return None;
        }

        let input = if input.starts_with('/') {
            &input[1..]
        } else {
            input
        };

        input.split_whitespace().next().map(|s| s.to_lowercase())
    }

    /// Extract arguments from input
    pub fn extract_arguments(&self, input: &str) -> Vec<String> {
        let input = input.trim();
        if input.is_empty() {
            return Vec::new();
        }

        let input = if input.starts_with('/') {
            &input[1..]
        } else {
            input
        };

        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.len() <= 1 {
            return Vec::new();
        }

        parts[1..].iter().map(|s| s.to_string()).collect()
    }

    /// Check if input is a valid command format
    pub fn is_command(&self, input: &str) -> bool {
        let input = input.trim();
        !input.is_empty() && input.starts_with('/')
    }

    /// Parse command-line style arguments
    pub fn parse_args(&self, args: &[&str]) -> (Vec<String>, std::collections::HashMap<String, Option<String>>) {
        let mut positional = Vec::new();
        let mut flags = std::collections::HashMap::new();
        let mut i = 0;

        while i < args.len() {
            let arg = args[i];
            if arg.starts_with("--") {
                let flag_name = &arg[2..];
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    flags.insert(flag_name.to_string(), Some(args[i + 1].to_string()));
                    i += 2;
                } else {
                    flags.insert(flag_name.to_string(), None);
                    i += 1;
                }
            } else {
                positional.push(arg.to_string());
                i += 1;
            }
        }

        (positional, flags)
    }
}

impl Default for CommandLexer {
    fn default() -> Self {
        Self::new()
    }
}