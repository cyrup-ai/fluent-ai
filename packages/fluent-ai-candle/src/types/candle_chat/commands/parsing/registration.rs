//! Command registration and builtin commands
//!
//! Contains logic for registering commands and defining builtin command metadata.

use super::parser::CommandParser;
use super::registry::{CommandInfo, ParameterInfo, ParameterType};

impl CommandParser {
    /// Register built-in commands
    pub(super) fn register_builtin_commands(&mut self) {
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
                },
                ParameterInfo {
                    name: "extended".to_string(),
                    description: "Show extended help".to_string(),
                    parameter_type: ParameterType::Boolean,
                    required: false,
                    default_value: Some("false".to_string()),
                },
            ],
            aliases: vec!["h".to_string(), "?".to_string()],
            category: "General".to_string(),
            examples: vec![
                "/help".to_string(),
                "/help config".to_string(),
                "/help --extended".to_string(),
            ],
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
                },
                ParameterInfo {
                    name: "keep-last".to_string(),
                    description: "Keep last N messages".to_string(),
                    parameter_type: ParameterType::Integer,
                    required: false,
                    default_value: None,
                },
            ],
            aliases: vec!["cls".to_string(), "reset".to_string()],
            category: "History".to_string(),
            examples: vec![
                "/clear".to_string(),
                "/clear --confirm".to_string(),
                "/clear --keep-last 10".to_string(),
            ],
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
                    parameter_type: ParameterType::Enum,
                    required: true,
                    default_value: None,
                },
                ParameterInfo {
                    name: "output".to_string(),
                    description: "Output file path".to_string(),
                    parameter_type: ParameterType::Path,
                    required: false,
                    default_value: None,
                },
                ParameterInfo {
                    name: "include-metadata".to_string(),
                    description: "Include metadata in export".to_string(),
                    parameter_type: ParameterType::Boolean,
                    required: false,
                    default_value: Some("true".to_string()),
                },
            ],
            aliases: vec!["save".to_string()],
            category: "Export".to_string(),
            examples: vec![
                "/export --format json".to_string(),
                "/export --format markdown --output chat.md".to_string(),
                "/export --format pdf --include-metadata".to_string(),
            ],
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
                },
                ParameterInfo {
                    name: "value".to_string(),
                    description: "Configuration value".to_string(),
                    parameter_type: ParameterType::String,
                    required: false,
                    default_value: None,
                },
                ParameterInfo {
                    name: "show".to_string(),
                    description: "Show current configuration".to_string(),
                    parameter_type: ParameterType::Boolean,
                    required: false,
                    default_value: Some("false".to_string()),
                },
                ParameterInfo {
                    name: "reset".to_string(),
                    description: "Reset to defaults".to_string(),
                    parameter_type: ParameterType::Boolean,
                    required: false,
                    default_value: Some("false".to_string()),
                },
            ],
            aliases: vec!["cfg".to_string(), "settings".to_string()],
            category: "Configuration".to_string(),
            examples: vec![
                "/config --show".to_string(),
                "/config theme dark".to_string(),
                "/config --reset".to_string(),
            ],
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
                },
                ParameterInfo {
                    name: "scope".to_string(),
                    description: "Search scope (all, current, recent)".to_string(),
                    parameter_type: ParameterType::Enum,
                    required: false,
                    default_value: Some("all".to_string()),
                },
                ParameterInfo {
                    name: "limit".to_string(),
                    description: "Maximum results".to_string(),
                    parameter_type: ParameterType::Integer,
                    required: false,
                    default_value: Some("10".to_string()),
                },
                ParameterInfo {
                    name: "include-context".to_string(),
                    description: "Include context in results".to_string(),
                    parameter_type: ParameterType::Boolean,
                    required: false,
                    default_value: Some("true".to_string()),
                },
            ],
            aliases: vec!["find".to_string(), "grep".to_string()],
            category: "Search".to_string(),
            examples: vec![
                "/search rust".to_string(),
                "/search \"error handling\" --scope recent".to_string(),
                "/search async --limit 5 --include-context".to_string(),
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
}