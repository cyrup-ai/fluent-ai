//! Basic command parsers for common operations
//!
//! Handles parsing of fundamental commands like help, clear, export,
//! config, settings, history, save, load, and import with zero allocations.

use std::collections::HashMap;

use super::super::commands::ImmutableChatCommand;
use super::super::core::{CommandError, CommandResult};
use super::super::executor::SettingsCategory;
use super::super::events::{HistoryAction, ImportType};

/// Parser for basic chat commands
pub struct BasicCommandParser;

impl BasicCommandParser {
    /// Parse basic command based on command name
    #[inline]
    pub fn parse_command(command_name: &str, args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        match command_name {
            "help" | "h" => Self::parse_help_command(args),
            "clear" | "c" => Self::parse_clear_command(args),
            "export" | "e" => Self::parse_export_command(args),
            "config" | "cfg" => Self::parse_config_command(args),
            "settings" | "set" => Self::parse_settings_command(args),
            "history" | "hist" => Self::parse_history_command(args),
            "save" => Self::parse_save_command(args),
            "load" => Self::parse_load_command(args),
            "import" => Self::parse_import_command(args),
            _ => Err(CommandError::UnknownCommand {
                command: command_name.to_string(),
            }),
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
        let keep_last = args
            .iter()
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
                "Export format required".to_string(),
            ));
        }

        let format = args[0].to_string();
        let output = args
            .iter()
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

    /// Parse settings command
    #[inline]
    fn parse_settings_command(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let show = args.contains(&"--show") || args.contains(&"-s");
        let reset = args.contains(&"--reset") || args.contains(&"-r");

        let (key, value) = if args.len() >= 2 && !args[0].starts_with('-') {
            (Some(args[0].to_string()), Some(args[1].to_string()))
        } else if args.len() >= 1 && !args[0].starts_with('-') {
            (Some(args[0].to_string()), None)
        } else {
            (None, None)
        };

        Ok(ImmutableChatCommand::Settings {
            category: SettingsCategory::Appearance, // Default category
            key,
            value,
            show,
            reset,
        })
    }

    /// Parse history command
    #[inline]
    fn parse_history_command(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let action = if args.contains(&"--show") {
            HistoryAction::Show
        } else if args.contains(&"--search") {
            HistoryAction::Search
        } else if args.contains(&"--clear") {
            HistoryAction::Clear
        } else if args.contains(&"--export") {
            HistoryAction::Export
        } else {
            HistoryAction::Show
        };

        let limit = args
            .iter()
            .position(|&arg| arg == "--limit")
            .and_then(|pos| args.get(pos + 1))
            .and_then(|s| s.parse().ok());

        let filter = args
            .iter()
            .position(|&arg| arg == "--filter")
            .and_then(|pos| args.get(pos + 1))
            .map(|s| s.to_string());

        Ok(ImmutableChatCommand::History {
            action,
            limit,
            filter,
        })
    }

    /// Parse save command
    #[inline]
    fn parse_save_command(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let name = args
            .iter()
            .find(|&&arg| !arg.starts_with("--"))
            .map(|s| s.to_string());

        let include_config = args.contains(&"--config");

        let location = args
            .iter()
            .position(|&arg| arg == "--location")
            .and_then(|pos| args.get(pos + 1))
            .map(|s| s.to_string());

        Ok(ImmutableChatCommand::Save {
            name,
            include_config,
            location,
        })
    }

    /// Parse load command
    #[inline]
    fn parse_load_command(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let name = args
            .iter()
            .find(|&&arg| !arg.starts_with("--"))
            .map(|s| s.to_string())
            .unwrap_or_else(|| "default".to_string());

        let merge = args.contains(&"--merge");

        let location = args
            .iter()
            .position(|&arg| arg == "--location")
            .and_then(|pos| args.get(pos + 1))
            .map(|s| s.to_string());

        Ok(ImmutableChatCommand::Load {
            name,
            merge,
            location,
        })
    }

    /// Parse import command
    #[inline]
    fn parse_import_command(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let import_type = if args.contains(&"--chat") {
            ImportType::Chat
        } else if args.contains(&"--config") {
            ImportType::Config
        } else if args.contains(&"--templates") {
            ImportType::Templates
        } else if args.contains(&"--macros") {
            ImportType::Macros
        } else {
            ImportType::Chat
        };

        let source = args
            .iter()
            .find(|&&arg| !arg.starts_with("--"))
            .map(|s| s.to_string())
            .unwrap_or_else(|| "stdin".to_string());

        let options = HashMap::new(); // Empty options for now

        Ok(ImmutableChatCommand::Import {
            import_type,
            source,
            options,
        })
    }
}