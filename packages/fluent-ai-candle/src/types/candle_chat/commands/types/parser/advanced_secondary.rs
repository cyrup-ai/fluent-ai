//! Secondary advanced command parsers
//!
//! Handles parsing of secondary advanced commands like session, tool,
//! stats, theme, and debug with zero-allocation patterns.

use std::collections::HashMap;

use super::super::commands::ImmutableChatCommand;
use super::super::core::CommandResult;
use super::super::events::{
    SessionAction, ToolAction, StatsType, ThemeAction, DebugAction
};

/// Parser for secondary advanced commands
pub struct AdvancedSecondaryParser;

impl AdvancedSecondaryParser {
    /// Parse session command
    #[inline]
    pub fn parse_session_command(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let action = if args.contains(&"--new") {
            SessionAction::New
        } else if args.contains(&"--switch") {
            SessionAction::Switch
        } else if args.contains(&"--delete") {
            SessionAction::Delete
        } else if args.contains(&"--export") {
            SessionAction::Export
        } else if args.contains(&"--import") {
            SessionAction::Import
        } else {
            SessionAction::List
        };

        let name = args
            .iter()
            .find(|&&arg| !arg.starts_with("--"))
            .map(|s| s.to_string());

        Ok(ImmutableChatCommand::Session {
            action,
            name,
            include_config: false,
        })
    }

    /// Parse tool command
    #[inline]
    pub fn parse_tool_command(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let action = if args.contains(&"--list") {
            ToolAction::List
        } else if args.contains(&"--install") {
            ToolAction::Install
        } else if args.contains(&"--remove") {
            ToolAction::Remove
        } else if args.contains(&"--configure") {
            ToolAction::Configure
        } else if args.contains(&"--update") {
            ToolAction::Update
        } else {
            ToolAction::Execute
        };

        let name = args
            .iter()
            .find(|&&arg| !arg.starts_with("--"))
            .map(|s| s.to_string());

        let args_map = args
            .iter()
            .skip_while(|&&arg| arg.starts_with("--") || Some(arg) == name.as_deref())
            .enumerate()
            .map(|(i, &arg)| (format!("arg_{}", i), arg.to_string()))
            .collect();

        Ok(ImmutableChatCommand::Tool {
            action,
            name,
            args: args_map,
        })
    }

    /// Parse stats command
    #[inline]
    pub fn parse_stats_command(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let stat_type = if args.contains(&"--usage") {
            StatsType::Usage
        } else if args.contains(&"--performance") {
            StatsType::Performance
        } else if args.contains(&"--history") {
            StatsType::History
        } else if args.contains(&"--tokens") {
            StatsType::Tokens
        } else if args.contains(&"--costs") {
            StatsType::Costs
        } else if args.contains(&"--errors") {
            StatsType::Errors
        } else {
            StatsType::Usage
        };

        let period = args
            .iter()
            .position(|&arg| arg == "--period")
            .and_then(|pos| args.get(pos + 1))
            .map(|s| s.to_string());

        let detailed = args.contains(&"--detailed");

        Ok(ImmutableChatCommand::Stats {
            stat_type,
            period,
            detailed,
        })
    }

    /// Parse theme command
    #[inline]
    pub fn parse_theme_command(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let action = if args.contains(&"--set") {
            ThemeAction::Set
        } else if args.contains(&"--list") {
            ThemeAction::List
        } else if args.contains(&"--create") {
            ThemeAction::Create
        } else if args.contains(&"--export") {
            ThemeAction::Export
        } else if args.contains(&"--import") {
            ThemeAction::Import
        } else if args.contains(&"--edit") {
            ThemeAction::Edit
        } else {
            ThemeAction::Set
        };

        let name = args
            .iter()
            .find(|&&arg| !arg.starts_with("--"))
            .map(|s| s.to_string());

        let properties = HashMap::new(); // Empty properties for now

        Ok(ImmutableChatCommand::Theme {
            action,
            name,
            properties,
        })
    }

    /// Parse debug command
    #[inline]
    pub fn parse_debug_command(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let action = if args.contains(&"--info") {
            DebugAction::Info
        } else if args.contains(&"--logs") {
            DebugAction::Logs
        } else if args.contains(&"--performance") {
            DebugAction::Performance
        } else if args.contains(&"--memory") {
            DebugAction::Memory
        } else if args.contains(&"--network") {
            DebugAction::Network
        } else if args.contains(&"--cache") {
            DebugAction::Cache
        } else {
            DebugAction::Info
        };

        let level = args
            .iter()
            .position(|&arg| arg == "--level")
            .and_then(|pos| args.get(pos + 1))
            .map(|s| s.to_string());

        let system_info = args.contains(&"--system");

        Ok(ImmutableChatCommand::Debug {
            action,
            level,
            system_info,
        })
    }

    /// Parse custom command
    #[inline]
    pub fn parse_custom_command(name: &str, args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let args_map = args
            .iter()
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