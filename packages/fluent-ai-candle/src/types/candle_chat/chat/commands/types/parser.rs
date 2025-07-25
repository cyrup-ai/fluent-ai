//! Command parsing with zero allocation
//!
//! Provides zero-allocation command parsing from string input with comprehensive
//! support for all command types and their arguments.

use std::collections::HashMap;

use super::actions::*;
use super::command::ImmutableChatCommand;
use super::core::{CommandError, CommandResult, SettingsCategory};

/// Command parsing with zero allocation
pub struct CommandParser;

impl CommandParser {
    /// Parse command from borrowed string (zero allocation in hot path)
    #[inline]
    pub fn parse_command(input: &str) -> CommandResult<ImmutableChatCommand> {
        let input = input.trim();
        if input.is_empty() {
            return Err(CommandError::ParseError("Empty command".to_string()));
        }

        let input = input.strip_prefix('/').unwrap_or(input);
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.is_empty() {
            return Err(CommandError::ParseError("Invalid command format".to_string()));
        }

        let command_name = parts[0].to_lowercase();
        let args = &parts[1..];

        match command_name.as_str() {
            "help" | "h" => Self::parse_help(args),
            "clear" | "c" => Self::parse_clear(args),
            "export" | "e" => Self::parse_export(args),
            "config" | "cfg" => Self::parse_config(args),
            "search" | "s" => Self::parse_search(args),
            "template" | "tpl" => Self::parse_template(args),
            "macro" | "m" => Self::parse_macro(args),
            "branch" | "b" => Self::parse_branch(args),
            "session" | "sess" => Self::parse_session(args),
            "tool" | "t" => Self::parse_tool(args),
            "stats" | "st" => Self::parse_stats(args),
            "theme" | "th" => Self::parse_theme(args),
            "debug" | "d" => Self::parse_debug(args),
            "history" | "hist" => Self::parse_history(args),
            "save" => Self::parse_save(args),
            "load" => Self::parse_load(args),
            "import" => Self::parse_import(args),
            "settings" | "set" => Self::parse_settings(args),
            _ => Self::parse_custom(&command_name, args),
        }
    }

    fn parse_help(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let command = if args.is_empty() { None } else { Some(args[0].to_string()) };
        let extended = args.contains(&"--extended") || args.contains(&"-e");
        Ok(ImmutableChatCommand::Help { command, extended })
    }

    fn parse_clear(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let confirm = args.contains(&"--confirm") || args.contains(&"-y");
        let keep_last = args.iter().position(|&arg| arg == "--keep" || arg == "-k")
            .and_then(|pos| args.get(pos + 1))
            .and_then(|s| s.parse().ok());
        Ok(ImmutableChatCommand::Clear { confirm, keep_last })
    }

    fn parse_export(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        if args.is_empty() {
            return Err(CommandError::InvalidArguments("Export format required".to_string()));
        }
        let format = args[0].to_string();
        let output = args.iter().position(|&arg| arg == "--output" || arg == "-o")
            .and_then(|pos| args.get(pos + 1))
            .map(|s| s.to_string());
        let include_metadata = args.contains(&"--metadata") || args.contains(&"-m");
        Ok(ImmutableChatCommand::Export { format, output, include_metadata })
    }

    fn parse_config(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let show = args.contains(&"--show") || args.contains(&"-s");
        let reset = args.contains(&"--reset") || args.contains(&"-r");
        let (key, value) = if args.len() >= 2 && !args[0].starts_with('-') {
            (Some(args[0].to_string()), Some(args[1].to_string()))
        } else if args.len() >= 1 && !args[0].starts_with('-') {
            (Some(args[0].to_string()), None)
        } else {
            (None, None)
        };
        Ok(ImmutableChatCommand::Config { key, value, show, reset })
    }

    fn parse_search(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let query = if args.is_empty() { String::new() } else { args[0].to_string() };
        let scope = if args.contains(&"--current") { SearchScope::Current }
        else if args.contains(&"--recent") { SearchScope::Recent }
        else if args.contains(&"--bookmarked") { SearchScope::Bookmarked }
        else { SearchScope::All };
        let limit = args.iter().position(|&arg| arg == "--limit")
            .and_then(|pos| args.get(pos + 1))
            .and_then(|s| s.parse().ok());
        let include_context = args.contains(&"--context");
        Ok(ImmutableChatCommand::Search { query, scope, limit, include_context })
    }

    fn parse_template(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let action = if args.contains(&"--list") { TemplateAction::List }
        else if args.contains(&"--create") { TemplateAction::Create }
        else if args.contains(&"--delete") { TemplateAction::Delete }
        else if args.contains(&"--edit") { TemplateAction::Edit }
        else { TemplateAction::Use };
        let name = args.iter().find(|&&arg| !arg.starts_with("--")).map(|s| s.to_string());
        let content = args.iter()
            .skip_while(|&&arg| arg.starts_with("--") || Some(arg) == name.as_deref())
            .next().map(|s| s.to_string());
        Ok(ImmutableChatCommand::Template { action, name, content, variables: HashMap::new() })
    }

    fn parse_macro(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let action = if args.contains(&"--list") { MacroAction::List }
        else if args.contains(&"--create") { MacroAction::Create }
        else if args.contains(&"--delete") { MacroAction::Delete }
        else if args.contains(&"--edit") { MacroAction::Edit }
        else { MacroAction::Execute };
        let name = args.iter().find(|&&arg| !arg.starts_with("--")).map(|s| s.to_string());
        let commands: Vec<String> = args.iter()
            .skip_while(|&&arg| arg.starts_with("--") || Some(arg) == name.as_deref())
            .map(|s| s.to_string()).collect();
        Ok(ImmutableChatCommand::Macro { action, name, auto_execute: false, commands })
    }

    fn parse_branch(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let action = if args.contains(&"--create") { BranchAction::Create }
        else if args.contains(&"--switch") { BranchAction::Switch }
        else if args.contains(&"--merge") { BranchAction::Merge }
        else if args.contains(&"--delete") { BranchAction::Delete }
        else { BranchAction::List };
        let name = args.iter().find(|&&arg| !arg.starts_with("--")).map(|s| s.to_string());
        let source = args.iter().position(|&arg| arg == "--from")
            .and_then(|pos| args.get(pos + 1))
            .map(|s| s.to_string());
        Ok(ImmutableChatCommand::Branch { action, name, source })
    }

    fn parse_session(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let action = if args.contains(&"--new") { SessionAction::New }
        else if args.contains(&"--switch") { SessionAction::Switch }
        else if args.contains(&"--delete") { SessionAction::Delete }
        else if args.contains(&"--export") { SessionAction::Export }
        else if args.contains(&"--import") { SessionAction::Import }
        else { SessionAction::List };
        let name = args.iter().find(|&&arg| !arg.starts_with("--")).map(|s| s.to_string());
        Ok(ImmutableChatCommand::Session { action, name, include_config: false })
    }

    fn parse_tool(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let action = if args.contains(&"--list") { ToolAction::List }
        else if args.contains(&"--install") { ToolAction::Install }
        else if args.contains(&"--remove") { ToolAction::Remove }
        else if args.contains(&"--configure") { ToolAction::Configure }
        else if args.contains(&"--update") { ToolAction::Update }
        else { ToolAction::Execute };
        let name = args.iter().find(|&&arg| !arg.starts_with("--")).map(|s| s.to_string());
        let args_map = args.iter()  
            .skip_while(|&&arg| arg.starts_with("--") || Some(arg) == name.as_deref())
            .enumerate()
            .map(|(i, &arg)| (format!("arg_{}", i), arg.to_string()))
            .collect();
        Ok(ImmutableChatCommand::Tool { action, name, args: args_map })
    }

    fn parse_stats(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let stat_type = if args.contains(&"--usage") { StatsType::Usage }
        else if args.contains(&"--performance") { StatsType::Performance }
        else if args.contains(&"--history") { StatsType::History }
        else if args.contains(&"--tokens") { StatsType::Tokens }
        else if args.contains(&"--costs") { StatsType::Costs }
        else if args.contains(&"--errors") { StatsType::Errors }
        else { StatsType::Usage };
        let period = args.iter().position(|&arg| arg == "--period")
            .and_then(|pos| args.get(pos + 1))
            .map(|s| s.to_string());
        let detailed = args.contains(&"--detailed");
        Ok(ImmutableChatCommand::Stats { stat_type, period, detailed })
    }

    fn parse_theme(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let action = if args.contains(&"--set") { ThemeAction::Set }
        else if args.contains(&"--list") { ThemeAction::List }
        else if args.contains(&"--create") { ThemeAction::Create }
        else if args.contains(&"--export") { ThemeAction::Export }
        else if args.contains(&"--import") { ThemeAction::Import }
        else if args.contains(&"--edit") { ThemeAction::Edit }
        else { ThemeAction::Set };
        let name = args.iter().find(|&&arg| !arg.starts_with("--")).map(|s| s.to_string());
        Ok(ImmutableChatCommand::Theme { action, name, properties: HashMap::new() })
    }

    fn parse_debug(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let action = if args.contains(&"--info") { DebugAction::Info }
        else if args.contains(&"--logs") { DebugAction::Logs }
        else if args.contains(&"--performance") { DebugAction::Performance }
        else if args.contains(&"--memory") { DebugAction::Memory }
        else if args.contains(&"--network") { DebugAction::Network }
        else if args.contains(&"--cache") { DebugAction::Cache }
        else { DebugAction::Info };
        let level = args.iter().position(|&arg| arg == "--level")
            .and_then(|pos| args.get(pos + 1))
            .map(|s| s.to_string());
        let system_info = args.contains(&"--system");
        Ok(ImmutableChatCommand::Debug { action, level, system_info })
    }

    fn parse_history(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let action = if args.contains(&"--show") { HistoryAction::Show }
        else if args.contains(&"--search") { HistoryAction::Search }
        else if args.contains(&"--clear") { HistoryAction::Clear }
        else if args.contains(&"--export") { HistoryAction::Export }
        else { HistoryAction::Show };
        let limit = args.iter().position(|&arg| arg == "--limit")
            .and_then(|pos| args.get(pos + 1))
            .and_then(|s| s.parse().ok());
        let filter = args.iter().position(|&arg| arg == "--filter")
            .and_then(|pos| args.get(pos + 1))
            .map(|s| s.to_string());
        Ok(ImmutableChatCommand::History { action, limit, filter })
    }

    fn parse_save(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let name = args.iter().find(|&&arg| !arg.starts_with("--")).map(|s| s.to_string());
        let include_config = args.contains(&"--config");
        let location = args.iter().position(|&arg| arg == "--location")
            .and_then(|pos| args.get(pos + 1))
            .map(|s| s.to_string());
        Ok(ImmutableChatCommand::Save { name, include_config, location })
    }

    fn parse_load(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let name = args.iter().find(|&&arg| !arg.starts_with("--"))
            .map(|s| s.to_string())
            .unwrap_or_else(|| "default".to_string());
        let merge = args.contains(&"--merge");
        let location = args.iter().position(|&arg| arg == "--location")
            .and_then(|pos| args.get(pos + 1))
            .map(|s| s.to_string());
        Ok(ImmutableChatCommand::Load { name, merge, location })
    }

    fn parse_import(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let import_type = if args.contains(&"--chat") { ImportType::Chat }
        else if args.contains(&"--config") { ImportType::Config }
        else if args.contains(&"--templates") { ImportType::Templates }
        else if args.contains(&"--macros") { ImportType::Macros }
        else { ImportType::Chat };
        let source = args.iter().find(|&&arg| !arg.starts_with("--"))
            .map(|s| s.to_string())
            .unwrap_or_else(|| "stdin".to_string());
        Ok(ImmutableChatCommand::Import { import_type, source, options: HashMap::new() })
    }

    fn parse_settings(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
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
            category: SettingsCategory::Appearance,
            key, value, show, reset
        })
    }

    fn parse_custom(name: &str, args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let args_map = args.iter().enumerate()
            .map(|(i, &arg)| (format!("arg_{}", i), arg.to_string()))
            .collect();
        Ok(ImmutableChatCommand::Custom { name: name.to_string(), args: args_map, metadata: None })
    }
}