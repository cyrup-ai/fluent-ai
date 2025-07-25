//! Primary advanced command parsers
//!
//! Handles parsing of primary advanced commands like search, template,
//! macro, and branch with zero-allocation patterns.

use std::collections::HashMap;

use super::super::commands::ImmutableChatCommand;
use super::super::core::CommandResult;
use super::super::events::{
    SearchScope, TemplateAction, MacroAction, BranchAction
};

/// Parser for primary advanced commands
pub struct AdvancedPrimaryParser;

impl AdvancedPrimaryParser {
    /// Parse search command
    #[inline]
    pub fn parse_search_command(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let query = if args.is_empty() {
            String::new()
        } else {
            args[0].to_string()
        };

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
            .and_then(|pos| args.get(pos + 1))
            .and_then(|s| s.parse().ok());

        let include_context = args.contains(&"--context");

        Ok(ImmutableChatCommand::Search {
            query,
            scope,
            limit,
            include_context,
        })
    }

    /// Parse template command
    #[inline]
    pub fn parse_template_command(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let action = if args.contains(&"--list") {
            TemplateAction::List
        } else if args.contains(&"--create") {
            TemplateAction::Create
        } else if args.contains(&"--delete") {
            TemplateAction::Delete
        } else if args.contains(&"--edit") {
            TemplateAction::Edit
        } else {
            TemplateAction::Use
        };

        let name = args
            .iter()
            .find(|&&arg| !arg.starts_with("--"))
            .map(|s| s.to_string());

        let content = args
            .iter()
            .skip_while(|&&arg| arg.starts_with("--") || Some(arg) == name.as_deref())
            .next()
            .map(|s| s.to_string());

        Ok(ImmutableChatCommand::Template {
            action,
            name,
            content,
            variables: HashMap::new(),
        })
    }

    /// Parse macro command
    #[inline]
    pub fn parse_macro_command(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let action = if args.contains(&"--list") {
            MacroAction::List
        } else if args.contains(&"--create") {
            MacroAction::Create
        } else if args.contains(&"--delete") {
            MacroAction::Delete
        } else if args.contains(&"--edit") {
            MacroAction::Edit
        } else {
            MacroAction::Execute
        };

        let name = args
            .iter()
            .find(|&&arg| !arg.starts_with("--"))
            .map(|s| s.to_string());

        let commands: Vec<String> = args
            .iter()
            .skip_while(|&&arg| arg.starts_with("--") || Some(arg) == name.as_deref())
            .map(|s| s.to_string())
            .collect();

        Ok(ImmutableChatCommand::Macro {
            action,
            name,
            auto_execute: false,
            commands,
        })
    }

    /// Parse branch command
    #[inline]
    pub fn parse_branch_command(args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        let action = if args.contains(&"--create") {
            BranchAction::Create
        } else if args.contains(&"--switch") {
            BranchAction::Switch
        } else if args.contains(&"--merge") {
            BranchAction::Merge
        } else if args.contains(&"--delete") {
            BranchAction::Delete
        } else {
            BranchAction::List
        };

        let name = args
            .iter()
            .find(|&&arg| !arg.starts_with("--"))
            .map(|s| s.to_string());

        let source = args
            .iter()
            .position(|&arg| arg == "--from")
            .and_then(|pos| args.get(pos + 1))
            .map(|s| s.to_string());

        Ok(ImmutableChatCommand::Branch {
            action,
            name,
            source,
        })
    }
}