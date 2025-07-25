//! Advanced command parser coordinator
//!
//! Routes advanced commands to specialized parsers and provides
//! a unified interface for all advanced command parsing operations.

use super::super::commands::ImmutableChatCommand;
use super::super::core::{CommandError, CommandResult};
use super::{advanced_primary::AdvancedPrimaryParser, advanced_secondary::AdvancedSecondaryParser};

/// Parser for advanced chat commands
pub struct AdvancedCommandParser;

impl AdvancedCommandParser {
    /// Parse advanced command based on command name
    #[inline]
    pub fn parse_command(command_name: &str, args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        match command_name {
            // Primary advanced commands
            "search" | "s" => AdvancedPrimaryParser::parse_search_command(args),
            "template" | "tpl" => AdvancedPrimaryParser::parse_template_command(args),
            "macro" | "m" => AdvancedPrimaryParser::parse_macro_command(args),
            "branch" | "b" => AdvancedPrimaryParser::parse_branch_command(args),
            
            // Secondary advanced commands
            "session" | "sess" => AdvancedSecondaryParser::parse_session_command(args),
            "tool" | "t" => AdvancedSecondaryParser::parse_tool_command(args),
            "stats" | "st" => AdvancedSecondaryParser::parse_stats_command(args),
            "theme" | "th" => AdvancedSecondaryParser::parse_theme_command(args),
            "debug" | "d" => AdvancedSecondaryParser::parse_debug_command(args),
            
            // Unknown commands
            _ => Self::parse_custom_command(command_name, args),
        }
    }

    /// Parse custom command (delegation to secondary parser)
    #[inline]
    pub fn parse_custom_command(name: &str, args: &[&str]) -> CommandResult<ImmutableChatCommand> {
        AdvancedSecondaryParser::parse_custom_command(name, args)
    }
}