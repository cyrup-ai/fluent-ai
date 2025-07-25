//! Individual command parsing methods
//!
//! Contains parsing methods for each specific command type (help, clear, export, etc.)

use super::errors::{ParseError, ParseResult};
use super::parser::CommandParser;
use super::super::types::{ImmutableChatCommand, SearchScope};

impl CommandParser {
    /// Parse help command
    pub(super) fn parse_help_command(&self, args: &[&str]) -> ParseResult<ImmutableChatCommand> {
        let mut command = None;
        let mut extended = false;

        let mut i = 0;
        while i < args.len() {
            match args[i] {
                "--extended" => extended = true,
                arg if !arg.starts_with('-') => command = Some(arg.to_string()),
                _ => {
                    return Err(ParseError::UnknownParameter {
                        parameter: args[i].to_string()});
                }
            }
            i += 1;
        }

        Ok(ImmutableChatCommand::Help { command, extended })
    }

    /// Parse clear command
    pub(super) fn parse_clear_command(&self, args: &[&str]) -> ParseResult<ImmutableChatCommand> {
        let mut confirm = false;
        let mut keep_last = None;

        let mut i = 0;
        while i < args.len() {
            match args[i] {
                "--confirm" => confirm = true,
                "--keep-last" => {
                    i += 1;
                    if i >= args.len() {
                        return Err(ParseError::MissingParameter {
                            parameter: "keep-last".to_string()});
                    }
                    keep_last =
                        Some(
                            args[i]
                                .parse()
                                .map_err(|_| ParseError::InvalidParameterValue {
                                    parameter: "keep-last".to_string(),
                                    value: args[i].to_string()})?,
                        );
                }
                _ => {
                    return Err(ParseError::UnknownParameter {
                        parameter: args[i].to_string()});
                }
            }
            i += 1;
        }

        Ok(ImmutableChatCommand::Clear { confirm, keep_last })
    }

    /// Parse export command
    pub(super) fn parse_export_command(&self, args: &[&str]) -> ParseResult<ImmutableChatCommand> {
        let mut format = None;
        let mut output = None;
        let mut include_metadata = false;

        let mut i = 0;
        while i < args.len() {
            match args[i] {
                "--format" => {
                    i += 1;
                    if i >= args.len() {
                        return Err(ParseError::MissingParameter {
                            parameter: "format".to_string()});
                    }
                    format = Some(args[i].to_string());
                }
                "--output" => {
                    i += 1;
                    if i >= args.len() {
                        return Err(ParseError::MissingParameter {
                            parameter: "output".to_string()});
                    }
                    output = Some(args[i].to_string());
                }
                "--include-metadata" => include_metadata = true,
                _ => {
                    return Err(ParseError::UnknownParameter {
                        parameter: args[i].to_string()});
                }
            }
            i += 1;
        }

        let format = format.ok_or_else(|| ParseError::MissingParameter {
            parameter: "format".to_string()})?;

        Ok(ImmutableChatCommand::Export {
            format,
            output,
            include_metadata})
    }

    /// Parse config command
    pub(super) fn parse_config_command(&self, args: &[&str]) -> ParseResult<ImmutableChatCommand> {
        let mut key = None;
        let mut value = None;
        let mut show = false;
        let mut reset = false;

        let mut i = 0;
        while i < args.len() {
            match args[i] {
                "--show" => show = true,
                "--reset" => reset = true,
                arg if !arg.starts_with('-') => {
                    if key.is_none() {
                        key = Some(arg.to_string());
                    } else if value.is_none() {
                        value = Some(arg.to_string());
                    } else {
                        return Err(ParseError::InvalidSyntax {
                            detail: "Too many positional arguments".to_string()});
                    }
                }
                _ => {
                    return Err(ParseError::UnknownParameter {
                        parameter: args[i].to_string()});
                }
            }
            i += 1;
        }

        Ok(ImmutableChatCommand::Config {
            key,
            value,
            show,
            reset})
    }

    /// Parse search command
    pub(super) fn parse_search_command(&self, args: &[&str]) -> ParseResult<ImmutableChatCommand> {
        let mut query = None;
        let mut scope = SearchScope::All;
        let mut limit = None;
        let mut include_context = false;

        let mut i = 0;
        while i < args.len() {
            match args[i] {
                "--scope" => {
                    i += 1;
                    if i >= args.len() {
                        return Err(ParseError::MissingParameter {
                            parameter: "scope".to_string()});
                    }
                    scope = match args[i] {
                        "all" => SearchScope::All,
                        "current" => SearchScope::Current,
                        "recent" => SearchScope::Recent,
                        _ => {
                            return Err(ParseError::InvalidParameterValue {
                                parameter: "scope".to_string(),
                                value: args[i].to_string()});
                        }
                    };
                }
                "--limit" => {
                    i += 1;
                    if i >= args.len() {
                        return Err(ParseError::MissingParameter {
                            parameter: "limit".to_string()});
                    }
                    limit =
                        Some(
                            args[i]
                                .parse()
                                .map_err(|_| ParseError::InvalidParameterValue {
                                    parameter: "limit".to_string(),
                                    value: args[i].to_string()})?,
                        );
                }
                "--include-context" => include_context = true,
                arg if !arg.starts_with('-') => {
                    if query.is_none() {
                        query = Some(arg.to_string());
                    } else {
                        return Err(ParseError::InvalidSyntax {
                            detail: "Multiple query arguments not supported".to_string()});
                    }
                }
                _ => {
                    return Err(ParseError::UnknownParameter {
                        parameter: args[i].to_string()});
                }
            }
            i += 1;
        }

        let query = query.ok_or_else(|| ParseError::MissingParameter {
            parameter: "query".to_string()})?;

        Ok(ImmutableChatCommand::Search {
            query,
            scope,
            limit,
            include_context})
    }
}