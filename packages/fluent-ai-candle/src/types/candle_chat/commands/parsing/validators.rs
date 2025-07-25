//! Command validation logic
//!
//! Provides validation for parsed commands to ensure parameters are within valid ranges
//! and meet business logic requirements.

use super::errors::{ParseError, ParseResult};
use super::super::types::ImmutableChatCommand;

/// Validate command parameters
pub fn validate_command(command: &ImmutableChatCommand) -> ParseResult<()> {
    match command {
        ImmutableChatCommand::Export { format, .. } => {
            let valid_formats = ["json", "markdown", "pdf", "html"];
            if !valid_formats.contains(&format.as_str()) {
                return Err(ParseError::InvalidParameterValue {
                    parameter: "format".to_string(),
                    value: format.clone(),
                });
            }
        }
        ImmutableChatCommand::Clear {
            keep_last: Some(n), ..
        } => {
            if *n == 0 {
                return Err(ParseError::InvalidParameterValue {
                    parameter: "keep-last".to_string(),
                    value: "0".to_string(),
                });
            }
        }
        _ => {}
    }
    Ok(())
}