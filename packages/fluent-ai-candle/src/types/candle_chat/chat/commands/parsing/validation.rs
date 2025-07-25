//! Command parameter validation and constraint checking
//!
//! Provides comprehensive validation for command parameters with zero-allocation
//! patterns and blazing-fast constraint checking with production-ready error handling.

use super::error_handling::{ParseError, ParseResult};
use crate::types::candle_chat::chat::commands::types::ImmutableChatCommand;
use std::collections::HashMap;

/// Command parameter validator
pub struct CommandValidator {
    /// Custom validation rules
    validation_rules: HashMap<String, Box<dyn Fn(&str) -> bool + Send + Sync>>,
}

impl CommandValidator {
    /// Create a new command validator
    pub fn new() -> Self {
        let mut validator = Self {
            validation_rules: HashMap::new(),
        };
        validator.register_default_rules();
        validator
    }

    /// Register default validation rules
    fn register_default_rules(&mut self) {
        // Export format validation
        self.validation_rules.insert(
            "export_format".to_string(),
            Box::new(|value| {
                let valid_formats = ["json", "markdown", "pdf", "html"];
                valid_formats.contains(&value)
            }),
        );

        // Positive integer validation
        self.validation_rules.insert(
            "positive_integer".to_string(),
            Box::new(|value| {
                value.parse::<u32>().map(|n| n > 0).unwrap_or(false)
            }),
        );

        // Non-empty string validation
        self.validation_rules.insert(
            "non_empty_string".to_string(),
            Box::new(|value| !value.trim().is_empty()),
        );
    }

    /// Validate command parameters
    pub fn validate_command(&self, command: &ImmutableChatCommand) -> ParseResult<()> {
        match command {
            ImmutableChatCommand::Export { format, .. } => {
                self.validate_export_format(format)?;
            }
            ImmutableChatCommand::Clear {
                keep_last: Some(n), ..
            } => {
                self.validate_keep_last_value(*n)?;
            }
            ImmutableChatCommand::Search { query, limit, .. } => {
                self.validate_search_query(query)?;
                if let Some(limit) = limit {
                    self.validate_search_limit(*limit)?;
                }
            }
            ImmutableChatCommand::Config { key, value, .. } => {
                if let Some(key) = key {
                    self.validate_config_key(key)?;
                }
                if let Some(value) = value {
                    self.validate_config_value(value)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Validate export format parameter
    fn validate_export_format(&self, format: &str) -> ParseResult<()> {
        let valid_formats = ["json", "markdown", "pdf", "html"];
        if !valid_formats.contains(&format) {
            return Err(ParseError::InvalidParameterValue {
                parameter: "format".to_string(),
                value: format.to_string(),
            });
        }
        Ok(())
    }

    /// Validate keep_last parameter for clear command
    fn validate_keep_last_value(&self, n: u32) -> ParseResult<()> {
        if n == 0 {
            return Err(ParseError::InvalidParameterValue {
                parameter: "keep-last".to_string(),
                value: "0".to_string(),
            });
        }
        Ok(())
    }

    /// Validate search query parameter
    fn validate_search_query(&self, query: &str) -> ParseResult<()> {
        if query.trim().is_empty() {
            return Err(ParseError::InvalidParameterValue {
                parameter: "query".to_string(),
                value: query.to_string(),
            });
        }
        Ok(())
    }

    /// Validate search limit parameter
    fn validate_search_limit(&self, limit: u32) -> ParseResult<()> {
        if limit == 0 || limit > 1000 {
            return Err(ParseError::InvalidParameterValue {
                parameter: "limit".to_string(),
                value: limit.to_string(),
            });
        }
        Ok(())
    }

    /// Validate configuration key
    fn validate_config_key(&self, key: &str) -> ParseResult<()> {
        if key.trim().is_empty() {
            return Err(ParseError::InvalidParameterValue {
                parameter: "key".to_string(),
                value: key.to_string(),
            });
        }

        // Check for valid configuration key format
        if !key.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.') {
            return Err(ParseError::InvalidParameterValue {
                parameter: "key".to_string(),
                value: key.to_string(),
            });
        }

        Ok(())
    }

    /// Validate configuration value
    fn validate_config_value(&self, value: &str) -> ParseResult<()> {
        if value.trim().is_empty() {
            return Err(ParseError::InvalidParameterValue {
                parameter: "value".to_string(),
                value: value.to_string(),
            });
        }
        Ok(())
    }

    /// Register custom validation rule
    pub fn register_rule<F>(&mut self, name: String, rule: F)
    where
        F: Fn(&str) -> bool + Send + Sync + 'static,
    {
        self.validation_rules.insert(name, Box::new(rule));
    }

    /// Apply custom validation rule
    pub fn apply_rule(&self, rule_name: &str, value: &str) -> ParseResult<()> {
        if let Some(rule) = self.validation_rules.get(rule_name) {
            if !rule(value) {
                return Err(ParseError::InvalidParameterValue {
                    parameter: rule_name.to_string(),
                    value: value.to_string(),
                });
            }
        }
        Ok(())
    }

    /// Check parameter type compatibility
    pub fn check_type_compatibility(&self, expected: &str, actual: &str) -> ParseResult<()> {
        match expected {
            "string" => Ok(()),
            "integer" => {
                if actual.parse::<i64>().is_err() {
                    Err(ParseError::TypeMismatch {
                        expected: expected.to_string(),
                        actual: actual.to_string(),
                    })
                } else {
                    Ok(())
                }
            }
            "float" => {
                if actual.parse::<f64>().is_err() {
                    Err(ParseError::TypeMismatch {
                        expected: expected.to_string(),
                        actual: actual.to_string(),
                    })
                } else {
                    Ok(())
                }
            }
            "boolean" => {
                if !matches!(actual.to_lowercase().as_str(), "true" | "false" | "yes" | "no" | "1" | "0") {
                    Err(ParseError::TypeMismatch {
                        expected: expected.to_string(),
                        actual: actual.to_string(),
                    })
                } else {
                    Ok(())
                }
            }
            _ => Ok(()),
        }
    }
}

impl Default for CommandValidator {
    fn default() -> Self {
        Self::new()
    }
}