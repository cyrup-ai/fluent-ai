//! Command validation and sanitization
//!
//! Provides comprehensive input validation with zero-allocation patterns and blazing-fast
//! validation algorithms for production-ready security and error handling.

use std::collections::HashMap;
use std::sync::Arc;

use once_cell::sync::Lazy;
use regex::Regex;

use super::types::ImmutableChatCommand;

/// Command validator with comprehensive validation rules
#[derive(Debug, Clone)]
pub struct CommandValidator {
    /// Maximum command length
    max_command_length: usize,
    /// Maximum parameter count
    max_parameter_count: usize,
    /// Maximum parameter value length
    max_parameter_value_length: usize,
    /// Allowed file extensions for path parameters
    allowed_extensions: Vec<Arc<str>>,
    /// Blocked patterns for security
    blocked_patterns: Vec<Regex>,
}

impl Default for CommandValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandValidator {
    /// Create a new command validator with default settings
    pub fn new() -> Self {
        Self {
            max_command_length: 1024,
            max_parameter_count: 50,
            max_parameter_value_length: 512,
            allowed_extensions: vec![
                Arc::from("txt"),
                Arc::from("md"),
                Arc::from("json"),
                Arc::from("csv"),
                Arc::from("html"),
                Arc::from("pdf"),
            ],
            blocked_patterns: vec![
                // Prevent command injection
                Regex::new(r"[;&|`$()]").unwrap(),
                // Prevent path traversal
                Regex::new(r"\.\.[\\/]").unwrap(),
                // Prevent script injection
                Regex::new(r"<script[^>]*>").unwrap(),
            ],
        }
    }

    /// Validate a command with comprehensive checks
    pub fn validate_command(&self, command: &ImmutableChatCommand) -> Result<(), ValidationError> {
        match command {
            ImmutableChatCommand::Help { command, .. } => {
                if let Some(cmd) = command {
                    self.validate_string_parameter("command", cmd, false)?;
                }
            }
            ImmutableChatCommand::Clear { keep_last, .. } => {
                if let Some(n) = keep_last {
                    self.validate_integer_parameter("keep_last", *n as i64, Some(1), Some(1000))?;
                }
            }
            ImmutableChatCommand::Export { format, output, .. } => {
                self.validate_enum_parameter(
                    "format",
                    format,
                    &["json", "markdown", "pdf", "html"],
                )?;
                if let Some(path) = output {
                    self.validate_path_parameter("output", path)?;
                }
            }
            ImmutableChatCommand::Config { key, value, .. } => {
                if let Some(k) = key {
                    self.validate_config_key(k)?;
                }
                if let Some(v) = value {
                    self.validate_config_value(v)?;
                }
            }
            ImmutableChatCommand::Search { query, limit, .. } => {
                self.validate_string_parameter("query", query, false)?;
                if let Some(n) = limit {
                    self.validate_integer_parameter("limit", *n as i64, Some(1), Some(100))?;
                }
            }
            ImmutableChatCommand::Template {
                name,
                content,
                variables,
                ..
            } => {
                if let Some(n) = name {
                    self.validate_name_parameter("name", n)?;
                }
                if let Some(c) = content {
                    self.validate_content_parameter("content", c)?;
                }
                self.validate_variables(variables)?;
            }
            ImmutableChatCommand::Macro { name, .. } => {
                if let Some(n) = name {
                    self.validate_name_parameter("name", n)?;
                }
            }
            ImmutableChatCommand::Branch { name, source, .. } => {
                if let Some(n) = name {
                    self.validate_name_parameter("name", n)?;
                }
                if let Some(s) = source {
                    self.validate_name_parameter("source", s)?;
                }
            }
            ImmutableChatCommand::Session { name, .. } => {
                if let Some(n) = name {
                    self.validate_name_parameter("name", n)?;
                }
            }
            ImmutableChatCommand::Tool { name, args, .. } => {
                if let Some(n) = name {
                    self.validate_name_parameter("name", n)?;
                }
                self.validate_tool_args(args)?;
            }
            ImmutableChatCommand::Stats { period, .. } => {
                if let Some(p) = period {
                    self.validate_enum_parameter("period", p, &["day", "week", "month", "all"])?;
                }
            }
            ImmutableChatCommand::Theme {
                name, properties, ..
            } => {
                if let Some(n) = name {
                    self.validate_name_parameter("name", n)?;
                }
                self.validate_theme_properties(properties)?;
            }
            ImmutableChatCommand::Debug { level, .. } => {
                if let Some(l) = level {
                    self.validate_enum_parameter(
                        "level",
                        l,
                        &["error", "warn", "info", "debug", "trace"],
                    )?;
                }
            }
            ImmutableChatCommand::History { limit, filter, .. } => {
                if let Some(n) = limit {
                    self.validate_integer_parameter("limit", *n as i64, Some(1), Some(10000))?;
                }
                if let Some(f) = filter {
                    self.validate_string_parameter("filter", f, false)?;
                }
            }
            ImmutableChatCommand::Save { name, location, .. } => {
                if let Some(n) = name {
                    self.validate_name_parameter("name", n)?;
                }
                if let Some(l) = location {
                    self.validate_path_parameter("location", l)?;
                }
            }
            ImmutableChatCommand::Load { name, location, .. } => {
                self.validate_name_parameter("name", name)?;
                if let Some(l) = location {
                    self.validate_path_parameter("location", l)?;
                }
            }
            ImmutableChatCommand::Import { source, .. } => {
                self.validate_path_parameter("source", source)?;
            }
            ImmutableChatCommand::Settings { key, .. } => {
                if let Some(k) = key {
                    self.validate_string_parameter("key", k, false)?;
                }
            }
            ImmutableChatCommand::Custom { name, .. } => {
                self.validate_string_parameter("name", name, false)?;
            }
        }

        Ok(())
    }

    /// Validate string parameter
    fn validate_string_parameter(
        &self,
        name: &str,
        value: &str,
        allow_empty: bool,
    ) -> Result<(), ValidationError> {
        if !allow_empty && value.is_empty() {
            return Err(ValidationError::EmptyParameter {
                parameter: Arc::from(name),
            });
        }

        if value.len() > self.max_parameter_value_length {
            return Err(ValidationError::ParameterTooLong {
                parameter: Arc::from(name),
                max_length: self.max_parameter_value_length,
                actual_length: value.len(),
            });
        }

        // Check for blocked patterns
        for pattern in &self.blocked_patterns {
            if pattern.is_match(value) {
                return Err(ValidationError::SecurityViolation {
                    parameter: Arc::from(name),
                    detail: Arc::from("Contains blocked pattern"),
                });
            }
        }

        Ok(())
    }

    /// Validate integer parameter
    fn validate_integer_parameter(
        &self,
        name: &str,
        value: i64,
        min: Option<i64>,
        max: Option<i64>,
    ) -> Result<(), ValidationError> {
        if let Some(min_val) = min {
            if value < min_val {
                return Err(ValidationError::ParameterOutOfRange {
                    parameter: Arc::from(name),
                    value: value.to_string(),
                    min: Some(min_val.to_string()),
                    max: max.map(|m| m.to_string()),
                });
            }
        }

        if let Some(max_val) = max {
            if value > max_val {
                return Err(ValidationError::ParameterOutOfRange {
                    parameter: Arc::from(name),
                    value: value.to_string(),
                    min: min.map(|m| m.to_string()),
                    max: Some(max_val.to_string()),
                });
            }
        }

        Ok(())
    }

    /// Validate enum parameter
    fn validate_enum_parameter(
        &self,
        name: &str,
        value: &str,
        allowed: &[&str],
    ) -> Result<(), ValidationError> {
        if !allowed.contains(&value) {
            return Err(ValidationError::InvalidEnumValue {
                parameter: Arc::from(name),
                value: Arc::from(value),
                allowed_values: allowed.iter().map(|s| Arc::from(*s)).collect(),
            });
        }
        Ok(())
    }

    /// Validate path parameter
    fn validate_path_parameter(&self, name: &str, path: &str) -> Result<(), ValidationError> {
        // Basic string validation
        self.validate_string_parameter(name, path, false)?;

        // Check for path traversal attempts
        if path.contains("..") {
            return Err(ValidationError::SecurityViolation {
                parameter: Arc::from(name),
                detail: Arc::from("Path traversal attempt detected"),
            });
        }

        // Validate file extension if present
        if let Some(ext_pos) = path.rfind('.') {
            let extension = &path[ext_pos + 1..];
            if !self
                .allowed_extensions
                .iter()
                .any(|ext| ext.as_ref() == extension)
            {
                return Err(ValidationError::InvalidFileExtension {
                    parameter: Arc::from(name),
                    extension: Arc::from(extension),
                    allowed_extensions: self.allowed_extensions.clone(),
                });
            }
        }

        Ok(())
    }

    /// Validate configuration key
    fn validate_config_key(&self, key: &str) -> Result<(), ValidationError> {
        self.validate_string_parameter("key", key, false)?;

        // Config keys should be alphanumeric with dots and underscores
        let config_key_regex = Regex::new(r"^[a-zA-Z0-9._-]+$").unwrap();
        if !config_key_regex.is_match(key) {
            return Err(ValidationError::InvalidParameterFormat {
                parameter: Arc::from("key"),
                value: Arc::from(key),
                expected_format: Arc::from("alphanumeric with dots, underscores, and hyphens"),
            });
        }

        Ok(())
    }

    /// Validate configuration value
    fn validate_config_value(&self, value: &str) -> Result<(), ValidationError> {
        self.validate_string_parameter("value", value, true)?;
        Ok(())
    }

    /// Validate name parameter (for templates, macros, etc.)
    fn validate_name_parameter(&self, param_name: &str, name: &str) -> Result<(), ValidationError> {
        self.validate_string_parameter(param_name, name, false)?;

        // Names should be alphanumeric with underscores and hyphens
        let name_regex = Regex::new(r"^[a-zA-Z0-9_-]+$").unwrap();
        if !name_regex.is_match(name) {
            return Err(ValidationError::InvalidParameterFormat {
                parameter: Arc::from(param_name),
                value: Arc::from(name),
                expected_format: Arc::from("alphanumeric with underscores and hyphens"),
            });
        }

        Ok(())
    }

    /// Validate content parameter
    fn validate_content_parameter(&self, name: &str, content: &str) -> Result<(), ValidationError> {
        // Allow longer content but still validate
        if content.len() > self.max_parameter_value_length * 4 {
            return Err(ValidationError::ParameterTooLong {
                parameter: Arc::from(name),
                max_length: self.max_parameter_value_length * 4,
                actual_length: content.len(),
            });
        }

        // Check for script injection attempts
        let script_regex = Regex::new(r"<script[^>]*>.*?</script>").unwrap();
        if script_regex.is_match(content) {
            return Err(ValidationError::SecurityViolation {
                parameter: Arc::from(name),
                detail: Arc::from("Script injection attempt detected"),
            });
        }

        Ok(())
    }

    /// Validate template/macro variables
    fn validate_variables(
        &self,
        variables: &HashMap<Arc<str>, Arc<str>>,
    ) -> Result<(), ValidationError> {
        if variables.len() > self.max_parameter_count {
            return Err(ValidationError::TooManyParameters {
                max_count: self.max_parameter_count,
                actual_count: variables.len(),
            });
        }

        for (key, value) in variables {
            self.validate_name_parameter("variable_key", key)?;
            self.validate_string_parameter("variable_value", value, true)?;
        }

        Ok(())
    }

    /// Validate tool arguments
    fn validate_tool_args(
        &self,
        args: &HashMap<Arc<str>, Arc<str>>,
    ) -> Result<(), ValidationError> {
        if args.len() > self.max_parameter_count {
            return Err(ValidationError::TooManyParameters {
                max_count: self.max_parameter_count,
                actual_count: args.len(),
            });
        }

        for (key, value) in args {
            self.validate_string_parameter("arg_key", key, false)?;
            self.validate_string_parameter("arg_value", value, true)?;
        }

        Ok(())
    }

    /// Validate theme properties
    fn validate_theme_properties(
        &self,
        properties: &HashMap<Arc<str>, Arc<str>>,
    ) -> Result<(), ValidationError> {
        if properties.len() > self.max_parameter_count {
            return Err(ValidationError::TooManyParameters {
                max_count: self.max_parameter_count,
                actual_count: properties.len(),
            });
        }

        for (key, value) in properties {
            self.validate_string_parameter("property_key", key, false)?;
            self.validate_string_parameter("property_value", value, true)?;
        }

        Ok(())
    }

    /// Sanitize input string
    pub fn sanitize_input(&self, input: &str) -> String {
        // Remove null bytes
        let sanitized = input.replace('\0', "");

        // Limit length
        if sanitized.len() > self.max_command_length {
            sanitized[..self.max_command_length].to_string()
        } else {
            sanitized
        }
    }

    /// Check if input is safe
    pub fn is_safe_input(&self, input: &str) -> bool {
        // Check length
        if input.len() > self.max_command_length {
            return false;
        }

        // Check for blocked patterns
        for pattern in &self.blocked_patterns {
            if pattern.is_match(input) {
                return false;
            }
        }

        true
    }
}

/// Validation error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum ValidationError {
    #[error("Parameter '{parameter}' cannot be empty")]
    EmptyParameter { parameter: Arc<str> },

    #[error("Parameter '{parameter}' is too long: {actual_length} > {max_length}")]
    ParameterTooLong {
        parameter: Arc<str>,
        max_length: usize,
        actual_length: usize,
    },

    #[error("Parameter '{parameter}' is out of range: {value} (min: {min:?}, max: {max:?})")]
    ParameterOutOfRange {
        parameter: Arc<str>,
        value: String,
        min: Option<String>,
        max: Option<String>,
    },

    #[error("Parameter '{parameter}' has invalid value '{value}', allowed: {allowed_values:?}")]
    InvalidEnumValue {
        parameter: Arc<str>,
        value: Arc<str>,
        allowed_values: Vec<Arc<str>>,
    },

    #[error("Parameter '{parameter}' has invalid format '{value}', expected: {expected_format}")]
    InvalidParameterFormat {
        parameter: Arc<str>,
        value: Arc<str>,
        expected_format: Arc<str>,
    },

    #[error(
        "Parameter '{parameter}' has invalid file extension '{extension}', allowed: {allowed_extensions:?}"
    )]
    InvalidFileExtension {
        parameter: Arc<str>,
        extension: Arc<str>,
        allowed_extensions: Vec<Arc<str>>,
    },

    #[error("Too many parameters: {actual_count} > {max_count}")]
    TooManyParameters {
        max_count: usize,
        actual_count: usize,
    },

    #[error("Security violation in parameter '{parameter}': {detail}")]
    SecurityViolation {
        parameter: Arc<str>,
        detail: Arc<str>,
    },
}

/// Global validator instance
static GLOBAL_VALIDATOR: Lazy<CommandValidator> = Lazy::new(CommandValidator::new);

/// Get global validator
pub fn get_global_validator() -> &'static CommandValidator {
    &GLOBAL_VALIDATOR
}

/// Validate command using global validator
pub fn validate_global_command(command: &ImmutableChatCommand) -> Result<(), ValidationError> {
    get_global_validator().validate_command(command)
}

/// Sanitize input using global validator
pub fn sanitize_global_input(input: &str) -> String {
    get_global_validator().sanitize_input(input)
}

/// Check if input is safe using global validator
pub fn is_global_safe_input(input: &str) -> bool {
    get_global_validator().is_safe_input(input)
}
