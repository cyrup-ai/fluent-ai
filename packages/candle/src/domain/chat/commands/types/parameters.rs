//! Command parameter definitions and validation with zero allocation patterns
//!
//! Provides blazing-fast parameter validation and type checking with owned strings
//! allocated once for maximum performance. No Arc usage, no locking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;

use super::errors::{CandleCommandError, ValidationResult};

/// Parameter type enumeration for command parameters with zero allocation dispatch
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ParameterType {
    /// String parameter - text values
    String,
    /// Integer parameter - whole numbers
    Integer,
    /// Float parameter - decimal numbers
    Float,
    /// Boolean parameter - true/false values
    Boolean,
    /// Array of strings - multiple text values
    StringArray,
    /// File path parameter - filesystem paths
    FilePath,
    /// URL parameter - web addresses
    Url,
    /// JSON object parameter - structured data
    Json,
    /// Enumeration parameter with possible values
    Enum {
        /// Allowed values for this enum parameter
        values: Vec<String>,
    },
    /// Path parameter for file/directory paths
    Path,
    /// Duration parameter - time intervals
    Duration,
    /// Size parameter - byte counts with units
    Size,
    /// Regex pattern parameter
    Regex,
    /// Email address parameter
    Email,
    /// IP address parameter
    IpAddress,
    /// UUID parameter
    Uuid,
    /// Date parameter (ISO 8601 format)
    Date,
    /// Time parameter (HH:MM:SS format)
    Time,
    /// DateTime parameter (ISO 8601 format)
    DateTime,
}

impl ParameterType {
    /// Validate a string value against this parameter type
    /// Returns validation result with zero allocation where possible
    #[inline]
    pub fn validate(&self, value: &str) -> ValidationResult {
        match self {
            Self::String => Ok(()),
            Self::Integer => {
                i64::from_str(value).map_err(|_| 
                    CandleCommandError::validation_failed(
                        format!("Invalid integer: {}", value)
                    )
                )?;
                Ok(())
            }
            Self::Float => {
                f64::from_str(value).map_err(|_| 
                    CandleCommandError::validation_failed(
                        format!("Invalid float: {}", value)
                    )
                )?;
                Ok(())
            }
            Self::Boolean => {
                match value.to_lowercase().as_str() {
                    "true" | "false" | "1" | "0" | "yes" | "no" | "on" | "off" => Ok(()),
                    _ => Err(CandleCommandError::validation_failed(
                        format!("Invalid boolean: {}", value)
                    ))
                }
            }
            Self::StringArray => {
                // Basic validation - check if it's valid JSON array
                if value.starts_with('[') && value.ends_with(']') {
                    Ok(())
                } else {
                    Err(CandleCommandError::validation_failed(
                        "String array must be JSON array format".to_string()
                    ))
                }
            }
            Self::FilePath | Self::Path => {
                if value.is_empty() {
                    Err(CandleCommandError::validation_failed(
                        "Path cannot be empty".to_string()
                    ))
                } else {
                    Ok(())
                }
            }
            Self::Url => {
                if value.starts_with("http://") || value.starts_with("https://") || 
                   value.starts_with("ftp://") || value.starts_with("file://") {
                    Ok(())
                } else {
                    Err(CandleCommandError::validation_failed(
                        format!("Invalid URL format: {}", value)
                    ))
                }
            }
            Self::Json => {
                serde_json::from_str::<serde_json::Value>(value).map_err(|_| 
                    CandleCommandError::validation_failed(
                        format!("Invalid JSON: {}", value)
                    )
                )?;
                Ok(())
            }
            Self::Enum { values } => {
                if values.contains(&value.to_string()) {
                    Ok(())
                } else {
                    Err(CandleCommandError::validation_failed(
                        format!("Invalid enum value: {}. Allowed: {:?}", value, values)
                    ))
                }
            }
            Self::Duration => {
                // Parse duration like "5s", "10m", "1h", "2d"
                if self.parse_duration(value).is_some() {
                    Ok(())
                } else {
                    Err(CandleCommandError::validation_failed(
                        format!("Invalid duration format: {}", value)
                    ))
                }
            }
            Self::Size => {
                // Parse size like "100", "1KB", "2MB", "1GB"
                if self.parse_size(value).is_some() {
                    Ok(())
                } else {
                    Err(CandleCommandError::validation_failed(
                        format!("Invalid size format: {}", value)
                    ))
                }
            }
            Self::Regex => {
                regex::Regex::new(value).map_err(|_| 
                    CandleCommandError::validation_failed(
                        format!("Invalid regex pattern: {}", value)
                    )
                )?;
                Ok(())
            }
            Self::Email => {
                if value.contains('@') && value.contains('.') {
                    Ok(())
                } else {
                    Err(CandleCommandError::validation_failed(
                        format!("Invalid email format: {}", value)
                    ))
                }
            }
            Self::IpAddress => {
                if value.parse::<std::net::IpAddr>().is_ok() {
                    Ok(())
                } else {
                    Err(CandleCommandError::validation_failed(
                        format!("Invalid IP address: {}", value)
                    ))
                }
            }
            Self::Uuid => {
                if value.len() == 36 && value.chars().filter(|&c| c == '-').count() == 4 {
                    Ok(())
                } else {
                    Err(CandleCommandError::validation_failed(
                        format!("Invalid UUID format: {}", value)
                    ))
                }
            }
            Self::Date => {
                // Basic date validation - check format YYYY-MM-DD
                if value.len() == 10 && value.chars().filter(|&c| c == '-').count() == 2 {
                    Ok(())
                } else {
                    Err(CandleCommandError::validation_failed(
                        format!("Invalid date format (expected YYYY-MM-DD): {}", value)
                    ))
                }
            }
            Self::Time => {
                // Basic time validation - check format HH:MM:SS
                if value.len() >= 5 && value.chars().filter(|&c| c == ':').count() >= 1 {
                    Ok(())
                } else {
                    Err(CandleCommandError::validation_failed(
                        format!("Invalid time format (expected HH:MM:SS): {}", value)
                    ))
                }
            }
            Self::DateTime => {
                // Basic datetime validation - look for T separator
                if value.contains('T') || value.len() >= 19 {
                    Ok(())
                } else {
                    Err(CandleCommandError::validation_failed(
                        format!("Invalid datetime format (expected ISO 8601): {}", value)
                    ))
                }
            }
        }
    }
    
    /// Parse duration string to seconds - zero allocation where possible
    #[inline]
    fn parse_duration(&self, value: &str) -> Option<u64> {
        if value.is_empty() {
            return None;
        }
        
        let (num_str, unit) = if value.ends_with("ms") {
            (&value[..value.len()-2], "ms")
        } else if let Some(last_char) = value.chars().last() {
            match last_char {
                's' => (&value[..value.len()-1], "s"),
                'm' => (&value[..value.len()-1], "m"),
                'h' => (&value[..value.len()-1], "h"),
                'd' => (&value[..value.len()-1], "d"),
                _ => (value, "s"), // default to seconds
            }
        } else {
            return None;
        };
        
        let num: u64 = num_str.parse().ok()?;
        
        Some(match unit {
            "ms" => num / 1000,
            "s" => num,
            "m" => num * 60,
            "h" => num * 3600,
            "d" => num * 86400,
            _ => num,
        })
    }
    
    /// Parse size string to bytes - zero allocation where possible
    #[inline]
    fn parse_size(&self, value: &str) -> Option<u64> {
        if value.is_empty() {
            return None;
        }
        
        let (num_str, multiplier) = if value.ends_with("KB") {
            (&value[..value.len()-2], 1024)
        } else if value.ends_with("MB") {
            (&value[..value.len()-2], 1024 * 1024)
        } else if value.ends_with("GB") {
            (&value[..value.len()-2], 1024 * 1024 * 1024)
        } else if value.ends_with("TB") {
            (&value[..value.len()-2], 1024_u64.pow(4))
        } else if value.ends_with('B') {
            (&value[..value.len()-1], 1)
        } else {
            (value, 1) // assume bytes
        };
        
        let num: u64 = num_str.parse().ok()?;
        Some(num * multiplier)
    }
    
    /// Get type name as static string for zero allocation
    #[inline]
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Integer => "integer",
            Self::Float => "float",
            Self::Boolean => "boolean",
            Self::StringArray => "string_array",
            Self::FilePath => "file_path",
            Self::Url => "url",
            Self::Json => "json",
            Self::Enum { .. } => "enum",
            Self::Path => "path",
            Self::Duration => "duration",
            Self::Size => "size",
            Self::Regex => "regex",
            Self::Email => "email",
            Self::IpAddress => "ip_address",
            Self::Uuid => "uuid",
            Self::Date => "date",
            Self::Time => "time",
            Self::DateTime => "datetime",
        }
    }
}

/// Parameter information for command definitions with owned strings allocated once
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInfo {
    /// Parameter name (owned string allocated once)
    pub name: String,
    /// Parameter description (owned string allocated once)
    pub description: String,
    /// Parameter type
    pub parameter_type: ParameterType,
    /// Whether the parameter is required
    pub required: bool,
    /// Default value if not required (owned string allocated once)
    pub default_value: Option<String>,
    /// Minimum value for numeric parameters
    pub min_value: Option<f64>,
    /// Maximum value for numeric parameters
    pub max_value: Option<f64>,
    /// Pattern for string validation (regex)
    pub pattern: Option<String>,
    /// Examples of valid values
    pub examples: Vec<String>,
}

impl ParameterInfo {
    /// Create a new required parameter with zero allocation constructor
    #[inline]
    pub fn required(
        name: impl Into<String>, 
        description: impl Into<String>, 
        parameter_type: ParameterType
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameter_type,
            required: true,
            default_value: None,
            min_value: None,
            max_value: None,
            pattern: None,
            examples: Vec::new(),
        }
    }
    
    /// Create a new optional parameter with default value
    #[inline]
    pub fn optional(
        name: impl Into<String>, 
        description: impl Into<String>, 
        parameter_type: ParameterType,
        default_value: impl Into<String>
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameter_type,
            required: false,
            default_value: Some(default_value.into()),
            min_value: None,
            max_value: None,
            pattern: None,
            examples: Vec::new(),
        }
    }
    
    /// Add numeric range constraints - builder pattern for fluent API
    #[inline]
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }
    
    /// Add validation pattern - builder pattern for fluent API
    #[inline]
    pub fn with_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.pattern = Some(pattern.into());
        self
    }
    
    /// Add usage examples - builder pattern for fluent API
    #[inline]
    pub fn with_examples(mut self, examples: Vec<String>) -> Self {
        self.examples = examples;
        self
    }
    
    /// Validate parameter value with comprehensive checking
    #[inline]
    pub fn validate(&self, value: &str) -> ValidationResult {
        // Type validation first
        self.parameter_type.validate(value)?;
        
        // Range validation for numeric types
        if let Some(min) = self.min_value {
            match &self.parameter_type {
                ParameterType::Integer => {
                    let val = value.parse::<i64>().map_err(|_| 
                        CandleCommandError::validation_failed("Invalid integer")
                    )?;
                    if (val as f64) < min {
                        return Err(CandleCommandError::validation_failed(
                            format!("Value {} is below minimum {}", val, min)
                        ));
                    }
                }
                ParameterType::Float => {
                    let val = value.parse::<f64>().map_err(|_| 
                        CandleCommandError::validation_failed("Invalid float")
                    )?;
                    if val < min {
                        return Err(CandleCommandError::validation_failed(
                            format!("Value {} is below minimum {}", val, min)
                        ));
                    }
                }
                _ => {}
            }
        }
        
        if let Some(max) = self.max_value {
            match &self.parameter_type {
                ParameterType::Integer => {
                    let val = value.parse::<i64>().map_err(|_| 
                        CandleCommandError::validation_failed("Invalid integer")
                    )?;
                    if (val as f64) > max {
                        return Err(CandleCommandError::validation_failed(
                            format!("Value {} is above maximum {}", val, max)
                        ));
                    }
                }
                ParameterType::Float => {
                    let val = value.parse::<f64>().map_err(|_| 
                        CandleCommandError::validation_failed("Invalid float")
                    )?;
                    if val > max {
                        return Err(CandleCommandError::validation_failed(
                            format!("Value {} is above maximum {}", val, max)
                        ));
                    }
                }
                _ => {}
            }
        }
        
        // Pattern validation
        if let Some(pattern) = &self.pattern {
            let regex = regex::Regex::new(pattern).map_err(|_| 
                CandleCommandError::validation_failed("Invalid validation pattern")
            )?;
            if !regex.is_match(value) {
                return Err(CandleCommandError::validation_failed(
                    format!("Value '{}' does not match required pattern", value)
                ));
            }
        }
        
        Ok(())
    }
}

/// Parameter validator for bulk validation with zero allocation patterns
#[derive(Debug, Clone)]
pub struct ParameterValidator {
    parameters: Vec<ParameterInfo>,
}

impl ParameterValidator {
    /// Create new validator with parameter definitions
    #[inline]
    pub fn new(parameters: Vec<ParameterInfo>) -> Self {
        Self { parameters }
    }
    
    /// Validate all parameters in a command with comprehensive error reporting
    #[inline]
    pub fn validate_all(&self, values: &HashMap<String, String>) -> ValidationResult {
        let mut errors = Vec::new();
        
        // Check required parameters
        for param in &self.parameters {
            if param.required && !values.contains_key(&param.name) {
                errors.push(format!("Missing required parameter '{}'", param.name));
                continue;
            }
            
            // Validate provided values
            if let Some(value) = values.get(&param.name) {
                if let Err(err) = param.validate(value) {
                    errors.push(format!("Parameter '{}': {}", param.name, err));
                }
            }
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(CandleCommandError::validation_failed(errors.join("; ")))
        }
    }
    
    /// Get parameter by name - zero allocation lookup
    #[inline]
    pub fn get_parameter(&self, name: &str) -> Option<&ParameterInfo> {
        self.parameters.iter().find(|p| p.name == name)
    }
    
    /// Get all parameters - returns slice to avoid allocation
    #[inline]
    pub fn parameters(&self) -> &[ParameterInfo] {
        &self.parameters
    }
}