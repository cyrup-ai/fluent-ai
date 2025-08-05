//! Command parameter definitions and metadata
//!
//! This module defines parameter types, validation, and metadata structures
//! for command definitions and parsing.

use serde::{Deserialize, Serialize};

/// Parameter type enumeration for command parameters
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ParameterType {
    /// String parameter
    String,
    /// Integer parameter
    Integer,
    /// Float parameter
    Float,
    /// Boolean parameter
    Boolean,
    /// Array of strings
    StringArray,
    /// File path parameter
    FilePath,
    /// URL parameter
    Url,
    /// JSON object parameter
    Json,
    /// Enumeration parameter with possible values
    Enum,
    /// Path parameter for file/directory paths
    Path,
}

impl ParameterType {
    /// Get display name for parameter type
    pub fn display_name(&self) -> &'static str {
        match self {
            ParameterType::String => "string",
            ParameterType::Integer => "integer",
            ParameterType::Float => "float",
            ParameterType::Boolean => "boolean",
            ParameterType::StringArray => "string[]",
            ParameterType::FilePath => "file_path",
            ParameterType::Url => "url",
            ParameterType::Json => "json",
            ParameterType::Enum => "enum",
            ParameterType::Path => "path",
        }
    }

    /// Check if parameter type accepts array values
    pub fn is_array_type(&self) -> bool {
        matches!(self, ParameterType::StringArray)
    }

    /// Check if parameter type is numeric
    pub fn is_numeric(&self) -> bool {
        matches!(self, ParameterType::Integer | ParameterType::Float)
    }
}

/// Parameter information for command definitions with owned strings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInfo {
    /// Parameter name
    pub name: String,
    /// Parameter description  
    pub description: String,
    /// Parameter type
    pub parameter_type: ParameterType,
    /// Whether the parameter is required
    pub required: bool,
    /// Default value if not required
    pub default_value: Option<String>,
    /// Allowed values for enum parameters
    pub allowed_values: Option<Vec<String>>,
    /// Minimum value for numeric parameters
    pub min_value: Option<f64>,
    /// Maximum value for numeric parameters
    pub max_value: Option<f64>,
}

impl ParameterInfo {
    /// Create a new required parameter
    pub fn required(name: impl Into<String>, description: impl Into<String>, parameter_type: ParameterType) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameter_type,
            required: true,
            default_value: None,
            allowed_values: None,
            min_value: None,
            max_value: None,
        }
    }

    /// Create a new optional parameter with default value
    pub fn optional(name: impl Into<String>, description: impl Into<String>, parameter_type: ParameterType, default_value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameter_type,
            required: false,
            default_value: Some(default_value.into()),
            allowed_values: None,
            min_value: None,
            max_value: None,
        }
    }

    /// Create an enum parameter with allowed values
    pub fn enum_param(name: impl Into<String>, description: impl Into<String>, allowed_values: Vec<String>, required: bool) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameter_type: ParameterType::Enum,
            required,
            default_value: if !required && !allowed_values.is_empty() { Some(allowed_values[0].clone()) } else { None },
            allowed_values: Some(allowed_values),
            min_value: None,
            max_value: None,
        }
    }

    /// Add numeric constraints
    pub fn with_range(mut self, min: Option<f64>, max: Option<f64>) -> Self {
        self.min_value = min;
        self.max_value = max;
        self
    }

    /// Check if a value is valid for this parameter
    pub fn validate_value(&self, value: &str) -> Result<(), String> {
        match self.parameter_type {
            ParameterType::Integer => {
                let parsed: i64 = value.parse().map_err(|_| "Invalid integer value")?;
                if let Some(min) = self.min_value {
                    if (parsed as f64) < min {
                        return Err(format!("Value must be at least {}", min));
                    }
                }
                if let Some(max) = self.max_value {
                    if (parsed as f64) > max {
                        return Err(format!("Value must be at most {}", max));
                    }
                }
            }
            ParameterType::Float => {
                let parsed: f64 = value.parse().map_err(|_| "Invalid float value")?;
                if let Some(min) = self.min_value {
                    if parsed < min {
                        return Err(format!("Value must be at least {}", min));
                    }
                }
                if let Some(max) = self.max_value {
                    if parsed > max {
                        return Err(format!("Value must be at most {}", max));
                    }
                }
            }
            ParameterType::Boolean => {
                if !matches!(value.to_lowercase().as_str(), "true" | "false" | "1" | "0" | "yes" | "no") {
                    return Err("Invalid boolean value. Use true/false, yes/no, or 1/0".to_string());
                }
            }
            ParameterType::Enum => {
                if let Some(ref allowed) = self.allowed_values {
                    if !allowed.contains(&value.to_string()) {
                        return Err(format!("Value must be one of: {}", allowed.join(", ")));
                    }
                }
            }
            ParameterType::Url => {
                if !value.starts_with("http://") && !value.starts_with("https://") && !value.starts_with("ftp://") {
                    return Err("URL must start with http://, https://, or ftp://".to_string());
                }
            }
            _ => {} // Other types don't have specific validation rules in this simplified version
        }
        Ok(())
    }
}