//! Command parameter types and definitions
//!
//! Provides type-safe parameter handling for command parsing and validation
//! with zero-allocation patterns and efficient parameter processing.

use serde::{Deserialize, Serialize};

/// Parameter type enumeration for command parameters
///
/// Defines the expected data type for command parameters with support
/// for basic types, collections, and specialized parameter types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
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
    /// Get type name as string
    #[inline(always)]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Integer => "integer", 
            Self::Float => "float",
            Self::Boolean => "boolean",
            Self::StringArray => "string_array",
            Self::FilePath => "file_path",
            Self::Url => "url",
            Self::Json => "json",
            Self::Enum => "enum",
            Self::Path => "path",
        }
    }

    /// Check if type supports multiple values
    #[inline(always)]
    pub fn is_array(&self) -> bool {
        matches!(self, Self::StringArray)
    }

    /// Check if type expects structured data
    #[inline(always)]
    pub fn is_structured(&self) -> bool {
        matches!(self, Self::Json)
    }

    /// Check if type represents a filesystem path
    #[inline(always)]
    pub fn is_path(&self) -> bool {
        matches!(self, Self::FilePath | Self::Path)
    }

    /// Check if type is numeric
    #[inline(always)]
    pub fn is_numeric(&self) -> bool {
        matches!(self, Self::Integer | Self::Float)
    }

    /// Get type description for help text
    #[inline(always)]
    pub fn description(&self) -> &'static str {
        match self {
            Self::String => "A text string value",
            Self::Integer => "An integer number",
            Self::Float => "A floating point number",
            Self::Boolean => "A true/false value",
            Self::StringArray => "An array of text strings",
            Self::FilePath => "A path to a file",
            Self::Url => "A valid URL",
            Self::Json => "A JSON object",
            Self::Enum => "One of several predefined values",
            Self::Path => "A filesystem path",
        }
    }
}

impl std::fmt::Display for ParameterType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for ParameterType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "string" => Ok(Self::String),
            "integer" | "int" => Ok(Self::Integer),
            "float" | "double" => Ok(Self::Float),
            "boolean" | "bool" => Ok(Self::Boolean),
            "string_array" | "strings" => Ok(Self::StringArray),
            "file_path" | "file" => Ok(Self::FilePath),
            "url" => Ok(Self::Url),
            "json" => Ok(Self::Json),
            "enum" => Ok(Self::Enum),
            "path" => Ok(Self::Path),
            _ => Err(format!("Unknown parameter type: {}", s)),
        }
    }
}

/// Parameter information for command definitions with owned strings
///
/// Contains all metadata needed to define, validate, and document
/// command parameters with efficient owned string storage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParameterInfo {
    /// Parameter name
    pub name: String,
    /// Parameter description for help text
    pub description: String,
    /// Parameter type
    pub parameter_type: ParameterType,
    /// Whether the parameter is required
    pub required: bool,
    /// Default value if not required
    pub default_value: Option<String>,
    /// Possible values for enum parameters
    pub possible_values: Option<Vec<String>>,
    /// Minimum value for numeric parameters
    pub min_value: Option<f64>,
    /// Maximum value for numeric parameters  
    pub max_value: Option<f64>,
    /// Regular expression pattern for validation
    pub pattern: Option<String>,
}

impl ParameterInfo {
    /// Create a new required parameter
    #[inline(always)]
    pub fn required(name: impl Into<String>, parameter_type: ParameterType, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameter_type,
            required: true,
            default_value: None,
            possible_values: None,
            min_value: None,
            max_value: None,
            pattern: None,
        }
    }

    /// Create a new optional parameter with default value
    #[inline(always)]
    pub fn optional(
        name: impl Into<String>,
        parameter_type: ParameterType,
        description: impl Into<String>,
        default_value: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameter_type,
            required: false,
            default_value: Some(default_value.into()),
            possible_values: None,
            min_value: None,
            max_value: None,
            pattern: None,
        }
    }

    /// Create an enum parameter with possible values
    #[inline(always)]
    pub fn enum_param(
        name: impl Into<String>,
        description: impl Into<String>,
        possible_values: Vec<String>,
        required: bool,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameter_type: ParameterType::Enum,
            required,
            default_value: if required { None } else { possible_values.first().cloned() },
            possible_values: Some(possible_values),
            min_value: None,
            max_value: None,
            pattern: None,
        }
    }

    /// Add numeric constraints
    #[inline(always)]
    pub fn with_range(mut self, min: Option<f64>, max: Option<f64>) -> Self {
        self.min_value = min;
        self.max_value = max;
        self
    }

    /// Add validation pattern
    #[inline(always)]
    pub fn with_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.pattern = Some(pattern.into());
        self
    }

    /// Validate a parameter value
    pub fn validate(&self, value: &str) -> Result<(), String> {
        // Type-specific validation
        match self.parameter_type {
            ParameterType::Integer => {
                let parsed = value.parse::<i64>()
                    .map_err(|_| format!("Invalid integer value: {}", value))?;
                
                if let Some(min) = self.min_value {
                    if (parsed as f64) < min {
                        return Err(format!("Value {} is below minimum {}", parsed, min));
                    }
                }
                
                if let Some(max) = self.max_value {
                    if (parsed as f64) > max {
                        return Err(format!("Value {} is above maximum {}", parsed, max));
                    }
                }
            }
            
            ParameterType::Float => {
                let parsed = value.parse::<f64>()
                    .map_err(|_| format!("Invalid float value: {}", value))?;
                
                if let Some(min) = self.min_value {
                    if parsed < min {
                        return Err(format!("Value {} is below minimum {}", parsed, min));
                    }
                }
                
                if let Some(max) = self.max_value {
                    if parsed > max {
                        return Err(format!("Value {} is above maximum {}", parsed, max));
                    }
                }
            }
            
            ParameterType::Boolean => {
                if !matches!(value.to_lowercase().as_str(), "true" | "false" | "1" | "0" | "yes" | "no") {
                    return Err(format!("Invalid boolean value: {}", value));
                }
            }
            
            ParameterType::Enum => {
                if let Some(ref possible) = self.possible_values {
                    if !possible.contains(&value.to_string()) {
                        return Err(format!("Invalid enum value: {}. Possible values: {:?}", value, possible));
                    }
                }
            }
            
            ParameterType::Url => {
                if value.parse::<url::Url>().is_err() {
                    return Err(format!("Invalid URL: {}", value));
                }
            }
            
            ParameterType::Json => {
                if serde_json::from_str::<serde_json::Value>(value).is_err() {
                    return Err(format!("Invalid JSON: {}", value));
                }
            }
            
            _ => {} // Other types accept any string
        }

        // Pattern validation
        if let Some(ref pattern) = self.pattern {
            let regex = regex::Regex::new(pattern)
                .map_err(|_| format!("Invalid validation pattern: {}", pattern))?;
            
            if !regex.is_match(value) {
                return Err(format!("Value '{}' does not match required pattern", value));
            }
        }

        Ok(())
    }

    /// Get usage string for help text
    pub fn usage_string(&self) -> String {
        let name = if self.required {
            format!("<{}>", self.name)
        } else {
            format!("[{}]", self.name)
        };

        let type_info = match &self.parameter_type {
            ParameterType::Enum => {
                if let Some(ref values) = self.possible_values {
                    format!(" ({})", values.join("|"))
                } else {
                    String::new()
                }
            }
            _ => format!(" ({})", self.parameter_type),
        };

        format!("{}{}", name, type_info)
    }
}

impl Default for ParameterInfo {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            parameter_type: ParameterType::String,
            required: false,
            default_value: None,
            possible_values: None,
            min_value: None,
            max_value: None,
            pattern: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_type_as_str() {
        assert_eq!(ParameterType::String.as_str(), "string");
        assert_eq!(ParameterType::Integer.as_str(), "integer");
        assert_eq!(ParameterType::Boolean.as_str(), "boolean");
    }

    #[test]
    fn test_parameter_validation() {
        let int_param = ParameterInfo::required("test", ParameterType::Integer, "Test integer")
            .with_range(Some(1.0), Some(10.0));
        
        assert!(int_param.validate("5").is_ok());
        assert!(int_param.validate("0").is_err());
        assert!(int_param.validate("11").is_err());
        assert!(int_param.validate("not_a_number").is_err());
    }

    #[test]
    fn test_enum_parameter() {
        let enum_param = ParameterInfo::enum_param(
            "mode",
            "Operation mode",
            vec!["fast".to_string(), "slow".to_string(), "auto".to_string()],
            true,
        );
        
        assert!(enum_param.validate("fast").is_ok());
        assert!(enum_param.validate("invalid").is_err());
    }
}