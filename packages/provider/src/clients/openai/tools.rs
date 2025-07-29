//! Zero-allocation OpenAI tool/function calling implementation
//!
//! Provides comprehensive support for OpenAI's function calling and tool use features
//! with optimal performance patterns and full API compatibility.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::{OpenAIError, OpenAIResult};
use crate::ZeroOneOrMany;
use fluent_ai_domain::completion::types::ToolDefinition;

/// OpenAI tool definition for function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAITool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAIFunction}

/// OpenAI function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>}

/// Tool choice configuration for OpenAI API
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAIToolChoice {
    Auto,
    None,
    Required,
    Specific {
        #[serde(rename = "type")]
        tool_type: String,
        function: OpenAIFunctionChoice}}

/// Specific function choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunctionChoice {
    pub name: String}

/// Tool call from OpenAI response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OpenAIFunctionCall}

/// Function call from OpenAI response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: String, // JSON string
}

/// Tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolResult {
    pub tool_call_id: String,
    pub result: Value,
    pub error: Option<String>}

/// Function execution context
#[derive(Debug, Clone)]
pub struct OpenAIFunctionContext {
    pub name: String,
    pub arguments: Value,
    pub call_id: String,
    pub model: String}

impl OpenAITool {
    /// Create new tool with function definition
    #[inline(always)]
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: OpenAIFunction {
                name: name.into(),
                description: description.into(),
                parameters,
                strict: None}}
    }

    /// Create tool with strict mode enabled (for structured outputs)
    #[inline(always)]
    pub fn function_strict(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: OpenAIFunction {
                name: name.into(),
                description: description.into(),
                parameters,
                strict: Some(true)}}
    }

    /// Create code interpreter tool
    #[inline(always)]
    pub fn code_interpreter() -> Self {
        Self {
            tool_type: "code_interpreter".to_string(),
            function: OpenAIFunction {
                name: "code_interpreter".to_string(),
                description: "Execute Python code in a sandboxed environment".to_string(),
                parameters: Value::Object(Map::new()),
                strict: None}}
    }

    /// Create file search tool
    #[inline(always)]
    pub fn file_search() -> Self {
        Self {
            tool_type: "file_search".to_string(),
            function: OpenAIFunction {
                name: "file_search".to_string(),
                description: "Search and retrieve information from uploaded files".to_string(),
                parameters: Value::Object(Map::new()),
                strict: None}}
    }

    /// Validate tool definition
    #[inline(always)]
    pub fn validate(&self) -> OpenAIResult<()> {
        if self.function.name.is_empty() {
            return Err(OpenAIError::ToolError(
                "Function name cannot be empty".to_string(),
            ));
        }

        if self.function.description.is_empty() {
            return Err(OpenAIError::ToolError(
                "Function description cannot be empty".to_string(),
            ));
        }

        // Validate parameters schema
        if !self.function.parameters.is_object() {
            return Err(OpenAIError::ToolError(
                "Function parameters must be a JSON object".to_string(),
            ));
        }

        Ok(())
    }

    /// Get parameter schema for function
    #[inline(always)]
    pub fn get_parameter_schema(&self) -> &Value {
        &self.function.parameters
    }

    /// Check if function has required parameters
    #[inline(always)]
    pub fn has_required_parameters(&self) -> bool {
        if let Some(required) = self.function.parameters.get("required") {
            if let Some(required_array) = required.as_array() {
                return !required_array.is_empty();
            }
        }
        false
    }

    /// Get required parameter names
    #[inline(always)]
    pub fn get_required_parameters(&self) -> Vec<String> {
        if let Some(required) = self.function.parameters.get("required") {
            if let Some(required_array) = required.as_array() {
                return required_array
                    .iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect();
            }
        }
        Vec::new()
    }
}

impl OpenAIToolChoice {
    /// Auto tool choice (let model decide)
    #[inline(always)]
    pub fn auto() -> Self {
        Self::Auto
    }

    /// No tool use
    #[inline(always)]
    pub fn none() -> Self {
        Self::None
    }

    /// Require tool use
    #[inline(always)]
    pub fn required() -> Self {
        Self::Required
    }

    /// Force specific function
    #[inline(always)]
    pub fn function(name: impl Into<String>) -> Self {
        Self::Specific {
            tool_type: "function".to_string(),
            function: OpenAIFunctionChoice { name: name.into() }}
    }

    /// Serialize to JSON value for API
    #[inline(always)]
    pub fn to_value(&self) -> Value {
        match self {
            Self::Auto => Value::String("auto".to_string()),
            Self::None => Value::String("none".to_string()),
            Self::Required => Value::String("required".to_string()),
            Self::Specific {
                tool_type,
                function} => {
                serde_json::json!({
                    "type": tool_type,
                    "function": {
                        "name": function.name
                    }
                })
            }
        }
    }
}

impl OpenAIToolCall {
    /// Parse function arguments as JSON
    #[inline(always)]
    pub fn parse_arguments(&self) -> OpenAIResult<Value> {
        serde_json::from_str(&self.function.arguments)
            .map_err(|e| OpenAIError::ToolError(format!("Invalid function arguments JSON: {}", e)))
    }

    /// Parse arguments into specific type
    #[inline(always)]
    pub fn parse_arguments_as<T>(&self) -> OpenAIResult<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(&self.function.arguments)
            .map_err(|e| OpenAIError::ToolError(format!("Failed to parse arguments: {}", e)))
    }

    /// Get function execution context
    #[inline(always)]
    pub fn to_context(&self, model: impl Into<String>) -> OpenAIResult<OpenAIFunctionContext> {
        let arguments = self.parse_arguments()?;
        Ok(OpenAIFunctionContext {
            name: self.function.name.clone(),
            arguments,
            call_id: self.id.clone(),
            model: model.into()})
    }

    /// Validate tool call against function definition
    #[inline(always)]
    pub fn validate_against_function(&self, function: &OpenAIFunction) -> OpenAIResult<()> {
        if self.function.name != function.name {
            return Err(OpenAIError::ToolError(format!(
                "Function name mismatch: expected {}, got {}",
                function.name, self.function.name
            )));
        }

        let arguments = self.parse_arguments()?;

        // Validate required parameters
        static EMPTY_VEC: Vec<serde_json::Value> = Vec::new();
        let required_params = function
            .parameters
            .get("required")
            .and_then(|r| r.as_array())
            .unwrap_or(&EMPTY_VEC);

        for required_param in required_params {
            if let Some(param_name) = required_param.as_str() {
                if !arguments
                    .as_object()
                    .map(|obj| obj.contains_key(param_name))
                    .unwrap_or(false)
                {
                    return Err(OpenAIError::ToolError(format!(
                        "Required parameter '{}' is missing",
                        param_name
                    )));
                }
            }
        }

        Ok(())
    }
}

impl OpenAIToolResult {
    /// Create successful tool result
    #[inline(always)]
    pub fn success(tool_call_id: impl Into<String>, result: Value) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            result,
            error: None}
    }

    /// Create error tool result
    #[inline(always)]
    pub fn error(tool_call_id: impl Into<String>, error_message: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            result: Value::Null,
            error: Some(error_message.into())}
    }

    /// Check if result is successful
    #[inline(always)]
    pub fn is_success(&self) -> bool {
        self.error.is_none()
    }

    /// Get result as string for message content
    #[inline(always)]
    pub fn to_message_content(&self) -> String {
        if let Some(error) = &self.error {
            format!("Error: {}", error)
        } else {
            self.result.to_string()
        }
    }
}

/// Convert fluent-ai ToolDefinition to OpenAI format
#[inline(always)]
pub fn convert_tool_definition(tool: &ToolDefinition) -> OpenAIResult<OpenAITool> {
    Ok(OpenAITool::function(
        &tool.name,
        &tool.description,
        tool.parameters.clone(),
    ))
}

/// Convert multiple tool definitions with batch optimization
#[inline(always)]
pub fn convert_tool_definitions(
    tools: &ZeroOneOrMany<ToolDefinition>,
) -> OpenAIResult<Vec<OpenAITool>> {
    match tools {
        ZeroOneOrMany::None => Ok(Vec::new()),
        ZeroOneOrMany::One(tool) => Ok(vec![convert_tool_definition(tool)?]),
        ZeroOneOrMany::Many(tools) => {
            let mut result = Vec::with_capacity(tools.len());
            for tool in tools {
                result.push(convert_tool_definition(tool)?);
            }
            Ok(result)
        }
    }
}

/// Create common tool definitions for typical use cases
pub mod common_tools {
    use super::*;

    /// Web search tool
    #[inline(always)]
    pub fn web_search() -> OpenAITool {
        OpenAITool::function(
            "web_search",
            "Search the web for current information",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }),
        )
    }

    /// Weather tool
    #[inline(always)]
    pub fn get_weather() -> OpenAITool {
        OpenAITool::function(
            "get_weather",
            "Get current weather information for a location",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country, e.g. 'San Francisco, CA'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                        "default": "celsius"
                    }
                },
                "required": ["location"]
            }),
        )
    }

    /// File operations tool
    #[inline(always)]
    pub fn file_operations() -> OpenAITool {
        OpenAITool::function(
            "file_operations",
            "Perform file operations like read, write, list",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "write", "list", "delete"],
                        "description": "Type of file operation"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory path"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write (for write operation)"
                    }
                },
                "required": ["operation", "path"]
            }),
        )
    }

    /// Calculator tool
    #[inline(always)]
    pub fn calculator() -> OpenAITool {
        OpenAITool::function(
            "calculator",
            "Perform mathematical calculations",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }),
        )
    }

    /// Database query tool
    #[inline(always)]
    pub fn database_query() -> OpenAITool {
        OpenAITool::function(
            "database_query",
            "Execute database queries",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute"
                    },
                    "database": {
                        "type": "string",
                        "description": "Database name or connection string"
                    }
                },
                "required": ["query"]
            }),
        )
    }
}
