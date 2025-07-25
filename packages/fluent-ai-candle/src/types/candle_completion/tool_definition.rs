//! Tool definition types for completion requests

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Tool definition for completion requests with zero-allocation design
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Tool parameters as JSON schema
    pub parameters: Value}

impl ToolDefinition {
    /// Create a new tool definition with elegant ergonomic API
    #[inline(always)]
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters}
    }

    /// Builder pattern for name with blazing-fast inline optimization
    #[inline(always)]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Builder pattern for description  
    #[inline(always)]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Builder pattern for parameters
    #[inline(always)]
    pub fn with_parameters(mut self, parameters: Value) -> Self {
        self.parameters = parameters;
        self
    }

    /// Get tool name reference for zero-allocation access
    #[inline(always)]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get tool description reference for zero-allocation access
    #[inline(always)]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Get parameters reference for zero-allocation access
    #[inline(always)]
    pub fn parameters(&self) -> &Value {
        &self.parameters
    }
}
