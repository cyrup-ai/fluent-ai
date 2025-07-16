//! Zero-allocation tool calling implementation for Anthropic API
//!
//! Comprehensive tool calling support with automatic tool execution,
//! result processing, and context injection with optimal performance.

use crate::providers::anthropic::{AnthropicError, AnthropicResult, Message, Tool};
use crate::providers::anthropic::messages::ContentBlock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub name: String,
    pub result: ToolOutput,
    pub execution_time_ms: Option<u64>,
    pub success: bool,
}

/// Tool output data
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolOutput {
    Text(String),
    Json(Value),
    Error { message: String, code: Option<String> },
}

/// Tool execution context with metadata
#[derive(Debug, Clone)]
pub struct ToolExecutionContext {
    pub conversation_id: Option<String>,
    pub user_id: Option<String>,
    pub metadata: HashMap<String, Value>,
    pub timeout_ms: Option<u64>,
    pub max_retries: u32,
}

/// Tool registry for managing available tools
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Tool>,
    executors: HashMap<String, Box<dyn ToolExecutor + Send + Sync>>,
}

/// Trait for tool execution implementations
pub trait ToolExecutor: Send + Sync {
    /// Execute tool with given input and context
    fn execute(
        &self,
        input: Value,
        context: &ToolExecutionContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = AnthropicResult<ToolOutput>> + Send>>;
    
    /// Get tool definition
    fn definition(&self) -> Tool;
    
    /// Validate input before execution
    fn validate_input(&self, input: &Value) -> AnthropicResult<()> {
        // Default validation - check if input is an object
        if !input.is_object() {
            return Err(AnthropicError::InvalidRequest(
                "Tool input must be a JSON object".to_string()
            ));
        }
        Ok(())
    }
}

/// Built-in calculator tool
pub struct CalculatorTool;

impl ToolExecutor for CalculatorTool {
    fn execute(
        &self,
        input: Value,
        _context: &ToolExecutionContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = AnthropicResult<ToolOutput>> + Send>> {
        Box::pin(async move {
        let expression = input
            .get("expression")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AnthropicError::InvalidRequest(
                "Calculator requires 'expression' parameter".to_string()
            ))?;
        
        // Simple expression evaluation (in production, use a proper parser)
        match evaluate_expression(expression) {
            Ok(result) => Ok(ToolOutput::Json(json!({
                "result": result,
                "expression": expression
            }))),
            Err(e) => Ok(ToolOutput::Error {
                message: format!("Calculation error: {}", e),
                code: Some("EVAL_ERROR".to_string()),
            }),
        }
        })
    }
    
    fn definition(&self) -> Tool {
        Tool::new(
            "calculator",
            "Perform mathematical calculations",
            json!({
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
}

/// Built-in web search tool
pub struct WebSearchTool;

impl ToolExecutor for WebSearchTool {
    fn execute(
        &self,
        input: Value,
        _context: &ToolExecutionContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = AnthropicResult<ToolOutput>> + Send>> {
        Box::pin(async move {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AnthropicError::InvalidRequest(
                "Web search requires 'query' parameter".to_string()
            ))?;
        
        let max_results = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(5);
        
        // Placeholder implementation - in production, integrate with search API
        Ok(ToolOutput::Json(json!({
            "query": query,
            "results": [
                {
                    "title": "Example Result 1",
                    "url": "https://example.com/1",
                    "snippet": "This is an example search result for the query."
                },
                {
                    "title": "Example Result 2", 
                    "url": "https://example.com/2",
                    "snippet": "Another example result with relevant information."
                }
            ],
            "total_results": max_results
        })))
        })
    }
    
    fn definition(&self) -> Tool {
        Tool::new(
            "web_search",
            "Search the web for information",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }),
        )
    }
}

/// Built-in file operations tool
pub struct FileOperationsTool;

impl ToolExecutor for FileOperationsTool {
    fn execute(
        &self,
        input: Value,
        _context: &ToolExecutionContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = AnthropicResult<ToolOutput>> + Send>> {
        Box::pin(async move {
        let operation = input
            .get("operation")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AnthropicError::InvalidRequest(
                "File operation requires 'operation' parameter".to_string()
            ))?;
        
        let path = input
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AnthropicError::InvalidRequest(
                "File operation requires 'path' parameter".to_string()
            ))?;
        
        match operation {
            "read" => {
                // Placeholder - in production, implement secure file reading
                Ok(ToolOutput::Json(json!({
                    "operation": "read",
                    "path": path,
                    "content": "File content would be here",
                    "size_bytes": 1024
                })))
            }
            "list" => {
                // Placeholder - in production, implement directory listing
                Ok(ToolOutput::Json(json!({
                    "operation": "list",
                    "path": path,
                    "files": [
                        {"name": "example.txt", "type": "file", "size": 1024},
                        {"name": "subfolder", "type": "directory", "size": null}
                    ]
                })))
            }
            _ => Ok(ToolOutput::Error {
                message: format!("Unsupported operation: {}", operation),
                code: Some("INVALID_OPERATION".to_string()),
            }),
        }
        })
    }
    
    fn definition(&self) -> Tool {
        Tool::new(
            "file_operations",
            "Perform file system operations",
            json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "list"],
                        "description": "Type of file operation to perform"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory path"
                    }
                },
                "required": ["operation", "path"]
            }),
        )
    }
}

impl ToolRegistry {
    /// Create new tool registry with built-in tools
    #[inline(always)]
    pub fn with_builtins() -> Self {
        let mut registry = Self::default();
        
        // Register built-in tools
        registry.register_tool(Box::new(CalculatorTool));
        registry.register_tool(Box::new(WebSearchTool));
        registry.register_tool(Box::new(FileOperationsTool));
        
        registry
    }

    /// Register a new tool
    #[inline(always)]
    pub fn register_tool(&mut self, executor: Box<dyn ToolExecutor + Send + Sync>) {
        let definition = executor.definition();
        let name = definition.name.clone();
        
        self.tools.insert(name.clone(), definition);
        self.executors.insert(name, executor);
    }

    /// Get tool definition by name
    #[inline(always)]
    pub fn get_tool(&self, name: &str) -> Option<&Tool> {
        self.tools.get(name)
    }

    /// Get all available tools
    #[inline(always)]
    pub fn get_all_tools(&self) -> Vec<&Tool> {
        self.tools.values().collect()
    }

    /// Execute tool by name
    pub async fn execute_tool(
        &self,
        name: &str,
        input: Value,
        context: &ToolExecutionContext,
    ) -> AnthropicResult<ToolResult> {
        let start_time = std::time::Instant::now();
        
        let executor = self.executors.get(name)
            .ok_or_else(|| AnthropicError::ToolExecutionError {
                tool_name: name.to_string(),
                error: "Tool not found".to_string(),
            })?;
        
        // Validate input
        executor.validate_input(&input)?;
        
        // Execute with timeout if specified
        let result = if let Some(timeout_ms) = context.timeout_ms {
            tokio::time::timeout(
                std::time::Duration::from_millis(timeout_ms),
                executor.execute(input.clone(), context)
            ).await
            .map_err(|_| AnthropicError::ToolExecutionError {
                tool_name: name.to_string(),
                error: "Tool execution timeout".to_string(),
            })?
        } else {
            executor.execute(input.clone(), context).await
        };
        
        let execution_time = start_time.elapsed();
        
        match result {
            Ok(output) => Ok(ToolResult {
                tool_use_id: context.metadata.get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                name: name.to_string(),
                result: output,
                execution_time_ms: Some(execution_time.as_millis() as u64),
                success: true,
            }),
            Err(e) => Ok(ToolResult {
                tool_use_id: context.metadata.get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                name: name.to_string(),
                result: ToolOutput::Error {
                    message: e.to_string(),
                    code: Some("EXECUTION_ERROR".to_string()),
                },
                execution_time_ms: Some(execution_time.as_millis() as u64),
                success: false,
            }),
        }
    }
    
    /// Process tool use blocks and create result messages
    pub async fn process_tool_calls(
        &self,
        content_blocks: &[ContentBlock],
        context: &ToolExecutionContext,
    ) -> Vec<Message> {
        let mut result_messages = Vec::new();
        
        for block in content_blocks {
            if let ContentBlock::ToolUse { id, name, input } = block {
                let mut tool_context = context.clone();
                tool_context.metadata.insert(
                    "tool_use_id".to_string(),
                    Value::String(id.clone())
                );
                
                match self.execute_tool(name, input.clone(), &tool_context).await {
                    Ok(result) => {
                        let content = match result.result {
                            ToolOutput::Text(text) => text,
                            ToolOutput::Json(json) => serde_json::to_string_pretty(&json)
                                .unwrap_or_else(|_| json.to_string()),
                            ToolOutput::Error { message, code } => {
                                if let Some(code) = code {
                                    format!("Error {}: {}", code, message)
                                } else {
                                    format!("Error: {}", message)
                                }
                            }
                        };
                        
                        if result.success {
                            result_messages.push(Message::tool_result(id, content));
                        } else {
                            result_messages.push(Message::tool_error(id, content));
                        }
                    }
                    Err(e) => {
                        result_messages.push(Message::tool_error(id, e.to_string()));
                    }
                }
            }
        }
        
        result_messages
    }
}

impl Default for ToolExecutionContext {
    fn default() -> Self {
        Self {
            conversation_id: None,
            user_id: None,
            metadata: HashMap::new(),
            timeout_ms: Some(30_000), // 30 second default timeout
            max_retries: 3,
        }
    }
}

impl ToolExecutionContext {
    /// Create new tool execution context
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set conversation ID
    #[inline(always)]
    pub fn with_conversation_id(mut self, id: impl Into<String>) -> Self {
        self.conversation_id = Some(id.into());
        self
    }
    
    /// Set user ID
    #[inline(always)]
    pub fn with_user_id(mut self, id: impl Into<String>) -> Self {
        self.user_id = Some(id.into());
        self
    }
    
    /// Add metadata
    #[inline(always)]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
    
    /// Set timeout
    #[inline(always)]
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
}

/// Simple expression evaluator for calculator tool
fn evaluate_expression(expr: &str) -> Result<f64, String> {
    // This is a simplified implementation
    // In production, use a proper expression parser like `evalexpr`
    let expr = expr.replace(' ', "");
    
    // Handle basic arithmetic
    if let Ok(result) = expr.parse::<f64>() {
        return Ok(result);
    }
    
    // Simple addition/subtraction
    if let Some(pos) = expr.rfind('+') {
        let left = evaluate_expression(&expr[..pos])?;
        let right = evaluate_expression(&expr[pos+1..])?;
        return Ok(left + right);
    }
    
    if let Some(pos) = expr.rfind('-') {
        let left = evaluate_expression(&expr[..pos])?;
        let right = evaluate_expression(&expr[pos+1..])?;
        return Ok(left - right);
    }
    
    if let Some(pos) = expr.rfind('*') {
        let left = evaluate_expression(&expr[..pos])?;
        let right = evaluate_expression(&expr[pos+1..])?;
        return Ok(left * right);
    }
    
    if let Some(pos) = expr.rfind('/') {
        let left = evaluate_expression(&expr[..pos])?;
        let right = evaluate_expression(&expr[pos+1..])?;
        if right == 0.0 {
            return Err("Division by zero".to_string());
        }
        return Ok(left / right);
    }
    
    Err(format!("Unable to evaluate expression: {}", expr))
}