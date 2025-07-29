//! Core Tool Implementation - EXACT API from ARCHITECTURE.md
//!
//! This module provides the Tool trait and implementations that support
//! the transparent JSON syntax: Tool<Perplexity>::new({"citations" => "true"})
//!
//! The syntax works automatically without exposing any macros to users.

use std::marker::PhantomData;
use serde_json::Value;
use hashbrown::HashMap;

// Note: The transparent JSON syntax {"key" => "value"} should work automatically
// through cyrup_sugars transformation without requiring explicit macro imports

/// Marker type for Perplexity
pub struct Perplexity;

/// Tool collection for managing multiple tools
#[derive(Debug, Clone, Default)]
pub struct ToolSet(Vec<ToolDefinition>);

impl ToolSet {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn push(&mut self, tool: ToolDefinition) {
        self.0.push(tool);
    }
}

/// Unified tool definition for all tool types with proper zero-allocation typing
#[derive(Debug, Clone)]
pub enum ToolDefinition {
    /// Typed tool with proper static dispatch instead of Box<dyn Any>
    Perplexity(PerplexityTool),
    /// Other typed tools can be added here with proper static dispatch
    Generic(GenericTool),
    /// Named tool for command-line tools
    Named(NamedTool)}

/// Perplexity tool with typed configuration
#[derive(Debug, Clone)]
pub struct PerplexityTool {
    pub config: HashMap<String, Value>,
}

/// Generic tool with typed configuration for extensibility
#[derive(Debug, Clone)]
pub struct GenericTool {
    pub tool_type: String,
    pub config: HashMap<String, Value>,
}

/// Generic Tool with type parameter
#[derive(Debug, Clone)]
pub struct Tool<T> {
    #[allow(dead_code)] // TODO: Use for type-level tool differentiation (Perplexity, etc.)
    _phantom: PhantomData<T>,
    #[allow(dead_code)] // TODO: Use for tool configuration and parameter storage
    config: HashMap<String, Value>,
    /// Cached parameters as JSON Value
    parameters: Value,
}

impl<T> Tool<T> {
    /// Create new tool with config - EXACT syntax: Tool<Perplexity>::new({"citations" => "true"})
    ///
    /// This method accepts the transparent JSON syntax {"key" => "value"} which is
    /// automatically transformed by cyrup_sugars into the appropriate HashMap.
    ///
    /// Examples:
    /// ```rust
    /// // Single parameter
    /// Tool::<Perplexity>::new({"citations" => "true"})
    ///
    /// // Multiple parameters
    /// Tool::<CustomTool>::new({"param1" => "value1", "param2" => "value2"})
    /// ```
    #[inline]
    pub fn new<P>(config: P) -> Self
    where
        P: Into<hashbrown::HashMap<&'static str, &'static str>>,
    {
        let config_map = config.into();
        let mut map = HashMap::with_capacity(config_map.len());

        for (k, v) in config_map {
            map.insert(k.to_string(), Value::String(v.to_string()));
        }

        let parameters = Value::Object(map.iter().map(|(k, v)| (k.clone(), v.clone())).collect());

        Self {
            _phantom: PhantomData,
            config: map,
            parameters,
        }
    }

}

/// Named tool builder
#[derive(Debug, Clone)]
pub struct NamedTool {
    #[allow(dead_code)] // TODO: Use for tool name identification and registration
    name: String,
    #[allow(dead_code)] // TODO: Use for tool binary executable path
    bin_path: Option<String>,
    #[allow(dead_code)] // TODO: Use for tool functionality description
    description: Option<String>,
    /// Cached parameters as JSON Value
    parameters: Value,
}

impl Tool<()> {
    /// Create named tool - EXACT syntax: Tool::named("cargo")
    pub fn named(name: impl Into<String>) -> NamedTool {
        NamedTool {
            name: name.into(),
            bin_path: None,
            description: None,
            parameters: Value::Object(serde_json::Map::new()),
        }
    }
}

impl NamedTool {
    /// Set binary path - EXACT syntax: .bin("~/.cargo/bin")
    pub fn bin(mut self, path: impl Into<String>) -> Self {
        self.bin_path = Some(path.into());
        self
    }

    /// Set description - EXACT syntax: .description("cargo --help".exec_to_text())
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

}

/// Extension trait for executing strings as shell commands and returning text output.
///
/// This trait provides a convenient way to execute shell commands from string-like types
/// and capture their stdout as a String. Primarily used for tool execution in the AI framework.
pub trait ExecToText {
    /// Execute the string as a shell command and return the stdout as a String.
    ///
    /// # Returns
    ///
    /// Returns the command's stdout as a String, or an error message if execution fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use fluent_ai_domain::tool::core::ExecToText;
    ///
    /// let output = "echo hello".exec_to_text();
    /// assert!(output.contains("hello"));
    /// ```
    fn exec_to_text(&self) -> String;
}

impl ExecToText for &str {
    fn exec_to_text(&self) -> String {
        // Execute command and return output
        std::process::Command::new("sh")
            .arg("-c")
            .arg(self)
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
            .unwrap_or_else(|_| format!("Failed to execute: {}", self))
    }
}

/// Dynamic tool embedding trait for runtime tool handling
pub trait ToolEmbeddingDyn: Send + Sync {
    /// Get tool name
    fn name(&self) -> String;

    /// Get embedding documentation strings
    fn embedding_docs(&self) -> Vec<String>;

    /// Get tool context as JSON value
    fn context(&self) -> Result<Value, Box<dyn std::error::Error + Send + Sync>>;
}

// Import the Tool trait
use crate::tool::traits::Tool as ToolTrait;
use fluent_ai_async::AsyncStream;

// Implement Tool trait for Tool<T>
impl<T> ToolTrait for Tool<T> 
where
    T: Send + Sync + std::fmt::Debug + Clone + 'static,
{
    fn name(&self) -> &str {
        std::any::type_name::<T>()
    }
    
    fn description(&self) -> &str {
        "Generic tool"
    }
    
    fn parameters(&self) -> &Value {
        // Return the stored parameters
        &self.parameters
    }
    
    fn execute(&self, args: Value) -> AsyncStream<Value> {
        let config = self.config.clone();
        AsyncStream::with_channel(move |sender| {
            // Execute tool with config and args
            let mut result = config;
            if let Value::Object(ref args_obj) = args {
                for (key, value) in args_obj {
                    result.insert(key.clone(), value.clone());
                }
            }
            let _ = sender.send(Value::Object(result.into_iter().collect()));
        })
    }
}

// Implement Tool trait for NamedTool
impl ToolTrait for NamedTool {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        self.description.as_deref().unwrap_or("Named tool")
    }
    
    fn parameters(&self) -> &Value {
        // Return stored parameters
        &self.parameters
    }
    
    fn execute(&self, args: Value) -> AsyncStream<Value> {
        let name = self.name.clone();
        let bin_path = self.bin_path.clone();
        let _description = self.description.clone();
        
        AsyncStream::with_channel(move |sender| {
            // Execute named tool using configured path and args
            let command = bin_path.as_deref().unwrap_or(&name);
            
            // Build command with args if provided
            let mut cmd_args = Vec::new();
            if let Value::Object(ref args_obj) = args {
                for (key, value) in args_obj {
                    cmd_args.push(format!("--{}", key));
                    if let Value::String(val) = value {
                        cmd_args.push(val.clone());
                    }
                }
            }
            
            // Execute command and capture output
            let output = std::process::Command::new(command)
                .args(&cmd_args)
                .output();
                
            let result = match output {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    
                    if output.status.success() {
                        Value::String(stdout.into_owned())
                    } else {
                        Value::Object(serde_json::Map::from_iter([
                            ("error".to_string(), Value::String(stderr.into_owned())),
                            ("exit_code".to_string(), Value::Number(output.status.code().unwrap_or(-1).into()))
                        ]))
                    }
                }
                Err(e) => {
                    Value::Object(serde_json::Map::from_iter([
                        ("error".to_string(), Value::String(format!("Failed to execute {}: {}", command, e))),
                        ("exit_code".to_string(), Value::Number((-1).into()))
                    ]))
                }
            };
            
            let _ = sender.send(result);
        })
    }
}

// Send + Sync are automatically implemented for these types
// since all fields are Send + Sync (PhantomData, HashMap, String, Option)
