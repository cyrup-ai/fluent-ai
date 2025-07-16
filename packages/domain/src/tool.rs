//! Tool v2 implementation - EXACT API from ARCHITECTURE.md

use crate::HashMap;
use serde_json::Value;
use std::marker::PhantomData;

/// Macro to create hashbrown::HashMap from JSON-like syntax
/// Usage: json_map!{"key" => "value", "key2" => "value2"}
#[macro_export]
macro_rules! json_map {
    ({$($key:expr => $value:expr),* $(,)?}) => {{
        let mut map = hashbrown::HashMap::new();
        $(
            map.insert($key, $value);
        )*
        map
    }};
}

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

/// Unified tool definition for all tool types
#[derive(Debug)]
pub enum ToolDefinition {
    Typed(Box<dyn std::any::Any + Send + Sync>),
    Named(NamedTool),
}

impl Clone for ToolDefinition {
    fn clone(&self) -> Self {
        match self {
            ToolDefinition::Typed(_) => {
                // Can't clone Box<dyn Any>, so create a new empty one
                ToolDefinition::Typed(Box::new(()))
            }
            ToolDefinition::Named(named) => ToolDefinition::Named(named.clone()),
        }
    }
}

/// Generic Tool with type parameter
pub struct Tool<T> {
    #[allow(dead_code)] // TODO: Use for type-level tool differentiation (Perplexity, etc.)
    _phantom: PhantomData<T>,
    #[allow(dead_code)] // TODO: Use for tool configuration and parameter storage
    config: HashMap<String, Value>,
}

impl<T> Tool<T> {
    /// Create new tool with config - EXACT syntax: Tool<Perplexity>::new({"citations" => "true"})
    pub fn new<P>(config: P) -> Self
    where
        P: Into<hashbrown::HashMap<&'static str, &'static str>>,
    {
        let config_map = config.into();
        let mut map = HashMap::new();
        for (k, v) in config_map {
            map.insert(k.to_string(), Value::String(v.to_string()));
        }
        Self {
            _phantom: PhantomData,
            config: map,
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
}

impl Tool<()> {
    /// Create named tool - EXACT syntax: Tool::named("cargo")
    pub fn named(name: impl Into<String>) -> NamedTool {
        NamedTool {
            name: name.into(),
            bin_path: None,
            description: None,
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

// String extension for exec_to_text
pub trait ExecToText {
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

// Implement Send + Sync
unsafe impl<T> Send for Tool<T> {}
unsafe impl<T> Sync for Tool<T> {}
unsafe impl Send for NamedTool {}
unsafe impl Sync for NamedTool {}
