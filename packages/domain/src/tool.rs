//! Tool v2 implementation - EXACT API from ARCHITECTURE.md
//!
//! This module provides the Tool trait and implementations that support
//! the transparent JSON syntax: Tool<Perplexity>::new({"citations" => "true"})
//! 
//! The syntax works automatically without exposing any macros to users.

use crate::HashMap;
use serde_json::Value;
use std::marker::PhantomData;

// Import Cylo execution environment types
// Note: Using conditional compilation to handle optional cylo dependency
#[cfg(feature = "cylo")]
use fluent_ai_cylo::{CyloInstance, execution_env::Cylo};

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
    /// Optional Cylo execution environment instance
    #[cfg(feature = "cylo")]
    cylo_instance: Option<CyloInstance>,
    #[cfg(not(feature = "cylo"))]
    cylo_instance: Option<()>,
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
        
        Self {
            _phantom: PhantomData,
            config: map,
            cylo_instance: None,
        }
    }
    
    /// Set Cylo execution environment - EXACT syntax: .cylo(Cylo::Apple("python:alpine3.20").instance("env_name"))
    /// 
    /// Allows specifying a specific execution environment for the tool's code execution.
    /// 
    /// Examples:
    /// ```rust
    /// // Apple containerization
    /// Tool::<CustomTool>::new({"param" => "value"})
    ///     .cylo(Cylo::Apple("python:alpine3.20").instance("python_env"))
    /// 
    /// // LandLock sandboxing
    /// Tool::<CustomTool>::new({"param" => "value"})
    ///     .cylo(Cylo::LandLock("/path/to/jail").instance("secure_env"))
    /// 
    /// // FireCracker microVM
    /// Tool::<CustomTool>::new({"param" => "value"})
    ///     .cylo(Cylo::FireCracker("rust:alpine3.20").instance("vm_env"))
    /// ```
    #[cfg(feature = "cylo")]
    pub fn cylo(mut self, instance: CyloInstance) -> Self {
        self.cylo_instance = Some(instance);
        self
    }
    
    /// Set Cylo execution environment (no-op when cylo feature is disabled)
    #[cfg(not(feature = "cylo"))]
    pub fn cylo(self, _instance: ()) -> Self {
        self
    }
    
    /// Get the Cylo execution environment instance if set
    #[cfg(feature = "cylo")]
    pub fn get_cylo_instance(&self) -> Option<&CyloInstance> {
        self.cylo_instance.as_ref()
    }
    
    /// Get the Cylo execution environment instance (returns None when cylo feature is disabled)
    #[cfg(not(feature = "cylo"))]
    pub fn get_cylo_instance(&self) -> Option<&()> {
        None
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
    /// Optional Cylo execution environment instance
    #[cfg(feature = "cylo")]
    cylo_instance: Option<CyloInstance>,
    #[cfg(not(feature = "cylo"))]
    cylo_instance: Option<()>,
}

impl Tool<()> {
    /// Create named tool - EXACT syntax: Tool::named("cargo")
    pub fn named(name: impl Into<String>) -> NamedTool {
        NamedTool {
            name: name.into(),
            bin_path: None,
            description: None,
            cylo_instance: None,
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
    
    /// Set Cylo execution environment - EXACT syntax: .cylo(Cylo::Apple("python:alpine3.20").instance("env_name"))
    /// 
    /// Allows specifying a specific execution environment for the named tool's execution.
    /// 
    /// Examples:
    /// ```rust
    /// // Apple containerization with named tool
    /// Tool::named("cargo")
    ///     .bin("~/.cargo/bin")
    ///     .cylo(Cylo::Apple("rust:alpine3.20").instance("rust_env"))
    /// 
    /// // LandLock sandboxing with named tool
    /// Tool::named("python")
    ///     .bin("/usr/bin/python3")
    ///     .cylo(Cylo::LandLock("/tmp/sandbox").instance("py_env"))
    /// ```
    #[cfg(feature = "cylo")]
    pub fn cylo(mut self, instance: CyloInstance) -> Self {
        self.cylo_instance = Some(instance);
        self
    }
    
    /// Set Cylo execution environment (no-op when cylo feature is disabled)
    #[cfg(not(feature = "cylo"))]
    pub fn cylo(self, _instance: ()) -> Self {
        self
    }
    
    /// Get the Cylo execution environment instance if set
    #[cfg(feature = "cylo")]
    pub fn get_cylo_instance(&self) -> Option<&CyloInstance> {
        self.cylo_instance.as_ref()
    }
    
    /// Get the Cylo execution environment instance (returns None when cylo feature is disabled)
    #[cfg(not(feature = "cylo"))]
    pub fn get_cylo_instance(&self) -> Option<&()> {
        None
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
