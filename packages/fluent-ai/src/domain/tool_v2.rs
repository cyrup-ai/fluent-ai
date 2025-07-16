//! Tool v2 implementation - EXACT API from ARCHITECTURE.md

use crate::HashMap;
use serde_json::Value;
use std::marker::PhantomData;

/// Marker type for Perplexity
pub struct Perplexity;

/// Generic Tool with type parameter
pub struct Tool<T> {
    #[allow(dead_code)] // TODO: Use for type-level tool differentiation (Perplexity, etc.)
    _phantom: PhantomData<T>,
    #[allow(dead_code)] // TODO: Use for tool configuration and parameter storage
    config: HashMap<String, Value>,
}

impl<T> Tool<T> {
    /// Create new tool with config - EXACT syntax: Tool<Perplexity>::new({"citations" => "true"})
    pub fn new<F>(config: F) -> Self
    where
        F: FnOnce() -> HashMap<String, Value>,
    {
        Self {
            _phantom: PhantomData,
            config: config(),
        }
    }
}

/// Named tool builder
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

// Implement Send + Sync
unsafe impl<T> Send for Tool<T> {}
unsafe impl<T> Sync for Tool<T> {}
unsafe impl Send for NamedTool {}
unsafe impl Sync for NamedTool {}
