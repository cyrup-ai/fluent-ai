use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::error::Error as StdError;

/// Configuration for creating an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub model: String,
    pub system_prompt: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub tools: Vec<String>,
}

/// Configuration for extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    pub model: String,
    pub prompt: String,
    pub schema: Option<Value>,
    pub temperature: Option<f64>,
}

/// Response from completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub content: String,
    pub usage: Option<Usage>,
}

/// Usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// A dummy agent trait for the interface
pub trait Agent {
    fn name(&self) -> &str;
}

/// Core trait for backend engines - object-safe version
#[async_trait::async_trait]
pub trait Engine: Send + Sync + 'static {
    async fn create_agent(&self, config: AgentConfig) -> Result<Box<dyn Agent + Send>, Box<dyn StdError + Send + Sync>>;
    async fn complete(&self, request: crate::domain::completion::CompletionRequest) -> Result<CompletionResponse, Box<dyn StdError + Send + Sync>>;
    async fn extract_json(&self, config: ExtractionConfig) -> Result<Value, Box<dyn StdError + Send + Sync>>;
    async fn execute_tool(&self, tool_name: &str, args: Value) -> Result<Value, Box<dyn StdError + Send + Sync>>;
    async fn available_tools(&self) -> Result<Vec<String>, Box<dyn StdError + Send + Sync>>;
}

/// Engine registry for managing multiple engines
pub struct EngineRegistry {
    engines: RwLock<HashMap<String, Arc<dyn Engine>>>,
    default_engine: RwLock<Option<Arc<dyn Engine>>>,
}

impl EngineRegistry {
    pub fn new() -> Self {
        Self {
            engines: RwLock::new(HashMap::new()),
            default_engine: RwLock::new(None),
        }
    }

    /// Register a new engine with a given name
    pub fn register(&self, name: &str, engine: Arc<dyn Engine>) -> Result<(), Box<dyn StdError + Send + Sync>> {
        let mut engines = self.engines.write().map_err(|_| "Lock poisoned")?;
        engines.insert(name.to_string(), engine);
        Ok(())
    }

    /// Get an engine by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn Engine>> {
        let engines = self.engines.read().ok()?;
        engines.get(name).cloned()
    }

    /// Set the default engine
    pub fn set_default(&self, engine: Arc<dyn Engine>) -> Result<(), Box<dyn StdError + Send + Sync>> {
        let mut default = self.default_engine.write().map_err(|_| "Lock poisoned")?;
        *default = Some(engine);
        Ok(())
    }

    /// Get the default engine
    pub fn get_default(&self) -> Option<Arc<dyn Engine>> {
        let default = self.default_engine.read().ok()?;
        default.clone()
    }

    /// List all registered engine names
    pub fn list_engines(&self) -> Vec<String> {
        let engines = self.engines.read().unwrap_or_else(|_| panic!("Lock poisoned"));
        engines.keys().cloned().collect()
    }
}

// Global registry instance
static GLOBAL_REGISTRY: std::sync::LazyLock<EngineRegistry> = std::sync::LazyLock::new(|| {
    let registry = EngineRegistry::new();
    // Set no-op engine as default
    let _ = registry.set_default(Arc::new(NoOpEngine));
    registry
});

/// Get the global engine registry
pub fn registry() -> &'static EngineRegistry {
    &GLOBAL_REGISTRY
}

/// Convenience function to register an engine globally
pub fn register_engine(name: &str, engine: Arc<dyn Engine>) -> Result<(), Box<dyn StdError + Send + Sync>> {
    registry().register(name, engine)
}

/// Convenience function to set the global default engine
pub fn set_default_engine(engine: Arc<dyn Engine>) -> Result<(), Box<dyn StdError + Send + Sync>> {
    registry().set_default(engine)
}

/// Get the default engine or return an error
pub fn get_default_engine() -> Result<Arc<dyn Engine>, Box<dyn StdError + Send + Sync>> {
    registry().get_default()
        .ok_or_else(|| "No default engine set. Use set_default_engine() or register an engine first.".into())
}

/// Get a specific engine by name
pub fn get_engine(name: &str) -> Result<Arc<dyn Engine>, Box<dyn StdError + Send + Sync>> {
    registry().get(name)
        .ok_or_else(|| format!("Engine '{}' not found. Available engines: {:?}", name, registry().list_engines()).into())
}

/// No-op engine for testing and default behavior
pub struct NoOpEngine;

#[async_trait::async_trait]
impl Engine for NoOpEngine {
    async fn create_agent(&self, _config: AgentConfig) -> Result<Box<dyn Agent + Send>, Box<dyn StdError + Send + Sync>> {
        Err("No-op engine: agent creation not implemented".into())
    }

    async fn complete(&self, _request: crate::domain::completion::CompletionRequest) -> Result<CompletionResponse, Box<dyn StdError + Send + Sync>> {
        Err("No-op engine: completion not implemented".into())
    }

    async fn extract_json(&self, _config: ExtractionConfig) -> Result<Value, Box<dyn StdError + Send + Sync>> {
        Err("No-op engine: extraction not implemented".into())
    }

    async fn execute_tool(&self, _tool_name: &str, _args: Value) -> Result<Value, Box<dyn StdError + Send + Sync>> {
        Err("No-op engine: tool execution not implemented".into())
    }

    async fn available_tools(&self) -> Result<Vec<String>, Box<dyn StdError + Send + Sync>> {
        Ok(vec![])
    }
}
