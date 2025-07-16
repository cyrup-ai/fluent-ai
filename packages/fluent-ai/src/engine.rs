use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use crate::async_task::{AsyncTask, spawn_async};
use crate::domain::completion::CompletionRequest;
use crate::ZeroOneOrMany;
use fluent_ai_provider::Models;

// Typesafe builder module
pub mod builder;
// Concrete engine implementations
pub mod fluent_engine;

// Re-export builder items for convenience
pub use builder::{EngineBuilder, EngineConfig, engine_builder, states};
// Re-export concrete engines
pub use fluent_engine::FluentEngine;

/// Configuration for creating an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub model: Models,
    pub system_prompt: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub tools: ZeroOneOrMany<String>,
}

/// Configuration for extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    pub model: Models,
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
pub trait Agent: Send + Sync {
    fn model(&self) -> &Models;
}

/// A no-op agent implementation
pub struct NoOpAgent {
    model: Models,
}

impl NoOpAgent {
    pub fn new(config: AgentConfig) -> Self {
        Self {
            model: config.model,
        }
    }
}

impl Agent for NoOpAgent {
    fn model(&self) -> &Models {
        &self.model
    }
}

/// Core trait for backend engines - using AsyncTask for NotResult constraint
pub trait Engine: Send + Sync + 'static {
    fn create_agent(
        &self,
        config: AgentConfig,
    ) -> AsyncTask<Box<dyn Agent + Send>>
    where
        Box<dyn Agent + Send>: crate::async_task::NotResult;

    fn complete(
        &self,
        request: CompletionRequest,
    ) -> AsyncTask<CompletionResponse>
    where
        CompletionResponse: crate::async_task::NotResult;

    fn extract_json(
        &self,
        config: ExtractionConfig,
    ) -> AsyncTask<Value>
    where
        Value: crate::async_task::NotResult;

    fn execute_tool(
        &self,
        tool_name: &str,
        args: Value,
    ) -> AsyncTask<Value>
    where
        Value: crate::async_task::NotResult;

    fn available_tools(
        &self,
    ) -> AsyncTask<ZeroOneOrMany<String>>
    where
        ZeroOneOrMany<String>: crate::async_task::NotResult;
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
    pub fn register(
        &self,
        name: &str,
        engine: Arc<dyn Engine>,
    ) -> bool {
        match self.engines.write() {
            Ok(mut engines) => {
                engines.insert(name.to_string(), engine);
                true
            },
            Err(_) => false, // Lock poisoned - return failure instead of panic
        }
    }

    /// Get an engine by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn Engine>> {
        let engines = self.engines.read().ok()?;
        engines.get(name).cloned()
    }

    /// Set the default engine
    pub fn set_default(
        &self,
        engine: Arc<dyn Engine>,
    ) -> bool {
        match self.default_engine.write() {
            Ok(mut default) => {
                *default = Some(engine);
                true
            },
            Err(_) => false, // Lock poisoned - return failure instead of panic
        }
    }

    /// Get the default engine
    pub fn get_default(&self) -> Option<Arc<dyn Engine>> {
        let default = self.default_engine.read().ok()?;
        default.clone()
    }

    /// List all registered engine names
    pub fn list_engines(&self) -> ZeroOneOrMany<String> {
        match self.engines.read() {
            Ok(engines) => {
                let names: Vec<String> = engines.keys().cloned().collect();
                ZeroOneOrMany::from_vec(names)
            },
            Err(_) => ZeroOneOrMany::None, // Lock poisoned - return empty instead of panic
        }
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
pub fn register_engine(
    name: &str,
    engine: Arc<dyn Engine>,
) -> bool {
    registry().register(name, engine)
}

/// Convenience function to set the global default engine
pub fn set_default_engine(engine: Arc<dyn Engine>) -> bool {
    registry().set_default(engine)
}

/// Get the default engine
pub fn get_default_engine() -> Option<Arc<dyn Engine>> {
    registry().get_default()
}

/// Get a specific engine by name
pub fn get_engine(name: &str) -> Option<Arc<dyn Engine>> {
    registry().get(name)
}

/// No-op engine for testing and default behavior
pub struct NoOpEngine;

impl Engine for NoOpEngine {
    fn create_agent(
        &self,
        config: AgentConfig,
    ) -> AsyncTask<Box<dyn Agent + Send>>
    where
        Box<dyn Agent + Send>: crate::async_task::NotResult,
    {
        spawn_async(async move { Box::new(NoOpAgent::new(config)) as Box<dyn Agent + Send> })
    }

    fn complete(
        &self,
        _request: CompletionRequest,
    ) -> AsyncTask<CompletionResponse>
    where
        CompletionResponse: crate::async_task::NotResult,
    {
        spawn_async(async move {
            CompletionResponse {
                content: "No-op response".to_string(),
                usage: None,
            }
        })
    }

    fn extract_json(
        &self,
        _config: ExtractionConfig,
    ) -> AsyncTask<Value>
    where
        Value: crate::async_task::NotResult,
    {
        spawn_async(async move { Value::Object(serde_json::Map::new()) })
    }

    fn execute_tool(
        &self,
        _tool_name: &str,
        _args: Value,
    ) -> AsyncTask<Value>
    where
        Value: crate::async_task::NotResult,
    {
        spawn_async(async move { Value::Null })
    }

    fn available_tools(
        &self,
    ) -> AsyncTask<ZeroOneOrMany<String>>
    where
        ZeroOneOrMany<String>: crate::async_task::NotResult,
    {
        spawn_async(async move { ZeroOneOrMany::None })
    }
}
