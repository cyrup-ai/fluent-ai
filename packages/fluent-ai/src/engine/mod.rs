pub mod builder;
pub mod fluent_engine;

use std::collections::HashMap;
use std::sync::Arc;

pub use builder::*;
pub use fluent_engine::*;

use crate::ZeroOneOrMany;
use crate::domain::agent::Agent as DomainAgent;

/// Engine trait for AI operations
pub trait Engine: Send + Sync {
    /// Get engine name
    fn name(&self) -> &str;
    /// Execute completion
    fn complete(
        &self,
        request: &crate::domain::completion::CompletionRequest,
    ) -> crate::runtime::AsyncTask<Result<CompletionResponse, crate::completion::CompletionError>>;
}

/// Agent configuration for engine operations

/// Completion response from engine
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub text: String,
    pub usage: Option<Usage>,
    pub model: String,
}

/// Usage statistics for completion
#[derive(Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Extraction configuration for structured data
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    pub schema: serde_json::Value,
    pub model: String,
    pub temperature: f64,
}

/// Agent trait for engine operations
pub trait Agent: Send + Sync {
    /// Get the model associated with this agent
    fn model(&self) -> &fluent_ai_provider::Models;
}

/// Global engine registry
static mut REGISTRY: Option<Arc<EngineRegistry>> = None;
static mut DEFAULT_ENGINE: Option<String> = None;

/// Engine registry for managing multiple engines
pub struct EngineRegistry {
    engines: HashMap<String, Arc<dyn Engine>>,
}

impl EngineRegistry {
    pub fn new() -> Self {
        Self {
            engines: HashMap::new(),
        }
    }

    pub fn register<E: Engine + 'static>(&mut self, name: String, engine: E) {
        self.engines.insert(name, Arc::new(engine));
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Engine>> {
        self.engines.get(name).cloned()
    }

    pub fn engines(&self) -> impl Iterator<Item = (&String, &Arc<dyn Engine>)> {
        self.engines.iter()
    }
}

/// Register an engine with the global registry
pub fn register_engine<E: Engine + 'static>(name: String, engine: E) {
    unsafe {
        if REGISTRY.is_none() {
            REGISTRY = Some(Arc::new(EngineRegistry::new()));
        }
        if let Some(registry) = &mut REGISTRY {
            match Arc::get_mut(registry) {
                Some(registry_mut) => registry_mut.register(name, engine),
                None => {
                    // Registry is shared, create new one with the engine
                    let mut new_registry = EngineRegistry::new();
                    new_registry.register(name, engine);
                    REGISTRY = Some(Arc::new(new_registry));
                }
            }
        }
    }
}

/// Get the global engine registry
pub fn registry() -> Arc<EngineRegistry> {
    unsafe {
        if REGISTRY.is_none() {
            REGISTRY = Some(Arc::new(EngineRegistry::new()));
        }
        match REGISTRY.as_ref() {
            Some(registry) => registry.clone(),
            None => {
                // Initialize with empty registry
                let registry = Arc::new(EngineRegistry::new());
                REGISTRY = Some(registry.clone());
                registry
            }
        }
    }
}

/// Get an engine by name
pub fn get_engine(name: &str) -> Option<Arc<dyn Engine>> {
    registry().get(name)
}

/// Set the default engine
pub fn set_default_engine(name: String) {
    unsafe {
        DEFAULT_ENGINE = Some(name);
    }
}

/// Get the default engine
pub fn get_default_engine() -> Option<Arc<dyn Engine>> {
    unsafe {
        if let Some(default_name) = &DEFAULT_ENGINE {
            get_engine(default_name)
        } else {
            None
        }
    }
}
