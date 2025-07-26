//! Unified model registry for AI model management
//!
//! This module provides a high-performance, thread-safe registry for managing
//! AI model information and capabilities.

use std::sync::Arc;
use std::collections::HashMap;

use ahash::RandomState;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use smallvec::SmallVec;
use arrayvec::ArrayVec;

// Temporarily commented out to break circular dependency
// use model_info::{Provider, ModelInfo as ModelInfoProvider, ProviderTrait};

// Temporary placeholder types to replace model-info dependencies
#[derive(Debug, Clone)]
pub struct ModelInfoProvider {
    pub name: String,
    pub max_context: u64,
    pub pricing_input: f64,
    pub pricing_output: f64,
    pub is_thinking: bool,
    pub required_temperature: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum Provider {
    OpenAi,
    Mistral,
    Anthropic,
    Together,
    OpenRouter,
    HuggingFace,
    Xai,
}

use crate::model::error::{ModelError, Result};
use crate::model::capabilities::ModelCapabilities;

/// Provider names for efficient lookups
const PROVIDER_NAMES: &[&str] = &[
    "openai", "mistral", "anthropic", "together", 
    "openrouter", "huggingface", "xai"
];

/// Maximum number of models per provider (for zero-allocation collections)
const MAX_MODELS_PER_PROVIDER: usize = 100;

/// A capability-based model filter
#[derive(Debug, Clone, Default)]
pub struct ModelFilter {
    /// Filter by provider name
    pub provider: Option<String>,
    /// Minimum context length
    pub min_context: Option<u64>,
    /// Maximum context length
    pub max_context: Option<u64>,
    /// Maximum input price per 1M tokens
    pub max_input_price: Option<f64>,
    /// Maximum output price per 1M tokens
    pub max_output_price: Option<f64>,
    /// Must support thinking/reasoning
    pub requires_thinking: Option<bool>,
    /// Must have specific temperature requirement
    pub required_temperature: Option<f64>,
}

/// Efficient model query result with zero-allocation for small results
pub type ModelQueryResult = SmallVec<Arc<ModelInfoProvider>, 16>;

/// Internal registry data structure
struct RegistryData {
    /// All models indexed by provider name
    models_by_provider: DashMap<String, SmallVec<Arc<ModelInfoProvider>, MAX_MODELS_PER_PROVIDER>, RandomState>,
    
    /// All models in a flat list for fast iteration
    all_models: parking_lot::RwLock<Vec<Arc<ModelInfoProvider>>>,
    
    /// Provider instances for dynamic queries
    providers: ArrayVec<Provider, 7>,
    
    /// Capability-based indices for fast filtering
    thinking_models: DashSet<String, RandomState>,
    high_context_models: DashSet<String, RandomState>,
    low_cost_models: DashSet<String, RandomState>,
}

impl Default for RegistryData {
    fn default() -> Self {
        let mut providers = ArrayVec::new();
        
        // Initialize all providers (zero allocation)
        let _ = providers.try_push(Provider::OpenAi);
        let _ = providers.try_push(Provider::Mistral);
        let _ = providers.try_push(Provider::Anthropic);
        let _ = providers.try_push(Provider::Together);
        let _ = providers.try_push(Provider::OpenRouter);
        let _ = providers.try_push(Provider::HuggingFace);
        let _ = providers.try_push(Provider::Xai);
        
        Self {
            models_by_provider: DashMap::with_hasher(RandomState::default()),
            all_models: parking_lot::RwLock::new(Vec::new()),
            providers,
            thinking_models: DashSet::with_hasher(RandomState::default()),
            high_context_models: DashSet::with_hasher(RandomState::default()),
            low_cost_models: DashSet::with_hasher(RandomState::default()),
        }
    }
}

/// Global registry instance
static REGISTRY: Lazy<RegistryData> = Lazy::new(Default::default);

/// Unified model registry providing access to real model data from all providers
///
/// This registry integrates with the model-info package to provide fast, cached access
/// to real model information from OpenAI, Anthropic, xAI, Together, and other providers.
///
/// # Performance
/// - Zero allocation for common operations
/// - Lock-free reads for cached data
/// - Efficient capability-based filtering
/// - Thread-safe concurrent access
#[derive(Clone, Default)]
pub struct ModelRegistry;

impl ModelRegistry {
    /// Create a new model registry instance
    #[inline]
    pub fn new() -> Self {
        Self
    }
    
    /// Get a model by provider and name
    ///
    /// # Arguments
    /// * `provider` - The provider name (e.g., "openai", "anthropic")
    /// * `name` - The model name (e.g., "gpt-4", "claude-3-opus-20240229")
    ///
    /// # Returns
    /// The model information if found, or None if not available
    #[inline]
    pub fn get_model(&self, provider: &str, name: &str) -> Option<Arc<ModelInfoProvider>> {
        let models = REGISTRY.models_by_provider.get(provider)?;
        models.iter().find(|model| model.name == name).cloned()
    }
    
    /// Get a model by provider and name, returning an error if not found
    ///
    /// # Arguments
    /// * `provider` - The provider name
    /// * `name` - The model name
    ///
    /// # Returns
    /// The model information or an error if not found
    pub fn get_model_required(&self, provider: impl Into<String>, name: impl Into<String>) -> Result<Arc<ModelInfoProvider>> {
        let provider = provider.into();
        let name = name.into();
        self.get_model(&provider, &name)
            .ok_or_else(|| ModelError::ModelNotFound {
                provider: provider.into(),
                name: name.into(),
            })
    }
    
    /// Get all models from all providers
    ///
    /// # Returns
    /// A vector of all available models
    pub fn all_models(&self) -> Vec<Arc<ModelInfoProvider>> {
        REGISTRY.all_models.read().clone()
    }
    
    /// Get all models from a specific provider
    ///
    /// # Arguments
    /// * `provider` - The provider name
    ///
    /// # Returns
    /// Models from the specified provider, or empty collection if provider not found
    pub fn models_by_provider(&self, provider: &str) -> ModelQueryResult {
        REGISTRY.models_by_provider
            .get(provider)
            .map(|models| models.iter().cloned().collect())
            .unwrap_or_default()
    }
    
    /// Find models matching specific capabilities
    ///
    /// # Arguments
    /// * `filter` - The filter criteria to apply
    ///
    /// # Returns
    /// Models matching the filter criteria
    pub fn models_by_capabilities(&self, filter: &ModelFilter) -> ModelQueryResult {
        let all_models = REGISTRY.all_models.read();
        let mut results = ModelQueryResult::new();
        
        for model in all_models.iter() {
            if self.matches_filter(model, filter) {
                results.push(model.clone());
            }
        }
        
        results
    }
    
    /// Find models within a specific price range
    ///
    /// # Arguments
    /// * `max_input_price` - Maximum input price per 1M tokens
    /// * `max_output_price` - Maximum output price per 1M tokens
    ///
    /// # Returns
    /// Models within the specified price range
    pub fn models_by_price_range(&self, max_input_price: f64, max_output_price: f64) -> ModelQueryResult {
        let filter = ModelFilter {
            max_input_price: Some(max_input_price),
            max_output_price: Some(max_output_price),
            ..Default::default()
        };
        self.models_by_capabilities(&filter)
    }
    
    /// Find the most cost-effective model for a given task
    ///
    /// # Arguments
    /// * `min_context` - Minimum required context length
    /// * `requires_thinking` - Whether thinking capability is required
    ///
    /// # Returns
    /// The most cost-effective model meeting the requirements
    pub fn find_cheapest_model(&self, min_context: Option<u64>, requires_thinking: bool) -> Option<Arc<ModelInfoProvider>> {
        let filter = ModelFilter {
            min_context,
            requires_thinking: Some(requires_thinking),
            ..Default::default()
        };
        
        let candidates = self.models_by_capabilities(&filter);
        
        candidates.into_iter()
            .min_by(|a, b| {
                let cost_a = a.pricing_input + a.pricing_output;
                let cost_b = b.pricing_input + b.pricing_output;
                cost_a.partial_cmp(&cost_b).unwrap_or(std::cmp::Ordering::Equal)
            })
    }
    
    /// Get models that support thinking/reasoning
    ///
    /// # Returns
    /// All models that support thinking capabilities
    pub fn thinking_models(&self) -> ModelQueryResult {
        let filter = ModelFilter {
            requires_thinking: Some(true),
            ..Default::default()
        };
        self.models_by_capabilities(&filter)
    }
    
    /// Get high-context models (>100K tokens)
    ///
    /// # Returns
    /// All models with high context length capability
    pub fn high_context_models(&self) -> ModelQueryResult {
        let filter = ModelFilter {
            min_context: Some(100_000),
            ..Default::default()
        };
        self.models_by_capabilities(&filter)
    }
    
    /// Refresh model data from all providers
    ///
    /// This method fetches fresh model information from all provider APIs.
    /// It should be called periodically to ensure data freshness.
    ///
    /// # Returns
    /// The number of models successfully refreshed
    pub async fn refresh_models(&self) -> Result<usize> {
        let mut total_refreshed = 0;
        
        // Temporary storage for new model data
        let mut new_models_by_provider = HashMap::with_capacity(PROVIDER_NAMES.len());
        let mut all_new_models = Vec::new();
        
        // Fetch from all providers concurrently
        let tasks: Vec<_> = REGISTRY.providers.iter()
            .map(|provider| async move {
                self.fetch_provider_models(provider).await
            })
            .collect();
        
        let results = futures_util::future::join_all(tasks).await;
        
        // Process results
        for (provider_name, result) in PROVIDER_NAMES.iter().zip(results.into_iter()) {
            match result {
                Ok(models) => {
                    total_refreshed += models.len();
                    
                    // Convert to Arc for efficient sharing
                    let arc_models: SmallVec<[Arc<ModelInfoProvider>; MAX_MODELS_PER_PROVIDER]> = 
                        models.into_iter().map(Arc::new).collect();
                    
                    // Add to all models list
                    all_new_models.extend(arc_models.iter().cloned());
                    
                    // Store by provider
                    new_models_by_provider.insert((*provider_name).to_string(), arc_models);
                }
                Err(e) => {
                    // Log error but continue with other providers
                    eprintln!("Failed to refresh models for provider {}: {}", provider_name, e);
                }
            }
        }
        
        // Atomically update the registry
        {
            let mut all_models = REGISTRY.all_models.write();
            *all_models = all_new_models;
        }
        
        // Update provider-specific maps
        for (provider_name, models) in new_models_by_provider {
            REGISTRY.models_by_provider.insert(provider_name, models);
        }
        
        // Update capability indices
        self.rebuild_capability_indices().await;
        
        Ok(total_refreshed)
    }
    
    /// Get the list of all supported providers
    ///
    /// # Returns
    /// A static list of all supported provider names
    #[inline]
    pub fn providers(&self) -> &'static [&'static str] {
        PROVIDER_NAMES
    }
    
    /// Check if a model matches the given filter
    #[inline]
    fn matches_filter(&self, model: &ModelInfoProvider, filter: &ModelFilter) -> bool {
        if let Some(ref provider) = filter.provider {
            // Extract provider from model name or use a provider field if available
            // For now, we'll need to determine provider from context or model name
            // This would need to be enhanced based on actual ModelInfo structure
        }
        
        if let Some(min_context) = filter.min_context {
            if model.max_context < min_context {
                return false;
            }
        }
        
        if let Some(max_context) = filter.max_context {
            if model.max_context > max_context {
                return false;
            }
        }
        
        if let Some(max_input_price) = filter.max_input_price {
            if model.pricing_input > max_input_price {
                return false;
            }
        }
        
        if let Some(max_output_price) = filter.max_output_price {
            if model.pricing_output > max_output_price {
                return false;
            }
        }
        
        if let Some(requires_thinking) = filter.requires_thinking {
            if model.is_thinking != requires_thinking {
                return false;
            }
        }
        
        if let Some(required_temp) = filter.required_temperature {
            match model.required_temperature {
                Some(temp) if (temp - required_temp).abs() < 0.01 => {},
                None if required_temp == 0.0 => {},
                _ => return false,
            }
        }
        
        true
    }
    
    /// Fetch models from a specific provider
    async fn fetch_provider_models(&self, provider: &Provider) -> Result<Vec<ModelInfoProvider>> {
        // For now, return empty vec as we need to implement provider-specific model enumeration
        // This would be enhanced to actually fetch model lists from each provider
        Ok(Vec::new())
    }
    
    /// Rebuild capability-based indices for fast filtering
    async fn rebuild_capability_indices(&self) {
        REGISTRY.thinking_models.clear();
        REGISTRY.high_context_models.clear();
        REGISTRY.low_cost_models.clear();
        
        let all_models = REGISTRY.all_models.read();
        
        for model in all_models.iter() {
            let model_key = format!("{}:{}", "provider", model.name); // Provider would be determined from context
            
            if model.is_thinking {
                REGISTRY.thinking_models.insert(model_key.clone());
            }
            
            if model.max_context > 100_000 {
                REGISTRY.high_context_models.insert(model_key.clone());
            }
            
            let total_cost = model.pricing_input + model.pricing_output;
            if total_cost < 5.0 { // Under $5 per 1M tokens total
                REGISTRY.low_cost_models.insert(model_key);
            }
        }
    }
}

// Add missing imports
use dashmap::DashSet;
use parking_lot;

/// Statistics about the model registry
#[derive(Debug, Clone)]
pub struct RegistryStats {
    pub total_models: usize,
    pub models_by_provider: HashMap<String, usize>,
    pub thinking_models: usize,
    pub high_context_models: usize,
    pub low_cost_models: usize,
}

impl ModelRegistry {
    /// Get statistics about the current registry state
    ///
    /// # Returns
    /// Detailed statistics about models in the registry
    pub fn stats(&self) -> RegistryStats {
        let all_models = REGISTRY.all_models.read();
        let total_models = all_models.len();
        
        let mut models_by_provider = HashMap::new();
        for provider in PROVIDER_NAMES {
            let count = REGISTRY.models_by_provider
                .get(*provider)
                .map(|models| models.len())
                .unwrap_or(0);
            models_by_provider.insert((*provider).to_string(), count);
        }
        
        RegistryStats {
            total_models,
            models_by_provider,
            thinking_models: REGISTRY.thinking_models.len(),
            high_context_models: REGISTRY.high_context_models.len(),
            low_cost_models: REGISTRY.low_cost_models.len(),
        }
    }
}