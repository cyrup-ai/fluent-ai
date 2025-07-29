use model_info::{ModelInfo, OpenAi, Mistral, Anthropic, Together, OpenRouter, HuggingFace, Xai, common::Model};
use fluent_ai_async::{AsyncStream, emit};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Unified model registry supporting both static enum lookup and dynamic provider calls
/// ALL STREAMS architecture - no Result types, only unwrapped values and streaming patterns
pub struct UnifiedModelRegistry {
    static_cache: HashMap<String, ModelInfo>,
    provider_cache: HashMap<String, Vec<String>>,
}

impl UnifiedModelRegistry {
    /// Create new unified registry
    pub fn new() -> Self {
        let mut registry = Self {
            static_cache: HashMap::new(),
            provider_cache: HashMap::new(),
        };
        
        // Pre-populate with all static model enum data
        registry.register_static_models();
        registry
    }
    
    /// Get global registry instance
    pub fn global() -> &'static Self {
        static REGISTRY: OnceLock<UnifiedModelRegistry> = OnceLock::new();
        REGISTRY.get_or_init(|| Self::new())
    }
    
    /// Register all static model enum data
    fn register_static_models(&mut self) {
        // Register OpenAI models
        for model in OpenAi::all_models() {
            let info = <OpenAi as Model>::to_model_info(&model);
            self.static_cache.insert(model.name().to_string(), info);
            self.provider_cache
                .entry("openai".to_string())
                .or_insert_with(Vec::new)
                .push(model.name().to_string());
        }
        
        // Register Mistral models
        for model in Mistral::all_models() {
            let info = <Mistral as Model>::to_model_info(&model);
            self.static_cache.insert(model.name().to_string(), info);
            self.provider_cache
                .entry("mistral".to_string())
                .or_insert_with(Vec::new)
                .push(model.name().to_string());
        }
        
        // Register Anthropic models
        for model in Anthropic::all_models() {
            let info = <Anthropic as Model>::to_model_info(&model);
            self.static_cache.insert(model.name().to_string(), info);
            self.provider_cache
                .entry("anthropic".to_string())
                .or_insert_with(Vec::new)
                .push(model.name().to_string());
        }
        
        // Register Together models
        for model in Together::all_models() {
            let info = <Together as Model>::to_model_info(&model);
            self.static_cache.insert(model.name().to_string(), info);
            self.provider_cache
                .entry("together".to_string())
                .or_insert_with(Vec::new)
                .push(model.name().to_string());
        }
        
        // Register OpenRouter models
        for model in OpenRouter::all_models() {
            let info = <OpenRouter as Model>::to_model_info(&model);
            self.static_cache.insert(model.name().to_string(), info);
            self.provider_cache
                .entry("openrouter".to_string())
                .or_insert_with(Vec::new)
                .push(model.name().to_string());
        }
        
        // Register HuggingFace models
        for model in HuggingFace::all_models() {
            let info = <HuggingFace as Model>::to_model_info(&model);
            self.static_cache.insert(model.name().to_string(), info);
            self.provider_cache
                .entry("huggingface".to_string())
                .or_insert_with(Vec::new)
                .push(model.name().to_string());
        }
        
        // Register XAI models
        for model in Xai::all_models() {
            let info = <Xai as Model>::to_model_info(&model);
            self.static_cache.insert(model.name().to_string(), info);
            self.provider_cache
                .entry("xai".to_string())
                .or_insert_with(Vec::new)
                .push(model.name().to_string());
        }
    }
    
    /// Resolve model by name - returns default if not found (no Option)
    pub fn resolve_model(&self, name: &str) -> ModelInfo {
        // First try static cache
        if let Some(info) = self.static_cache.get(name) {
            return info.clone();
        }
        
        // TODO: Try dynamic provider lookup here
        // For now return default fallback
        ModelInfo {
            provider_name: "unknown",
            name: Box::leak(name.to_string().into_boxed_str()),
            max_input_tokens: Some(std::num::NonZeroU32::new(4096).unwrap()),
            max_output_tokens: Some(std::num::NonZeroU32::new(4096).unwrap()),
            input_price: Some(0.0),
            output_price: Some(0.0),
            supports_vision: false,
            supports_function_calling: false,
            supports_streaming: true,
            supports_embeddings: false,
            requires_max_tokens: false,
            supports_thinking: false,
            optimal_thinking_budget: None,
            system_prompt_prefix: None,
            real_name: None,
            model_type: None,
            patch: None,
            required_temperature: None,
        }
    }
    
    /// Stream all available model names
    pub fn stream_available_models(&self) -> AsyncStream<String> {
        let all_models: Vec<String> = self.static_cache.keys().cloned().collect();
        AsyncStream::with_channel(move |sender| {
            for model in all_models {
                emit!(sender, model);
            }
        })
    }
    
    /// Get all available models as Vec (non-streaming fallback)
    pub fn list_available_models(&self) -> Vec<String> {
        self.static_cache.keys().cloned().collect()
    }
    
    /// Stream models by provider
    pub fn stream_models_by_provider(&self, provider: &str) -> AsyncStream<String> {
        let models = self.provider_cache.get(provider).cloned().unwrap_or_default();
        AsyncStream::with_channel(move |sender| {
            for model in models {
                emit!(sender, model);
            }
        })
    }
    
    /// Get models by provider as Vec (non-streaming fallback)
    pub fn models_by_provider(&self, provider: &str) -> Vec<String> {
        self.provider_cache.get(provider).cloned().unwrap_or_default()
    }
    
    /// Stream all providers
    pub fn stream_providers(&self) -> AsyncStream<String> {
        let providers: Vec<String> = self.provider_cache.keys().cloned().collect();
        AsyncStream::with_channel(move |sender| {
            for provider in providers {
                emit!(sender, provider);
            }
        })
    }
    
    /// Get all providers as Vec (non-streaming fallback)
    pub fn providers(&self) -> Vec<String> {
        self.provider_cache.keys().cloned().collect()
    }
    
    /// Stream all model info
    pub fn stream_all_model_info(&self) -> AsyncStream<ModelInfo> {
        let all_info: Vec<ModelInfo> = self.static_cache.values().cloned().collect();
        AsyncStream::with_channel(move |sender| {
            for info in all_info {
                emit!(sender, info);
            }
        })
    }
    
    /// Get total number of registered models
    pub fn model_count(&self) -> usize {
        self.static_cache.len()
    }
    
    /// Get provider count
    pub fn provider_count(&self) -> usize {
        self.provider_cache.len()
    }
    
    /// Check if model exists
    pub fn has_model(&self, name: &str) -> bool {
        self.static_cache.contains_key(name)
    }
    
    /// Check if provider exists
    pub fn has_provider(&self, provider: &str) -> bool {
        self.provider_cache.contains_key(provider)
    }
}

impl Default for UnifiedModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Registry statistics for monitoring and debugging
#[derive(Debug, Clone)]
pub struct RegistryStats {
    pub total_models: usize,
    pub total_providers: usize,
    pub models_per_provider: HashMap<String, usize>,
}

impl UnifiedModelRegistry {
    /// Get registry statistics
    pub fn stats(&self) -> RegistryStats {
        let models_per_provider = self.provider_cache
            .iter()
            .map(|(provider, models)| (provider.clone(), models.len()))
            .collect();
            
        RegistryStats {
            total_models: self.model_count(),
            total_providers: self.provider_count(),
            models_per_provider,
        }
    }
    
    /// Stream registry statistics
    pub fn stream_stats(&self) -> AsyncStream<RegistryStats> {
        let stats = self.stats();
        AsyncStream::with_channel(move |sender| {
            emit!(sender, stats);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_registry_creation() {
        let registry = UnifiedModelRegistry::new();
        assert!(registry.model_count() > 0);
        assert!(registry.provider_count() > 0);
    }
    
    #[test]
    fn test_model_resolution() {
        let registry = UnifiedModelRegistry::new();
        
        // Test resolving existing model
        let gpt4_info = registry.resolve_model("gpt-4");
        assert_eq!(gpt4_info.name, "gpt-4");
        assert!(gpt4_info.max_context > 0);
        
        // Test resolving non-existing model (should return default)
        let unknown_info = registry.resolve_model("unknown-model");
        assert_eq!(unknown_info.name, "unknown-model");
        assert_eq!(unknown_info.max_context, 4096);
    }
    
    #[test]
    fn test_provider_operations() {
        let registry = UnifiedModelRegistry::new();
        
        assert!(registry.has_provider("openai"));
        assert!(registry.has_provider("anthropic"));
        assert!(!registry.has_provider("nonexistent"));
        
        let openai_models = registry.models_by_provider("openai");
        assert!(!openai_models.is_empty());
        
        let unknown_models = registry.models_by_provider("unknown");
        assert!(unknown_models.is_empty());
    }
    
    #[test]
    fn test_streaming_operations() {
        let registry = UnifiedModelRegistry::new();
        
        let all_models = registry.stream_available_models().collect();
        assert!(!all_models.is_empty());
        
        let providers = registry.stream_providers().collect();
        assert!(!providers.is_empty());
        
        let openai_models = registry.stream_models_by_provider("openai").collect();
        assert!(!openai_models.is_empty());
    }
    
    #[test]
    fn test_registry_stats() {
        let registry = UnifiedModelRegistry::new();
        let stats = registry.stats();
        
        assert!(stats.total_models > 0);
        assert!(stats.total_providers > 0);
        assert!(!stats.models_per_provider.is_empty());
        
        // Test streaming stats
        let streamed_stats = registry.stream_stats().collect();
        assert_eq!(streamed_stats.len(), 1);
        assert_eq!(streamed_stats[0].total_models, stats.total_models);
    }
    
    #[test]
    fn test_global_registry() {
        let registry1 = UnifiedModelRegistry::global();
        let registry2 = UnifiedModelRegistry::global();
        
        // Should be the same instance
        assert_eq!(registry1.model_count(), registry2.model_count());
        assert_eq!(registry1.provider_count(), registry2.provider_count());
    }
}