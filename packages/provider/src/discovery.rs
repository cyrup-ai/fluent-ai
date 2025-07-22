//! Provider model discovery and registration

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use futures_util::future::BoxFuture;
use once_cell::sync::Lazy;
use thiserror::Error;
use tracing::{debug, error, info, instrument, trace, warn};

use crate::model::{
    error::ModelError,
    info::ModelInfo,
    registry::{ModelRegistry, RegisteredModel},
    traits::Model,
};

/// Error type for model discovery operations
#[derive(Debug, Error)]
pub enum DiscoveryError {
    /// Failed to discover models from provider
    #[error("Failed to discover models from provider: {0}")]
    DiscoveryFailed(String),

    /// Provider not found
    #[error("Provider not found: {0}")]
    ProviderNotFound(String),

    /// Model registration failed
    #[error("Failed to register model: {0}")]
    RegistrationFailed(String),

    /// Wrapper for underlying errors
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

/// Result type for discovery operations
pub type DiscoveryResult<T> = std::result::Result<T, DiscoveryError>;

/// Trait for discovering and registering models from a provider
#[async_trait]
pub trait ProviderModelDiscovery: Send + Sync + 'static {
    /// Get the name of the provider
    fn provider_name(&self) -> &'static str;

    /// Discover and register all available models from this provider
    async fn discover_and_register(&self) -> DiscoveryResult<()>;

    /// Register a single model with the global registry
    async fn register_model<M>(&self, model: M) -> DiscoveryResult<RegisteredModel<M>>
    where
        M: Model + 'static,
    {
        let registry = ModelRegistry::global();
        registry
            .register(self.provider_name(), model)
            .map_err(|e| DiscoveryError::RegistrationFailed(e.to_string()))
    }

    /// Get the list of supported model names
    fn supported_models(&self) -> &'static [&'static str];

    /// Check if a model is supported by this provider
    fn supports_model(&self, model_name: &str) -> bool {
        self.supported_models()
            .iter()
            .any(|&name| name.eq_ignore_ascii_case(model_name))
    }
}

/// Global registry of model providers
static PROVIDER_REGISTRY: Lazy<DashMap<&'static str, Arc<dyn ProviderModelDiscovery>>> =
    Lazy::new(DashMap::new);

/// Register a model provider with the global registry
pub fn register_provider(provider: impl ProviderModelDiscovery) -> DiscoveryResult<()> {
    let provider_name = provider.provider_name();
    let provider = Arc::new(provider);

    PROVIDER_REGISTRY.entry(provider_name).or_insert(provider);
    info!("Registered model provider: {}", provider_name);

    Ok(())
}

/// Get a registered provider by name
pub fn get_provider(provider_name: &str) -> Option<Arc<dyn ProviderModelDiscovery>> {
    PROVIDER_REGISTRY
        .get(provider_name)
        .map(|p| p.value().clone())
}

/// Discover and register all models from all registered providers
pub async fn discover_all_models() -> DiscoveryResult<()> {
    let providers: Vec<_> = PROVIDER_REGISTRY
        .iter()
        .map(|p| p.value().clone())
        .collect();

    for provider in providers {
        let provider_name = provider.provider_name();
        info!("Discovering models for provider: {}", provider_name);

        if let Err(e) = provider.discover_and_register().await {
            error!(
                "Failed to discover models for provider {}: {}",
                provider_name, e
            );
            return Err(e);
        }
    }

    Ok(())
}

/// Initialize the default set of model providers
pub fn initialize_default_providers() -> DiscoveryResult<()> {
    // Register default providers here
    // Example:
    // register_provider(OpenAIDiscovery::new())?;
    // register_provider(AnthropicDiscovery::new())?;
    // register_provider(CandleDiscovery::new())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct TestProvider {
        name: &'static str,
        models: &'static [&'static str],
        discover_count: AtomicUsize,
    }

    #[async_trait]
    impl ProviderModelDiscovery for TestProvider {
        fn provider_name(&self) -> &'static str {
            self.name
        }

        async fn discover_and_register(&self) -> DiscoveryResult<()> {
            self.discover_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn supported_models(&self) -> &'static [&'static str] {
            self.models
        }
    }

    #[tokio::test]
    async fn test_provider_registration() {
        let provider = TestProvider {
            name: "test",
            models: &["model1", "model2"],
            discover_count: AtomicUsize::new(0),
        };

        // Register the provider
        register_provider(provider).expect("Failed to register test provider");

        // Retrieve the provider
        let provider = get_provider("test").expect("Failed to get test provider");
        assert_eq!(provider.provider_name(), "test");
        assert!(provider.supports_model("model1"));
        assert!(!provider.supports_model("unknown"));

        // Test discovery
        provider
            .discover_and_register()
            .await
            .expect("Failed to discover and register models");
        assert_eq!(
            provider
                .as_any()
                .downcast_ref::<TestProvider>()
                .expect("Failed to downcast to TestProvider")
                .discover_count
                .load(Ordering::SeqCst),
            1
        );
    }

    #[tokio::test]
    async fn test_discover_all_models() {
        let provider1 = TestProvider {
            name: "test1",
            models: &["model1"],
            discover_count: AtomicUsize::new(0),
        };

        let provider2 = TestProvider {
            name: "test2",
            models: &["model2"],
            discover_count: AtomicUsize::new(0),
        };

        // Register providers
        register_provider(provider1).expect("Failed to register test provider1");
        register_provider(provider2).expect("Failed to register test provider2");

        // Discover all models
        discover_all_models()
            .await
            .expect("Failed to discover all models");

        // Verify both providers were called
        let p1 = get_provider("test1").expect("Failed to get test1 provider");
        let p2 = get_provider("test2").expect("Failed to get test2 provider");

        assert_eq!(
            p1.as_any()
                .downcast_ref::<TestProvider>()
                .expect("Failed to downcast p1 to TestProvider")
                .discover_count
                .load(Ordering::SeqCst),
            1
        );

        assert_eq!(
            p2.as_any()
                .downcast_ref::<TestProvider>()
                .expect("Failed to downcast p2 to TestProvider")
                .discover_count
                .load(Ordering::SeqCst),
            1
        );
    }
}
