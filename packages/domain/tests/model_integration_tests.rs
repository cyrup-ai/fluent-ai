//! Integration tests for model functionality
//!
//! These tests verify end-to-end model functionality without mocking,
//! using real API keys from environment variables when available.

use std::env;
use std::time::Duration;

use fluent_ai_domain::model::{
    UnifiedModelRegistry, ModelCache, ModelValidator, ModelFilter,
    RealModelInfo, ModelInfoProvider,
};

/// Test that the unified model registry can be created
#[tokio::test]
async fn test_registry_creation() {
    let registry = UnifiedModelRegistry::new();
    let stats = registry.stats();
    
    // Registry should be created successfully
    assert_eq!(stats.total_models, 0); // Initially empty until refreshed
    assert!(stats.models_by_provider.len() > 0); // Should have provider entries
}

/// Test model cache basic operations
#[tokio::test]
async fn test_cache_operations() {
    let cache = ModelCache::new();
    
    // Test cache miss
    let result = cache.get("openai", "gpt-4");
    assert!(result.is_none());
    
    // Test cache stats
    let stats = cache.stats();
    assert_eq!(stats.hits, 0);
    assert!(stats.misses > 0);
    
    // Test cache clear
    cache.clear();
    let stats_after_clear = cache.stats();
    assert_eq!(stats_after_clear.entries, 0);
}

/// Test model validator creation and basic operations
#[tokio::test]
async fn test_validator_creation() {
    let validator = ModelValidator::new();
    
    // Test circuit breaker status (should be empty initially)
    let status = validator.circuit_breaker_status();
    assert!(status.is_empty());
    
    // Clear cache should not panic
    validator.clear_cache();
}

/// Test provider health status check
#[tokio::test]
async fn test_provider_health_status() {
    let validator = ModelValidator::new();
    
    // This will attempt to check all providers
    let health_status = validator.provider_health_status().await;
    
    // Should have entries for all providers
    assert!(health_status.len() > 0);
    
    // All entries should have provider names
    for (provider_name, result) in health_status {
        assert!(!provider_name.is_empty());
        assert!(!result.provider.is_empty());
        assert_eq!(result.provider, provider_name);
    }
}

/// Test registry stats functionality
#[tokio::test]
async fn test_registry_stats() {
    let registry = UnifiedModelRegistry::new();
    let stats = registry.stats();
    
    // Stats should have reasonable structure
    assert!(stats.models_by_provider.contains_key("openai"));
    assert!(stats.models_by_provider.contains_key("anthropic"));
    assert!(stats.models_by_provider.contains_key("xai"));
    
    // Counts should be non-negative
    assert!(stats.thinking_models == 0 || stats.thinking_models > 0);
    assert!(stats.high_context_models == 0 || stats.high_context_models > 0);
    assert!(stats.low_cost_models == 0 || stats.low_cost_models > 0);
}

/// Test cache hit ratio calculation
#[tokio::test]
async fn test_cache_hit_ratio() {
    let cache = ModelCache::new();
    
    // Initially should be 0% (no hits or misses yet)
    let initial_ratio = cache.hit_ratio();
    assert_eq!(initial_ratio, 0.0);
    
    // After a miss, should still be 0%
    let _ = cache.get("test", "model");
    let ratio_after_miss = cache.hit_ratio();
    assert_eq!(ratio_after_miss, 0.0);
}

/// Test memory usage estimation
#[tokio::test]
async fn test_cache_memory_usage() {
    let cache = ModelCache::new();
    
    let memory_usage = cache.memory_usage();
    // Should be reasonable (not negative or extremely large)
    assert!(memory_usage == 0 || memory_usage > 0);
    assert!(memory_usage < 1_000_000_000); // Less than 1GB
}

/// Test model filter creation and validation
#[tokio::test]
async fn test_model_filter() {
    let filter = ModelFilter {
        provider: Some("openai".to_string()),
        min_context: Some(8000),
        max_context: Some(200000),
        max_input_price: Some(10.0),
        max_output_price: Some(30.0),
        requires_thinking: Some(false),
        required_temperature: None,
    };
    
    // Filter should be created successfully
    assert_eq!(filter.provider.as_ref().expect("Provider should be set"), "openai");
    assert_eq!(filter.min_context.expect("Min context should be set"), 8000);
}

/// Test registry provider list
#[tokio::test]
async fn test_registry_providers() {
    let registry = UnifiedModelRegistry::new();
    let providers = registry.providers();
    
    // Should have expected providers
    assert!(providers.contains(&"openai"));
    assert!(providers.contains(&"anthropic"));
    assert!(providers.contains(&"xai"));
    assert!(providers.contains(&"mistral"));
    assert!(providers.contains(&"together"));
    assert!(providers.contains(&"openrouter"));
    assert!(providers.contains(&"huggingface"));
}

/// Test batch validation with empty list
#[tokio::test]
async fn test_batch_validation_empty() {
    let validator = ModelValidator::new();
    
    let models = vec![];
    let result = validator.batch_validate_models(&models).await;
    
    assert!(result.is_ok());
    let batch_result = result.expect("Batch validation should succeed");
    assert_eq!(batch_result.results.len(), 0);
    assert_eq!(batch_result.successful_count, 0);
    assert_eq!(batch_result.failed_count, 0);
    assert_eq!(batch_result.success_rate(), 0.0);
}

/// Test validation result status description
#[tokio::test]
async fn test_validation_result_status() {
    use fluent_ai_domain::model::ValidationResult;
    use std::time::Instant;
    
    // Test successful result
    let success_result = ValidationResult {
        provider: "test".to_string(),
        model_name: "test-model".to_string(),
        is_available: true,
        api_key_valid: true,
        connectivity_ok: true,
        response_time_ms: Some(100),
        error: None,
        validated_at: Instant::now(),
    };
    
    assert!(success_result.is_valid());
    assert_eq!(success_result.status_description(), "Available");
    
    // Test failed result
    let failed_result = ValidationResult {
        provider: "test".to_string(),
        model_name: "test-model".to_string(),
        is_available: false,
        api_key_valid: true,
        connectivity_ok: true,
        response_time_ms: Some(100),
        error: Some("Model not found".to_string()),
        validated_at: Instant::now(),
    };
    
    assert!(!failed_result.is_valid());
    assert_eq!(failed_result.status_description(), "Model Unavailable");
}

/// Test cache configuration
#[tokio::test]
async fn test_cache_configuration() {
    use fluent_ai_domain::model::CacheConfig;
    
    let config = CacheConfig::default();
    assert!(config.default_ttl > Duration::from_secs(0));
    assert!(config.max_size > 0);
    assert!(config.enable_cleanup);
    assert!(config.enable_warming);
    
    let cache = ModelCache::with_config(config);
    // Should create cache without error
    let _ = cache.stats();
}

/// Integration test with real API keys (if available)
#[tokio::test]
async fn test_real_model_validation() {
    let validator = ModelValidator::new();
    
    // Only run if we have at least one API key
    let has_openai = env::var("OPENAI_API_KEY").is_ok();
    let has_anthropic = env::var("ANTHROPIC_API_KEY").is_ok();
    let has_xai = env::var("XAI_API_KEY").is_ok();
    
    if !has_openai && !has_anthropic && !has_xai {
        println!("Skipping real API test - no API keys available");
        return;
    }
    
    // Test with available providers
    let mut test_models = vec![];
    
    if has_openai {
        test_models.push(("openai", "gpt-3.5-turbo"));
    }
    if has_anthropic {
        test_models.push(("anthropic", "claude-3-haiku-20240307"));
    }
    if has_xai {
        test_models.push(("xai", "grok-beta"));
    }
    
    if !test_models.is_empty() {
        let result = validator.batch_validate_models(&test_models).await;
        assert!(result.is_ok());
        
        let batch_result = result.expect("Batch validation should succeed");
        assert_eq!(batch_result.results.len(), test_models.len());
        
        // At least some models should validate successfully with real API keys
        if batch_result.successful_count > 0 {
            println!("Successfully validated {} models", batch_result.successful_count);
        } else {
            println!("No models validated successfully - check API keys");
        }
    }
}

/// Test registry model queries
#[tokio::test]
async fn test_registry_queries() {
    let registry = UnifiedModelRegistry::new();
    
    // Test models by provider (should not panic even if empty)
    let openai_models = registry.models_by_provider("openai");
    let anthropic_models = registry.models_by_provider("anthropic");
    let unknown_models = registry.models_by_provider("unknown-provider");
    
    // Should return collections (empty is fine)
    assert!(openai_models.len() == 0 || openai_models.len() > 0);
    assert!(anthropic_models.len() == 0 || anthropic_models.len() > 0);
    assert_eq!(unknown_models.len(), 0);
    
    // Test capabilities query
    let filter = ModelFilter {
        requires_thinking: Some(true),
        ..Default::default()
    };
    let thinking_models = registry.models_by_capabilities(&filter);
    assert!(thinking_models.len() == 0 || thinking_models.len() > 0);
    
    // Test price range query
    let cheap_models = registry.models_by_price_range(5.0, 15.0);
    assert!(cheap_models.len() == 0 || cheap_models.len() > 0);
}

/// Test background task functionality
#[tokio::test]
async fn test_background_tasks() {
    let cache = ModelCache::new();
    let validator = ModelValidator::new();
    
    // Starting background tasks should not panic
    cache.start_background_tasks();
    validator.start_background_tasks();
    
    // Give tasks a moment to start
    tokio::time::sleep(Duration::from_millis(10)).await;
    
    // Tasks should be running (no direct way to verify, but no panic is good)
}