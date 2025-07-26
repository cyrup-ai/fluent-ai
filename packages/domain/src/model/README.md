# Model System Documentation

This document provides comprehensive documentation for the fluent-ai model system, which integrates with the model-info package to provide real-time access to AI model data from multiple providers.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start Guide](#quick-start-guide)
- [API Reference](#api-reference)
- [Performance Considerations](#performance-considerations)
- [Error Handling](#error-handling)
- [Provider Configuration](#provider-configuration)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Overview

The fluent-ai model system provides a unified interface for accessing real model information from AI providers including OpenAI, Anthropic, xAI, Mistral, Together AI, OpenRouter, and Hugging Face. The system features:

- **Real-time model data**: Auto-generated model enums from live provider APIs
- **High-performance caching**: Thread-safe, lock-free caching with TTL
- **Intelligent validation**: Provider health monitoring with circuit breakers
- **Zero-allocation queries**: Optimized for high-throughput applications
- **Comprehensive filtering**: Query models by capabilities, pricing, and features

## Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ UnifiedModel    │    │ ModelCache      │    │ ModelValidator  │
│ Registry        │◄──►│                 │◄──►│                 │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ model-info      │    │ DashMap Cache   │    │ Circuit         │
│ Providers       │    │ (Thread-safe)   │    │ Breakers        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Model Registry** aggregates model data from all providers
2. **Cache Layer** provides fast access with TTL-based invalidation
3. **Validation Layer** ensures model availability and API health
4. **Provider Integration** connects to real AI provider APIs

## Quick Start Guide

### Basic Usage

```rust
use fluent_ai_domain::model::{
    UnifiedModelRegistry, ModelCache, ModelValidator, ModelFilter
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create registry instance
    let registry = UnifiedModelRegistry::new();
    
    // Get all available models
    let all_models = registry.all_models();
    println!("Found {} models", all_models.len());
    
    // Get models from specific provider
    let openai_models = registry.models_by_provider("openai");
    println!("OpenAI has {} models", openai_models.len());
    
    // Find thinking models
    let thinking_models = registry.thinking_models();
    println!("Found {} thinking models", thinking_models.len());
    
    Ok(())
}
```

### With Caching

```rust
use fluent_ai_domain::model::{ModelCache, CacheConfig};
use std::time::Duration;

// Create cache with custom configuration
let config = CacheConfig {
    default_ttl: Duration::from_secs(600), // 10 minutes
    max_size: 5000,
    enable_cleanup: true,
    enable_warming: true,
    cleanup_interval: Duration::from_secs(120),
};

let cache = ModelCache::with_config(config);

// Start background tasks for cleanup and warming
cache.start_background_tasks();

// Use cache for model lookups
if let Some(model) = cache.get("openai", "gpt-4") {
    println!("Cache hit: {}", model.name);
} else {
    println!("Cache miss - would fetch from provider");
}
```

### With Validation

```rust
use fluent_ai_domain::model::ModelValidator;

let validator = ModelValidator::new();

// Validate a specific model
let result = validator.validate_model_exists("openai", "gpt-4").await?;
if result.is_valid() {
    println!("Model is available: {}", result.status_description());
} else {
    println!("Model validation failed: {:?}", result.error);
}

// Check provider health
let health_status = validator.provider_health_status().await;
for (provider, status) in health_status {
    println!("{}: {}", provider, status.status_description());
}
```

## API Reference

### UnifiedModelRegistry

The main registry for model discovery and querying.

#### Methods

- `new() -> Self` - Create a new registry instance
- `all_models() -> Vec<Arc<ModelInfo>>` - Get all available models
- `models_by_provider(provider: &str) -> ModelQueryResult` - Get models from specific provider
- `get_model(provider: &str, name: &str) -> Option<Arc<ModelInfo>>` - Get specific model
- `models_by_capabilities(filter: &ModelFilter) -> ModelQueryResult` - Filter by capabilities
- `thinking_models() -> ModelQueryResult` - Get models with reasoning capabilities
- `high_context_models() -> ModelQueryResult` - Get models with >100K context
- `find_cheapest_model(min_context: Option<u64>, requires_thinking: bool) -> Option<Arc<ModelInfo>>` - Find most cost-effective model
- `refresh_models() -> Result<usize>` - Refresh data from all providers
- `stats() -> RegistryStats` - Get registry statistics

#### Example

```rust
let registry = UnifiedModelRegistry::new();

// Filter for high-context, affordable models
let filter = ModelFilter {
    min_context: Some(100_000),
    max_input_price: Some(10.0),
    max_output_price: Some(30.0),
    ..Default::default()
};

let models = registry.models_by_capabilities(&filter);
println!("Found {} matching models", models.len());
```

### ModelCache

High-performance caching layer with TTL and LRU eviction.

#### Methods

- `new() -> Self` - Create cache with default settings
- `with_config(config: CacheConfig) -> Self` - Create cache with custom config
- `get(provider: &str, model_name: &str) -> Option<Arc<ModelInfo>>` - Get cached model
- `put(provider: &str, model_name: &str, model_info: Arc<ModelInfo>, ttl: Option<Duration>)` - Cache model
- `invalidate(provider: &str, model_name: &str) -> bool` - Remove from cache
- `clear()` - Clear entire cache
- `stats() -> CacheStats` - Get cache statistics
- `hit_ratio() -> f64` - Get cache hit ratio percentage
- `memory_usage() -> usize` - Estimate memory usage in bytes
- `contains(provider: &str, model_name: &str) -> bool` - Check if cached and valid

#### Performance Tips

```rust
// Pre-warm cache for frequently accessed models
let cache = ModelCache::new();
let models = vec!["gpt-4".to_string(), "claude-3-opus-20240229".to_string()];
cache.warm("openai", &models);

// Monitor cache performance
let stats = cache.stats();
println!("Hit ratio: {:.1}%", cache.hit_ratio());
println!("Memory usage: {} bytes", cache.memory_usage());
```

### ModelValidator

Validation and health checking for models and providers.

#### Methods

- `new() -> Self` - Create new validator
- `validate_model_exists(provider: &str, model_name: &str) -> Result<ValidationResult>` - Check model availability
- `validate_provider_access(provider: &str) -> Result<ValidationResult>` - Check provider connectivity
- `batch_validate_models(models: &[(&str, &str)]) -> Result<BatchValidationResult>` - Validate multiple models
- `provider_health_status() -> HashMap<String, ValidationResult>` - Get all provider health
- `circuit_breaker_status() -> HashMap<String, String>` - Get circuit breaker states
- `clear_cache()` - Clear validation cache

#### Validation Results

```rust
// Understand validation results
let result = validator.validate_model_exists("openai", "gpt-4").await?;

println!("Available: {}", result.is_available);
println!("API Key Valid: {}", result.api_key_valid);
println!("Connectivity: {}", result.connectivity_ok);
println!("Response Time: {:?}ms", result.response_time_ms);
println!("Status: {}", result.status_description());

if let Some(error) = result.error {
    println!("Error: {}", error);
}
```

### ModelFilter

Criteria for filtering models by capabilities and requirements.

#### Fields

- `provider: Option<String>` - Filter by provider name
- `min_context: Option<u64>` - Minimum context length
- `max_context: Option<u64>` - Maximum context length  
- `max_input_price: Option<f64>` - Maximum input price per 1M tokens
- `max_output_price: Option<f64>` - Maximum output price per 1M tokens
- `requires_thinking: Option<bool>` - Must support reasoning
- `required_temperature: Option<f64>` - Specific temperature requirement

#### Example

```rust
// Find affordable thinking models with high context
let filter = ModelFilter {
    min_context: Some(50_000),
    max_input_price: Some(5.0),
    requires_thinking: Some(true),
    ..Default::default()
};

let models = registry.models_by_capabilities(&filter);
```

## Performance Considerations

### Optimization Guidelines

1. **Use caching**: Enable ModelCache for repeated model lookups
2. **Batch operations**: Use batch validation instead of individual calls
3. **Filter efficiently**: Use specific filters to reduce result sets
4. **Monitor metrics**: Track cache hit ratios and response times
5. **Circuit breakers**: Respect circuit breaker states for failing providers

### Benchmarks

Run benchmarks to measure performance:

```bash
cd packages/domain
cargo bench --bench model_performance
```

Target performance metrics:
- Registry queries: <1ms
- Cache hits: <100μs  
- Validation: <500ms per model
- Batch validation: <2s for 20 models

### Memory Usage

The system is designed for minimal memory allocation:

- Zero-allocation for cache hits
- SmallVec for small result sets
- Arc sharing for model data
- Lock-free data structures

Monitor memory usage:

```rust
let cache = ModelCache::new();
println!("Cache memory: {} bytes", cache.memory_usage());

let registry = UnifiedModelRegistry::new();
let stats = registry.stats();
println!("Registry has {} models across {} providers", 
         stats.total_models, stats.models_by_provider.len());
```

## Error Handling

### Error Types

The system uses comprehensive error handling:

```rust
use fluent_ai_domain::model::{ModelError, Result};

match registry.get_model_required("openai", "nonexistent") {
    Ok(model) => println!("Found: {}", model.name),
    Err(ModelError::ModelNotFound { provider, name }) => {
        println!("Model {}/{} not found", provider, name);
    }
    Err(ModelError::InvalidConfiguration(msg)) => {
        println!("Configuration error: {}", msg);
    }
    Err(e) => println!("Other error: {}", e),
}
```

### Validation Error Handling

```rust
// Handle validation failures gracefully
let result = validator.validate_model_exists("provider", "model").await;

match result {
    Ok(validation) if validation.is_valid() => {
        println!("Model is ready to use");
    }
    Ok(validation) => {
        println!("Model validation failed: {}", validation.status_description());
        if let Some(error) = validation.error {
            println!("Details: {}", error);
        }
    }
    Err(e) => {
        println!("Validation error: {}", e);
    }
}
```

### Circuit Breaker Handling

```rust
// Check circuit breaker status before operations
let status = validator.circuit_breaker_status();
for (provider, state) in status {
    if state.contains("Open") {
        println!("Provider {} is unavailable: {}", provider, state);
    }
}
```

## Provider Configuration

### API Keys

Set environment variables for providers you want to use:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export XAI_API_KEY="xai-..."
export MISTRAL_API_KEY="..."
export TOGETHER_API_KEY="..."
export OPENROUTER_API_KEY="sk-or-..."
export HF_API_TOKEN="hf_..."
```

### Provider-Specific Notes

#### OpenAI
- Supports: GPT-4, GPT-3.5-turbo, DALL-E models
- Rate limits: Respect tier-based limits
- Models: Auto-detected from /v1/models endpoint

#### Anthropic  
- Supports: Claude 3 family (Opus, Sonnet, Haiku)
- Headers: Requires anthropic-version header
- Thinking: Claude models support reasoning

#### xAI
- Supports: Grok models with reasoning capabilities
- Context: Large context windows (128K-256K)
- Reasoning: All models support thinking mode

#### Mistral
- Supports: Mistral Large, Small, and specialized models
- Performance: Fast inference, competitive pricing
- Features: Function calling, JSON mode

#### Together AI
- Supports: Open source models (Llama, Mixtral, etc.)
- Pricing: Very competitive for open models
- Context: Variable by model

#### OpenRouter
- Supports: Aggregated access to multiple providers
- Models: Unified access to OpenAI, Anthropic, etc.
- Pricing: Pay-per-use with transparent costs

#### Hugging Face
- Supports: Open source models and inference API
- Free tier: Available for many models
- Variety: Largest selection of models

## Troubleshooting

### Common Issues

#### "Model not found" errors
```rust
// Check if model exists in registry
let models = registry.models_by_provider("openai");
for model in models {
    println!("Available: {}", model.name);
}

// Refresh model data
let refreshed = registry.refresh_models().await?;
println!("Refreshed {} models", refreshed);
```

#### API key validation failures
```rust
// Test provider access
let result = validator.validate_provider_access("openai").await?;
if !result.api_key_valid {
    println!("Check OPENAI_API_KEY environment variable");
}
```

#### Performance issues
```rust
// Check cache performance
let stats = cache.stats();
if stats.hits == 0 && stats.misses > 100 {
    println!("Cache not being used effectively");
}

if cache.hit_ratio() < 50.0 {
    println!("Consider increasing cache TTL or size");
}
```

#### Network connectivity
```rust
// Check provider health
let health = validator.provider_health_status().await;
for (provider, status) in health {
    if !status.connectivity_ok {
        println!("Connectivity issues with {}", provider);
    }
}
```

### Debugging

Enable debug logging:

```rust
// Set environment variable
std::env::set_var("RUST_LOG", "fluent_ai_domain::model=debug");

// Or use tracing in your application
use tracing::info;
info!("Registry stats: {:?}", registry.stats());
```

### Performance Debugging

```rust
// Monitor operation timing
use std::time::Instant;

let start = Instant::now();
let models = registry.thinking_models();
println!("Query took: {:?}", start.elapsed());

// Check for bottlenecks
let stats = cache.stats();
if stats.average_lookup_time_nanos > 1_000_000 {
    println!("Cache lookups are slow: {}ns avg", stats.average_lookup_time_nanos);
}
```

## Advanced Usage

### Custom Caching Strategies

```rust
use std::time::Duration;

// Long-lived cache for stable models
let stable_cache = ModelCache::with_config(CacheConfig {
    default_ttl: Duration::from_secs(3600), // 1 hour
    max_size: 1000,
    enable_cleanup: true,
    enable_warming: false,
    cleanup_interval: Duration::from_secs(600),
});

// Short-lived cache for rapidly changing data
let dynamic_cache = ModelCache::with_config(CacheConfig {
    default_ttl: Duration::from_secs(60), // 1 minute
    max_size: 100,
    enable_cleanup: true,
    enable_warming: true,
    cleanup_interval: Duration::from_secs(30),
});
```

### Async Background Processing

```rust
// Set up background tasks
let cache = ModelCache::new();
let validator = ModelValidator::new();

// Start maintenance tasks
cache.start_background_tasks();
validator.start_background_tasks();

// Periodic model refresh
tokio::spawn(async move {
    let mut interval = tokio::time::interval(Duration::from_secs(3600));
    loop {
        interval.tick().await;
        if let Ok(count) = registry.refresh_models().await {
            println!("Refreshed {} models", count);
        }
    }
});
```

### Integration with Provider Selection

```rust
// Select optimal provider based on requirements
async fn select_optimal_model(
    registry: &UnifiedModelRegistry,
    validator: &ModelValidator,
    min_context: u64,
    max_cost: f64
) -> Option<Arc<ModelInfo>> {
    // Find candidates
    let filter = ModelFilter {
        min_context: Some(min_context),
        max_input_price: Some(max_cost),
        max_output_price: Some(max_cost * 2.0),
        ..Default::default()
    };
    
    let candidates = registry.models_by_capabilities(&filter);
    
    // Validate and select best option
    for model in candidates {
        // Extract provider from model info
        let provider = "openai"; // This would be determined from model info
        let result = validator.validate_model_exists(provider, &model.name).await;
        
        if let Ok(validation) = result {
            if validation.is_valid() {
                return Some(model);
            }
        }
    }
    
    None
}
```

### Metrics and Monitoring

```rust
// Comprehensive metrics collection
struct ModelMetrics {
    registry_stats: RegistryStats,
    cache_stats: CacheStats,
    validation_results: HashMap<String, ValidationResult>,
}

impl ModelMetrics {
    async fn collect(
        registry: &UnifiedModelRegistry,
        cache: &ModelCache,
        validator: &ModelValidator,
    ) -> Self {
        Self {
            registry_stats: registry.stats(),
            cache_stats: cache.stats(),
            validation_results: validator.provider_health_status().await,
        }
    }
    
    fn report(&self) {
        println!("=== Model System Metrics ===");
        println!("Total models: {}", self.registry_stats.total_models);
        println!("Cache hit ratio: {:.1}%", 
                 self.cache_stats.hits as f64 / 
                 (self.cache_stats.hits + self.cache_stats.misses) as f64 * 100.0);
        
        for (provider, result) in &self.validation_results {
            println!("{}: {}", provider, result.status_description());
        }
    }
}
```

This documentation provides a comprehensive guide to using the fluent-ai model system effectively. For additional examples and use cases, see the `examples/` directory.