//! Performance benchmarks for model operations
//!
//! This benchmark suite measures the performance of model registry operations,
//! cache performance, validation speed, and provider enumeration with and without caching.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;
use tokio::runtime::Runtime;

use fluent_ai_domain::model::{
    UnifiedModelRegistry, ModelCache, ModelValidator, ModelFilter,
    RealModelInfo, CacheConfig,
};

/// Setup function for benchmarks
fn setup_runtime() -> Runtime {
    Runtime::new().expect("Failed to create tokio runtime")
}

/// Setup test data for benchmarks
fn setup_test_data() -> Vec<(&'static str, &'static str)> {
    vec![
        ("openai", "gpt-4"),
        ("openai", "gpt-3.5-turbo"),
        ("anthropic", "claude-3-opus-20240229"),
        ("anthropic", "claude-3-sonnet-20240229"),
        ("anthropic", "claude-3-haiku-20240307"),
        ("xai", "grok-beta"),
        ("xai", "grok-2"),
        ("mistral", "mistral-large"),
        ("mistral", "mistral-small"),
        ("together", "meta-llama/Llama-2-7b-chat-hf"),
        ("together", "meta-llama/Llama-2-13b-chat-hf"),
        ("openrouter", "openai/gpt-4"),
        ("openrouter", "anthropic/claude-3-opus"),
        ("huggingface", "microsoft/DialoGPT-medium"),
        ("huggingface", "facebook/blenderbot-400M-distill"),
    ]
}

/// Benchmark registry creation and basic operations
fn bench_registry_operations(c: &mut Criterion) {
    let rt = setup_runtime();
    
    let mut group = c.benchmark_group("registry_operations");
    group.measurement_time(Duration::from_secs(10));
    
    // Benchmark registry creation
    group.bench_function("registry_creation", |b| {
        b.iter(|| {
            let registry = black_box(UnifiedModelRegistry::new());
            black_box(registry)
        })
    });
    
    // Benchmark stats collection
    group.bench_function("stats_collection", |b| {
        let registry = UnifiedModelRegistry::new();
        b.iter(|| {
            let stats = black_box(registry.stats());
            black_box(stats)
        })
    });
    
    // Benchmark provider enumeration
    group.bench_function("provider_enumeration", |b| {
        let registry = UnifiedModelRegistry::new();
        b.iter(|| {
            let providers = black_box(registry.providers());
            black_box(providers)
        })
    });
    
    // Benchmark models by provider (empty registry)
    group.bench_function("models_by_provider_empty", |b| {
        let registry = UnifiedModelRegistry::new();
        b.iter(|| {
            let models = black_box(registry.models_by_provider("openai"));
            black_box(models)
        })
    });
    
    group.finish();
}

/// Benchmark cache operations
fn bench_cache_operations(c: &mut Criterion) {
    let rt = setup_runtime();
    
    let mut group = c.benchmark_group("cache_operations");
    group.measurement_time(Duration::from_secs(10));
    
    // Benchmark cache creation
    group.bench_function("cache_creation", |b| {
        b.iter(|| {
            let cache = black_box(ModelCache::new());
            black_box(cache)
        })
    });
    
    // Benchmark cache miss (get operation on empty cache)
    group.bench_function("cache_miss", |b| {
        let cache = ModelCache::new();
        b.iter(|| {
            let result = black_box(cache.get("openai", "gpt-4"));
            black_box(result)
        })
    });
    
    // Benchmark cache stats
    group.bench_function("cache_stats", |b| {
        let cache = ModelCache::new();
        // Perform some operations to generate stats
        for _ in 0..10 {
            cache.get("test", "model");
        }
        
        b.iter(|| {
            let stats = black_box(cache.stats());
            black_box(stats)
        })
    });
    
    // Benchmark cache hit ratio calculation
    group.bench_function("cache_hit_ratio", |b| {
        let cache = ModelCache::new();
        // Generate some cache activity
        for i in 0..100 {
            cache.get("provider", &format!("model-{}", i));
        }
        
        b.iter(|| {
            let ratio = black_box(cache.hit_ratio());
            black_box(ratio)
        })
    });
    
    // Benchmark memory usage calculation
    group.bench_function("cache_memory_usage", |b| {
        let cache = ModelCache::new();
        b.iter(|| {
            let usage = black_box(cache.memory_usage());
            black_box(usage)
        })
    });
    
    // Benchmark cache contains check
    group.bench_function("cache_contains", |b| {
        let cache = ModelCache::new();
        b.iter(|| {
            let contains = black_box(cache.contains("openai", "gpt-4"));
            black_box(contains)
        })
    });
    
    // Benchmark cache invalidation
    group.bench_function("cache_invalidation", |b| {
        let cache = ModelCache::new();
        b.iter(|| {
            let invalidated = black_box(cache.invalidate("test", "model"));
            black_box(invalidated)
        })
    });
    
    // Benchmark cache clear
    group.bench_function("cache_clear", |b| {
        let cache = ModelCache::new();
        b.iter(|| {
            black_box(cache.clear());
        })
    });
    
    group.finish();
}

/// Benchmark validation operations
fn bench_validation_operations(c: &mut Criterion) {
    let rt = setup_runtime();
    
    let mut group = c.benchmark_group("validation_operations");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20); // Fewer samples for network operations
    
    // Benchmark validator creation
    group.bench_function("validator_creation", |b| {
        b.iter(|| {
            let validator = black_box(ModelValidator::new());
            black_box(validator)
        })
    });
    
    // Benchmark circuit breaker status
    group.bench_function("circuit_breaker_status", |b| {
        let validator = ModelValidator::new();
        b.iter(|| {
            let status = black_box(validator.circuit_breaker_status());
            black_box(status)
        })
    });
    
    // Benchmark cache clearing
    group.bench_function("validation_cache_clear", |b| {
        let validator = ModelValidator::new();
        b.iter(|| {
            black_box(validator.clear_cache());
        })
    });
    
    // Benchmark provider health status (async operation)
    group.bench_function("provider_health_status", |b| {
        let validator = ModelValidator::new();
        b.to_async(&rt).iter(|| async {
            let status = black_box(validator.provider_health_status().await);
            black_box(status)
        })
    });
    
    // Benchmark empty batch validation
    group.bench_function("batch_validation_empty", |b| {
        let validator = ModelValidator::new();
        let empty_models = vec![];
        
        b.to_async(&rt).iter(|| async {
            let result = black_box(validator.batch_validate_models(&empty_models).await);
            black_box(result)
        })
    });
    
    group.finish();
}

/// Benchmark model filtering operations
fn bench_filtering_operations(c: &mut Criterion) {
    let rt = setup_runtime();
    
    let mut group = c.benchmark_group("filtering_operations");
    group.measurement_time(Duration::from_secs(10));
    
    // Benchmark filter creation
    group.bench_function("filter_creation", |b| {
        b.iter(|| {
            let filter = black_box(ModelFilter {
                provider: Some("openai".to_string()),
                min_context: Some(8000),
                max_context: Some(200000),
                max_input_price: Some(10.0),
                max_output_price: Some(30.0),
                requires_thinking: Some(true),
                required_temperature: None,
            });
            black_box(filter)
        })
    });
    
    // Benchmark default filter creation
    group.bench_function("filter_default", |b| {
        b.iter(|| {
            let filter = black_box(ModelFilter::default());
            black_box(filter)
        })
    });
    
    // Benchmark capabilities filtering (empty registry)
    group.bench_function("models_by_capabilities_empty", |b| {
        let registry = UnifiedModelRegistry::new();
        let filter = ModelFilter {
            requires_thinking: Some(true),
            ..Default::default()
        };
        
        b.iter(|| {
            let models = black_box(registry.models_by_capabilities(&filter));
            black_box(models)
        })
    });
    
    // Benchmark price range filtering
    group.bench_function("models_by_price_range", |b| {
        let registry = UnifiedModelRegistry::new();
        
        b.iter(|| {
            let models = black_box(registry.models_by_price_range(5.0, 15.0));
            black_box(models)
        })
    });
    
    // Benchmark thinking models query
    group.bench_function("thinking_models_query", |b| {
        let registry = UnifiedModelRegistry::new();
        
        b.iter(|| {
            let models = black_box(registry.thinking_models());
            black_box(models)
        })
    });
    
    // Benchmark high context models query
    group.bench_function("high_context_models_query", |b| {
        let registry = UnifiedModelRegistry::new();
        
        b.iter(|| {
            let models = black_box(registry.high_context_models());
            black_box(models)
        })
    });
    
    // Benchmark cheapest model search
    group.bench_function("find_cheapest_model", |b| {
        let registry = UnifiedModelRegistry::new();
        
        b.iter(|| {
            let model = black_box(registry.find_cheapest_model(Some(8000), false));
            black_box(model)
        })
    });
    
    group.finish();
}

/// Benchmark batch operations with varying sizes
fn bench_batch_operations(c: &mut Criterion) {
    let rt = setup_runtime();
    let test_data = setup_test_data();
    
    let mut group = c.benchmark_group("batch_operations");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10); // Fewer samples for expensive operations
    
    // Benchmark batch validation with different sizes
    for size in [1, 5, 10, 15].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_validation", size),
            size,
            |b, &size| {
                let validator = ModelValidator::new();
                let models: Vec<_> = test_data.iter().take(size).cloned().collect();
                
                b.to_async(&rt).iter(|| async {
                    let result = black_box(validator.batch_validate_models(&models).await);
                    black_box(result)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark cache warming operations
fn bench_cache_warming(c: &mut Criterion) {
    let rt = setup_runtime();
    let test_data = setup_test_data();
    
    let mut group = c.benchmark_group("cache_warming");
    group.measurement_time(Duration::from_secs(10));
    
    // Benchmark cache warming with different batch sizes
    for size in [1, 5, 10, 20].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("cache_warm", size),
            size,
            |b, &size| {
                let cache = ModelCache::new();
                let model_names: Vec<String> = test_data
                    .iter()
                    .take(size)
                    .map(|(_, name)| name.to_string())
                    .collect();
                
                b.iter(|| {
                    black_box(cache.warm("openai", &model_names));
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark configuration operations
fn bench_configuration_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("configuration_operations");
    group.measurement_time(Duration::from_secs(5));
    
    // Benchmark cache config creation
    group.bench_function("cache_config_default", |b| {
        b.iter(|| {
            let config = black_box(CacheConfig::default());
            black_box(config)
        })
    });
    
    // Benchmark cache creation with config
    group.bench_function("cache_with_config", |b| {
        let config = CacheConfig::default();
        b.iter(|| {
            let cache = black_box(ModelCache::with_config(config.clone()));
            black_box(cache)
        })
    });
    
    group.finish();
}

/// Benchmark concurrent operations
fn bench_concurrent_operations(c: &mut Criterion) {
    let rt = setup_runtime();
    
    let mut group = c.benchmark_group("concurrent_operations");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);
    
    // Benchmark concurrent cache operations
    group.bench_function("concurrent_cache_gets", |b| {
        let cache = ModelCache::new();
        
        b.to_async(&rt).iter(|| async {
            let tasks: Vec<_> = (0..10)
                .map(|i| {
                    let cache = cache.clone();
                    tokio::spawn(async move {
                        cache.get("provider", &format!("model-{}", i))
                    })
                })
                .collect();
            
            let results = futures_util::future::join_all(tasks).await;
            black_box(results)
        })
    });
    
    // Benchmark concurrent validation operations
    group.bench_function("concurrent_validations", |b| {
        let validator = ModelValidator::new();
        
        b.to_async(&rt).iter(|| async {
            let tasks: Vec<_> = (0..5)
                .map(|_| {
                    let validator = validator.clone();
                    tokio::spawn(async move {
                        validator.circuit_breaker_status()
                    })
                })
                .collect();
            
            let results = futures_util::future::join_all(tasks).await;
            black_box(results)
        })
    });
    
    // Benchmark concurrent registry operations
    group.bench_function("concurrent_registry_stats", |b| {
        let registry = UnifiedModelRegistry::new();
        
        b.to_async(&rt).iter(|| async {
            let tasks: Vec<_> = (0..10)
                .map(|_| {
                    let registry = registry.clone();
                    tokio::spawn(async move {
                        registry.stats()
                    })
                })
                .collect();
            
            let results = futures_util::future::join_all(tasks).await;
            black_box(results)
        })
    });
    
    group.finish();
}

// Group all benchmarks
criterion_group!(
    benches,
    bench_registry_operations,
    bench_cache_operations,
    bench_validation_operations,
    bench_filtering_operations,
    bench_batch_operations,
    bench_cache_warming,
    bench_configuration_operations,
    bench_concurrent_operations
);

criterion_main!(benches);