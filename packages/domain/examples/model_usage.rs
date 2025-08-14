//! Practical usage examples for the fluent-ai model system
//!
//! This file demonstrates real-world usage patterns for the model registry,
//! caching, and validation systems. All examples use real functionality
//! without mocking and show proper error handling throughout.

use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};

use fluent_ai_domain::model::{
    BatchValidationResult, CacheConfig, ModelCache, ModelFilter, ModelValidator, RealModelInfo,
    UnifiedModelRegistry, ValidationResult,
};

/// Example 1: Basic model discovery and querying
///
/// This example shows how to discover available models and query
/// them by different criteria.
async fn example_basic_model_discovery() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Example 1: Basic Model Discovery ===");

    // Create a registry instance
    let registry = UnifiedModelRegistry::new();

    // Get statistics about available models
    let stats = registry.stats();
    println!(
        "Registry contains {} models across {} providers",
        stats.total_models,
        stats.models_by_provider.len()
    );

    // List all supported providers
    let providers = registry.providers();
    println!("Supported providers: {:?}", providers);

    // Get all models from OpenAI
    let openai_models = registry.models_by_provider("openai");
    println!("OpenAI has {} models available", openai_models.len());

    // Get all models that support reasoning/thinking
    let thinking_models = registry.thinking_models();
    println!(
        "Found {} models with thinking capabilities",
        thinking_models.len()
    );

    // Get high-context models (>100K tokens)
    let high_context_models = registry.high_context_models();
    println!("Found {} high-context models", high_context_models.len());

    // Find the most cost-effective model for a specific use case
    if let Some(cheapest) = registry.find_cheapest_model(Some(8000), false) {
        println!(
            "Most affordable model with 8K+ context: {} (${:.2} + ${:.2})",
            cheapest.name, cheapest.pricing_input, cheapest.pricing_output
        );
    }

    Ok(())
}

/// Example 2: Advanced filtering and model selection
///
/// This example demonstrates sophisticated model filtering based on
/// capabilities, pricing, and performance requirements.
async fn example_advanced_filtering() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Example 2: Advanced Model Filtering ===");

    let registry = UnifiedModelRegistry::new();

    // Find affordable models suitable for a chatbot application
    let chatbot_filter = ModelFilter {
        min_context: Some(16000),       // Need decent context
        max_input_price: Some(3.0),     // Budget constraint
        max_output_price: Some(10.0),   // Budget constraint
        requires_thinking: Some(false), // Simple responses, no reasoning needed
        ..Default::default()
    };

    let chatbot_models = registry.models_by_capabilities(&chatbot_filter);
    println!(
        "Found {} models suitable for chatbot use:",
        chatbot_models.len()
    );
    for model in chatbot_models.iter().take(3) {
        println!(
            "  - {} (context: {}, cost: ${:.2}/${:.2})",
            model.name, model.max_context, model.pricing_input, model.pricing_output
        );
    }

    // Find premium models for complex reasoning tasks
    let reasoning_filter = ModelFilter {
        min_context: Some(100000),     // Large context for complex tasks
        requires_thinking: Some(true), // Must support reasoning
        max_input_price: Some(20.0),   // Higher budget for quality
        max_output_price: Some(60.0),  // Higher budget for quality
        ..Default::default()
    };

    let reasoning_models = registry.models_by_capabilities(&reasoning_filter);
    println!(
        "\nFound {} models suitable for complex reasoning:",
        reasoning_models.len()
    );
    for model in reasoning_models.iter().take(3) {
        println!(
            "  - {} (context: {}, thinking: {}, cost: ${:.2}/${:.2})",
            model.name,
            model.max_context,
            model.is_thinking,
            model.pricing_input,
            model.pricing_output
        );
    }

    // Find models from a specific provider within budget
    let provider_filter = ModelFilter {
        provider: Some("anthropic".to_string()),
        max_input_price: Some(15.0),
        max_output_price: Some(75.0),
        ..Default::default()
    };

    let anthropic_budget_models = registry.models_by_capabilities(&provider_filter);
    println!(
        "\nAnthropic models within budget: {}",
        anthropic_budget_models.len()
    );

    Ok(())
}

/// Example 3: High-performance caching strategies
///
/// This example shows how to use the caching system effectively
/// for different usage patterns and performance requirements.
async fn example_caching_strategies() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Example 3: Caching Strategies ===");

    // Strategy 1: Default caching for general use
    let cache = ModelCache::new();

    println!("Testing cache performance...");
    let start = Instant::now();

    // Simulate cache misses
    for i in 0..5 {
        let result = cache.get("openai", &format!("model-{}", i));
        if result.is_none() {
            println!("  Cache miss for model-{}", i);
        }
    }

    let initial_stats = cache.stats();
    println!(
        "Initial cache stats: {} hits, {} misses",
        initial_stats.hits, initial_stats.misses
    );

    // Strategy 2: Custom configuration for high-performance applications
    let high_perf_config = CacheConfig {
        default_ttl: Duration::from_secs(300), // 5 minutes
        max_size: 10000,                       // Large cache
        enable_cleanup: true,
        enable_warming: true,
        cleanup_interval: Duration::from_secs(60),
    };

    let high_perf_cache = ModelCache::with_config(high_perf_config);
    high_perf_cache.start_background_tasks();

    // Strategy 3: Cache warming for predictable access patterns
    let models_to_warm = vec![
        "gpt-4".to_string(),
        "gpt-3.5-turbo".to_string(),
        "claude-3-opus-20240229".to_string(),
    ];

    println!("Warming cache with {} models...", models_to_warm.len());
    cache.warm("openai", &models_to_warm[..2]); // Warm OpenAI models
    cache.warm("anthropic", &models_to_warm[2..]); // Warm Anthropic models

    // Monitor cache performance
    let final_stats = cache.stats();
    println!("Cache hit ratio: {:.1}%", cache.hit_ratio());
    println!("Estimated memory usage: {} bytes", cache.memory_usage());
    println!(
        "Average lookup time: {}ns",
        final_stats.average_lookup_time_nanos
    );

    // Strategy 4: Cache management and maintenance
    println!("\nCache management operations:");

    // Check specific model in cache
    let is_cached = cache.contains("openai", "gpt-4");
    println!("GPT-4 is cached: {}", is_cached);

    // Invalidate specific model
    let was_invalidated = cache.invalidate("test", "model");
    println!("Test model invalidated: {}", was_invalidated);

    // Memory usage monitoring
    let memory_before = cache.memory_usage();
    cache.clear();
    let memory_after = cache.memory_usage();
    println!(
        "Memory usage: {} bytes -> {} bytes",
        memory_before, memory_after
    );

    Ok(())
}

/// Example 4: Model validation and health monitoring
///
/// This example demonstrates comprehensive model validation,
/// provider health checking, and error handling strategies.
async fn example_validation_and_health_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Example 4: Validation and Health Monitoring ===");

    let validator = ModelValidator::new();

    // Start background tasks for maintenance
    validator.start_background_tasks();

    // Example 4a: Single model validation
    println!("Validating individual models...");

    let test_models = vec![
        ("openai", "gpt-3.5-turbo"),
        ("anthropic", "claude-3-haiku-20240307"),
        ("xai", "grok-beta"),
    ];

    for (provider, model) in &test_models {
        match validator.validate_model_exists(provider, model).await {
            Ok(result) => {
                println!(
                    "  {}/{}: {} ({}ms)",
                    provider,
                    model,
                    result.status_description(),
                    result.response_time_ms.unwrap_or(0)
                );

                if !result.is_valid() {
                    if let Some(error) = &result.error {
                        println!("    Error: {}", error);
                    }
                }
            }
            Err(e) => {
                println!("  {}/{}: Validation failed - {}", provider, model, e);
            }
        }
    }

    // Example 4b: Batch validation for efficiency
    println!("\nBatch validation of multiple models...");
    let batch_start = Instant::now();

    let batch_result = validator.batch_validate_models(&test_models).await?;
    let batch_duration = batch_start.elapsed();

    println!("Batch validation completed in {:?}", batch_duration);
    println!(
        "Success rate: {:.1}% ({}/{} models)",
        batch_result.success_rate(),
        batch_result.successful_count,
        batch_result.results.len()
    );

    // Show failed models
    let failed_models = batch_result.failed_models();
    if !failed_models.is_empty() {
        println!("Failed validations:");
        for failed in failed_models {
            println!(
                "  - {}/{}: {}",
                failed.provider,
                failed.model_name,
                failed.status_description()
            );
        }
    }

    // Example 4c: Provider health monitoring
    println!("\nChecking provider health status...");
    let health_status = validator.provider_health_status().await;

    for (provider, status) in &health_status {
        let status_icon = if status.is_valid() { "✓" } else { "✗" };
        println!(
            "  {} {}: {} ({}ms)",
            status_icon,
            provider,
            status.status_description(),
            status.response_time_ms.unwrap_or(0)
        );
    }

    // Example 4d: Circuit breaker monitoring
    println!("\nCircuit breaker status:");
    let circuit_status = validator.circuit_breaker_status();

    if circuit_status.is_empty() {
        println!("  All circuit breakers are in normal state");
    } else {
        for (provider, state) in circuit_status {
            println!("  {}: {}", provider, state);
        }
    }

    Ok(())
}

/// Example 5: Real-world application integration
///
/// This example shows how to integrate the model system into a
/// real application with proper error handling and fallback strategies.
async fn example_application_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Example 5: Application Integration ===");

    // Initialize all components
    let registry = UnifiedModelRegistry::new();
    let cache = ModelCache::new();
    let validator = ModelValidator::new();

    // Start background tasks
    cache.start_background_tasks();
    validator.start_background_tasks();

    // Application function: Select optimal model for a task
    async fn select_model_for_task(
        registry: &UnifiedModelRegistry,
        cache: &ModelCache,
        validator: &ModelValidator,
        task_requirements: &TaskRequirements,
    ) -> Result<Arc<RealModelInfo>, String> {
        // Step 1: Check cache for previous selection
        let cache_key = format!(
            "task:{}:{}:{}",
            task_requirements.min_context,
            task_requirements.max_cost,
            task_requirements.needs_reasoning
        );

        // Step 2: Find candidate models
        let filter = ModelFilter {
            min_context: Some(task_requirements.min_context),
            max_input_price: Some(task_requirements.max_cost),
            max_output_price: Some(task_requirements.max_cost * 2.0),
            requires_thinking: Some(task_requirements.needs_reasoning),
            ..Default::default()
        };

        let candidates = registry.models_by_capabilities(&filter);

        if candidates.is_empty() {
            return Err("No models match the requirements".to_string());
        }

        // Step 3: Validate candidates and select the best one
        for model in candidates {
            // For this example, we'll assume provider can be determined from model name
            let provider = if model.name.starts_with("gpt") {
                "openai"
            } else if model.name.starts_with("claude") {
                "anthropic"
            } else if model.name.starts_with("grok") {
                "xai"
            } else {
                "unknown"
            };

            if provider == "unknown" {
                continue;
            }

            // Validate the model
            match validator.validate_model_exists(provider, &model.name).await {
                Ok(validation) if validation.is_valid() => {
                    println!("Selected model: {} from {}", model.name, provider);
                    return Ok(model);
                }
                Ok(validation) => {
                    println!(
                        "Model {}/{} failed validation: {}",
                        provider,
                        model.name,
                        validation.status_description()
                    );
                }
                Err(e) => {
                    println!("Validation error for {}/{}: {}", provider, model.name, e);
                }
            }
        }

        Err("No valid models found".to_string())
    }

    // Define task requirements
    struct TaskRequirements {
        min_context: u64,
        max_cost: f64,
        needs_reasoning: bool,
    }

    // Example tasks
    let tasks = vec![
        TaskRequirements {
            min_context: 8000,
            max_cost: 5.0,
            needs_reasoning: false,
        },
        TaskRequirements {
            min_context: 100000,
            max_cost: 20.0,
            needs_reasoning: true,
        },
    ];

    // Process each task
    for (i, task) in tasks.iter().enumerate() {
        println!("\nProcessing task {}:", i + 1);
        println!(
            "  Requirements: {}+ context, ${:.2} max cost, reasoning: {}",
            task.min_context, task.max_cost, task.needs_reasoning
        );

        match select_model_for_task(&registry, &cache, &validator, task).await {
            Ok(model) => {
                println!(
                    "  ✓ Selected: {} (context: {}, cost: ${:.2}/${:.2})",
                    model.name, model.max_context, model.pricing_input, model.pricing_output
                );
            }
            Err(e) => {
                println!("  ✗ Failed: {}", e);
            }
        }
    }

    Ok(())
}

/// Example 6: Performance monitoring and optimization
///
/// This example demonstrates how to monitor system performance
/// and optimize for specific usage patterns.
async fn example_performance_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Example 6: Performance Monitoring ===");

    let registry = UnifiedModelRegistry::new();
    let cache = ModelCache::new();
    let validator = ModelValidator::new();

    // Performance monitoring structure
    struct PerformanceMetrics {
        registry_stats: fluent_ai_domain::model::RegistryStats,
        cache_stats: fluent_ai_domain::model::CacheStats,
        validation_health: HashMap<String, ValidationResult>,
    }

    impl PerformanceMetrics {
        async fn collect(
            registry: &UnifiedModelRegistry,
            cache: &ModelCache,
            validator: &ModelValidator,
        ) -> Self {
            Self {
                registry_stats: registry.stats(),
                cache_stats: cache.stats(),
                validation_health: validator.provider_health_status().await,
            }
        }

        fn report_performance(&self) {
            println!("📊 Performance Metrics Report:");

            // Registry metrics
            println!("  Registry:");
            println!("    Total models: {}", self.registry_stats.total_models);
            println!(
                "    Thinking models: {}",
                self.registry_stats.thinking_models
            );
            println!(
                "    High-context models: {}",
                self.registry_stats.high_context_models
            );

            // Cache metrics
            println!("  Cache:");
            let total_requests = self.cache_stats.hits + self.cache_stats.misses;
            let hit_ratio = if total_requests > 0 {
                (self.cache_stats.hits as f64 / total_requests as f64) * 100.0
            } else {
                0.0
            };

            println!("    Hit ratio: {:.1}%", hit_ratio);
            println!("    Entries: {}", self.cache_stats.entries);
            println!("    Evictions: {}", self.cache_stats.evictions);
            println!(
                "    Avg lookup: {}ns",
                self.cache_stats.average_lookup_time_nanos
            );

            // Health metrics
            println!("  Provider Health:");
            let healthy_providers = self
                .validation_health
                .values()
                .filter(|v| v.is_valid())
                .count();

            println!(
                "    Healthy providers: {}/{}",
                healthy_providers,
                self.validation_health.len()
            );

            for (provider, health) in &self.validation_health {
                let status = if health.is_valid() { "🟢" } else { "🔴" };
                println!(
                    "    {} {}: {}ms",
                    status,
                    provider,
                    health.response_time_ms.unwrap_or(0)
                );
            }
        }

        fn identify_issues(&self) -> Vec<String> {
            let mut issues = Vec::new();

            // Check cache performance
            let total_requests = self.cache_stats.hits + self.cache_stats.misses;
            if total_requests > 0 {
                let hit_ratio = (self.cache_stats.hits as f64 / total_requests as f64) * 100.0;
                if hit_ratio < 50.0 {
                    issues.push(format!("Low cache hit ratio: {:.1}%", hit_ratio));
                }
            }

            // Check lookup performance
            if self.cache_stats.average_lookup_time_nanos > 1_000_000 {
                issues.push(format!(
                    "Slow cache lookups: {}ms avg",
                    self.cache_stats.average_lookup_time_nanos / 1_000_000
                ));
            }

            // Check provider health
            let unhealthy_providers: Vec<_> = self
                .validation_health
                .iter()
                .filter(|(_, health)| !health.is_valid())
                .map(|(provider, _)| provider.clone())
                .collect();

            if !unhealthy_providers.is_empty() {
                issues.push(format!("Unhealthy providers: {:?}", unhealthy_providers));
            }

            issues
        }
    }

    // Simulate some activity to generate metrics
    println!("Generating activity for metrics...");

    // Perform various operations
    let _ = registry.all_models();
    let _ = registry.thinking_models();
    let _ = cache.get("test", "model1");
    let _ = cache.get("test", "model2");
    let _ = cache.get("test", "model1"); // This should be a hit if we had put it

    // Collect and report metrics
    let metrics = PerformanceMetrics::collect(&registry, &cache, &validator).await;
    metrics.report_performance();

    // Identify potential issues
    let issues = metrics.identify_issues();
    if issues.is_empty() {
        println!("✅ No performance issues detected");
    } else {
        println!("⚠️  Performance issues detected:");
        for issue in issues {
            println!("    - {}", issue);
        }
    }

    // Performance recommendations
    println!("\n💡 Performance Recommendations:");
    println!("  - Use caching for frequently accessed models");
    println!("  - Monitor cache hit ratios and adjust TTL as needed");
    println!("  - Batch validation operations when possible");
    println!("  - Set up provider health monitoring alerts");
    println!("  - Consider cache warming for predictable access patterns");

    Ok(())
}

/// Main function demonstrating all examples
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Fluent AI Model System Usage Examples");
    println!("=========================================");

    // Check for API keys
    let mut available_providers = Vec::new();

    if env::var("OPENAI_API_KEY").is_ok() {
        available_providers.push("OpenAI");
    }
    if env::var("ANTHROPIC_API_KEY").is_ok() {
        available_providers.push("Anthropic");
    }
    if env::var("XAI_API_KEY").is_ok() {
        available_providers.push("xAI");
    }

    if available_providers.is_empty() {
        println!("ℹ️  Note: No API keys detected. Examples will use cached/mock data.");
        println!("   Set environment variables like OPENAI_API_KEY for live data.");
    } else {
        println!(
            "✅ Detected API keys for: {}",
            available_providers.join(", ")
        );
    }

    // Run all examples
    let examples = vec![
        (
            "Basic Model Discovery",
            example_basic_model_discovery as fn() -> _,
        ),
        ("Advanced Filtering", example_advanced_filtering),
        ("Caching Strategies", example_caching_strategies),
        (
            "Validation & Health Monitoring",
            example_validation_and_health_monitoring,
        ),
        ("Application Integration", example_application_integration),
        ("Performance Monitoring", example_performance_monitoring),
    ];

    for (name, example_fn) in examples {
        println!("\n🔄 Running: {}", name);
        match example_fn().await {
            Ok(()) => println!("✅ {} completed successfully", name),
            Err(e) => println!("❌ {} failed: {}", name, e),
        }
    }

    println!("\n🎉 All examples completed!");
    println!("\nFor more information, see:");
    println!("  - Documentation: packages/domain/src/model/README.md");
    println!("  - Tests: packages/domain/tests/model_integration_tests.rs");
    println!("  - Benchmarks: packages/domain/benches/model_performance.rs");

    Ok(())
}
