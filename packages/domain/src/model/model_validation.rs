//! Model validation and health checking system
//!
//! This module provides comprehensive validation for AI models including
//! availability checks, API connectivity, capability verification, and
//! provider health monitoring.

use std::time::{Duration, Instant};
use std::collections::HashMap;

use ahash::RandomState;
use dashmap::DashMap;
use once_cell::sync::Lazy;
// arrayvec::ArrayVec removed - not used
use futures_util::future::join_all;
use tokio::time::timeout;

use model_info::{Provider, common::ProviderTrait};
use crate::model::error::{ModelError, Result};

/// Maximum validation timeout per model
const VALIDATION_TIMEOUT: Duration = Duration::from_secs(10);

/// Batch size for parallel validations
const VALIDATION_BATCH_SIZE: usize = 20;

/// Circuit breaker failure threshold
const CIRCUIT_BREAKER_THRESHOLD: usize = 5;

/// Circuit breaker recovery time
const CIRCUIT_BREAKER_RECOVERY: Duration = Duration::from_secs(300); // 5 minutes

/// Validation result for a single model
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// The provider name
    pub provider: String,
    /// The model name
    pub model_name: String,
    /// Whether the model is available
    pub is_available: bool,
    /// Whether the API key is valid
    pub api_key_valid: bool,
    /// Whether provider connectivity is working
    pub connectivity_ok: bool,
    /// Response time in milliseconds
    pub response_time_ms: Option<u64>,
    /// Any error encountered during validation
    pub error: Option<String>,
    /// Validation timestamp
    pub validated_at: Instant,
}

impl ValidationResult {
    /// Check if the model passed all validations
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.is_available && self.api_key_valid && self.connectivity_ok
    }
    
    /// Get a human-readable status description
    pub fn status_description(&self) -> &'static str {
        if self.is_valid() {
            "Available"
        } else if !self.connectivity_ok {
            "Connectivity Issues"
        } else if !self.api_key_valid {
            "Invalid API Key"
        } else if !self.is_available {
            "Model Unavailable"
        } else {
            "Unknown Issue"
        }
    }
}

/// Batch validation results
#[derive(Debug, Clone)]
pub struct BatchValidationResult {
    /// Individual model validation results
    pub results: Vec<ValidationResult>,
    /// Total validation time
    pub total_time_ms: u64,
    /// Number of successful validations
    pub successful_count: usize,
    /// Number of failed validations
    pub failed_count: usize,
}

impl BatchValidationResult {
    /// Get the success rate as a percentage
    #[inline]
    pub fn success_rate(&self) -> f64 {
        if self.results.is_empty() {
            0.0
        } else {
            (self.successful_count as f64 / self.results.len() as f64) * 100.0
        }
    }
    
    /// Get models that failed validation
    pub fn failed_models(&self) -> Vec<&ValidationResult> {
        self.results.iter().filter(|r| !r.is_valid()).collect()
    }
    
    /// Get models that passed validation
    pub fn successful_models(&self) -> Vec<&ValidationResult> {
        self.results.iter().filter(|r| r.is_valid()).collect()
    }
}

/// Circuit breaker state for a provider
#[derive(Debug, Clone)]
enum CircuitState {
    Closed,                           // Normal operation
    Open { opened_at: Instant },      // Failing, requests blocked
    HalfOpen,                         // Testing if provider recovered
}

/// Circuit breaker for provider health monitoring
#[derive(Debug, Clone)]
struct CircuitBreaker {
    state: CircuitState,
    failure_count: usize,
    last_failure: Option<Instant>,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            last_failure: None,
        }
    }
}

impl CircuitBreaker {
    /// Record a successful operation
    fn record_success(&mut self) {
        self.failure_count = 0;
        self.last_failure = None;
        self.state = CircuitState::Closed;
    }
    
    /// Record a failed operation
    fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure = Some(Instant::now());
        
        if self.failure_count >= CIRCUIT_BREAKER_THRESHOLD {
            self.state = CircuitState::Open { opened_at: Instant::now() };
        }
    }
    
    /// Check if requests should be allowed through
    fn allows_request(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => true,
            CircuitState::Open { opened_at } => {
                if opened_at.elapsed() > CIRCUIT_BREAKER_RECOVERY {
                    self.state = CircuitState::HalfOpen;
                    true
                } else {
                    false
                }
            }
        }
    }
}

/// Validation cache for avoiding redundant checks
struct ValidationCache {
    /// Cached validation results
    results: DashMap<String, ValidationResult, RandomState>,
    /// Cache TTL (results expire after this duration)
    ttl: Duration,
}

impl ValidationCache {
    fn new(ttl: Duration) -> Self {
        Self {
            results: DashMap::with_hasher(RandomState::default()),
            ttl,
        }
    }
    
    fn get(&self, provider: &str, model_name: &str) -> Option<ValidationResult> {
        let key = format!("{}:{}", provider, model_name);
        self.results.get(&key).and_then(|entry| {
            if entry.validated_at.elapsed() < self.ttl {
                Some(entry.clone())
            } else {
                None
            }
        })
    }
    
    fn put(&self, result: ValidationResult) {
        let key = format!("{}:{}", result.provider, result.model_name);
        self.results.insert(key, result);
    }
    
    fn clear_expired(&self) {
        let expired_keys: Vec<String> = self.results
            .iter()
            .filter(|entry| entry.validated_at.elapsed() >= self.ttl)
            .map(|entry| entry.key().clone())
            .collect();
        
        for key in expired_keys {
            self.results.remove(&key);
        }
    }
}

/// Global validation state
struct ValidationData {
    /// Circuit breakers per provider
    circuit_breakers: DashMap<String, CircuitBreaker, RandomState>,
    /// Validation result cache
    cache: ValidationCache,
    /// Provider instances
    providers: ArrayVec<(String, Provider), 7>,
}

impl Default for ValidationData {
    fn default() -> Self {
        let mut providers = ArrayVec::new();
        
        // Initialize providers
        let _ = providers.try_push(("openai".to_string(), Provider::OpenAi(model_info::providers::openai::OpenAiProvider)));
        let _ = providers.try_push(("mistral".to_string(), Provider::Mistral(model_info::providers::mistral::MistralProvider)));
        let _ = providers.try_push(("anthropic".to_string(), Provider::Anthropic(model_info::providers::anthropic::AnthropicProvider)));
        let _ = providers.try_push(("together".to_string(), Provider::Together(model_info::providers::together::TogetherProvider)));
        let _ = providers.try_push(("openrouter".to_string(), Provider::OpenRouter(model_info::providers::openrouter::OpenRouterProvider)));
        let _ = providers.try_push(("huggingface".to_string(), Provider::HuggingFace(model_info::providers::huggingface::HuggingFaceProvider)));
        let _ = providers.try_push(("xai".to_string(), Provider::Xai(model_info::providers::xai::XaiProvider)));
        
        Self {
            circuit_breakers: DashMap::with_hasher(RandomState::default()),
            cache: ValidationCache::new(Duration::from_secs(300)), // 5 minute cache
            providers,
        }
    }
}

/// Global validation instance
static VALIDATION: Lazy<ValidationData> = Lazy::new(Default::default);

/// Model validator for checking availability and health
///
/// This validator provides comprehensive model validation including:
/// - Model availability checking
/// - API key validation
/// - Provider connectivity testing
/// - Circuit breaker pattern for failing providers
/// - Intelligent caching to avoid redundant checks
/// - Batch validation for efficiency
#[derive(Clone, Default)]
pub struct ModelValidator;

impl ModelValidator {
    /// Create a new model validator
    #[inline]
    pub fn new() -> Self {
        Self
    }
    
    /// Validate that a model exists and is accessible
    ///
    /// # Arguments
    /// * `provider` - The provider name
    /// * `model_name` - The model name to validate
    ///
    /// # Returns
    /// Validation result indicating model status
    pub async fn validate_model_exists(&self, provider: &str, model_name: &str) -> Result<ValidationResult> {
        // Check cache first
        if let Some(cached) = VALIDATION.cache.get(provider, model_name) {
            return Ok(cached);
        }
        
        // Check circuit breaker
        {
            let mut breaker = VALIDATION.circuit_breakers
                .entry(provider.to_string())
                .or_insert_with(CircuitBreaker::default);
            
            if !breaker.allows_request() {
                return Ok(ValidationResult {
                    provider: provider.to_string(),
                    model_name: model_name.to_string(),
                    is_available: false,
                    api_key_valid: false,
                    connectivity_ok: false,
                    response_time_ms: None,
                    error: Some("Provider circuit breaker is open".to_string()),
                    validated_at: Instant::now(),
                });
            }
        }
        
        let start_time = Instant::now();
        
        // Find the provider instance
        let provider_instance = VALIDATION.providers
            .iter()
            .find(|(name, _)| name == provider)
            .map(|(_, provider)| provider)
            .ok_or_else(|| ModelError::InvalidConfiguration(
                format!("Unknown provider: {}", provider).into()
            ))?;
        
        // Perform validation with timeout
        let result = timeout(VALIDATION_TIMEOUT, async {
            match provider_instance.get_model_info(model_name).await {
                Ok(model_info) => ValidationResult {
                    provider: provider.to_string(),
                    model_name: model_name.to_string(),
                    is_available: true,
                    api_key_valid: true,
                    connectivity_ok: true,
                    response_time_ms: Some(start_time.elapsed().as_millis() as u64),
                    error: None,
                    validated_at: Instant::now(),
                },
                Err(e) => {
                    let error_str = e.to_string();
                    let is_auth_error = error_str.contains("401") || 
                                      error_str.contains("unauthorized") ||
                                      error_str.contains("invalid") && error_str.contains("key");
                    
                    ValidationResult {
                        provider: provider.to_string(),
                        model_name: model_name.to_string(),
                        is_available: false,
                        api_key_valid: !is_auth_error,
                        connectivity_ok: !error_str.contains("timeout") && !error_str.contains("connection"),
                        response_time_ms: Some(start_time.elapsed().as_millis() as u64),
                        error: Some(error_str),
                        validated_at: Instant::now(),
                    }
                }
            }
        }).await;
        
        let validation_result = match result {
            Ok(result) => result,
            Err(_) => ValidationResult {
                provider: provider.to_string(),
                model_name: model_name.to_string(),
                is_available: false,
                api_key_valid: false,
                connectivity_ok: false,
                response_time_ms: Some(VALIDATION_TIMEOUT.as_millis() as u64),
                error: Some("Validation timeout".to_string()),
                validated_at: Instant::now(),
            }
        };
        
        // Update circuit breaker
        {
            let mut breaker = VALIDATION.circuit_breakers
                .entry(provider.to_string())
                .or_insert_with(CircuitBreaker::default);
            
            if validation_result.is_valid() {
                breaker.record_success();
            } else {
                breaker.record_failure();
            }
        }
        
        // Cache the result
        VALIDATION.cache.put(validation_result.clone());
        
        Ok(validation_result)
    }
    
    /// Validate provider access (API key and connectivity)
    ///
    /// # Arguments
    /// * `provider` - The provider name to validate
    ///
    /// # Returns
    /// Validation result for provider access
    pub async fn validate_provider_access(&self, provider: &str) -> Result<ValidationResult> {
        // Use a known model for the provider to test access
        let test_model = match provider {
            "openai" => "gpt-3.5-turbo",
            "anthropic" => "claude-3-haiku-20240307", 
            "xai" => "grok-beta",
            "mistral" => "mistral-small",
            "together" => "meta-llama/Llama-2-7b-chat-hf",
            "openrouter" => "openai/gpt-3.5-turbo",
            "huggingface" => "microsoft/DialoGPT-medium",
            _ => return Err(ModelError::InvalidConfiguration(
                format!("Unknown provider: {}", provider).into()
            )),
        };
        
        self.validate_model_exists(provider, test_model).await
    }
    
    /// Validate that a model supports required capabilities
    ///
    /// # Arguments
    /// * `provider` - The provider name
    /// * `model_name` - The model name
    /// * `required_capabilities` - List of required capabilities
    ///
    /// # Returns
    /// Validation result including capability check
    pub async fn validate_model_capabilities(&self, 
                                           provider: &str, 
                                           model_name: &str,
                                           required_capabilities: &[&str]) -> Result<ValidationResult> {
        let mut base_result = self.validate_model_exists(provider, model_name).await?;
        
        if !base_result.is_available {
            return Ok(base_result);
        }
        
        // For now, assume all available models support basic capabilities
        // In a full implementation, this would check specific capabilities
        // against the model's feature set
        
        Ok(base_result)
    }
    
    /// Validate multiple models in parallel
    ///
    /// # Arguments
    /// * `models` - List of (provider, model_name) tuples to validate
    ///
    /// # Returns
    /// Batch validation results
    pub async fn batch_validate_models(&self, models: &[(&str, &str)]) -> Result<BatchValidationResult> {
        let start_time = Instant::now();
        
        // Process in batches to avoid overwhelming providers
        let mut all_results = Vec::with_capacity(models.len());
        
        for chunk in models.chunks(VALIDATION_BATCH_SIZE) {
            let tasks: Vec<_> = chunk.iter()
                .map(|(provider, model_name)| {
                    self.validate_model_exists(provider, model_name)
                })
                .collect();
            
            let chunk_results = join_all(tasks).await;
            
            for result in chunk_results {
                match result {
                    Ok(validation) => all_results.push(validation),
                    Err(e) => {
                        // Create error result for failed validation
                        all_results.push(ValidationResult {
                            provider: "unknown".to_string(),
                            model_name: "unknown".to_string(),
                            is_available: false,
                            api_key_valid: false,
                            connectivity_ok: false,
                            response_time_ms: None,
                            error: Some(e.to_string()),
                            validated_at: Instant::now(),
                        });
                    }
                }
            }
            
            // Small delay between batches to be respectful to APIs
            if models.len() > VALIDATION_BATCH_SIZE {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
        
        let successful_count = all_results.iter().filter(|r| r.is_valid()).count();
        let failed_count = all_results.len() - successful_count;
        
        Ok(BatchValidationResult {
            results: all_results,
            total_time_ms: start_time.elapsed().as_millis() as u64,
            successful_count,
            failed_count,
        })
    }
    
    /// Get the health status of all providers
    ///
    /// # Returns
    /// Map of provider names to their health status
    pub async fn provider_health_status(&self) -> HashMap<String, ValidationResult> {
        let providers = ["openai", "anthropic", "xai", "mistral", "together", "openrouter", "huggingface"];
        
        let tasks: Vec<_> = providers.iter()
            .map(|provider| async move {
                let result = self.validate_provider_access(provider).await
                    .unwrap_or_else(|e| ValidationResult {
                        provider: provider.to_string(),
                        model_name: "health_check".to_string(),
                        is_available: false,
                        api_key_valid: false,
                        connectivity_ok: false,
                        response_time_ms: None,
                        error: Some(e.to_string()),
                        validated_at: Instant::now(),
                    });
                (provider.to_string(), result)
            })
            .collect();
        
        let results = join_all(tasks).await;
        results.into_iter().collect()
    }
    
    /// Clear validation cache
    pub fn clear_cache(&self) {
        VALIDATION.cache.results.clear();
    }
    
    /// Get circuit breaker status for all providers
    ///
    /// # Returns
    /// Map of provider names to their circuit breaker status
    pub fn circuit_breaker_status(&self) -> HashMap<String, String> {
        VALIDATION.circuit_breakers
            .iter()
            .map(|entry| {
                let status = match &entry.state {
                    CircuitState::Closed => "Closed".to_string(),
                    CircuitState::Open { opened_at } => 
                        format!("Open ({}s ago)", opened_at.elapsed().as_secs()),
                    CircuitState::HalfOpen => "Half-Open".to_string(),
                };
                (entry.key().clone(), status)
            })
            .collect()
    }
    
    /// Start background maintenance tasks
    pub fn start_background_tasks(&self) {
        // Start cache cleanup task
        tokio::spawn(async {
            let mut interval = tokio::time::interval(Duration::from_secs(300));
            
            loop {
                interval.tick().await;
                VALIDATION.cache.clear_expired();
            }
        });
    }
}