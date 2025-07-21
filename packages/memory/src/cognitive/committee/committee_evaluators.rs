//! Individual provider evaluator implementations for committee-based assessment
//!
//! This module provides the core evaluator logic that interfaces with individual
//! completion providers to perform assessment tasks. Each evaluator manages a single
//! model instance and handles prompt generation, response parsing, and error recovery.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use arrayvec::ArrayVec;
// AtomicCounter trait no longer needed since we use local RelaxedCounter
use crossbeam_queue::SegQueue;
use tokio::sync::{RwLock, Semaphore};
use uuid::Uuid;

pub use super::committee_types::{
    CommitteeError, CommitteeEvaluation, CommitteeResult, EvaluationPrompt, HealthStatus,
    MAX_COMMITTEE_SIZE, Model, ModelMetrics, ModelType, QualityTier,
};
// Import additional types for zero allocation patterns
pub use crate::cognitive::committee::committee_evaluators_extension::{
    EvaluationSessionMetrics, EvaluationTask, EvaluatorPoolMetrics,
};
use crate::cognitive::types::OptimizationSpec;
// Use domain types for traits and models  
use fluent_ai_domain::{
    completion::{CompletionProvider, CompletionError},
    chat::Message,
};

// Use provider clients for completion services
use fluent_ai_provider::{
    openai::OpenAIClient,
    anthropic::AnthropicClient,
};

/// Individual provider evaluator managing a single model instance
#[derive(Debug)]
pub struct ProviderEvaluator {
    /// Model instance with provider
    model: Model,
    /// Unique identifier for this evaluator
    evaluator_id: String,
    /// Semaphore for request rate limiting
    request_limiter: Arc<Semaphore>,
    /// Performance tracking
    metrics: Arc<RwLock<ModelMetrics>>,
    /// Health monitoring
    health_status: Arc<RwLock<HealthStatus>>,
}

impl ProviderEvaluator {
    /// Create a new provider evaluator instance
    ///
    /// # Arguments
    /// * `model_type` - Type of model to use for evaluation
    /// * `max_concurrent_requests` - Maximum concurrent requests allowed
    ///
    /// # Returns
    /// * ProviderEvaluator configured for the specified model type
    pub async fn new(
        model_type: ModelType,
        max_concurrent_requests: usize,
    ) -> CommitteeResult<Self> {
        let provider = Self::create_provider(&model_type).await.map_err(|e| {
            CommitteeError::ProviderError {
                model_type: model_type.clone(),
                source: e,
            }
        })?;

        let model = Model {
            model_type: model_type.clone(),
            api_key: Arc::from(""),
            base_url: None,
            timeout_ms: 30000,
            max_retries: 3,
            rate_limit_per_minute: 60,
            provider,
            health_status: Arc::new(RwLock::new(HealthStatus::default())),
            metrics: Arc::new(RwLock::new(ModelMetrics::default())),
        };

        let evaluator_id = format!(
            "{}-{}",
            model_type.display_name(),
            Uuid::new_v4().as_simple()
        );

        Ok(Self {
            model,
            evaluator_id,
            request_limiter: Arc::new(Semaphore::new(max_concurrent_requests)),
            metrics: Arc::new(RwLock::new(ModelMetrics::default())),
            health_status: Arc::new(RwLock::new(HealthStatus::default())),
        })
    }

    /// Perform evaluation of an optimization proposal
    ///
    /// # Arguments
    /// * `optimization_spec` - The optimization to evaluate
    /// * `current_code` - Current code state
    /// * `proposed_code` - Proposed optimized code
    /// * `timeout` - Maximum time to wait for evaluation
    ///
    /// # Returns
    /// * CommitteeEvaluation with detailed assessment
    pub async fn evaluate_optimization(
        &self,
        optimization_spec: &OptimizationSpec,
        current_code: &str,
        proposed_code: &str,
        timeout: Duration,
    ) -> CommitteeResult<CommitteeEvaluation> {
        let start_time = Instant::now();

        // Acquire rate limiting permit
        let _permit =
            self.request_limiter
                .acquire()
                .await
                .map_err(|_| CommitteeError::ModelUnavailable {
                    model_type: self.model.model_type.clone(),
                })?;

        // Check health status
        {
            let health = self.health_status.read().await;
            if !health.is_available {
                return Err(CommitteeError::ModelUnavailable {
                    model_type: self.model.model_type.clone(),
                });
            }
        }

        // Generate evaluation prompt
        let prompt = EvaluationPrompt::new_optimization_prompt(
            optimization_spec,
            current_code,
            proposed_code,
            None,
        );

        // Perform evaluation with timeout
        let evaluation_result =
            tokio::time::timeout(timeout, self.perform_evaluation_request(prompt)).await;

        let evaluation_time = start_time.elapsed();

        match evaluation_result {
            Ok(Ok(response)) => {
                self.update_success_metrics(evaluation_time).await;
                self.parse_evaluation_response(response, evaluation_time)
                    .await
            }
            Ok(Err(e)) => {
                self.update_error_metrics().await;
                Err(e)
            }
            Err(_) => {
                self.update_error_metrics().await;
                Err(CommitteeError::EvaluationTimeout {
                    timeout_ms: timeout.as_millis() as u64,
                })
            }
        }
    }

    /// Get current health status of this evaluator
    pub async fn health_status(&self) -> HealthStatus {
        self.health_status.read().await.clone()
    }

    /// Get current performance metrics
    pub async fn metrics(&self) -> ModelMetrics {
        self.metrics.read().await.clone()
    }

    /// Get evaluator identifier
    #[inline(always)]
    pub fn evaluator_id(&self) -> &str {
        &self.evaluator_id
    }

    /// Get model type
    #[inline(always)]
    pub fn model_type(&self) -> &ModelType {
        &self.model.model_type
    }

    /// Create appropriate provider for model type
    async fn create_provider(
        model_type: &ModelType,
    ) -> Result<Arc<dyn CompletionProvider>, Box<dyn std::error::Error + Send + Sync>> {
        match model_type {
            ModelType::Gpt35Turbo | ModelType::Gpt4O | ModelType::Gpt4Turbo => {
                // Create OpenAI provider using Provider package
                let api_key = std::env::var("OPENAI_API_KEY")
                    .map_err(|_| "OPENAI_API_KEY environment variable not set")?;
                let provider = OpenAIClient::new(api_key, model_type.display_name())
                    .map_err(|e| format!("Failed to create OpenAI client: {}", e))?;
                Ok(Arc::new(provider))
            }
            ModelType::Claude3Opus | ModelType::Claude3Sonnet | ModelType::Claude3Haiku => {
                // Create Anthropic provider using Provider package
                let api_key = std::env::var("ANTHROPIC_API_KEY")
                    .map_err(|_| "ANTHROPIC_API_KEY environment variable not set")?;
                let provider = AnthropicClient::new(api_key, model_type.display_name())
                    .map_err(|e| format!("Failed to create Anthropic client: {}", e))?;
                Ok(Arc::new(provider))
            }
            ModelType::GeminiPro
            | ModelType::Mixtral8x7B
            | ModelType::Llama270B
            | ModelType::Llama3 => {
                // These models are not yet implemented
                Err(format!("Model type {:?} is not yet implemented", model_type).into())
            }
        }
    }

    /// Perform the actual provider request
    async fn perform_evaluation_request(
        &self,
        prompt: EvaluationPrompt,
    ) -> CommitteeResult<String> {
        // Combine system and user prompts into a single prompt string
        let full_prompt = format!("{}\n\n{}", prompt.system_prompt, prompt.user_prompt);

        // Use the CompletionProvider streaming interface
        let stream = self.model.provider.prompt(&full_prompt);
        
        // Collect streaming response into complete text
        let mut result = String::new();
        let mut stream = stream;
        
        use futures::StreamExt;
        while let Some(chunk) = stream.next().await {
            if let Some(content) = chunk.content {
                result.push_str(&content);
            }
        }

        if result.is_empty() {
            Err(CommitteeError::ProviderError {
                model_type: self.model.model_type.clone(),
                source: Box::new(CompletionError::ProviderUnavailable("Empty response from provider".to_string())),
            })
        } else {
            Ok(result)
        }
    }

    /// Parse provider response into structured evaluation
    async fn parse_evaluation_response(
        &self,
        response: String,
        evaluation_time: Duration,
    ) -> CommitteeResult<CommitteeEvaluation> {
        let content = &response;

        // Parse structured response using regex or simple parsing
        let (
            makes_progress,
            objective_alignment,
            implementation_quality,
            risk_assessment,
            reasoning,
            _improvements,
        ) = self.extract_evaluation_components(content)?;

        // Calculate confidence based on model quality tier and response coherence
        let confidence = self.calculate_response_confidence(content, &reasoning);

        Ok(CommitteeEvaluation {
            model: self.model.model_type.clone(),
            score: objective_alignment, // Use objective_alignment as overall score
            reasoning: reasoning.into_bytes().into(), // Convert String to SmallVec<u8, 512>
            confidence,
            processing_time_ms: evaluation_time.as_millis() as u64,
            timestamp: chrono::Utc::now(),
            objective_alignment,
            implementation_quality,
            risk_assessment,
            makes_progress,
            evaluation_time: evaluation_time.as_micros() as u64,
        })
    }

    /// Extract evaluation components from response text
    fn extract_evaluation_components(
        &self,
        content: &str,
    ) -> CommitteeResult<(bool, f64, f64, f64, String, Vec<String>)> {
        let mut makes_progress = false;
        let mut objective_alignment = 0.5;
        let mut implementation_quality = 0.5;
        let mut risk_assessment = 0.5;
        let mut reasoning = String::new();
        let mut improvements = Vec::new();

        // Simple parsing logic (production would use more sophisticated parsing)
        for line in content.lines() {
            let line = line.trim();

            if line.to_lowercase().contains("makes progress") {
                makes_progress = line.to_lowercase().contains("true");
            } else if line.to_lowercase().contains("objective alignment") {
                if let Some(score) = self.extract_score_from_line(line) {
                    objective_alignment = score;
                }
            } else if line.to_lowercase().contains("implementation quality") {
                if let Some(score) = self.extract_score_from_line(line) {
                    implementation_quality = score;
                }
            } else if line.to_lowercase().contains("risk assessment") {
                if let Some(score) = self.extract_score_from_line(line) {
                    risk_assessment = score;
                }
            } else if line.to_lowercase().contains("reasoning") {
                reasoning = content
                    .lines()
                    .skip_while(|l| !l.to_lowercase().contains("reasoning"))
                    .skip(1)
                    .take_while(|l| !l.to_lowercase().contains("suggested improvements"))
                    .collect::<Vec<_>>()
                    .join("\n")
                    .trim()
                    .to_string();
            } else if line.to_lowercase().contains("suggested improvements") {
                improvements = content
                    .lines()
                    .skip_while(|l| !l.to_lowercase().contains("suggested improvements"))
                    .skip(1)
                    .filter(|l| l.trim().starts_with('-') || l.trim().starts_with('*'))
                    .map(|l| {
                        l.trim()
                            .trim_start_matches('-')
                            .trim_start_matches('*')
                            .trim()
                            .to_string()
                    })
                    .collect();
            }
        }

        // Ensure reasoning has some content
        if reasoning.is_empty() {
            reasoning = content
                .lines()
                .filter(|l| l.len() > 20)
                .take(3)
                .collect::<Vec<_>>()
                .join(" ");
        }

        Ok((
            makes_progress,
            objective_alignment,
            implementation_quality,
            risk_assessment,
            reasoning,
            improvements,
        ))
    }

    /// Extract numerical score from a line of text
    fn extract_score_from_line(&self, line: &str) -> Option<f64> {
        // Simple regex to find floating point numbers
        use regex::Regex;
        let re = Regex::new(r"(\d*\.?\d+)").ok()?;

        for cap in re.captures_iter(line) {
            if let Ok(score) = cap[1].parse::<f64>() {
                if score >= 0.0 && score <= 1.0 {
                    return Some(score);
                }
                // Handle scores given as percentages
                if score > 1.0 && score <= 100.0 {
                    return Some(score / 100.0);
                }
            }
        }
        None
    }

    /// Calculate confidence based on response quality and model characteristics
    fn calculate_response_confidence(&self, content: &str, reasoning: &str) -> f64 {
        let mut confidence: f64 = match self.model.model_type.quality_tier() {
            QualityTier::Draft => 0.6,
            QualityTier::Good => 0.7,
            QualityTier::High => 0.8,
            QualityTier::Premium => 0.9,
            QualityTier::Basic => 0.65,
            QualityTier::Standard => 0.75,
            QualityTier::Experimental => 0.55,
        };

        // Adjust based on response length and detail
        if reasoning.len() > 100 {
            confidence += 0.05;
        }
        if reasoning.len() > 300 {
            confidence += 0.05;
        }

        // Penalize very short responses
        if content.len() < 100 {
            confidence -= 0.2;
        }

        // Ensure confidence stays in valid range
        confidence.clamp(0.0, 1.0)
    }

    /// Update metrics after successful evaluation
    async fn update_success_metrics(&self, evaluation_time: Duration) {
        let mut health = self.health_status.write().await;
        health.is_available = true;
        health.last_success = Some(Instant::now());
        health.total_requests += 1;

        // Update running average of response time
        let total_time = health.avg_response_time.as_nanos() as f64
            * (health.total_requests - 1) as f64
            + evaluation_time.as_nanos() as f64;
        health.avg_response_time =
            Duration::from_nanos((total_time / health.total_requests as f64) as u64);

        // Update error rate
        health.error_rate = health.failed_requests as f64 / health.total_requests as f64;

        let mut metrics = self.metrics.write().await;
        metrics.evaluations_completed += 1;
        metrics.total_evaluation_time += evaluation_time;
    }

    /// Update metrics after failed evaluation
    async fn update_error_metrics(&self) {
        let mut health = self.health_status.write().await;
        health.total_requests += 1;
        health.failed_requests += 1;
        health.error_rate = health.failed_requests as f64 / health.total_requests as f64;

        // Mark as unavailable if error rate is too high
        if health.error_rate > 0.5 && health.total_requests > 5 {
            health.is_available = false;
        }
    }
}

/// Lock-free evaluator pool for blazing-fast load balancing and redundancy
#[derive(Debug)]
pub struct EvaluatorPool {
    /// Available evaluators by model type (stack-allocated for zero allocation)
    evaluators: HashMap<ModelType, ArrayVec<Arc<ProviderEvaluator>, MAX_COMMITTEE_SIZE>>,
    /// Atomic round-robin indices for lock-free load balancing
    round_robin_indices: HashMap<ModelType, AtomicUsize>,
    /// Lock-free task queue for evaluator distribution (TODO: Implement distributed task system)
    _task_queue: SegQueue<EvaluationTask>,
    /// Pool metrics with atomic counters
    metrics: EvaluatorPoolMetrics,
}

impl EvaluatorPool {
    /// Create a new evaluator pool with zero allocation
    #[inline]
    pub fn new() -> Self {
        Self {
            evaluators: HashMap::new(),
            round_robin_indices: HashMap::new(),
            _task_queue: SegQueue::new(),
            metrics: EvaluatorPoolMetrics::new(),
        }
    }

    /// Add evaluator to the pool with zero allocation
    #[inline]
    pub fn add_evaluator(&mut self, evaluator: Arc<ProviderEvaluator>) -> Result<(), CommitteeError> {
        let model_type = evaluator.model_type().clone();

        let evaluators = self
            .evaluators
            .entry(model_type.clone())
            .or_insert_with(ArrayVec::new);
        if evaluators.is_full() {
            return Err(CommitteeError::ResourceExhausted {
                resource: "evaluator pool capacity".into(),
            });
        }
        evaluators.push(evaluator);

        self.round_robin_indices
            .entry(model_type)
            .or_insert_with(|| AtomicUsize::new(0));
        self.metrics.evaluators_added.inc();
        Ok(())
    }

    /// Get next available evaluator using lock-free round-robin
    #[inline]
    pub fn get_evaluator(&self, model_type: &ModelType) -> Option<Arc<ProviderEvaluator>> {
        let evaluators = self.evaluators.get(model_type)?;
        if evaluators.is_empty() {
            return None;
        }

        let atomic_index = self.round_robin_indices.get(model_type)?;
        let current_index = atomic_index.fetch_add(1, Ordering::Relaxed);
        let index = current_index % evaluators.len();

        self.metrics.evaluators_accessed.inc();
        Some(evaluators[index].clone())
    }

    /// Get all available model types
    pub fn available_model_types(&self) -> Vec<ModelType> {
        self.evaluators.keys().cloned().collect()
    }

    /// Get pool health status with zero allocation
    pub async fn pool_health(
        &self,
    ) -> HashMap<ModelType, ArrayVec<HealthStatus, MAX_COMMITTEE_SIZE>> {
        let mut health_map = HashMap::new();

        for (model_type, evaluators) in &self.evaluators {
            let mut health_statuses = ArrayVec::new();
            for evaluator in evaluators {
                let health = evaluator.health_status().await;
                if let Ok(()) = health_statuses.try_push(health) {
                    // Successfully added health status
                } else {
                    // ArrayVec is full, skip remaining evaluators
                    break;
                }
            }
            health_map.insert(model_type.clone(), health_statuses);
        }

        health_map
    }
}

/// Zero-allocation session for coordinating multiple evaluations
#[derive(Debug)]
pub struct EvaluationSession {
    /// Active evaluators in this session (stack-allocated)
    evaluators: ArrayVec<Arc<ProviderEvaluator>, MAX_COMMITTEE_SIZE>,
    /// Session start time
    start_time: Instant,
    /// Session timeout
    timeout: Duration,
    /// Session metrics with atomic counters
    metrics: EvaluationSessionMetrics,
}

impl EvaluationSession {
    /// Create a new evaluation session with zero allocation
    #[inline]
    pub fn new(
        evaluators: ArrayVec<Arc<ProviderEvaluator>, MAX_COMMITTEE_SIZE>,
        timeout: Duration,
    ) -> Result<Self, CommitteeError> {
        if evaluators.is_empty() {
            return Err(CommitteeError::InsufficientMembers {
                available: 0,
                required: 1,
            });
        }

        Ok(Self {
            evaluators,
            start_time: Instant::now(),
            timeout,
            metrics: EvaluationSessionMetrics::new(),
        })
    }

    /// Run evaluation across all session evaluators with zero allocation
    pub async fn evaluate_all(
        &self,
        optimization_spec: &OptimizationSpec,
        current_code: &str,
        proposed_code: &str,
    ) -> ArrayVec<CommitteeResult<CommitteeEvaluation>, MAX_COMMITTEE_SIZE> {
        let remaining_timeout = self.timeout.saturating_sub(self.start_time.elapsed());

        // Pre-allocate futures array for zero allocation
        let mut evaluation_futures: ArrayVec<_, MAX_COMMITTEE_SIZE> = ArrayVec::new();

        for evaluator in &self.evaluators {
            let spec = optimization_spec.clone();
            // Use references instead of allocating strings
            let current_ref = current_code;
            let proposed_ref = proposed_code;

            let future = async move {
                evaluator
                    .evaluate_optimization(&spec, current_ref, proposed_ref, remaining_timeout)
                    .await
            };

            if evaluation_futures.try_push(future).is_err() {
                // ArrayVec is full, cannot add more futures
                break;
            }
        }

        // Execute all futures concurrently
        let results = futures::future::join_all(evaluation_futures).await;

        // Convert Vec<Result> to ArrayVec<Result> with zero allocation
        let mut result_array = ArrayVec::new();
        for result in results {
            if result_array.try_push(result).is_err() {
                break; // ArrayVec is full
            }
        }

        self.metrics.evaluations_completed.inc();
        result_array
    }

    /// Get session runtime
    pub fn session_runtime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Check if session has timed out
    pub fn is_timed_out(&self) -> bool {
        self.start_time.elapsed() > self.timeout
    }
}
