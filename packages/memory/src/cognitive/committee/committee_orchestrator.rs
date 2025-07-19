//! Main orchestration and coordination for committee-based evaluation
//!
//! This module provides the high-level orchestration logic that coordinates
//! multiple LLM evaluators, manages evaluation sessions, handles caching,
//! and provides the main public API for the committee evaluation system.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

// AtomicCounter trait no longer needed since we use local RelaxedCounter
use crossbeam_skiplist::SkipMap;
use sha2::{Digest, Sha256};
use tracing::{error, info, instrument, warn};
use uuid::Uuid;

use super::committee_consensus::{CommitteeConsensusEngine, ConsensusConfig, ConsensusDecision};
use super::committee_evaluators::{EvaluationSession, EvaluatorPool, LLMEvaluator};
use super::committee_types::{
    CacheEntry, CacheMetrics, CommitteeError, CommitteeEvaluation, CommitteeMetrics,
    CommitteeResult, EvaluationConfig, EvaluationResult, ModelType,
    MAX_COMMITTEE_SIZE, MAX_CACHE_LIFETIME_SECS, MetricsSnapshot, CacheMetricsSnapshot,
};
use arrayvec::ArrayVec;
use crate::cognitive::mcts::CodeState;
use crate::cognitive::types::OptimizationSpec;

/// Main committee evaluator orchestrating the entire evaluation process
#[derive(Debug)]
pub struct CommitteeEvaluator {
    /// Configuration for evaluation parameters
    config: EvaluationConfig,
    /// Pool of available evaluators (lock-free)
    evaluator_pool: EvaluatorPool,
    /// Consensus engine for decision aggregation
    consensus_engine: CommitteeConsensusEngine,
    /// Lock-free cache for storing evaluation results
    evaluation_cache: SkipMap<String, CacheEntry>,
    /// Performance metrics tracking (atomic counters)
    metrics: CommitteeMetrics,
    /// Cache performance metrics (atomic counters)
    cache_metrics: CacheMetrics,
}

impl CommitteeEvaluator {
    /// Create a new committee evaluator
    ///
    /// # Arguments
    /// * `config` - Configuration specifying models, timeout, and consensus threshold
    ///
    /// # Returns
    /// * CommitteeEvaluator ready for evaluation tasks
    #[instrument(skip(config))]
    pub async fn new(config: EvaluationConfig) -> CommitteeResult<Self> {
        info!(
            "Initializing committee evaluator with {} models",
            config.models.len()
        );

        // Validate configuration
        Self::validate_config(&config)?;

        // Initialize evaluator pool
        let mut evaluator_pool = EvaluatorPool::new();

        // Create evaluators for each specified model type
        for model_type in &config.models {
            let evaluator = LLMEvaluator::new(model_type.clone(), 3)
                .await
                .map_err(|e| {
                    error!("Failed to create evaluator for {:?}: {}", model_type, e);
                    e
                })?;

            evaluator_pool.add_evaluator(Arc::new(evaluator))?;
            info!("Added evaluator for model: {:?}", model_type);
        }

        let consensus_engine = CommitteeConsensusEngine::new(ConsensusConfig::default());

        Ok(Self {
            config,
            evaluator_pool,
            consensus_engine,
            evaluation_cache: SkipMap::new(),
            metrics: CommitteeMetrics::default(),
            cache_metrics: CacheMetrics::default(),
        })
    }

    /// Evaluate an optimization proposal against user objectives
    ///
    /// # Arguments
    /// * `optimization_spec` - The optimization to evaluate
    /// * `current_state` - Current code state
    /// * `proposed_state` - Proposed optimized state
    ///
    /// # Returns
    /// * ConsensusDecision with committee assessment
    #[instrument(skip(self, optimization_spec, current_state, proposed_state))]
    pub async fn evaluate_optimization(
        &self,
        optimization_spec: &OptimizationSpec,
        current_state: &CodeState,
        proposed_state: &CodeState,
    ) -> CommitteeResult<ConsensusDecision> {
        let start_time = Instant::now();

        // Generate cache key
        let cache_key = self.generate_cache_key(optimization_spec, current_state, proposed_state);

        // Check cache first
        if let Some(cached_result) = self.check_cache(&cache_key)? {
            info!("Cache hit for evaluation");
            self.update_cache_hit_metrics();
            return Ok(cached_result.decision);
        }

        self.update_cache_miss_metrics();

        // Perform evaluation
        let evaluation_result = self
            .perform_evaluation(
                optimization_spec,
                &current_state.code_content,
                &proposed_state.code_content,
            )
            .await?;

        // Cache the result
        self.cache_result(cache_key, evaluation_result.clone());

        // Update metrics
        self.update_evaluation_metrics(&evaluation_result, start_time.elapsed());

        info!(
            "Evaluation completed in {:?} with consensus score {:.3}",
            start_time.elapsed(),
            evaluation_result.decision.normalized_consensus_score()
        );

        Ok(evaluation_result.decision)
    }

    /// Get current committee performance metrics
    pub fn metrics(&self) -> MetricsSnapshot {
        self.metrics.snapshot()
    }

    /// Get cache performance metrics
    pub fn cache_metrics(&self) -> CacheMetricsSnapshot {
        self.cache_metrics.snapshot()
    }

    /// Clear evaluation cache
    pub fn clear_cache(&self) {
        self.evaluation_cache.clear();

        // Reset cache metrics atomically
        self.cache_metrics.total_entries.store(0, std::sync::atomic::Ordering::Relaxed);
        self.cache_metrics.evictions.store(0, std::sync::atomic::Ordering::Relaxed);
        self.cache_metrics
            .memory_usage_bytes
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.cache_metrics
            .avg_entry_age_seconds
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.cache_metrics.generation.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        info!("Evaluation cache cleared");
    }

    /// Get health status of all evaluators
    pub async fn health_status(&self) -> HashMap<ModelType, f64> {
        let health_map = self.evaluator_pool.pool_health().await;

        let mut status_scores = HashMap::new();
        for (model_type, health_statuses) in health_map {
            let avg_health = if health_statuses.is_empty() {
                0.0
            } else {
                health_statuses
                    .iter()
                    .map(|h| {
                        if h.is_available {
                            1.0 - h.error_rate
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
                    / health_statuses.len() as f64
            };
            status_scores.insert(model_type, avg_health);
        }

        status_scores
    }

    /// Validate configuration before initialization
    fn validate_config(config: &EvaluationConfig) -> CommitteeResult<()> {
        if config.models.is_empty() {
            return Err(CommitteeError::InvalidConfiguration {
                message: "No models specified in configuration".into(),
            });
        }

        if config.models.len() < 2 {
            return Err(CommitteeError::InvalidConfiguration {
                message: "At least 2 models required for committee evaluation".into(),
            });
        }

        if config.consensus_threshold < 0.5 || config.consensus_threshold > 1.0 {
            return Err(CommitteeError::InvalidConfiguration {
                message: "Consensus threshold must be between 0.5 and 1.0".into(),
            });
        }

        if config.timeout_ms < 5000 {
            return Err(CommitteeError::ConfigurationError {
                message: "Timeout must be at least 5 seconds".into(),
            });
        }

        Ok(())
    }

    /// Generate cache key for evaluation request
    fn generate_cache_key(
        &self,
        optimization_spec: &OptimizationSpec,
        current_state: &CodeState,
        proposed_state: &CodeState,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(optimization_spec.objective.as_bytes());
        hasher.update(current_state.code_content.as_bytes());
        hasher.update(proposed_state.code_content.as_bytes());

        // Include model configuration in cache key
        let model_names: Vec<String> = self
            .config
            .models
            .iter()
            .map(|m| m.display_name().to_string())
            .collect();
        hasher.update(model_names.join(",").as_bytes());

        format!("{:x}", hasher.finalize())
    }

    /// Check cache for existing evaluation result
    fn check_cache(&self, cache_key: &str) -> CommitteeResult<Option<EvaluationResult>> {
        if let Some(entry_ref) = self.evaluation_cache.get(cache_key) {
            let entry = entry_ref.value();
            // Check if entry is still fresh (24 hours)
            if entry.created_at.elapsed() < Duration::from_secs(24 * 3600) {
                // Note: We can't atomically update access statistics with SkipMap
                // This is acceptable as access count is primarily for debugging
                return Ok(Some(entry.result.load().as_ref().clone()));
            }
        }

        Ok(None)
    }

    /// Perform the actual evaluation with committee
    async fn perform_evaluation(
        &self,
        optimization_spec: &OptimizationSpec,
        current_code: &str,
        proposed_code: &str,
    ) -> CommitteeResult<EvaluationResult> {
        let start_time = Instant::now();

        // Get evaluators for session
        let evaluators = self.get_session_evaluators()?;

        // Create evaluation session
        let session = EvaluationSession::new(evaluators, Duration::from_millis(self.config.timeout_ms))?;

        // Run evaluations concurrently
        let evaluation_results = session
            .evaluate_all(optimization_spec, current_code, proposed_code)
            .await;

        // Collect successful evaluations
        let mut successful_evaluations = Vec::new();
        let mut failed_count = 0;

        for result in evaluation_results {
            match result {
                Ok(evaluation) => successful_evaluations.push(evaluation),
                Err(e) => {
                    warn!("Individual evaluation failed: {}", e);
                    failed_count += 1;
                }
            }
        }

        // Ensure we have enough successful evaluations
        if successful_evaluations.len() < 2 {
            return Err(CommitteeError::InsufficientMembers {
                available: successful_evaluations.len(),
                required: 2,
            });
        }

        info!(
            "Collected {} successful evaluations ({} failed)",
            successful_evaluations.len(),
            failed_count
        );

        // Build consensus from successful evaluations
        let decision = self
            .consensus_engine
            .calculate_consensus(&successful_evaluations)
            .await?;

        // Calculate evaluation metrics
        let _metrics = self.calculate_evaluation_metrics(&successful_evaluations, &decision);

        let total_time = start_time.elapsed();
        let cache_key = self.generate_cache_key(
            optimization_spec,
            &CodeState {
                code: current_code.to_string(),
                code_content: current_code.to_string(),
                latency: 0.0,
                memory: 0.0,
                relevance: 1.0,
            },
            &CodeState {
                code: proposed_code.to_string(),
                code_content: proposed_code.to_string(),
                latency: 0.0,
                memory: 0.0,
                relevance: 1.0,
            },
        );

        Ok(EvaluationResult {
            decision,
            cache_key: cache_key.into(),
            from_cache: false,
            request_timestamp: start_time,
            evaluation_duration_ms: total_time.as_millis() as u64,
            cache_generation: 0,
        })
    }

    /// Get evaluators for evaluation session
    fn get_session_evaluators(&self) -> CommitteeResult<ArrayVec<Arc<LLMEvaluator>, MAX_COMMITTEE_SIZE>> {
        let mut evaluators = ArrayVec::new();

        for model_type in &self.config.models {
            if let Some(evaluator_arc) = self.evaluator_pool.get_evaluator(model_type) {
                if evaluators.try_push(evaluator_arc).is_err() {
                    break; // ArrayVec is full
                }
            } else {
                warn!("No evaluator available for model type: {:?}", model_type);
            }
        }

        if evaluators.len() < 2 {
            return Err(CommitteeError::InsufficientMembers {
                available: evaluators.len(),
                required: 2,
            });
        }

        Ok(evaluators)
    }

    /// Calculate evaluation metrics
    fn calculate_evaluation_metrics(
        &self,
        evaluations: &[CommitteeEvaluation],
        decision: &super::committee_consensus::ConsensusDecision,
    ) -> super::committee_types::EvaluationMetrics {
        let participants = evaluations.len();
        let consensus_count = evaluations
            .iter()
            .filter(|e| e.makes_progress == decision.is_positive())
            .count();

        let average_response_time = if evaluations.is_empty() {
            Duration::from_millis(0)
        } else {
            let total_time: Duration = evaluations.iter().map(|e| Duration::from_millis(e.evaluation_time)).sum();
            total_time / evaluations.len() as u32
        };

        let scores: Vec<f64> = evaluations
            .iter()
            .map(|e| {
                (e.objective_alignment + e.implementation_quality + (1.0 - e.risk_assessment)) / 3.0
            })
            .collect();

        let score_variance = if scores.len() < 2 {
            0.0
        } else {
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let variance = scores
                .iter()
                .map(|score| (score - mean).powi(2))
                .sum::<f64>()
                / scores.len() as f64;
            variance.sqrt()
        };

        let reasoning_quality = evaluations
            .iter()
            .map(|e| {
                let length_factor = (e.reasoning.len() as f64 / 200.0).clamp(0.0, 1.0);
                let reasoning_str = String::from_utf8_lossy(&e.reasoning);
                let word_count = reasoning_str.split_whitespace().count();
                let detail_factor = if word_count > 20 {
                    1.0
                } else {
                    word_count as f64 / 20.0
                };
                (length_factor + detail_factor) / 2.0
            })
            .sum::<f64>()
            / evaluations.len() as f64;

        let completed_on_time = average_response_time < Duration::from_millis(self.config.timeout_ms);

        super::committee_types::EvaluationMetrics {
            participants,
            consensus_count,
            average_response_time,
            score_variance,
            reasoning_quality,
            completed_on_time,
        }
    }

    /// Cache evaluation result
    #[inline]
    fn cache_result(&self, cache_key: String, result: EvaluationResult) {
        // Implement simple size-based eviction if cache is getting too large
        if self.evaluation_cache.len() >= 1000 {
            // Remove approximately 10% of entries by clearing some entries
            // Note: SkipMap doesn't provide ordered iteration, so we use a simpler approach
            let mut removed_count = 0;
            let target_removals = 100;
            
            // Create iterator and remove entries (simplified LRU)
            for entry in self.evaluation_cache.iter() {
                if removed_count >= target_removals {
                    break;
                }
                
                // Remove entries that are expired
                if entry.value().is_expired() {
                    self.evaluation_cache.remove(entry.key());
                    removed_count += 1;
                    self.cache_metrics.record_entry_evicted(1024); // Estimated size
                }
            }
        }
        
        let entry = CacheEntry::new(result, MAX_CACHE_LIFETIME_SECS);
        self.evaluation_cache.insert(cache_key, entry);
        self.cache_metrics.record_entry_added(1024); // Estimated size in bytes
    }

    /// Update cache hit metrics
    #[inline]
    fn update_cache_hit_metrics(&self) {
        self.cache_metrics.total_entries.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Update cache miss metrics
    #[inline]
    fn update_cache_miss_metrics(&self) {
        // Cache misses are tracked by not incrementing hits
        // The cache hit ratio calculation handles this automatically
    }

    /// Update overall evaluation metrics
    #[inline]
    fn update_evaluation_metrics(&self, result: &EvaluationResult, total_time: Duration) {
        // Record evaluation using atomic operations
        let quality_score = result.decision.normalized_consensus_score();
        self.metrics.record_evaluation(
            total_time.as_millis() as u64,
            result.from_cache,
            quality_score as f64,
        );
    }
}

/// Evaluation workflow coordinator for complex multi-step evaluations
#[derive(Debug)]
pub struct EvaluationWorkflow {
    /// Committee evaluator instance
    evaluator: Arc<CommitteeEvaluator>,
    /// Workflow identifier
    workflow_id: String,
    /// Steps in the evaluation workflow
    steps: Vec<WorkflowStep>,
    /// Current step index
    current_step: usize,
}

/// Individual step in an evaluation workflow
#[derive(Debug, Clone)]
pub struct WorkflowStep {
    /// Step identifier
    pub step_id: String,
    /// Step description
    pub description: String,
    /// Optimization spec for this step
    pub optimization_spec: OptimizationSpec,
    /// Whether this step is required for workflow completion
    pub is_required: bool,
    /// Dependencies on other steps
    pub dependencies: Vec<String>,
}

impl EvaluationWorkflow {
    /// Create a new evaluation workflow
    pub fn new(evaluator: Arc<CommitteeEvaluator>, steps: Vec<WorkflowStep>) -> Self {
        Self {
            evaluator,
            workflow_id: Uuid::new_v4().to_string(),
            steps,
            current_step: 0,
        }
    }

    /// Execute the complete workflow
    pub async fn execute_workflow(
        &mut self,
        current_state: &CodeState,
        proposed_state: &CodeState,
    ) -> CommitteeResult<Vec<ConsensusDecision>> {
        let mut results = Vec::new();

        for (index, step) in self.steps.iter().enumerate() {
            info!(
                "Executing workflow step {}: {}",
                index + 1,
                step.description
            );

            let decision = self
                .evaluator
                .evaluate_optimization(&step.optimization_spec, current_state, proposed_state)
                .await?;

            results.push(decision.clone());

            // Check if required step failed
            if step.is_required && !decision.is_positive() {
                warn!("Required workflow step failed: {}", step.description);
                break;
            }

            self.current_step = index + 1;
        }

        info!("Workflow completed with {} steps executed", results.len());
        Ok(results)
    }

    /// Get workflow progress
    pub fn progress(&self) -> f64 {
        if self.steps.is_empty() {
            1.0
        } else {
            self.current_step as f64 / self.steps.len() as f64
        }
    }
}

/// Committee coordinator for managing multiple committee instances
#[derive(Debug)]
pub struct CommitteeCoordinator {
    /// Active committee instances
    committees: HashMap<String, Arc<CommitteeEvaluator>>,
    /// Default committee configuration
    default_config: EvaluationConfig,
}

impl CommitteeCoordinator {
    /// Create a new committee coordinator
    pub fn new(default_config: EvaluationConfig) -> Self {
        Self {
            committees: HashMap::new(),
            default_config,
        }
    }

    /// Get or create committee for specific configuration
    pub async fn get_committee(
        &mut self,
        config: Option<EvaluationConfig>,
    ) -> CommitteeResult<Arc<CommitteeEvaluator>> {
        let effective_config = config.unwrap_or_else(|| self.default_config.clone());
        let config_key = format!("{:?}", effective_config.models); // Simplified key

        if let Some(committee) = self.committees.get(&config_key) {
            Ok(committee.clone())
        } else {
            let committee = Arc::new(CommitteeEvaluator::new(effective_config).await?);
            self.committees.insert(config_key, committee.clone());
            Ok(committee)
        }
    }

    /// Get aggregated metrics across all committees
    pub async fn aggregated_metrics(&self) -> HashMap<String, MetricsSnapshot> {
        let mut all_metrics = HashMap::new();

        for (key, committee) in &self.committees {
            let metrics = committee.metrics();
            all_metrics.insert(key.clone(), metrics);
        }

        all_metrics
    }
}
