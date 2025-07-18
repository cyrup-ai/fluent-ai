//! Main orchestration and coordination for committee-based evaluation
//! 
//! This module provides the high-level orchestration logic that coordinates
//! multiple LLM evaluators, manages evaluation sessions, handles caching,
//! and provides the main public API for the committee evaluation system.

use super::committee_types::{
    EvaluationConfig, EvaluationResult, CommitteeMetrics, CacheEntry, CacheMetrics,
    CommitteeError, CommitteeResult, ModelType, ConsensusDecision, CommitteeEvaluation
};
use super::committee_evaluators::{LLMEvaluator, EvaluatorPool, EvaluationSession};
use super::committee_consensus::{ConsensusEngine, QualityMetrics};
use crate::cognitive::mcts::CodeState;
use crate::cognitive::types::{OptimizationSpec, CognitiveError};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error, instrument};
use uuid::Uuid;

/// Main committee evaluator orchestrating the entire evaluation process
#[derive(Debug)]
pub struct CommitteeEvaluator {
    /// Configuration for evaluation parameters
    config: EvaluationConfig,
    /// Pool of available evaluators
    evaluator_pool: Arc<RwLock<EvaluatorPool>>,
    /// Consensus engine for decision aggregation
    consensus_engine: ConsensusEngine,
    /// Cache for storing evaluation results
    evaluation_cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// Performance metrics tracking
    metrics: Arc<RwLock<CommitteeMetrics>>,
    /// Cache performance metrics
    cache_metrics: Arc<RwLock<CacheMetrics>>,
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
        info!("Initializing committee evaluator with {} models", config.models.len());
        
        // Validate configuration
        Self::validate_config(&config)?;
        
        // Initialize evaluator pool
        let mut evaluator_pool = EvaluatorPool::new();
        
        // Create evaluators for each specified model type
        for model_type in &config.models {
            let evaluator = LLMEvaluator::new(model_type.clone(), 3).await
                .map_err(|e| {
                    error!("Failed to create evaluator for {:?}: {}", model_type, e);
                    e
                })?;
            
            evaluator_pool.add_evaluator(evaluator).await;
            info!("Added evaluator for model: {:?}", model_type);
        }
        
        let consensus_engine = ConsensusEngine::new(config.consensus_threshold);
        
        Ok(Self {
            config,
            evaluator_pool: Arc::new(RwLock::new(evaluator_pool)),
            consensus_engine,
            evaluation_cache: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(CommitteeMetrics::default())),
            cache_metrics: Arc::new(RwLock::new(CacheMetrics::default())),
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
        if let Some(cached_result) = self.check_cache(&cache_key).await? {
            info!("Cache hit for evaluation");
            self.update_cache_hit_metrics().await;
            return Ok(cached_result.decision);
        }
        
        self.update_cache_miss_metrics().await;
        
        // Perform evaluation
        let evaluation_result = self.perform_evaluation(
            optimization_spec,
            &current_state.code_content,
            &proposed_state.code_content,
        ).await?;
        
        // Cache the result
        self.cache_result(cache_key, evaluation_result.clone()).await;
        
        // Update metrics
        self.update_evaluation_metrics(&evaluation_result, start_time.elapsed()).await;
        
        info!(
            "Evaluation completed in {:?} with confidence {:.3}",
            start_time.elapsed(),
            evaluation_result.decision.confidence
        );
        
        Ok(evaluation_result.decision)
    }
    
    /// Get current committee performance metrics
    pub async fn metrics(&self) -> CommitteeMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Get cache performance metrics
    pub async fn cache_metrics(&self) -> CacheMetrics {
        self.cache_metrics.read().await.clone()
    }
    
    /// Clear evaluation cache
    pub async fn clear_cache(&self) {
        let mut cache = self.evaluation_cache.write().await;
        cache.clear();
        
        let mut cache_metrics = self.cache_metrics.write().await;
        *cache_metrics = CacheMetrics::default();
        
        info!("Evaluation cache cleared");
    }
    
    /// Get health status of all evaluators
    pub async fn health_status(&self) -> HashMap<ModelType, f64> {
        let pool = self.evaluator_pool.read().await;
        let health_map = pool.pool_health().await;
        
        let mut status_scores = HashMap::new();
        for (model_type, health_statuses) in health_map {
            let avg_health = if health_statuses.is_empty() {
                0.0
            } else {
                health_statuses
                    .iter()
                    .map(|h| if h.is_available { 1.0 - h.error_rate } else { 0.0 })
                    .sum::<f64>() / health_statuses.len() as f64
            };
            status_scores.insert(model_type, avg_health);
        }
        
        status_scores
    }
    
    /// Validate configuration before initialization
    fn validate_config(config: &EvaluationConfig) -> CommitteeResult<()> {
        if config.models.is_empty() {
            return Err(CommitteeError::InvalidConfiguration {
                reason: "No models specified in configuration".to_string(),
            });
        }
        
        if config.models.len() < 2 {
            return Err(CommitteeError::InvalidConfiguration {
                reason: "At least 2 models required for committee evaluation".to_string(),
            });
        }
        
        if config.consensus_threshold < 0.5 || config.consensus_threshold > 1.0 {
            return Err(CommitteeError::InvalidConfiguration {
                reason: "Consensus threshold must be between 0.5 and 1.0".to_string(),
            });
        }
        
        if config.timeout < Duration::from_secs(5) {
            return Err(CommitteeError::InvalidConfiguration {
                reason: "Timeout must be at least 5 seconds".to_string(),
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
        let model_names: Vec<String> = self.config.models
            .iter()
            .map(|m| m.display_name().to_string())
            .collect();
        hasher.update(model_names.join(",").as_bytes());
        
        format!("{:x}", hasher.finalize())
    }
    
    /// Check cache for existing evaluation result
    async fn check_cache(&self, cache_key: &str) -> CommitteeResult<Option<&EvaluationResult>> {
        let cache = self.evaluation_cache.read().await;
        
        if let Some(entry) = cache.get(cache_key) {
            // Check if entry is still fresh (24 hours)
            if entry.created_at.elapsed() < Duration::from_secs(24 * 3600) {
                // Update access statistics
                drop(cache);
                let mut cache_mut = self.evaluation_cache.write().await;
                if let Some(entry_mut) = cache_mut.get_mut(cache_key) {
                    entry_mut.access_count += 1;
                    entry_mut.last_accessed = Instant::now();
                }
                
                // Return reference to cached result
                let cache_read = self.evaluation_cache.read().await;
                if let Some(entry) = cache_read.get(cache_key) {
                    return Ok(Some(&entry.result));
                }
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
        let evaluators = self.get_session_evaluators().await?;
        
        // Create evaluation session
        let session = EvaluationSession::new(evaluators, self.config.timeout);
        
        // Run evaluations concurrently
        let evaluation_results = session.evaluate_all(
            optimization_spec,
            current_code,
            proposed_code,
        ).await;
        
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
        let decision = self.consensus_engine.build_consensus_with_fallback(&successful_evaluations)?;
        
        // Calculate evaluation metrics
        let metrics = self.calculate_evaluation_metrics(&successful_evaluations, &decision);
        
        let total_time = start_time.elapsed();
        let cache_key = self.generate_cache_key(optimization_spec, 
            &CodeState { code_content: current_code.to_string() },
            &CodeState { code_content: proposed_code.to_string() }
        );
        
        Ok(EvaluationResult {
            decision,
            individual_evaluations: successful_evaluations,
            metrics,
            cache_key,
            total_time,
        })
    }
    
    /// Get evaluators for evaluation session
    async fn get_session_evaluators(&self) -> CommitteeResult<Vec<Arc<LLMEvaluator>>> {
        let pool = self.evaluator_pool.read().await;
        let mut evaluators = Vec::new();
        
        for model_type in &self.config.models {
            if let Some(evaluator) = pool.get_evaluator(model_type).await {
                evaluators.push(Arc::new(evaluator.clone()));
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
        decision: &ConsensusDecision,
    ) -> super::committee_types::EvaluationMetrics {
        let participants = evaluations.len();
        let consensus_count = evaluations
            .iter()
            .filter(|e| e.makes_progress == decision.makes_progress)
            .count();
        
        let average_response_time = if evaluations.is_empty() {
            Duration::from_millis(0)
        } else {
            let total_time: Duration = evaluations.iter().map(|e| e.evaluation_time).sum();
            total_time / evaluations.len() as u32
        };
        
        let scores: Vec<f64> = evaluations
            .iter()
            .map(|e| (e.objective_alignment + e.implementation_quality + (1.0 - e.risk_assessment)) / 3.0)
            .collect();
        
        let score_variance = if scores.len() < 2 {
            0.0
        } else {
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let variance = scores
                .iter()
                .map(|score| (score - mean).powi(2))
                .sum::<f64>() / scores.len() as f64;
            variance.sqrt()
        };
        
        let reasoning_quality = evaluations
            .iter()
            .map(|e| {
                let length_factor = (e.reasoning.len() as f64 / 200.0).clamp(0.0, 1.0);
                let word_count = e.reasoning.split_whitespace().count();
                let detail_factor = if word_count > 20 { 1.0 } else { word_count as f64 / 20.0 };
                (length_factor + detail_factor) / 2.0
            })
            .sum::<f64>() / evaluations.len() as f64;
        
        let completed_on_time = average_response_time < self.config.timeout;
        
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
    async fn cache_result(&self, cache_key: String, result: EvaluationResult) {
        let mut cache = self.evaluation_cache.write().await;
        
        // Implement simple LRU eviction if cache is getting too large
        if cache.len() >= 1000 {
            // Remove oldest 10% of entries
            let mut entries_by_age: Vec<_> = cache.iter().collect();
            entries_by_age.sort_by_key(|(_, entry)| entry.created_at);
            
            let to_remove: Vec<String> = entries_by_age
                .iter()
                .take(100)
                .map(|(key, _)| key.to_string())
                .collect();
            
            for key in to_remove {
                cache.remove(&key);
            }
        }
        
        let entry = CacheEntry {
            result,
            created_at: Instant::now(),
            access_count: 1,
            last_accessed: Instant::now(),
        };
        
        cache.insert(cache_key, entry);
    }
    
    /// Update cache hit metrics
    async fn update_cache_hit_metrics(&self) {
        let mut metrics = self.cache_metrics.write().await;
        metrics.hits += 1;
    }
    
    /// Update cache miss metrics
    async fn update_cache_miss_metrics(&self) {
        let mut metrics = self.cache_metrics.write().await;
        metrics.misses += 1;
    }
    
    /// Update overall evaluation metrics
    async fn update_evaluation_metrics(&self, result: &EvaluationResult, total_time: Duration) {
        let mut metrics = self.metrics.write().await;
        
        metrics.total_evaluations += 1;
        
        // Update running average of confidence
        let prev_avg = metrics.average_confidence;
        let count = metrics.total_evaluations as f64;
        metrics.average_confidence = (prev_avg * (count - 1.0) + result.decision.confidence) / count;
        
        // Update cache hit rate
        let cache_metrics = self.cache_metrics.read().await;
        metrics.cache_hit_rate = cache_metrics.hit_rate();
        
        // Update average evaluation time
        let prev_time = metrics.average_evaluation_time;
        metrics.average_evaluation_time = Duration::from_nanos(
            ((prev_time.as_nanos() as f64 * (count - 1.0) + total_time.as_nanos() as f64) / count) as u64
        );
        
        // Update success rate (simplified - assumes all cached results are successes)
        metrics.success_rate = 1.0; // Would be more sophisticated in real implementation
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
            info!("Executing workflow step {}: {}", index + 1, step.description);
            
            let decision = self.evaluator.evaluate_optimization(
                &step.optimization_spec,
                current_state,
                proposed_state,
            ).await?;
            
            results.push(decision.clone());
            
            // Check if required step failed
            if step.is_required && !decision.makes_progress {
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
    pub async fn get_committee(&mut self, config: Option<EvaluationConfig>) -> CommitteeResult<Arc<CommitteeEvaluator>> {
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
    pub async fn aggregated_metrics(&self) -> HashMap<String, CommitteeMetrics> {
        let mut all_metrics = HashMap::new();
        
        for (key, committee) in &self.committees {
            let metrics = committee.metrics().await;
            all_metrics.insert(key.clone(), metrics);
        }
        
        all_metrics
    }
}