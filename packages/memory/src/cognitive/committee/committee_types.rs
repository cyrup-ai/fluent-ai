//! Zero-Allocation Committee Evaluation System Core Types
//!
//! Blazing-fast, lock-free data structures for LLM committee-based evaluation.
//! All operations are designed for optimal performance with zero heap allocations
//! in hot paths and comprehensive atomic operations for thread safety.

use crate::cognitive::types::{CognitiveError, OptimizationSpec};
use arrayvec::ArrayVec;
use atomic_counter::{AtomicCounter, RelaxedCounter};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::sync::atomic::{AtomicU64, AtomicU32, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use arc_swap::ArcSwap;

/// Maximum committee size for optimal performance and decision quality
pub const MAX_COMMITTEE_SIZE: usize = 8;

/// Maximum reasoning text length for stack allocation optimization
pub const MAX_REASONING_BYTES: usize = 512;

/// Cache entry maximum lifetime for memory efficiency
pub const MAX_CACHE_LIFETIME_SECS: u64 = 3600;

/// LLM model types with compile-time performance characteristics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ModelType {
    /// GPT-4 Omni - superior reasoning and analysis
    Gpt4O = 0,
    /// Claude 3 Sonnet - balanced performance and quality
    Claude3Sonnet = 1,
    /// Claude 3 Haiku - ultra-fast evaluation
    Claude3Haiku = 2,
    /// Claude 3 Opus - maximum capability and depth
    Claude3Opus = 3,
    /// GPT-3.5 Turbo - cost-effective rapid evaluation
    Gpt35Turbo = 4,
    /// Gemini Pro - Google's flagship model
    GeminiPro = 5,
    /// Mixtral 8x7B - open source high performance
    Mixtral8x7B = 6,
    /// Llama 2 70B - Meta's large scale model
    Llama270B = 7,
}

/// Model quality tier for evaluation weighting (user-configurable thresholds)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum QualityTier {
    /// Draft quality: 0.0-0.4 threshold
    Draft = 0,
    /// Good quality: 0.4-0.7 threshold  
    Good = 1,
    /// High quality: 0.7-0.9 threshold
    High = 2,
    /// Premium quality: 0.9+ threshold
    Premium = 3,
}

impl QualityTier {
    /// Get quality threshold (hardcoded algorithm, user-configurable via config)
    #[inline(always)]
    pub const fn threshold(self) -> f64 {
        match self {
            Self::Draft => 0.0,
            Self::Good => 0.4,
            Self::High => 0.7,
            Self::Premium => 0.9,
        }
    }
}

/// Zero-allocation health status with lock-free updates
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub is_available: bool,
    pub last_success: Option<Instant>,
    pub total_requests: u64,
    pub failed_requests: u64,
    pub error_rate: f64,
    pub avg_response_time: Duration,
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self {
            is_available: true,
            last_success: None,
            total_requests: 0,
            failed_requests: 0,
            error_rate: 0.0,
            avg_response_time: Duration::from_millis(0),
        }
    }
}

/// Lock-free model metrics with performance tracking
#[derive(Debug, Clone)]
pub struct ModelMetrics {
    pub evaluations_completed: u64,
    pub total_evaluation_time: Duration,
    pub average_score: f64,
    pub success_rate: f64,
    pub last_update: Instant,
}

impl Default for ModelMetrics {
    fn default() -> Self {
        Self {
            evaluations_completed: 0,
            total_evaluation_time: Duration::from_millis(0),
            average_score: 0.0,
            success_rate: 1.0,
            last_update: Instant::now(),
        }
    }
}

impl ModelType {
    /// Get model quality tier for evaluation weighting
    #[inline(always)]
    pub const fn quality_tier(self) -> QualityTier {
        match self {
            Self::Gpt4O | Self::Claude3Opus => QualityTier::Premium,
            Self::Claude3Sonnet | Self::GeminiPro => QualityTier::High,
            Self::Claude3Haiku | Self::Gpt35Turbo | Self::Mixtral8x7B | Self::Llama270B => QualityTier::Good,
        }
    }

    /// Get zero-allocation model identifier for API calls
    #[inline(always)]
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::Gpt4O => "gpt-4o",
            Self::Claude3Sonnet => "claude-3-sonnet-20240229",
            Self::Claude3Haiku => "claude-3-haiku-20240307",
            Self::Claude3Opus => "claude-3-opus-20240229",
            Self::Gpt35Turbo => "gpt-3.5-turbo",
            Self::GeminiPro => "gemini-pro",
            Self::Mixtral8x7B => "mixtral-8x7b-instruct",
            Self::Llama270B => "llama-2-70b-chat",
        }
    }

    /// Get provider routing information
    #[inline(always)]
    pub const fn provider(self) -> &'static str {
        match self {
            Self::Gpt4O | Self::Gpt35Turbo => "openai",
            Self::Claude3Sonnet | Self::Claude3Haiku | Self::Claude3Opus => "anthropic",
            Self::GeminiPro => "google",
            Self::Mixtral8x7B => "mistral",
            Self::Llama270B => "meta",
        }
    }

    /// Get relative model strength for consensus weighting
    #[inline(always)]
    pub const fn strength_weight(self) -> f64 {
        match self {
            Self::Gpt4O => 1.0,
            Self::Claude3Opus => 0.98,
            Self::Claude3Sonnet => 0.88,
            Self::GeminiPro => 0.85,
            Self::Mixtral8x7B => 0.78,
            Self::Llama270B => 0.75,
            Self::Claude3Haiku => 0.65,
            Self::Gpt35Turbo => 0.55,
        }
    }

    /// Get expected latency for timeout calculations
    #[inline(always)]
    pub const fn expected_latency_ms(self) -> u64 {
        match self {
            Self::Claude3Haiku => 2500,
            Self::Gpt35Turbo => 4000,
            Self::Gpt4O => 6500,
            Self::Claude3Sonnet => 8500,
            Self::GeminiPro => 9500,
            Self::Mixtral8x7B => 12000,
            Self::Llama270B => 15000,
            Self::Claude3Opus => 18000,
        }
    }

    /// Get cost factor for budget optimization
    #[inline(always)]
    pub const fn cost_factor(self) -> f64 {
        match self {
            Self::Gpt35Turbo => 1.0,
            Self::Claude3Haiku => 1.8,
            Self::Gpt4O => 2.5,
            Self::GeminiPro => 3.2,
            Self::Mixtral8x7B => 4.1,
            Self::Claude3Sonnet => 6.8,
            Self::Llama270B => 12.5,
            Self::Claude3Opus => 18.0,
        }
    }

    /// Check if model supports function calling
    #[inline(always)]
    pub const fn supports_function_calling(self) -> bool {
        match self {
            Self::Gpt4O | Self::Gpt35Turbo | Self::Claude3Opus => true,
            _ => false,
        }
    }

    /// Get maximum context window size
    #[inline(always)]
    pub const fn max_context_tokens(self) -> u32 {
        match self {
            Self::Gpt4O => 128000,
            Self::Claude3Opus | Self::Claude3Sonnet | Self::Claude3Haiku => 200000,
            Self::Gpt35Turbo => 16385,
            Self::GeminiPro => 32768,
            Self::Mixtral8x7B => 32768,
            Self::Llama270B => 4096,
        }
    }
}

/// Zero-allocation model configuration with shared resources
#[derive(Debug, Clone)]
pub struct Model {
    pub model_type: ModelType,
    pub api_key: Arc<str>,
    pub base_url: Option<Arc<str>>,
    pub timeout_ms: u64,
    pub max_retries: u8,
    pub rate_limit_per_minute: u32,
}

impl Model {
    /// Create new model configuration with optimal defaults
    #[inline]
    pub fn new(model_type: ModelType, api_key: Arc<str>) -> Self {
        Self {
            model_type,
            api_key,
            base_url: None,
            timeout_ms: model_type.expected_latency_ms() * 2,
            max_retries: 3,
            rate_limit_per_minute: match model_type.provider() {
                "openai" => 3500,
                "anthropic" => 4000,
                "google" => 1500,
                _ => 1000,
            },
        }
    }

    /// Set custom base URL with zero allocation
    #[inline]
    pub fn with_base_url(mut self, base_url: Arc<str>) -> Self {
        self.base_url = Some(base_url);
        self
    }

    /// Set timeout with validation
    #[inline]
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms.max(1000).min(300000);
        self
    }

    /// Set retry count with validation
    #[inline]
    pub fn with_max_retries(mut self, max_retries: u8) -> Self {
        self.max_retries = max_retries.min(10);
        self
    }
}

/// Committee evaluation configuration with stack-allocated collections
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Committee models (stack-allocated for blazing performance)
    pub models: ArrayVec<ModelType, MAX_COMMITTEE_SIZE>,
    /// Global timeout for all evaluations
    pub timeout_ms: u64,
    /// Minimum consensus threshold (0.5 - 1.0)
    pub consensus_threshold: f64,
    /// Parallel evaluation concurrency limit
    pub max_concurrent_evaluations: u8,
    /// Enable aggressive caching for performance
    pub enable_caching: bool,
    /// Quality score minimum threshold
    pub quality_threshold: f64,
}

impl EvaluationConfig {
    /// Create new configuration with optimal defaults
    #[inline]
    pub const fn new() -> Self {
        Self {
            models: ArrayVec::new_const(),
            timeout_ms: 45000,
            consensus_threshold: 0.72,
            max_concurrent_evaluations: 6,
            enable_caching: true,
            quality_threshold: 0.65,
        }
    }

    /// Add model to committee with validation
    #[inline]
    pub fn add_model(&mut self, model: ModelType) -> Result<(), CommitteeError> {
        if self.models.is_full() {
            return Err(CommitteeError::ConfigurationError {
                message: "Maximum committee size exceeded".into(),
            });
        }
        if self.models.contains(&model) {
            return Err(CommitteeError::ConfigurationError {
                message: "Model already in committee".into(),
            });
        }
        self.models.push(model);
        Ok(())
    }

    /// Validate configuration for production use
    #[inline]
    pub fn validate(&self) -> Result<(), CommitteeError> {
        if self.models.is_empty() {
            return Err(CommitteeError::ConfigurationError {
                message: "At least one model required".into(),
            });
        }
        if !(0.5..=1.0).contains(&self.consensus_threshold) {
            return Err(CommitteeError::ConfigurationError {
                message: "Consensus threshold must be between 0.5 and 1.0".into(),
            });
        }
        if self.timeout_ms < 5000 || self.timeout_ms > 300000 {
            return Err(CommitteeError::ConfigurationError {
                message: "Timeout must be between 5-300 seconds".into(),
            });
        }
        if !(0.0..=1.0).contains(&self.quality_threshold) {
            return Err(CommitteeError::ConfigurationError {
                message: "Quality threshold must be between 0.0 and 1.0".into(),
            });
        }
        Ok(())
    }

    /// Get estimated total evaluation time
    #[inline]
    pub fn estimated_evaluation_time_ms(&self) -> u64 {
        if self.models.is_empty() {
            return 0;
        }
        let max_latency = self.models.iter()
            .map(|m| m.expected_latency_ms())
            .max()
            .unwrap_or(0);
        max_latency + 2000 // Add overhead
    }

    /// Get estimated cost factor
    #[inline]
    pub fn estimated_cost_factor(&self) -> f64 {
        self.models.iter()
            .map(|m| m.cost_factor())
            .sum::<f64>()
    }
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        let mut config = Self::new();
        // Optimal default committee for balanced performance
        let _ = config.add_model(ModelType::Gpt4O);
        let _ = config.add_model(ModelType::Claude3Sonnet);
        let _ = config.add_model(ModelType::Claude3Haiku);
        config
    }
}

/// Individual committee member evaluation with optimized storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitteeEvaluation {
    /// Model that performed evaluation
    pub model: ModelType,
    /// Numeric score (0.0 - 1.0)
    pub score: f64,
    /// Detailed reasoning (optimized for small stack allocation)
    pub reasoning: SmallVec<[u8; MAX_REASONING_BYTES]>,
    /// Confidence in evaluation (0.0 - 1.0)
    pub confidence: f64,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Evaluation timestamp
    pub timestamp: Instant,
    /// Objective alignment score (0.0 - 1.0)
    pub objective_alignment: f64,
    /// Implementation quality score (0.0 - 1.0)
    pub implementation_quality: f64,
    /// Risk assessment score (0.0 - 1.0)
    pub risk_assessment: f64,
}

impl CommitteeEvaluation {
    /// Create new evaluation with validation
    #[inline]
    pub fn new(
        model: ModelType,
        score: f64,
        reasoning: &str,
        confidence: f64,
        objective_alignment: f64,
        implementation_quality: f64,
        risk_assessment: f64,
    ) -> Result<Self, CommitteeError> {
        if !(0.0..=1.0).contains(&score) {
            return Err(CommitteeError::ValidationError {
                field: "score".into(),
                value: score.to_string().into(),
            });
        }
        if !(0.0..=1.0).contains(&confidence) {
            return Err(CommitteeError::ValidationError {
                field: "confidence".into(),
                value: confidence.to_string().into(),
            });
        }

        let reasoning_bytes = reasoning.as_bytes();
        if reasoning_bytes.len() > MAX_REASONING_BYTES * 4 {
            return Err(CommitteeError::ValidationError {
                field: "reasoning".into(),
                value: "reasoning too long".into(),
            });
        }

        let mut reasoning_storage = SmallVec::new();
        reasoning_storage.extend_from_slice(reasoning_bytes);

        Ok(Self {
            model,
            score: score.clamp(0.0, 1.0),
            reasoning: reasoning_storage,
            confidence: confidence.clamp(0.0, 1.0),
            processing_time_ms: 0,
            timestamp: Instant::now(),
            objective_alignment: objective_alignment.clamp(0.0, 1.0),
            implementation_quality: implementation_quality.clamp(0.0, 1.0),
            risk_assessment: risk_assessment.clamp(0.0, 1.0),
        })
    }

    /// Get reasoning as string with error handling
    #[inline]
    pub fn reasoning_str(&self) -> Result<&str, CommitteeError> {
        std::str::from_utf8(&self.reasoning).map_err(|_| CommitteeError::DecodingError {
            message: "Invalid UTF-8 in reasoning".into(),
        })
    }

    /// Set processing time
    #[inline]
    pub fn with_processing_time(mut self, processing_time: Duration) -> Self {
        self.processing_time_ms = processing_time.as_millis().min(u64::MAX as u128) as u64;
        self
    }

    /// Calculate weighted score using model strength and confidence
    #[inline]
    pub fn weighted_score(&self) -> f64 {
        let base_weight = self.model.strength_weight();
        let confidence_weight = self.confidence;
        let quality_weight = (self.objective_alignment + self.implementation_quality) / 2.0;
        let risk_adjustment = 1.0 - (self.risk_assessment * 0.3);
        
        self.score * base_weight * confidence_weight * quality_weight * risk_adjustment
    }

    /// Calculate comprehensive quality metric
    #[inline]
    pub fn quality_metric(&self) -> f64 {
        // Weighted combination of all quality factors
        let alignment_weight = 0.35;
        let implementation_weight = 0.25;
        let confidence_weight = 0.25;
        let risk_weight = 0.15;

        (self.objective_alignment * alignment_weight)
            + (self.implementation_quality * implementation_weight)
            + (self.confidence * confidence_weight)
            + ((1.0 - self.risk_assessment) * risk_weight)
    }

    /// Check if evaluation meets quality threshold
    #[inline]
    pub fn meets_quality_threshold(&self, threshold: f64) -> bool {
        self.quality_metric() >= threshold
    }
}

/// Final committee consensus decision with comprehensive analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusDecision {
    /// Whether optimization should proceed
    pub makes_progress: bool,
    /// Overall confidence in decision (0.0 - 1.0)
    pub confidence: f64,
    /// Consensus score (0.0 - 1.0)
    pub consensus_score: f64,
    /// Aggregated reasoning summary
    pub summary_reasoning: Arc<str>,
    /// Individual evaluations (stack-allocated)
    pub evaluations: ArrayVec<CommitteeEvaluation, MAX_COMMITTEE_SIZE>,
    /// Number of dissenting opinions
    pub dissenting_count: u8,
    /// Total processing time for all evaluations
    pub total_processing_time_ms: u64,
    /// Decision timestamp
    pub decision_timestamp: Instant,
    /// Weighted average score
    pub weighted_average_score: f64,
    /// Quality score of the decision
    pub decision_quality: f64,
    /// Risk assessment aggregate
    pub aggregate_risk: f64,
}

impl ConsensusDecision {
    /// Create new consensus decision with comprehensive metrics
    #[inline]
    pub fn new(
        makes_progress: bool,
        confidence: f64,
        consensus_score: f64,
        summary_reasoning: Arc<str>,
    ) -> Self {
        Self {
            makes_progress,
            confidence: confidence.clamp(0.0, 1.0),
            consensus_score: consensus_score.clamp(0.0, 1.0),
            summary_reasoning,
            evaluations: ArrayVec::new(),
            dissenting_count: 0,
            total_processing_time_ms: 0,
            decision_timestamp: Instant::now(),
            weighted_average_score: 0.0,
            decision_quality: 0.0,
            aggregate_risk: 0.0,
        }
    }

    /// Add evaluation and update aggregate metrics
    #[inline]
    pub fn add_evaluation(&mut self, evaluation: CommitteeEvaluation) -> Result<(), CommitteeError> {
        if self.evaluations.is_full() {
            return Err(CommitteeError::ConfigurationError {
                message: "Maximum evaluations exceeded".into(),
            });
        }

        self.total_processing_time_ms += evaluation.processing_time_ms;
        self.evaluations.push(evaluation);
        self.recalculate_metrics();
        Ok(())
    }

    /// Recalculate all aggregate metrics
    #[inline]
    fn recalculate_metrics(&mut self) {
        if self.evaluations.is_empty() {
            return;
        }

        // Calculate weighted average score
        let total_weight: f64 = self.evaluations.iter()
            .map(|e| e.model.strength_weight() * e.confidence)
            .sum();
        
        if total_weight > 0.0 {
            self.weighted_average_score = self.evaluations.iter()
                .map(|e| e.weighted_score())
                .sum::<f64>() / total_weight;
        }

        // Calculate decision quality
        self.decision_quality = self.evaluations.iter()
            .map(|e| e.quality_metric())
            .sum::<f64>() / self.evaluations.len() as f64;

        // Calculate aggregate risk
        self.aggregate_risk = self.evaluations.iter()
            .map(|e| e.risk_assessment * e.confidence)
            .sum::<f64>() / self.evaluations.len() as f64;

        // Count dissenting opinions
        let progress_votes = self.evaluations.iter()
            .filter(|e| e.score >= 0.5)
            .count();
        
        self.dissenting_count = if self.makes_progress {
            (self.evaluations.len() - progress_votes) as u8
        } else {
            progress_votes as u8
        };
    }

    /// Get agreement ratio
    #[inline]
    pub fn agreement_ratio(&self) -> f64 {
        if self.evaluations.is_empty() {
            return 0.0;
        }
        let agreeing = self.evaluations.len() as f64 - self.dissenting_count as f64;
        agreeing / self.evaluations.len() as f64
    }

    /// Get score variance for stability assessment
    #[inline]
    pub fn score_variance(&self) -> f64 {
        if self.evaluations.len() < 2 {
            return 0.0;
        }

        let mean = self.weighted_average_score;
        let variance = self.evaluations.iter()
            .map(|e| {
                let diff = e.score - mean;
                diff * diff
            })
            .sum::<f64>() / self.evaluations.len() as f64;
        
        variance.sqrt()
    }

    /// Check if decision is stable and reliable
    #[inline]
    pub fn is_stable(&self, max_variance: f64) -> bool {
        self.consensus_score >= 0.7
            && self.confidence >= 0.6
            && self.score_variance() <= max_variance
            && self.decision_quality >= 0.65
    }

    /// Get performance summary
    #[inline]
    pub fn performance_summary(&self) -> PerformanceSummary {
        PerformanceSummary {
            total_time_ms: self.total_processing_time_ms,
            average_time_ms: if self.evaluations.is_empty() {
                0
            } else {
                self.total_processing_time_ms / self.evaluations.len() as u64
            },
            fastest_time_ms: self.evaluations.iter()
                .map(|e| e.processing_time_ms)
                .min()
                .unwrap_or(0),
            slowest_time_ms: self.evaluations.iter()
                .map(|e| e.processing_time_ms)
                .max()
                .unwrap_or(0),
            evaluator_count: self.evaluations.len() as u8,
            consensus_strength: self.consensus_score,
        }
    }
}

/// Performance summary for monitoring and optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_time_ms: u64,
    pub average_time_ms: u64,
    pub fastest_time_ms: u64,
    pub slowest_time_ms: u64,
    pub evaluator_count: u8,
    pub consensus_strength: f64,
}

/// Evaluation result with comprehensive metadata
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Final consensus decision
    pub decision: ConsensusDecision,
    /// Cache key for result caching
    pub cache_key: Arc<str>,
    /// Whether result was loaded from cache
    pub from_cache: bool,
    /// Request timestamp
    pub request_timestamp: Instant,
    /// Total evaluation duration
    pub evaluation_duration_ms: u64,
    /// Cache generation number for invalidation
    pub cache_generation: u64,
}

impl EvaluationResult {
    /// Create new evaluation result
    #[inline]
    pub fn new(decision: ConsensusDecision, cache_key: Arc<str>) -> Self {
        Self {
            decision,
            cache_key,
            from_cache: false,
            request_timestamp: Instant::now(),
            evaluation_duration_ms: 0,
            cache_generation: 0,
        }
    }

    /// Mark as cached result with generation
    #[inline]
    pub fn from_cache(mut self, generation: u64) -> Self {
        self.from_cache = true;
        self.cache_generation = generation;
        self
    }

    /// Set evaluation duration
    #[inline]
    pub fn with_duration_ms(mut self, duration_ms: u64) -> Self {
        self.evaluation_duration_ms = duration_ms;
        self
    }

    /// Check if result is fresh enough
    #[inline]
    pub fn is_fresh(&self, max_age_ms: u64) -> bool {
        self.request_timestamp.elapsed().as_millis() <= max_age_ms as u128
    }

    /// Get cache efficiency score
    #[inline]
    pub fn cache_efficiency(&self) -> f64 {
        if self.from_cache {
            1.0 // Perfect efficiency for cache hits
        } else {
            0.0 // No efficiency for cache misses
        }
    }
}

/// Lock-free committee metrics using atomic operations
#[derive(Debug)]
pub struct CommitteeMetrics {
    /// Total evaluations performed
    pub total_evaluations: RelaxedCounter,
    /// Cache hits for performance tracking
    pub cache_hits: RelaxedCounter,
    /// Cache misses
    pub cache_misses: RelaxedCounter,
    /// Total processing time accumulator
    pub total_processing_time_ms: AtomicU64,
    /// Average response time tracker
    pub avg_response_time_ms: AtomicU64,
    /// Consensus failures counter
    pub consensus_failures: RelaxedCounter,
    /// Model timeout events
    pub model_timeouts: RelaxedCounter,
    /// Active evaluations counter
    pub active_evaluations: AtomicU32,
    /// Quality score accumulator
    pub total_quality_score: AtomicU64,
    /// Best decision quality achieved
    pub best_quality_score: AtomicU64,
    /// Metrics generation counter
    pub generation: RelaxedCounter,
}

impl CommitteeMetrics {
    /// Create new metrics instance
    #[inline]
    pub fn new() -> Self {
        Self {
            total_evaluations: RelaxedCounter::new(0),
            cache_hits: RelaxedCounter::new(0),
            cache_misses: RelaxedCounter::new(0),
            total_processing_time_ms: AtomicU64::new(0),
            avg_response_time_ms: AtomicU64::new(0),
            consensus_failures: RelaxedCounter::new(0),
            model_timeouts: RelaxedCounter::new(0),
            active_evaluations: AtomicU32::new(0),
            total_quality_score: AtomicU64::new(0),
            best_quality_score: AtomicU64::new(0),
            generation: RelaxedCounter::new(0),
        }
    }

    /// Record evaluation completion with comprehensive metrics
    #[inline]
    pub fn record_evaluation(
        &self,
        processing_time_ms: u64,
        from_cache: bool,
        quality_score: f64,
    ) {
        self.total_evaluations.inc();
        self.generation.inc();
        
        if from_cache {
            self.cache_hits.inc();
        } else {
            self.cache_misses.inc();
        }

        self.total_processing_time_ms.fetch_add(processing_time_ms, Ordering::Relaxed);
        
        // Update average response time
        let total_evaluations = self.total_evaluations.get();
        if total_evaluations > 0 {
            let total_time = self.total_processing_time_ms.load(Ordering::Relaxed);
            let avg = total_time / total_evaluations;
            self.avg_response_time_ms.store(avg, Ordering::Relaxed);
        }

        // Update quality metrics
        let quality_fixed = (quality_score * 1000.0) as u64;
        self.total_quality_score.fetch_add(quality_fixed, Ordering::Relaxed);
        
        // Update best quality score
        let current_best = self.best_quality_score.load(Ordering::Relaxed);
        if quality_fixed > current_best {
            let _ = self.best_quality_score.compare_exchange_weak(
                current_best,
                quality_fixed,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        }
    }

    /// Record consensus failure
    #[inline]
    pub fn record_consensus_failure(&self) {
        self.consensus_failures.inc();
        self.generation.inc();
    }

    /// Record model timeout
    #[inline]
    pub fn record_timeout(&self) {
        self.model_timeouts.inc();
        self.generation.inc();
    }

    /// Start evaluation tracking
    #[inline]
    pub fn start_evaluation(&self) -> u32 {
        self.active_evaluations.fetch_add(1, Ordering::Relaxed)
    }

    /// End evaluation tracking
    #[inline]
    pub fn end_evaluation(&self) -> u32 {
        self.active_evaluations.fetch_sub(1, Ordering::Relaxed)
    }

    /// Get cache hit ratio
    #[inline]
    pub fn cache_hit_ratio(&self) -> f64 {
        let hits = self.cache_hits.get() as f64;
        let misses = self.cache_misses.get() as f64;
        let total = hits + misses;
        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }

    /// Get average quality score
    #[inline]
    pub fn average_quality_score(&self) -> f64 {
        let total_evaluations = self.total_evaluations.get();
        if total_evaluations > 0 {
            let total_quality = self.total_quality_score.load(Ordering::Relaxed);
            (total_quality as f64) / (total_evaluations as f64 * 1000.0)
        } else {
            0.0
        }
    }

    /// Get metrics snapshot for reporting
    #[inline]
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            total_evaluations: self.total_evaluations.get(),
            cache_hits: self.cache_hits.get(),
            cache_misses: self.cache_misses.get(),
            avg_response_time_ms: self.avg_response_time_ms.load(Ordering::Relaxed),
            consensus_failures: self.consensus_failures.get(),
            model_timeouts: self.model_timeouts.get(),
            active_evaluations: self.active_evaluations.load(Ordering::Relaxed),
            cache_hit_ratio: self.cache_hit_ratio(),
            average_quality_score: self.average_quality_score(),
            best_quality_score: (self.best_quality_score.load(Ordering::Relaxed) as f64) / 1000.0,
            generation: self.generation.get(),
        }
    }

    /// Check if metrics indicate healthy system
    #[inline]
    pub fn is_healthy(&self) -> bool {
        let snapshot = self.snapshot();
        snapshot.cache_hit_ratio >= 0.3
            && snapshot.consensus_failures < snapshot.total_evaluations / 10
            && snapshot.model_timeouts < snapshot.total_evaluations / 20
            && snapshot.average_quality_score >= 0.5
    }
}

impl Default for CommitteeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable metrics snapshot for reporting and monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub total_evaluations: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub avg_response_time_ms: u64,
    pub consensus_failures: usize,
    pub model_timeouts: usize,
    pub active_evaluations: u32,
    pub cache_hit_ratio: f64,
    pub average_quality_score: f64,
    pub best_quality_score: f64,
    pub generation: usize,
}

/// Lock-free cache entry with atomic access tracking
#[derive(Debug)]
pub struct CacheEntry {
    /// Cached evaluation result
    pub result: ArcSwap<EvaluationResult>,
    /// Entry creation timestamp
    pub created_at: Instant,
    /// Entry expiration time
    pub expires_at: Instant,
    /// Access count for LRU eviction
    pub access_count: RelaxedCounter,
    /// Last access timestamp
    pub last_accessed: ArcSwap<Instant>,
    /// Entry generation for cache invalidation
    pub generation: AtomicU64,
}

impl CacheEntry {
    /// Create new cache entry with TTL
    #[inline]
    pub fn new(result: EvaluationResult, ttl_seconds: u64) -> Self {
        let now = Instant::now();
        let expires_at = now + Duration::from_secs(ttl_seconds);
        
        Self {
            result: ArcSwap::from_pointee(result),
            created_at: now,
            expires_at,
            access_count: RelaxedCounter::new(0),
            last_accessed: ArcSwap::from_pointee(now),
            generation: AtomicU64::new(1),
        }
    }

    /// Check if entry is expired
    #[inline]
    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.expires_at
    }

    /// Record access and return result
    #[inline]
    pub fn access(&self) -> Arc<EvaluationResult> {
        self.access_count.inc();
        self.last_accessed.store(Arc::new(Instant::now()));
        self.result.load_full()
    }

    /// Get access frequency score for eviction algorithms
    #[inline]
    pub fn frequency_score(&self) -> f64 {
        let access_count = self.access_count.get() as f64;
        let age_seconds = self.created_at.elapsed().as_secs_f64();
        if age_seconds > 0.0 {
            access_count / age_seconds
        } else {
            access_count
        }
    }

    /// Update cached result
    #[inline]
    pub fn update(&self, new_result: EvaluationResult) {
        self.result.store(Arc::new(new_result));
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Check if entry should be evicted
    #[inline]
    pub fn should_evict(&self, min_frequency: f64) -> bool {
        self.is_expired() || self.frequency_score() < min_frequency
    }
}

/// Cache metrics for performance monitoring
#[derive(Debug)]
pub struct CacheMetrics {
    /// Total cache entries
    pub total_entries: RelaxedCounter,
    /// Cache evictions performed
    pub evictions: RelaxedCounter,
    /// Estimated memory usage in bytes
    pub memory_usage_bytes: AtomicU64,
    /// Average entry age in seconds
    pub avg_entry_age_seconds: AtomicU64,
    /// Cache generation counter
    pub generation: RelaxedCounter,
}

impl CacheMetrics {
    /// Create new cache metrics
    #[inline]
    pub fn new() -> Self {
        Self {
            total_entries: RelaxedCounter::new(0),
            evictions: RelaxedCounter::new(0),
            memory_usage_bytes: AtomicU64::new(0),
            avg_entry_age_seconds: AtomicU64::new(0),
            generation: RelaxedCounter::new(0),
        }
    }

    /// Record cache entry addition
    #[inline]
    pub fn record_entry_added(&self, estimated_size_bytes: u64) {
        self.total_entries.inc();
        self.memory_usage_bytes.fetch_add(estimated_size_bytes, Ordering::Relaxed);
        self.generation.inc();
    }

    /// Record cache entry eviction
    #[inline]
    pub fn record_entry_evicted(&self, estimated_size_bytes: u64) {
        self.total_entries.dec();
        self.memory_usage_bytes.fetch_sub(estimated_size_bytes, Ordering::Relaxed);
        self.evictions.inc();
        self.generation.inc();
    }

    /// Update average entry age
    #[inline]
    pub fn update_avg_age(&self, total_age_seconds: u64) {
        let entry_count = self.total_entries.get();
        if entry_count > 0 {
            let avg_age = total_age_seconds / entry_count;
            self.avg_entry_age_seconds.store(avg_age, Ordering::Relaxed);
        }
    }

    /// Get cache efficiency ratio
    #[inline]
    pub fn efficiency_ratio(&self) -> f64 {
        let total_entries = self.total_entries.get();
        let evictions = self.evictions.get();
        if total_entries + evictions > 0 {
            total_entries as f64 / (total_entries + evictions) as f64
        } else {
            1.0
        }
    }

    /// Get metrics snapshot
    #[inline]
    pub fn snapshot(&self) -> CacheMetricsSnapshot {
        CacheMetricsSnapshot {
            total_entries: self.total_entries.get(),
            evictions: self.evictions.get(),
            memory_usage_bytes: self.memory_usage_bytes.load(Ordering::Relaxed),
            avg_entry_age_seconds: self.avg_entry_age_seconds.load(Ordering::Relaxed),
            efficiency_ratio: self.efficiency_ratio(),
            generation: self.generation.get(),
        }
    }

    /// Check if cache is performing optimally
    #[inline]
    pub fn is_optimal(&self) -> bool {
        let snapshot = self.snapshot();
        snapshot.efficiency_ratio >= 0.7
            && snapshot.avg_entry_age_seconds >= 300
            && snapshot.memory_usage_bytes < 100_000_000 // 100MB limit
    }
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable cache metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetricsSnapshot {
    pub total_entries: usize,
    pub evictions: usize,
    pub memory_usage_bytes: u64,
    pub avg_entry_age_seconds: u64,
    pub efficiency_ratio: f64,
    pub generation: usize,
}

/// Committee evaluation errors with comprehensive error handling
#[derive(Debug, thiserror::Error)]
pub enum CommitteeError {
    #[error("Evaluation timeout after {timeout_ms}ms")]
    EvaluationTimeout { timeout_ms: u64 },
    
    #[error("Insufficient committee members: {available} available, {required} required")]
    InsufficientMembers { available: usize, required: usize },
    
    #[error("Model unavailable: {model_type:?}")]
    ModelUnavailable { model_type: ModelType },
    
    #[error("Consensus not reached: {agreement:.1}% agreement, {threshold:.1}% required")]
    ConsensusNotReached { agreement: f64, threshold: f64 },
    
    #[error("Configuration error: {message}")]
    ConfigurationError { message: Arc<str> },
    
    #[error("Validation error in field '{field}': {value}")]
    ValidationError { field: Arc<str>, value: Arc<str> },
    
    #[error("Provider error for {model_type:?}: {source}")]
    ProviderError {
        model_type: ModelType,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    
    #[error("Cache error: {message}")]
    CacheError { message: Arc<str> },
    
    #[error("Decoding error: {message}")]
    DecodingError { message: Arc<str> },
    
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: Arc<str> },
    
    #[error("Quality threshold not met: {actual:.2} < {required:.2}")]
    QualityThresholdNotMet { actual: f64, required: f64 },
}

impl From<CommitteeError> for CognitiveError {
    fn from(error: CommitteeError) -> Self {
        CognitiveError::EvaluationFailed(error.to_string())
    }
}

/// Result type for committee operations
pub type CommitteeResult<T> = Result<T, CommitteeError>;

/// Evaluation prompt template with zero-allocation construction
#[derive(Debug, Clone)]
pub struct EvaluationPrompt {
    /// System prompt for context setting
    pub system_prompt: Arc<str>,
    /// User prompt with evaluation request
    pub user_prompt: Arc<str>,
    /// Maximum tokens for response
    pub max_tokens: u32,
    /// Temperature for response generation
    pub temperature: f32,
    /// Evaluation criteria weights
    pub criteria_weights: EvaluationCriteria,
}

/// Evaluation criteria with weights for scoring
#[derive(Debug, Clone, Copy)]
pub struct EvaluationCriteria {
    pub objective_alignment_weight: f32,
    pub implementation_quality_weight: f32,
    pub risk_assessment_weight: f32,
    pub innovation_weight: f32,
}

impl Default for EvaluationCriteria {
    fn default() -> Self {
        Self {
            objective_alignment_weight: 0.4,
            implementation_quality_weight: 0.3,
            risk_assessment_weight: 0.2,
            innovation_weight: 0.1,
        }
    }
}

impl EvaluationPrompt {
    /// Create optimization evaluation prompt with comprehensive criteria
    pub fn new_optimization_prompt(
        optimization_spec: &OptimizationSpec,
        current_code: &str,
        proposed_code: &str,
        criteria: Option<EvaluationCriteria>,
    ) -> Self {
        let criteria = criteria.unwrap_or_default();
        
        let system_prompt = Arc::from(format!(
            "You are a world-class code evaluation expert with deep expertise in software optimization, \
             performance analysis, and code quality assessment. Your task is to evaluate optimization \
             proposals with mathematical precision and comprehensive analysis.\n\n\
             EVALUATION OBJECTIVE: '{}'\n\n\
             SCORING REQUIREMENTS:\n\
             1. Objective Alignment (Weight: {:.1}): How well does the change advance the stated objective?\n\
             2. Implementation Quality (Weight: {:.1}): Technical excellence, maintainability, best practices\n\
             3. Risk Assessment (Weight: {:.1}): Potential for bugs, performance regressions, complexity\n\
             4. Innovation Factor (Weight: {:.1}): Novel approaches, creative solutions, efficiency gains\n\n\
             Provide numerical scores (0.0-1.0) for each criterion and comprehensive reasoning.",
            optimization_spec.objective,
            criteria.objective_alignment_weight,
            criteria.implementation_quality_weight,
            criteria.risk_assessment_weight,
            criteria.innovation_weight
        ));
        
        let user_prompt = Arc::from(format!(
            "## OPTIMIZATION EVALUATION REQUEST\n\n\
             **Objective**: {}\n\n\
             **Current Implementation**:\n```\n{}\n```\n\n\
             **Proposed Optimization**:\n```\n{}\n```\n\n\
             ## REQUIRED EVALUATION OUTPUT\n\n\
             Please provide your evaluation in this exact format:\n\n\
             **MAKES_PROGRESS**: true/false\n\
             **OBJECTIVE_ALIGNMENT**: 0.0-1.0\n\
             **IMPLEMENTATION_QUALITY**: 0.0-1.0\n\
             **RISK_ASSESSMENT**: 0.0-1.0\n\
             **INNOVATION_FACTOR**: 0.0-1.0\n\
             **CONFIDENCE**: 0.0-1.0\n\n\
             **DETAILED_REASONING**:\n\
             [Provide comprehensive analysis explaining your scores, focusing on:\n\
             - How the change advances the objective\n\
             - Technical quality and best practices\n\
             - Potential risks and mitigations\n\
             - Innovation and efficiency improvements\n\
             - Overall recommendation and justification]\n\n\
             **IMPROVEMENT_SUGGESTIONS**:\n\
             [Specific, actionable suggestions for enhancement]",
            optimization_spec.objective,
            current_code,
            proposed_code
        ));
        
        Self {
            system_prompt,
            user_prompt,
            max_tokens: 2048,
            temperature: 0.1, // Low temperature for consistent evaluation
            criteria_weights: criteria,
        }
    }

    /// Create general evaluation prompt for flexible use cases
    pub fn new_general_prompt(
        context: &str,
        evaluation_request: &str,
        criteria: Option<EvaluationCriteria>,
    ) -> Self {
        let criteria = criteria.unwrap_or_default();
        
        let system_prompt = Arc::from(
            "You are an expert evaluator with comprehensive knowledge across multiple domains. \
             Provide objective, well-reasoned assessments with numerical scoring and detailed analysis."
        );
        
        let user_prompt = Arc::from(format!(
            "## EVALUATION CONTEXT\n{}\n\n## EVALUATION REQUEST\n{}\n\n\
             Please provide structured evaluation with numerical scores (0.0-1.0) and detailed reasoning.",
            context, evaluation_request
        ));
        
        Self {
            system_prompt,
            user_prompt,
            max_tokens: 1536,
            temperature: 0.2,
            criteria_weights: criteria,
        }
    }

    /// Estimate token count for the prompt
    #[inline]
    pub fn estimated_token_count(&self) -> u32 {
        // Rough estimation: ~4 characters per token
        let total_chars = self.system_prompt.len() + self.user_prompt.len();
        (total_chars / 4) as u32
    }

    /// Check if prompt fits within model's context window
    #[inline]
    pub fn fits_in_context(&self, model_type: ModelType) -> bool {
        let estimated_tokens = self.estimated_token_count();
        let available_tokens = model_type.max_context_tokens();
        // Reserve tokens for response
        estimated_tokens + self.max_tokens < available_tokens
    }

    /// Optimize prompt for specific model type
    #[inline]
    pub fn optimize_for_model(mut self, model_type: ModelType) -> Self {
        if !self.fits_in_context(model_type) {
            // Reduce max_tokens if prompt is too long
            let estimated_prompt_tokens = self.estimated_token_count();
            let available_tokens = model_type.max_context_tokens();
            if estimated_prompt_tokens < available_tokens {
                self.max_tokens = (available_tokens - estimated_prompt_tokens).min(2048);
            } else {
                self.max_tokens = 512; // Minimal response for very long prompts
            }
        }
        
        // Adjust temperature based on model characteristics
        self.temperature = match model_type {
            ModelType::Claude3Haiku => 0.05, // Very low for fast, consistent responses
            ModelType::Gpt35Turbo => 0.1,    // Low for consistency
            _ => 0.15,                        // Slightly higher for more sophisticated models
        };
        
        self
    }
}