//! Core data structures and types for the LLM committee evaluation system
//! 
//! This module defines all the fundamental types used throughout the committee
//! evaluation system, including model configurations, evaluation results,
//! consensus decisions, and performance metrics.

use crate::cognitive::types::{CognitiveError, OptimizationSpec, ImpactFactors};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Model type for LLM evaluation with performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ModelType {
    /// GPT-3.5 Turbo - Fast and cost-effective
    Gpt35Turbo,
    /// GPT-4 - High quality reasoning
    Gpt4,
    /// GPT-4O - Optimized for speed and quality
    Gpt4O,
    /// Claude 3 Opus - Highest capability
    Claude3Opus,
    /// Claude 3 Sonnet - Balanced performance
    Claude3Sonnet,
    /// Claude 3 Haiku - Fastest response
    Claude3Haiku,
}

impl ModelType {
    /// Get the canonical display name for API calls
    #[inline(always)]
    pub fn display_name(&self) -> &'static str {
        match self {
            ModelType::Gpt35Turbo => "gpt-3.5-turbo",
            ModelType::Gpt4 => "gpt-4",
            ModelType::Gpt4O => "gpt-4o",
            ModelType::Claude3Opus => "claude-3-opus",
            ModelType::Claude3Sonnet => "claude-3-sonnet",
            ModelType::Claude3Haiku => "claude-3-haiku",
        }
    }
    
    /// Get expected response latency characteristics
    #[inline(always)]
    pub fn expected_latency(&self) -> Duration {
        match self {
            ModelType::Gpt35Turbo => Duration::from_secs(5),
            ModelType::Gpt4 => Duration::from_secs(20),
            ModelType::Gpt4O => Duration::from_secs(8),
            ModelType::Claude3Opus => Duration::from_secs(25),
            ModelType::Claude3Sonnet => Duration::from_secs(12),
            ModelType::Claude3Haiku => Duration::from_secs(3),
        }
    }
    
    /// Get relative cost factor for budget planning
    #[inline(always)]
    pub fn cost_factor(&self) -> f64 {
        match self {
            ModelType::Gpt35Turbo => 1.0,
            ModelType::Gpt4 => 15.0,
            ModelType::Gpt4O => 3.5,
            ModelType::Claude3Opus => 20.0,
            ModelType::Claude3Sonnet => 8.0,
            ModelType::Claude3Haiku => 2.0,
        }
    }
    
    /// Get quality tier for evaluation weighting
    #[inline(always)]
    pub fn quality_tier(&self) -> QualityTier {
        match self {
            ModelType::Gpt35Turbo => QualityTier::Standard,
            ModelType::Gpt4 => QualityTier::Premium,
            ModelType::Gpt4O => QualityTier::High,
            ModelType::Claude3Opus => QualityTier::Premium,
            ModelType::Claude3Sonnet => QualityTier::High,
            ModelType::Claude3Haiku => QualityTier::Standard,
        }
    }
}

/// Quality tier classification for model capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityTier {
    /// Standard quality - good for basic evaluations
    Standard,
    /// High quality - reliable for most evaluations
    High,
    /// Premium quality - best for critical evaluations
    Premium,
}

/// LLM provider wrapper with connection management
#[derive(Debug, Clone)]
pub struct Model {
    /// Model type identifier
    pub model_type: ModelType,
    /// Provider implementation for API calls
    pub provider: Arc<dyn crate::llm::LLMProvider>,
    /// Connection health status
    pub health_status: Arc<RwLock<HealthStatus>>,
    /// Performance metrics
    pub metrics: Arc<RwLock<ModelMetrics>>,
}

/// Health status tracking for model instances
#[derive(Debug, Clone)]
pub struct HealthStatus {
    /// Whether the model is currently available
    pub is_available: bool,
    /// Last successful request timestamp
    pub last_success: Option<Instant>,
    /// Current error rate (0.0 to 1.0)
    pub error_rate: f64,
    /// Average response time
    pub avg_response_time: Duration,
    /// Total requests made
    pub total_requests: u64,
    /// Failed requests count
    pub failed_requests: u64,
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self {
            is_available: true,
            last_success: None,
            error_rate: 0.0,
            avg_response_time: Duration::from_millis(100),
            total_requests: 0,
            failed_requests: 0,
        }
    }
}

/// Performance metrics for individual models
#[derive(Debug, Clone, Default)]
pub struct ModelMetrics {
    /// Total evaluation requests processed
    pub evaluations_completed: u64,
    /// Total evaluation time
    pub total_evaluation_time: Duration,
    /// Average evaluation score given
    pub average_score: f64,
    /// Agreement rate with committee majority
    pub agreement_rate: f64,
    /// Quality consistency score
    pub consistency_score: f64,
}

/// Configuration for committee evaluation system
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// List of models to include in committee
    pub models: Vec<ModelType>,
    /// Maximum time to wait for all evaluations
    pub timeout: Duration,
    /// Minimum agreement level required for consensus (0.0 to 1.0)
    pub consensus_threshold: f64,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            models: vec![
                ModelType::Gpt4O,
                ModelType::Claude3Sonnet,
                ModelType::Claude3Haiku,
            ],
            timeout: Duration::from_secs(30),
            consensus_threshold: 0.7,
        }
    }
}

/// Individual committee member's evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitteeEvaluation {
    /// Unique identifier for the evaluating agent
    pub agent_id: String,
    /// Model type used for this evaluation
    pub model_type: ModelType,
    /// Whether the optimization makes progress toward objectives
    pub makes_progress: bool,
    /// How well the optimization aligns with stated objectives (0.0 to 1.0)
    pub objective_alignment: f64,
    /// Quality of the implementation approach (0.0 to 1.0)
    pub implementation_quality: f64,
    /// Risk assessment of the proposed change (0.0 to 1.0, higher = riskier)
    pub risk_assessment: f64,
    /// Detailed reasoning behind the evaluation
    pub reasoning: String,
    /// Specific suggestions for improvement
    pub suggested_improvements: Vec<String>,
    /// Time taken to complete evaluation
    pub evaluation_time: Duration,
    /// Confidence in this evaluation (0.0 to 1.0)
    pub confidence: f64,
}

/// Final consensus decision from the committee
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusDecision {
    /// Whether the committee believes optimization makes progress
    pub makes_progress: bool,
    /// Confidence level in the decision (0.0 to 1.0)
    pub confidence: f64,
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f64,
    /// Aggregated improvement suggestions
    pub improvement_suggestions: Vec<String>,
    /// Dissenting opinions from minority committee members
    pub dissenting_opinions: Vec<String>,
}

/// Complete evaluation result with metadata
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// The final consensus decision
    pub decision: ConsensusDecision,
    /// All individual committee evaluations
    pub individual_evaluations: Vec<CommitteeEvaluation>,
    /// Performance metrics for this evaluation
    pub metrics: EvaluationMetrics,
    /// Cache key for future reference
    pub cache_key: String,
    /// Total time taken for evaluation
    pub total_time: Duration,
}

/// Performance metrics for a complete evaluation session
#[derive(Debug, Clone, Default)]
pub struct EvaluationMetrics {
    /// Number of committee members that participated
    pub participants: usize,
    /// Number of committee members that agreed with final decision
    pub consensus_count: usize,
    /// Average response time across all participants
    pub average_response_time: Duration,
    /// Standard deviation of scores
    pub score_variance: f64,
    /// Quality of reasoning provided
    pub reasoning_quality: f64,
    /// Whether evaluation completed within timeout
    pub completed_on_time: bool,
}

/// Committee-wide performance metrics
#[derive(Debug, Clone, Default)]
pub struct CommitteeMetrics {
    /// Total evaluations completed
    pub total_evaluations: u64,
    /// Average consensus confidence
    pub average_confidence: f64,
    /// Cache hit rate for duplicate evaluations
    pub cache_hit_rate: f64,
    /// Average evaluation time
    pub average_evaluation_time: Duration,
    /// Success rate (non-error evaluations)
    pub success_rate: f64,
    /// Model reliability scores
    pub model_reliability: HashMap<ModelType, f64>,
}

/// Cache entry for storing evaluation results
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The cached evaluation result
    pub result: EvaluationResult,
    /// When this entry was created
    pub created_at: Instant,
    /// Number of times this cache entry has been accessed
    pub access_count: u64,
    /// Last time this entry was accessed
    pub last_accessed: Instant,
}

/// Cache performance metrics
#[derive(Debug, Clone, Default)]
pub struct CacheMetrics {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Number of entries currently in cache
    pub entry_count: usize,
    /// Total memory usage estimate
    pub memory_usage_bytes: usize,
    /// Average age of cache entries
    pub average_entry_age: Duration,
}

impl CacheMetrics {
    /// Calculate cache hit rate as percentage
    #[inline(always)]
    pub fn hit_rate(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f64 / (self.hits + self.misses) as f64
        }
    }
    
    /// Check if cache is performing well
    #[inline(always)]
    pub fn is_performing_well(&self) -> bool {
        self.hit_rate() > 0.3 && self.entry_count < 1000
    }
}

/// Error types specific to committee evaluation
#[derive(Debug, thiserror::Error)]
pub enum CommitteeError {
    #[error("Evaluation timeout after {timeout:?}")]
    EvaluationTimeout { timeout: Duration },
    
    #[error("Insufficient committee members: {available} available, {required} required")]
    InsufficientMembers { available: usize, required: usize },
    
    #[error("Model unavailable: {model_type:?}")]
    ModelUnavailable { model_type: ModelType },
    
    #[error("Consensus not reached: {agreement:.1}% agreement, {threshold:.1}% required")]
    ConsensusNotReached { agreement: f64, threshold: f64 },
    
    #[error("Invalid configuration: {reason}")]
    InvalidConfiguration { reason: String },
    
    #[error("Provider error for {model_type:?}: {source}")]
    ProviderError {
        model_type: ModelType,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    
    #[error("Cache error: {message}")]
    CacheError { message: String },
}

impl From<CommitteeError> for CognitiveError {
    fn from(error: CommitteeError) -> Self {
        CognitiveError::EvaluationFailed(error.to_string())
    }
}

/// Result type for committee operations
pub type CommitteeResult<T> = Result<T, CommitteeError>;

/// Evaluation prompt template for LLM requests
#[derive(Debug, Clone)]
pub struct EvaluationPrompt {
    /// System prompt setting up the evaluation context
    pub system_prompt: String,
    /// User prompt with specific evaluation request
    pub user_prompt: String,
    /// Maximum tokens for response
    pub max_tokens: u32,
    /// Temperature for response generation
    pub temperature: f32,
}

impl EvaluationPrompt {
    /// Create a new evaluation prompt for optimization assessment
    pub fn new_optimization_prompt(
        optimization_spec: &OptimizationSpec,
        current_code: &str,
        proposed_code: &str,
    ) -> Self {
        let system_prompt = format!(
            "You are an expert code reviewer evaluating optimization proposals. \
             Analyze the proposed changes against the objective: '{}'. \
             Provide numerical scores (0.0-1.0) for alignment, quality, and risk. \
             Be objective and thorough in your assessment.",
            optimization_spec.objective
        );
        
        let user_prompt = format!(
            "Objective: {}\n\n\
             Current Code:\n```\n{}\n```\n\n\
             Proposed Code:\n```\n{}\n```\n\n\
             Please evaluate this optimization and respond with:\n\
             1. Makes Progress: true/false\n\
             2. Objective Alignment: 0.0-1.0\n\
             3. Implementation Quality: 0.0-1.0\n\
             4. Risk Assessment: 0.0-1.0\n\
             5. Reasoning: Detailed explanation\n\
             6. Suggested Improvements: List of specific suggestions",
            optimization_spec.objective,
            current_code,
            proposed_code
        );
        
        Self {
            system_prompt,
            user_prompt,
            max_tokens: 1000,
            temperature: 0.1, // Low temperature for consistent evaluation
        }
    }
}