//! LLM Committee-based Evaluation System
//!
//! This module provides a sophisticated committee-based evaluation system that uses
//! multiple LLM agents to score and analyze code optimizations against user objectives.
//! The system employs prompt-based evaluation with rubric scoring to ensure
//! objective and comprehensive assessment.
//!
//! ## Architecture
//!
//! The committee system is organized into specialized modules:
//!
//! - **committee_types**: Core data structures and type definitions
//! - **committee_evaluators**: Individual LLM evaluator implementations
//! - **committee_consensus**: Consensus building and decision aggregation
//! - **committee_orchestrator**: Main coordination and workflow orchestration
//!
//! ## Usage
//!
//! ```rust
//! use crate::cognitive::committee::{CommitteeEvaluator, EvaluationConfig};
//!
//! // Create committee with multiple model types
//! let committee = CommitteeEvaluator::new(EvaluationConfig {
//!     models: vec![ModelType::Gpt4O, ModelType::Claude3Sonnet],
//!     timeout: Duration::from_secs(30),
//!     consensus_threshold: 0.7,
//! }).await?;
//!
//! // Evaluate optimization against objectives
//! let decision = committee.evaluate_optimization(
//!     &optimization_spec,
//!     &current_state,
//!     &proposed_state,
//! ).await?;
//! ```
//!
//! ## Performance Characteristics
//!
//! - **Parallel Evaluation**: All committee members evaluate simultaneously
//! - **Zero-allocation Consensus**: Pre-allocated buffers for decision aggregation
//! - **Lock-free Metrics**: Atomic counters for performance statistics
//! - **Efficient Caching**: SHA256-based evaluation result caching
//!
//! ## Production Features
//!
//! - **Comprehensive Error Handling**: Graceful handling of LLM provider failures
//! - **Timeout Management**: Configurable timeouts for evaluation operations
//! - **Quality Assurance**: Multi-dimensional scoring with dissenting opinion tracking
//! - **Performance Monitoring**: Detailed metrics and evaluation statistics

// Re-export core types and functions for backward compatibility
pub use committee_consensus::{
    CommitteeConsensusEngine, ConsensusConfig, ConsensusDecision, ConsensusStatistics,
};
pub use committee_evaluators::{EvaluationSession, EvaluatorPool, LLMEvaluator};
pub use committee_orchestrator::{CommitteeCoordinator, CommitteeEvaluator, EvaluationWorkflow};
pub use committee_types::EvaluationPrompt;
pub use committee_types::{
    CacheEntry, CacheMetrics, CommitteeEvaluation, CommitteeMetrics,
    EvaluationConfig, EvaluationMetrics, EvaluationResult, Model, ModelType,
};

// Internal modules (not exposed in public API)
mod committee_consensus;
mod committee_evaluators;
mod committee_evaluators_extension;
mod committee_orchestrator;
mod committee_types;

// Re-export legacy types from committee_old.rs for backward compatibility
pub use crate::cognitive::committee_old::{CommitteeEvent, EvaluationCommittee};

/// High-level API for common committee operations
///
/// This provides convenient wrapper functions for the most common committee
/// evaluation operations, combining functionality from multiple modules.
pub mod api {
    use std::time::Duration;

    use super::*;
    use crate::cognitive::mcts::CodeState;
    use crate::cognitive::types::{CognitiveError, OptimizationSpec};

    /// Create a balanced committee with recommended model diversity
    ///
    /// # Arguments
    /// * `timeout` - Maximum time to wait for evaluations
    /// * `consensus_threshold` - Minimum agreement level for decisions
    ///
    /// # Returns
    /// * CommitteeEvaluator configured with optimal model selection
    pub async fn create_balanced_committee(
        timeout: Duration,
        consensus_threshold: f64,
    ) -> Result<CommitteeEvaluator, CognitiveError> {
        let config = EvaluationConfig {
            models: vec![
                ModelType::Gpt4O,         // Strong reasoning
                ModelType::Claude3Sonnet, // Balanced analysis
                ModelType::Claude3Haiku,  // Fast evaluation
            ],
            timeout,
            consensus_threshold,
        };

        CommitteeEvaluator::new(config).await
    }

    /// Quick evaluation for simple optimization decisions
    ///
    /// # Arguments
    /// * `optimization_spec` - The optimization to evaluate
    /// * `current_state` - Current code state
    /// * `proposed_state` - Proposed optimized state
    ///
    /// # Returns
    /// * Boolean decision indicating if optimization should proceed
    pub async fn quick_evaluate(
        optimization_spec: &OptimizationSpec,
        current_state: &CodeState,
        proposed_state: &CodeState,
    ) -> Result<bool, CognitiveError> {
        let committee = create_balanced_committee(
            Duration::from_secs(15),
            0.6, // Lower threshold for quick decisions
        )
        .await?;

        let decision = committee
            .evaluate_optimization(optimization_spec, current_state, proposed_state)
            .await?;

        Ok(decision.makes_progress && decision.confidence > 0.6)
    }

    /// Comprehensive evaluation with detailed analysis
    ///
    /// # Arguments
    /// * `optimization_spec` - The optimization to evaluate
    /// * `current_state` - Current code state
    /// * `proposed_state` - Proposed optimized state
    ///
    /// # Returns
    /// * Full ConsensusDecision with detailed analysis and suggestions
    pub async fn comprehensive_evaluate(
        optimization_spec: &OptimizationSpec,
        current_state: &CodeState,
        proposed_state: &CodeState,
    ) -> Result<ConsensusDecision, CognitiveError> {
        let committee = create_balanced_committee(
            Duration::from_secs(45),
            0.8, // Higher threshold for comprehensive analysis
        )
        .await?;

        committee
            .evaluate_optimization(optimization_spec, current_state, proposed_state)
            .await
    }
}

/// Constants for committee evaluation system
pub mod constants {
    use std::time::Duration;

    /// Default evaluation timeout
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

    /// Default consensus threshold
    pub const DEFAULT_CONSENSUS_THRESHOLD: f64 = 0.7;

    /// Maximum number of committee members
    pub const MAX_COMMITTEE_SIZE: usize = 7;

    /// Minimum number of committee members for valid evaluation
    pub const MIN_COMMITTEE_SIZE: usize = 3;

    /// Cache entry expiration time
    pub const CACHE_EXPIRY_HOURS: u64 = 24;

    /// Maximum cached evaluations
    pub const MAX_CACHE_ENTRIES: usize = 1000;

    /// Scoring weight for objective alignment
    pub const ALIGNMENT_WEIGHT: f64 = 0.5;

    /// Scoring weight for implementation quality
    pub const QUALITY_WEIGHT: f64 = 0.3;

    /// Scoring weight for risk assessment
    pub const RISK_WEIGHT: f64 = 0.2;
}

/// Utility functions for committee evaluation processing
pub mod utils {
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::cognitive::types::OptimizationSpec;

    /// Generate cache key for evaluation request
    ///
    /// # Arguments
    /// * `optimization_spec` - The optimization specification
    /// * `current_code` - Current code content
    /// * `proposed_code` - Proposed code content
    ///
    /// # Returns
    /// * SHA256 hash string for caching
    pub fn generate_cache_key(
        optimization_spec: &OptimizationSpec,
        current_code: &str,
        proposed_code: &str,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(optimization_spec.objective.as_bytes());
        hasher.update(current_code.as_bytes());
        hasher.update(proposed_code.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Validate committee configuration
    ///
    /// # Arguments
    /// * `config` - Configuration to validate
    ///
    /// # Returns
    /// * true if configuration is valid for production use
    pub fn validate_committee_config(config: &EvaluationConfig) -> bool {
        config.models.len() >= constants::MIN_COMMITTEE_SIZE
            && config.models.len() <= constants::MAX_COMMITTEE_SIZE
            && config.consensus_threshold >= 0.5
            && config.consensus_threshold <= 1.0
            && config.timeout_ms >= 5000
    }

    /// Calculate evaluation confidence score
    ///
    /// # Arguments
    /// * `agreement_ratio` - Ratio of agreeing committee members
    /// * `score_variance` - Variance in scoring across committee
    ///
    /// # Returns
    /// * Confidence score between 0.0 and 1.0
    pub fn calculate_confidence(agreement_ratio: f64, score_variance: f64) -> f64 {
        agreement_ratio * (1.0 / (1.0 + score_variance))
    }

    /// Extract key insights from committee evaluations
    ///
    /// # Arguments
    /// * `evaluations` - All committee member evaluations
    ///
    /// # Returns
    /// * Summary of key insights and patterns
    pub fn extract_key_insights(evaluations: &[CommitteeEvaluation]) -> Vec<String> {
        let mut insights = Vec::new();

        // Identify common themes in reasoning
        let reasoning_terms: Vec<_> = evaluations
            .iter()
            .flat_map(|e| {
                String::from_utf8_lossy(&e.reasoning)
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            })
            .filter(|word| word.len() > 4)
            .collect();

        // Find most frequent terms (simple implementation)
        let mut term_counts = std::collections::HashMap::new();
        for term in reasoning_terms {
            *term_counts.entry(term.to_lowercase()).or_insert(0) += 1;
        }

        let mut frequent_terms: Vec<_> = term_counts
            .into_iter()
            .filter(|(_, count)| *count >= 2)
            .collect();
        frequent_terms.sort_by_key(|(_, count)| std::cmp::Reverse(*count));

        if let Some((term, _)) = frequent_terms.first() {
            insights.push(format!("Common concern: {}", term));
        }

        insights
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_committee_config_validation() {
        let valid_config = EvaluationConfig {
            models: vec![
                ModelType::Gpt4O,
                ModelType::Claude3Sonnet,
                ModelType::Claude3Haiku,
            ],
            timeout: Duration::from_secs(30),
            consensus_threshold: 0.7,
        };

        assert!(utils::validate_committee_config(&valid_config));

        let invalid_config = EvaluationConfig {
            models: vec![ModelType::Gpt4O], // Too few models
            timeout: Duration::from_secs(30),
            consensus_threshold: 0.7,
        };

        assert!(!utils::validate_committee_config(&invalid_config));
    }

    #[test]
    fn test_confidence_calculation() {
        assert_eq!(utils::calculate_confidence(1.0, 0.0), 1.0);
        assert_eq!(utils::calculate_confidence(0.5, 0.0), 0.5);
        assert!(utils::calculate_confidence(1.0, 1.0) < 1.0);
    }

    #[test]
    fn test_cache_key_generation() {
        let spec = OptimizationSpec {
            objective: "Improve performance".to_string(),
            impact_factors: Default::default(),
        };

        let key1 = utils::generate_cache_key(&spec, "code1", "code2");
        let key2 = utils::generate_cache_key(&spec, "code1", "code2");
        let key3 = utils::generate_cache_key(&spec, "code1", "code3");

        assert_eq!(key1, key2); // Same inputs should generate same key
        assert_ne!(key1, key3); // Different inputs should generate different keys
        assert_eq!(key1.len(), 64); // SHA256 hex string length
    }
}

// NOTE: This modularized structure provides:
// 1. Clear separation of concerns with focused modules
// 2. Maintainable codebase with logical boundaries
// 3. Backward compatibility with existing code
// 4. Production-ready APIs with comprehensive error handling
// 5. Zero-allocation performance characteristics where possible
// 6. Comprehensive testing and validation utilities
// 7. Support for different evaluation workflows and use cases
