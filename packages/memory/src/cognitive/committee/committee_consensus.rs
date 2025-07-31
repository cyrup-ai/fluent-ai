//! Committee consensus engine for cognitive evaluations
//!
//! This module provides comprehensive consensus building capabilities for committee evaluations
//! using zero-allocation patterns and production-ready error handling.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::committee_types::{
    CommitteeError, CommitteeEvaluation, EvaluationMetrics, ModelType, QualityTier};

/// Convert ConsensusError to CommitteeError
impl From<ConsensusError> for CommitteeError {
    fn from(error: ConsensusError) -> Self {
        match error {
            ConsensusError::InsufficientEvaluations {
                available,
                required} => CommitteeError::InsufficientMembers {
                available,
                required},
            ConsensusError::ConsensusThresholdNotMet { achieved, required } => {
                CommitteeError::ConsensusNotReached {
                    agreement: achieved * 100.0,
                    threshold: required * 100.0}
            }
            ConsensusError::EvaluationTimeout { duration } => CommitteeError::EvaluationTimeout {
                timeout_ms: duration.as_millis() as u64},
            ConsensusError::ConflictingEvaluations { detail } => {
                CommitteeError::ConfigurationError { message: detail }
            }
            ConsensusError::InvalidConfiguration { detail } => {
                CommitteeError::ConfigurationError { message: detail }
            }
        }
    }
}

/// Consensus decision error types
#[derive(Debug, Clone, Error)]
pub enum ConsensusError {
    #[error("Insufficient evaluations: {available}/{required}")]
    InsufficientEvaluations { available: usize, required: usize },

    #[error("Consensus threshold not met: {achieved}/{required}")]
    ConsensusThresholdNotMet { achieved: f64, required: f64 },

    #[error("Conflicting evaluations: {detail}")]
    ConflictingEvaluations { detail: Arc<str> },

    #[error("Evaluation timeout: {duration:?}")]
    EvaluationTimeout { duration: Duration },

    #[error("Invalid consensus configuration: {detail}")]
    InvalidConfiguration { detail: Arc<str> }}

/// Consensus algorithm type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusAlgorithm {
    /// Simple majority voting
    Majority,
    /// Weighted by model quality tier
    WeightedByQuality,
    /// Byzantine fault tolerant consensus
    ByzantineFaultTolerant,
    /// Confidence-weighted consensus
    ConfidenceWeighted}

/// Consensus configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Minimum number of evaluations required
    pub min_evaluations: usize,
    /// Consensus threshold (0.0-1.0)
    pub threshold: f64,
    /// Algorithm to use for consensus
    pub algorithm: ConsensusAlgorithm,
    /// Maximum time to wait for consensus
    pub timeout: Duration,
    /// Quality tier weights
    pub quality_weights: HashMap<QualityTier, f64>}

impl Default for ConsensusConfig {
    fn default() -> Self {
        let mut quality_weights = HashMap::new();
        quality_weights.insert(QualityTier::Premium, 1.0);
        quality_weights.insert(QualityTier::Standard, 0.8);
        quality_weights.insert(QualityTier::Basic, 0.6);
        quality_weights.insert(QualityTier::Experimental, 0.4);

        Self {
            min_evaluations: 3,
            threshold: 0.7,
            algorithm: ConsensusAlgorithm::WeightedByQuality,
            timeout: Duration::from_secs(30),
            quality_weights}
    }
}

/// Consensus decision with comprehensive metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusDecision {
    /// Final decision score (0.0-1.0)
    pub decision_score: f64,
    /// Confidence in the decision (0.0-1.0)
    pub confidence: f64,
    /// Consensus strength (0.0-1.0)
    pub consensus_strength: f64,
    /// Individual evaluation scores
    pub evaluation_scores: Vec<f64>,
    /// Participating model types
    pub participating_models: Vec<ModelType>,
    /// Dissenting opinions
    pub dissenting_opinions: Vec<Arc<str>>,
    /// Consensus algorithm used
    pub algorithm_used: ConsensusAlgorithm,
    /// Time taken to reach consensus
    pub consensus_duration: Duration,
    /// Additional metadata
    pub metadata: HashMap<Arc<str>, Arc<str>>}

impl ConsensusDecision {
    /// Create a new consensus decision
    #[inline]
    pub fn new(decision_score: f64, confidence: f64) -> Self {
        Self {
            decision_score,
            confidence,
            consensus_strength: 0.0,
            evaluation_scores: Vec::new(),
            participating_models: Vec::new(),
            dissenting_opinions: Vec::new(),
            algorithm_used: ConsensusAlgorithm::Majority,
            consensus_duration: Duration::from_millis(0),
            metadata: HashMap::new()}
    }

    /// Check if the decision is positive (score > 0.5)
    #[inline]
    pub fn is_positive(&self) -> bool {
        self.decision_score > 0.5
    }

    /// Check if the decision has high confidence (> 0.8)
    #[inline]
    pub fn is_high_confidence(&self) -> bool {
        self.confidence > 0.8
    }

    /// Check if consensus is strong (> 0.7)
    #[inline]
    pub fn is_strong_consensus(&self) -> bool {
        self.consensus_strength > 0.7
    }

    /// Add metadata to the decision
    #[inline]
    pub fn with_metadata(mut self, key: impl Into<Arc<str>>, value: impl Into<Arc<str>>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get normalized consensus score (0.0-1.0)
    #[inline]
    pub fn normalized_consensus_score(&self) -> f64 {
        self.decision_score
    }
}

/// Committee consensus engine
#[derive(Debug)]
pub struct CommitteeConsensusEngine {
    config: ConsensusConfig,
    /// Evaluation metrics for performance tracking
    metrics: EvaluationMetrics,
    /// Start time for timeout tracking
    start_time: Instant}

impl CommitteeConsensusEngine {
    /// Create a new consensus engine
    #[inline]
    pub fn new(config: ConsensusConfig) -> Self {
        Self {
            config,
            metrics: EvaluationMetrics::default(),
            start_time: Instant::now()}
    }

    /// Create with default configuration
    #[inline]
    pub fn with_default_config() -> Self {
        Self::new(ConsensusConfig::default())
    }

    /// Calculate consensus from committee evaluations
    pub async fn calculate_consensus(
        &self,
        evaluations: &[CommitteeEvaluation],
    ) -> Result<ConsensusDecision, ConsensusError> {
        let start_time = Instant::now();

        // Validate input
        if evaluations.len() < self.config.min_evaluations {
            return Err(ConsensusError::InsufficientEvaluations {
                available: evaluations.len(),
                required: self.config.min_evaluations});
        }

        // Check timeout
        if start_time.duration_since(self.start_time) > self.config.timeout {
            return Err(ConsensusError::EvaluationTimeout {
                duration: self.config.timeout});
        }

        // Calculate consensus based on algorithm
        let decision = match self.config.algorithm {
            ConsensusAlgorithm::Majority => self.calculate_majority_consensus(evaluations)?,
            ConsensusAlgorithm::WeightedByQuality => {
                self.calculate_weighted_consensus(evaluations)?
            }
            ConsensusAlgorithm::ByzantineFaultTolerant => {
                self.calculate_bft_consensus(evaluations)?
            }
            ConsensusAlgorithm::ConfidenceWeighted => {
                self.calculate_confidence_weighted_consensus(evaluations)?
            }
        };

        // Validate consensus threshold
        if decision.consensus_strength < self.config.threshold {
            return Err(ConsensusError::ConsensusThresholdNotMet {
                achieved: decision.consensus_strength,
                required: self.config.threshold});
        }

        Ok(decision)
    }

    /// Calculate majority consensus
    fn calculate_majority_consensus(
        &self,
        evaluations: &[CommitteeEvaluation],
    ) -> Result<ConsensusDecision, ConsensusError> {
        let scores: Vec<f64> = evaluations.iter().map(|e| e.score).collect();
        let average_score = scores.iter().sum::<f64>() / scores.len() as f64;

        // Calculate consensus strength based on score variance
        let variance = scores
            .iter()
            .map(|s| (s - average_score).powi(2))
            .sum::<f64>()
            / scores.len() as f64;
        let consensus_strength = 1.0 - variance.sqrt();

        // Calculate confidence based on score distribution
        let confidence = self.calculate_confidence_from_scores(&scores);

        Ok(ConsensusDecision {
            decision_score: average_score,
            confidence,
            consensus_strength,
            evaluation_scores: scores,
            participating_models: evaluations.iter().map(|e| e.model.clone()).collect(),
            dissenting_opinions: self.identify_dissenting_opinions(evaluations, average_score),
            algorithm_used: ConsensusAlgorithm::Majority,
            consensus_duration: Instant::now().duration_since(self.start_time),
            metadata: HashMap::new()})
    }

    /// Calculate weighted consensus by quality tier
    fn calculate_weighted_consensus(
        &self,
        evaluations: &[CommitteeEvaluation],
    ) -> Result<ConsensusDecision, ConsensusError> {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        let mut scores = Vec::new();

        for evaluation in evaluations {
            let weight = self.get_quality_weight(&evaluation.model);
            weighted_sum += evaluation.score * weight;
            total_weight += weight;
            scores.push(evaluation.score);
        }

        let weighted_average = weighted_sum / total_weight;
        let consensus_strength = self.calculate_weighted_consensus_strength(evaluations);
        let confidence = self.calculate_confidence_from_scores(&scores);

        Ok(ConsensusDecision {
            decision_score: weighted_average,
            confidence,
            consensus_strength,
            evaluation_scores: scores,
            participating_models: evaluations.iter().map(|e| e.model.clone()).collect(),
            dissenting_opinions: self.identify_dissenting_opinions(evaluations, weighted_average),
            algorithm_used: ConsensusAlgorithm::WeightedByQuality,
            consensus_duration: Instant::now().duration_since(self.start_time),
            metadata: HashMap::new()})
    }

    /// Calculate Byzantine fault tolerant consensus
    fn calculate_bft_consensus(
        &self,
        evaluations: &[CommitteeEvaluation],
    ) -> Result<ConsensusDecision, ConsensusError> {
        // For BFT, we need at least 3f+1 evaluations to tolerate f faults
        let f = (evaluations.len() - 1) / 3;
        if evaluations.len() < 3 * f + 1 {
            return Err(ConsensusError::InsufficientEvaluations {
                available: evaluations.len(),
                required: 3 * f + 1});
        }

        // Sort scores and use median for fault tolerance
        let mut scores: Vec<f64> = evaluations.iter().map(|e| e.score).collect();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median_score = scores[scores.len() / 2];

        // Calculate consensus strength based on how many evaluations are close to median
        let close_to_median = scores
            .iter()
            .filter(|&&s| (s - median_score).abs() < 0.1)
            .count();
        let consensus_strength = close_to_median as f64 / scores.len() as f64;

        let confidence = self.calculate_confidence_from_scores(&scores);

        Ok(ConsensusDecision {
            decision_score: median_score,
            confidence,
            consensus_strength,
            evaluation_scores: scores,
            participating_models: evaluations.iter().map(|e| e.model.clone()).collect(),
            dissenting_opinions: self.identify_dissenting_opinions(evaluations, median_score),
            algorithm_used: ConsensusAlgorithm::ByzantineFaultTolerant,
            consensus_duration: Instant::now().duration_since(self.start_time),
            metadata: HashMap::new()})
    }

    /// Calculate confidence-weighted consensus
    fn calculate_confidence_weighted_consensus(
        &self,
        evaluations: &[CommitteeEvaluation],
    ) -> Result<ConsensusDecision, ConsensusError> {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        let mut scores = Vec::new();

        for evaluation in evaluations {
            let weight = evaluation.confidence;
            weighted_sum += evaluation.score * weight;
            total_weight += weight;
            scores.push(evaluation.score);
        }

        let weighted_average = weighted_sum / total_weight;
        let consensus_strength = self.calculate_confidence_consensus_strength(evaluations);
        let confidence = self.calculate_confidence_from_scores(&scores);

        Ok(ConsensusDecision {
            decision_score: weighted_average,
            confidence,
            consensus_strength,
            evaluation_scores: scores,
            participating_models: evaluations.iter().map(|e| e.model.clone()).collect(),
            dissenting_opinions: self.identify_dissenting_opinions(evaluations, weighted_average),
            algorithm_used: ConsensusAlgorithm::ConfidenceWeighted,
            consensus_duration: Instant::now().duration_since(self.start_time),
            metadata: HashMap::new()})
    }

    /// Get quality weight for a model type
    fn get_quality_weight(&self, model: &ModelType) -> f64 {
        let quality_tier = self.get_model_quality_tier(model);
        self.config
            .quality_weights
            .get(&quality_tier)
            .copied()
            .unwrap_or(1.0)
    }

    /// Get quality tier for a model type
    fn get_model_quality_tier(&self, model: &ModelType) -> QualityTier {
        match model {
            ModelType::Gpt4O | ModelType::Claude3Opus => QualityTier::Premium,
            ModelType::Gpt4Turbo | ModelType::Claude3Sonnet => QualityTier::Standard,
            ModelType::Gpt35Turbo | ModelType::Claude3Haiku => QualityTier::Basic,
            ModelType::GeminiPro
            | ModelType::Llama3
            | ModelType::Mixtral8x7B
            | ModelType::Llama270B => QualityTier::Experimental}
    }

    /// Calculate consensus strength for weighted consensus
    fn calculate_weighted_consensus_strength(&self, evaluations: &[CommitteeEvaluation]) -> f64 {
        let scores: Vec<f64> = evaluations.iter().map(|e| e.score).collect();
        let average = scores.iter().sum::<f64>() / scores.len() as f64;

        let variance =
            scores.iter().map(|s| (s - average).powi(2)).sum::<f64>() / scores.len() as f64;

        1.0 - variance.sqrt()
    }

    /// Calculate consensus strength for confidence-weighted consensus
    fn calculate_confidence_consensus_strength(&self, evaluations: &[CommitteeEvaluation]) -> f64 {
        let confidence_scores: Vec<f64> = evaluations.iter().map(|e| e.confidence).collect();
        let average_confidence =
            confidence_scores.iter().sum::<f64>() / confidence_scores.len() as f64;

        // Higher average confidence means stronger consensus
        average_confidence
    }

    /// Calculate confidence from score distribution
    fn calculate_confidence_from_scores(&self, scores: &[f64]) -> f64 {
        if scores.is_empty() {
            return 0.0;
        }

        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;

        // Lower variance means higher confidence
        1.0 - variance.sqrt().min(1.0)
    }

    /// Identify dissenting opinions
    fn identify_dissenting_opinions(
        &self,
        evaluations: &[CommitteeEvaluation],
        consensus_score: f64,
    ) -> Vec<Arc<str>> {
        let mut dissenting = Vec::new();

        for evaluation in evaluations {
            if (evaluation.score - consensus_score).abs() > 0.2 {
                dissenting.push(Arc::from(format!(
                    "Model {} dissents with score {:.2} vs consensus {:.2}",
                    evaluation.model.display_name(),
                    evaluation.score,
                    consensus_score
                )));
            }
        }

        dissenting
    }

    /// Get current configuration
    #[inline]
    pub fn config(&self) -> &ConsensusConfig {
        &self.config
    }

    /// Update configuration
    #[inline]
    pub fn update_config(&mut self, config: ConsensusConfig) {
        self.config = config;
    }

    /// Get evaluation metrics
    #[inline]
    pub fn metrics(&self) -> &EvaluationMetrics {
        &self.metrics
    }
}

/// Consensus result summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResultSummary {
    /// Final decision
    pub decision: ConsensusDecision,
    /// Evaluation count
    pub evaluation_count: usize,
    /// Consensus algorithm used
    pub algorithm: ConsensusAlgorithm,
    /// Time taken
    pub duration: Duration,
    /// Success indicator
    pub success: bool}

impl ConsensusResultSummary {
    /// Create from consensus decision
    #[inline]
    pub fn from_decision(decision: ConsensusDecision, evaluation_count: usize) -> Self {
        Self {
            algorithm: decision.algorithm_used.clone(),
            duration: decision.consensus_duration,
            success: decision.is_positive() && decision.is_high_confidence(),
            decision,
            evaluation_count}
    }
}
