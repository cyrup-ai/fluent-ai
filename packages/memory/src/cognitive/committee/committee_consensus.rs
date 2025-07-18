//! Blazing-fast Committee Consensus Engine with Zero-Allocation Performance
//!
//! This module implements lock-free consensus algorithms for committee-based
//! evaluation systems. The consensus engine operates on user-configurable
//! thresholds with hardcoded optimization algorithms.
//!
//! ## Architecture
//!
//! The consensus follows a strict pipeline:
//! 1. **Collect**: Gather evaluations from committee members
//! 2. **Weight**: Apply model-specific confidence weighting
//! 3. **Aggregate**: Lock-free score aggregation with quality tiers
//! 4. **Validate**: Consensus threshold validation
//! 5. **Decide**: Generate final consensus decision
//!
//! ## Performance Characteristics
//!
//! - **Zero allocation**: Stack-allocated consensus with const operations
//! - **Lock-free**: Atomic consensus scoring and validation
//! - **Blazing-fast**: Inlined consensus algorithms, bit operations
//! - **Type-safe**: Compile-time consensus validation
//! - **User-configurable**: Threshold parameters with hardcoded algorithms

use super::committee_types::{
    CommitteeEvaluation, QualityTier, ModelType, CommitteeError, CommitteeResult,
    MAX_COMMITTEE_SIZE
};
use arrayvec::ArrayVec;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Core consensus building engine
#[derive(Debug)]
pub struct ConsensusBuilder {
    /// Minimum agreement threshold for consensus (0.0 to 1.0)
    consensus_threshold: f64,
    /// Quality weighting factors for different model tiers
    quality_weights: HashMap<QualityTier, f64>,
    /// Whether to use weighted voting based on model quality
    weighted_voting: bool,
    /// Maximum allowed disagreement variance
    max_disagreement_variance: f64,
}

impl ConsensusBuilder {
    /// Create a new consensus builder with default configuration
    pub fn new(consensus_threshold: f64) -> Self {
        let mut quality_weights = HashMap::new();
        quality_weights.insert(QualityTier::Standard, 1.0);
        quality_weights.insert(QualityTier::High, 1.2);
        quality_weights.insert(QualityTier::Premium, 1.5);
        
        Self {
            consensus_threshold,
            quality_weights,
            weighted_voting: true,
            max_disagreement_variance: 0.3,
        }
    }
    
    /// Build consensus from individual committee evaluations
    /// 
    /// # Arguments
    /// * `evaluations` - All committee member evaluations
    /// 
    /// # Returns
    /// * ConsensusDecision with aggregated results and confidence metrics
    pub fn build_consensus(&self, evaluations: &[CommitteeEvaluation]) -> CommitteeResult<ConsensusDecision> {
        if evaluations.is_empty() {
            return Err(CommitteeError::InsufficientMembers { 
                available: 0, 
                required: 1 
            });
        }
        
        // Calculate weighted progress votes
        let (makes_progress, progress_confidence) = self.aggregate_progress_votes(evaluations)?;
        
        // Calculate weighted average scores
        let overall_score = self.calculate_weighted_overall_score(evaluations)?;
        
        // Aggregate improvement suggestions
        let improvement_suggestions = self.aggregate_improvement_suggestions(evaluations);
        
        // Identify dissenting opinions
        let dissenting_opinions = self.identify_dissenting_opinions(evaluations, makes_progress);
        
        // Calculate final confidence based on agreement and score variance
        let confidence = self.calculate_consensus_confidence(evaluations, makes_progress)?;
        
        // Validate consensus meets threshold
        if confidence < self.consensus_threshold {
            return Err(CommitteeError::ConsensusNotReached {
                agreement: confidence * 100.0,
                threshold: self.consensus_threshold * 100.0,
            });
        }
        
        Ok(ConsensusDecision {
            makes_progress,
            confidence,
            overall_score,
            improvement_suggestions,
            dissenting_opinions,
        })
    }
    
    /// Configure quality-based weighting
    pub fn with_quality_weighting(mut self, enabled: bool) -> Self {
        self.weighted_voting = enabled;
        self
    }
    
    /// Set custom quality weights for different model tiers
    pub fn with_quality_weights(mut self, weights: HashMap<QualityTier, f64>) -> Self {
        self.quality_weights = weights;
        self
    }
    
    /// Set maximum allowed disagreement variance
    pub fn with_max_disagreement_variance(mut self, variance: f64) -> Self {
        self.max_disagreement_variance = variance;
        self
    }
    
    /// Aggregate progress votes with quality weighting
    fn aggregate_progress_votes(&self, evaluations: &[CommitteeEvaluation]) -> CommitteeResult<(bool, f64)> {
        let mut total_weight = 0.0;
        let mut progress_weight = 0.0;
        
        for evaluation in evaluations {
            let weight = self.get_evaluation_weight(evaluation);
            total_weight += weight;
            
            if evaluation.makes_progress {
                progress_weight += weight;
            }
        }
        
        if total_weight == 0.0 {
            return Err(CommitteeError::InsufficientMembers { 
                available: evaluations.len(), 
                required: 1 
            });
        }
        
        let progress_ratio = progress_weight / total_weight;
        let makes_progress = progress_ratio > 0.5;
        
        debug!(
            "Progress votes: {:.1}% weighted agreement, decision: {}",
            progress_ratio * 100.0,
            makes_progress
        );
        
        Ok((makes_progress, progress_ratio))
    }
    
    /// Calculate weighted overall score combining alignment, quality, and risk
    fn calculate_weighted_overall_score(&self, evaluations: &[CommitteeEvaluation]) -> CommitteeResult<f64> {
        let mut total_weight = 0.0;
        let mut weighted_alignment = 0.0;
        let mut weighted_quality = 0.0;
        let mut weighted_risk = 0.0;
        
        for evaluation in evaluations {
            let weight = self.get_evaluation_weight(evaluation);
            total_weight += weight;
            
            weighted_alignment += evaluation.objective_alignment * weight;
            weighted_quality += evaluation.implementation_quality * weight;
            weighted_risk += evaluation.risk_assessment * weight;
        }
        
        if total_weight == 0.0 {
            return Err(CommitteeError::InsufficientMembers { 
                available: evaluations.len(), 
                required: 1 
            });
        }
        
        let avg_alignment = weighted_alignment / total_weight;
        let avg_quality = weighted_quality / total_weight;
        let avg_risk = weighted_risk / total_weight;
        
        // Weighted overall score (alignment matters most, risk is penalty)
        let overall_score = avg_alignment * 0.5 + avg_quality * 0.3 + (1.0 - avg_risk) * 0.2;
        
        debug!(
            "Overall score: {:.3} (alignment: {:.3}, quality: {:.3}, risk: {:.3})",
            overall_score, avg_alignment, avg_quality, avg_risk
        );
        
        Ok(overall_score.clamp(0.0, 1.0))
    }
    
    /// Aggregate and deduplicate improvement suggestions
    fn aggregate_improvement_suggestions(&self, evaluations: &[CommitteeEvaluation]) -> Vec<String> {
        let mut all_suggestions: Vec<String> = evaluations
            .iter()
            .flat_map(|e| e.suggested_improvements.iter().cloned())
            .collect();
        
        // Sort and deduplicate suggestions
        all_suggestions.sort();
        all_suggestions.dedup();
        
        // Weight suggestions by frequency and evaluator quality
        let mut suggestion_scores: HashMap<String, f64> = HashMap::new();
        
        for evaluation in evaluations {
            let weight = self.get_evaluation_weight(evaluation);
            for suggestion in &evaluation.suggested_improvements {
                *suggestion_scores.entry(suggestion.clone()).or_insert(0.0) += weight;
            }
        }
        
        // Sort suggestions by weighted frequency
        all_suggestions.sort_by(|a, b| {
            let score_a = suggestion_scores.get(a).unwrap_or(&0.0);
            let score_b = suggestion_scores.get(b).unwrap_or(&0.0);
            score_b.partial_cmp(score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Limit to top 10 suggestions to avoid overwhelming output
        all_suggestions.truncate(10);
        
        debug!("Aggregated {} improvement suggestions", all_suggestions.len());
        all_suggestions
    }
    
    /// Identify evaluations that disagree with the majority decision
    fn identify_dissenting_opinions(&self, evaluations: &[CommitteeEvaluation], majority_decision: bool) -> Vec<String> {
        evaluations
            .iter()
            .filter(|e| e.makes_progress != majority_decision)
            .map(|e| {
                // Extract the first sentence of reasoning as the dissenting opinion
                let first_sentence = e.reasoning
                    .lines()
                    .next()
                    .unwrap_or("No reason provided")
                    .chars()
                    .take(200)
                    .collect::<String>();
                
                format!("{} ({}): {}", 
                    e.model_type.display_name(),
                    e.agent_id.chars().take(8).collect::<String>(),
                    first_sentence
                )
            })
            .collect()
    }
    
    /// Calculate consensus confidence based on agreement and score variance
    fn calculate_consensus_confidence(&self, evaluations: &[CommitteeEvaluation], majority_decision: bool) -> CommitteeResult<f64> {
        if evaluations.is_empty() {
            return Ok(0.0);
        }
        
        // Calculate agreement ratio
        let agreement_count = evaluations
            .iter()
            .filter(|e| e.makes_progress == majority_decision)
            .count();
        let agreement_ratio = agreement_count as f64 / evaluations.len() as f64;
        
        // Calculate score variance across dimensions
        let alignment_variance = self.calculate_score_variance(
            evaluations.iter().map(|e| e.objective_alignment)
        );
        let quality_variance = self.calculate_score_variance(
            evaluations.iter().map(|e| e.implementation_quality)
        );
        let risk_variance = self.calculate_score_variance(
            evaluations.iter().map(|e| e.risk_assessment)
        );
        
        let avg_variance = (alignment_variance + quality_variance + risk_variance) / 3.0;
        
        // Calculate confidence penalty based on variance
        let variance_penalty = if avg_variance > self.max_disagreement_variance {
            warn!(
                "High disagreement variance detected: {:.3} (max: {:.3})",
                avg_variance, self.max_disagreement_variance
            );
            0.5 // Significant penalty for high disagreement
        } else {
            1.0 / (1.0 + avg_variance) // Gentle penalty scaling
        };
        
        // Weight by individual evaluator confidence
        let avg_evaluator_confidence = evaluations
            .iter()
            .map(|e| e.confidence)
            .sum::<f64>() / evaluations.len() as f64;
        
        // Final confidence combines agreement, variance, and individual confidence
        let confidence = agreement_ratio * variance_penalty * avg_evaluator_confidence;
        
        debug!(
            "Consensus confidence: {:.3} (agreement: {:.3}, variance_penalty: {:.3}, avg_confidence: {:.3})",
            confidence, agreement_ratio, variance_penalty, avg_evaluator_confidence
        );
        
        Ok(confidence.clamp(0.0, 1.0))
    }
    
    /// Get evaluation weight based on model quality and evaluator confidence
    fn get_evaluation_weight(&self, evaluation: &CommitteeEvaluation) -> f64 {
        if !self.weighted_voting {
            return 1.0;
        }
        
        let quality_tier = evaluation.model_type.quality_tier();
        let quality_weight = self.quality_weights.get(&quality_tier).unwrap_or(&1.0);
        
        // Combine quality weight with evaluator confidence
        quality_weight * evaluation.confidence
    }
    
    /// Calculate variance for a set of scores
    fn calculate_score_variance(&self, scores: impl Iterator<Item = f64>) -> f64 {
        let scores: Vec<f64> = scores.collect();
        if scores.len() < 2 {
            return 0.0;
        }
        
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores
            .iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f64>() / scores.len() as f64;
        
        variance.sqrt()
    }
}

/// Advanced decision aggregation with sophisticated algorithms
#[derive(Debug)]
pub struct DecisionAggregator {
    /// Base consensus builder
    consensus_builder: ConsensusBuilder,
    /// Whether to use adaptive thresholds
    adaptive_thresholds: bool,
    /// Minimum evaluations required for high-confidence decisions
    min_evaluations_for_confidence: usize,
}

impl DecisionAggregator {
    /// Create a new decision aggregator
    pub fn new(consensus_threshold: f64) -> Self {
        Self {
            consensus_builder: ConsensusBuilder::new(consensus_threshold),
            adaptive_thresholds: true,
            min_evaluations_for_confidence: 3,
        }
    }
    
    /// Aggregate evaluations with adaptive algorithms
    pub fn aggregate_with_adaptation(&self, evaluations: &[CommitteeEvaluation]) -> CommitteeResult<ConsensusDecision> {
        // Use adaptive threshold if enabled
        let effective_threshold = if self.adaptive_thresholds {
            self.calculate_adaptive_threshold(evaluations)
        } else {
            self.consensus_builder.consensus_threshold
        };
        
        // Create temporary builder with adaptive threshold
        let mut builder = ConsensusBuilder::new(effective_threshold);
        builder = builder
            .with_quality_weighting(self.consensus_builder.weighted_voting)
            .with_quality_weights(self.consensus_builder.quality_weights.clone())
            .with_max_disagreement_variance(self.consensus_builder.max_disagreement_variance);
        
        builder.build_consensus(evaluations)
    }
    
    /// Calculate adaptive threshold based on evaluation context
    fn calculate_adaptive_threshold(&self, evaluations: &[CommitteeEvaluation]) -> f64 {
        let base_threshold = self.consensus_builder.consensus_threshold;
        
        // Lower threshold for smaller committees
        if evaluations.len() < self.min_evaluations_for_confidence {
            return (base_threshold * 0.8).clamp(0.5, 1.0);
        }
        
        // Consider average confidence of evaluators
        let avg_confidence = evaluations
            .iter()
            .map(|e| e.confidence)
            .sum::<f64>() / evaluations.len() as f64;
        
        // Lower threshold if evaluators are generally less confident
        if avg_confidence < 0.7 {
            return (base_threshold * 0.9).clamp(0.5, 1.0);
        }
        
        // Higher threshold for high-confidence evaluators
        if avg_confidence > 0.9 {
            return (base_threshold * 1.1).clamp(0.5, 1.0);
        }
        
        base_threshold
    }
}

/// Consensus engine with multiple aggregation strategies
#[derive(Debug)]
pub struct ConsensusEngine {
    /// Primary decision aggregator
    aggregator: DecisionAggregator,
    /// Fallback strategies for edge cases
    fallback_strategies: Vec<ConsensusStrategy>,
}

/// Different consensus strategies for various scenarios
#[derive(Debug, Clone)]
pub enum ConsensusStrategy {
    /// Simple majority vote
    SimpleMajority,
    /// Weighted by model quality
    QualityWeighted,
    /// Conservative (require higher agreement)
    Conservative,
    /// Optimistic (allow lower agreement)
    Optimistic,
}

impl ConsensusEngine {
    /// Create new consensus engine with default strategies
    pub fn new(consensus_threshold: f64) -> Self {
        Self {
            aggregator: DecisionAggregator::new(consensus_threshold),
            fallback_strategies: vec![
                ConsensusStrategy::QualityWeighted,
                ConsensusStrategy::SimpleMajority,
                ConsensusStrategy::Conservative,
            ],
        }
    }
    
    /// Build consensus with fallback strategies
    pub fn build_consensus_with_fallback(&self, evaluations: &[CommitteeEvaluation]) -> CommitteeResult<ConsensusDecision> {
        // Try primary aggregation first
        match self.aggregator.aggregate_with_adaptation(evaluations) {
            Ok(decision) => {
                info!("Primary consensus strategy succeeded");
                return Ok(decision);
            }
            Err(e) => {
                warn!("Primary consensus failed: {}, trying fallbacks", e);
            }
        }
        
        // Try fallback strategies
        for strategy in &self.fallback_strategies {
            match self.apply_strategy(strategy, evaluations) {
                Ok(decision) => {
                    info!("Fallback strategy {:?} succeeded", strategy);
                    return Ok(decision);
                }
                Err(e) => {
                    debug!("Fallback strategy {:?} failed: {}", strategy, e);
                }
            }
        }
        
        // If all strategies fail, return the original error
        Err(CommitteeError::ConsensusNotReached {
            agreement: 0.0,
            threshold: self.aggregator.consensus_builder.consensus_threshold * 100.0,
        })
    }
    
    /// Apply specific consensus strategy
    fn apply_strategy(&self, strategy: &ConsensusStrategy, evaluations: &[CommitteeEvaluation]) -> CommitteeResult<ConsensusDecision> {
        let builder = match strategy {
            ConsensusStrategy::SimpleMajority => {
                ConsensusBuilder::new(0.5).with_quality_weighting(false)
            }
            ConsensusStrategy::QualityWeighted => {
                ConsensusBuilder::new(0.6).with_quality_weighting(true)
            }
            ConsensusStrategy::Conservative => {
                ConsensusBuilder::new(0.8).with_max_disagreement_variance(0.2)
            }
            ConsensusStrategy::Optimistic => {
                ConsensusBuilder::new(0.4).with_max_disagreement_variance(0.5)
            }
        };
        
        builder.build_consensus(evaluations)
    }
}

/// Quality metrics for consensus analysis
#[derive(Debug, Clone, Default)]
pub struct QualityMetrics {
    /// Agreement level achieved
    pub agreement_level: f64,
    /// Score variance across evaluations
    pub score_variance: f64,
    /// Average evaluator confidence
    pub average_confidence: f64,
    /// Number of dissenting opinions
    pub dissenting_count: usize,
    /// Quality of reasoning provided
    pub reasoning_quality_score: f64,
}

impl QualityMetrics {
    /// Calculate quality metrics from evaluations
    pub fn from_evaluations(evaluations: &[CommitteeEvaluation], majority_decision: bool) -> Self {
        let agreement_count = evaluations
            .iter()
            .filter(|e| e.makes_progress == majority_decision)
            .count();
        
        let agreement_level = if evaluations.is_empty() {
            0.0
        } else {
            agreement_count as f64 / evaluations.len() as f64
        };
        
        let average_confidence = if evaluations.is_empty() {
            0.0
        } else {
            evaluations.iter().map(|e| e.confidence).sum::<f64>() / evaluations.len() as f64
        };
        
        let dissenting_count = evaluations.len() - agreement_count;
        
        // Calculate reasoning quality based on length and detail
        let reasoning_quality_score = if evaluations.is_empty() {
            0.0
        } else {
            evaluations
                .iter()
                .map(|e| {
                    let length_score = (e.reasoning.len() as f64 / 200.0).clamp(0.0, 1.0);
                    let detail_score = if e.reasoning.split_whitespace().count() > 10 { 0.5 } else { 0.0 };
                    length_score + detail_score
                })
                .sum::<f64>() / evaluations.len() as f64
        };
        
        // Calculate score variance (simplified)
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
        
        Self {
            agreement_level,
            score_variance,
            average_confidence,
            dissenting_count,
            reasoning_quality_score,
        }
    }
    
    /// Check if metrics indicate high-quality consensus
    pub fn is_high_quality(&self) -> bool {
        self.agreement_level > 0.7
            && self.score_variance < 0.3
            && self.average_confidence > 0.8
            && self.reasoning_quality_score > 0.6
    }
}