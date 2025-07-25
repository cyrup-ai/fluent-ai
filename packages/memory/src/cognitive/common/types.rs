// src/cognitive/common/types.rs
//! Defines common data structures for the cognitive evaluation committee.

use serde::{Deserialize, Serialize};

use crate::cognitive::types::{ImpactFactors, OptimizationSpec, RoutingDecision};

/// Consensus decision from committee
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusDecision {
    pub makes_progress: bool,
    pub confidence: f64,
    pub overall_score: f64, // Weighted combination of alignment, quality, safety
    pub improvement_suggestions: Vec<String>,
    pub dissenting_opinions: Vec<String>,
    pub latency_factor: f64,
    pub memory_factor: f64,
    pub relevance_factor: f64}

impl From<ConsensusDecision> for ImpactFactors {
    fn from(decision: ConsensusDecision) -> Self {
        ImpactFactors {
            alignment_score: decision.overall_score,
            quality_score: decision.overall_score,
            safety_score: decision.overall_score,
            confidence: decision.confidence,
            improvement_suggestions: decision.improvement_suggestions,
            potential_risks: decision.dissenting_opinions,
            latency_factor: decision.latency_factor,
            memory_factor: decision.memory_factor,
            relevance_factor: decision.relevance_factor}
    }
}

/// Individual agent's evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvaluation {
    pub agent_id: String,
    pub action: String,
    pub makes_progress: bool,        // Core question: does this help?
    pub objective_alignment: f64,    // 0-1: How well aligned with objective
    pub implementation_quality: f64, // 0-1: How well implemented
    pub risk_assessment: f64,        // 0-1: How safe/risky (1 = safe)
    pub reasoning: String,           // Detailed explanation
    pub suggested_improvements: Vec<String>, // What could be better
}

/// Evaluation rubric provided to agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationRubric {
    pub objective: String,
    pub success_criteria: Vec<String>,
    pub constraints: Vec<String>,
    pub scoring_guidelines: HashMap<String, String>}

impl EvaluationRubric {
    pub fn from_spec(spec: &OptimizationSpec, user_objective: &str) -> Self {
        let mut scoring_guidelines = HashMap::new();
        scoring_guidelines.insert(
            "latency".to_string(),
            format!("Score 0.0-2.0: How much does this change improve speed? (1.0 = no change, <1.0 = faster, >1.0 = slower). Max acceptable: {:.2}", 
                1.0 + spec.content_type.restrictions.max_latency_increase / 100.0)
        );
        scoring_guidelines.insert(
            "memory".to_string(),
            format!("Score 0.0-2.0: How much does this change affect memory usage? (1.0 = no change, <1.0 = less memory, >1.0 = more memory). Max acceptable: {:.2}",
                1.0 + spec.content_type.restrictions.max_memory_increase / 100.0)
        );
        scoring_guidelines.insert(
            "relevance".to_string(),
            format!("Score 0.0-2.0: How much does this improve achieving the objective? (1.0 = no change, >1.0 = better, <1.0 = worse). Min required: {:.2}",
                1.0 + spec.content_type.restrictions.min_relevance_improvement / 100.0)
        );

        Self {
            objective: user_objective.to_string(),
            success_criteria: vec![
                format!("Achieve: {}", user_objective),
                format!(
                    "Maintain latency within {}% increase",
                    spec.content_type.restrictions.max_latency_increase
                ),
                format!(
                    "Maintain memory within {}% increase",
                    spec.content_type.restrictions.max_memory_increase
                ),
                format!(
                    "Improve relevance by at least {}%",
                    spec.content_type.restrictions.min_relevance_improvement
                ),
            ],
            constraints: spec.constraints.clone(),
            scoring_guidelines}
    }
}

/// Multi-round evaluation phase
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationPhase {
    Initial,  // First independent evaluation
    Review,   // Review others' evaluations
    Refine,   // Refine based on committee feedback
    Finalize, // Final scoring round
}

/// Round of evaluations with phase tracking
#[derive(Debug, Clone)]
pub struct EvaluationRound {
    pub phase: EvaluationPhase,
    pub evaluations: Vec<AgentEvaluation>,
    pub consensus: Option<ConsensusDecision>,
    pub steering_feedback: Option<String>}

/// Configuration for the evaluation committee
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitteeConfig {
    pub rounds: usize,
    pub concurrency: usize,
    pub consensus_threshold: f64,
    pub steering_strength: f64,
    pub agent_perspectives: HashMap<String, String>}

impl Default for CommitteeConfig {
    fn default() -> Self {
        let mut agent_perspectives = HashMap::new();
        agent_perspectives.insert("performance".to_string(), "gpt-4".to_string());
        agent_perspectives.insert("safety".to_string(), "claude-3".to_string());
        agent_perspectives.insert("maintainability".to_string(), "gpt-4".to_string());

        Self {
            rounds: 3,
            concurrency: 4,
            consensus_threshold: 0.7,
            steering_strength: 0.8,
            agent_perspectives}
    }
}

/// Events emitted by the evaluation committee
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommitteeEvent {
    SteeringDecision(RoutingDecision),
    EvaluationStarted {
        action: String,
        agent_count: usize},
    RoundCompleted {
        round: usize,
        consensus: Option<ConsensusDecision>},
    ConsensusReached {
        action: String,
        decision: ConsensusDecision,
        factors: ImpactFactors,
        rounds_taken: usize},
    ConsensusFailed {
        action: String,
        rounds: usize}}
