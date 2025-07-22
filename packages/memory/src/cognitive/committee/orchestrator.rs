// src/cognitive/committee/orchestrator.rs
//! Orchestrates the evaluation committee to achieve consensus.

use std::sync::Arc;

use futures_util::stream::{FuturesUnordered, StreamExt};
use tokio::sync::{RwLock, Semaphore, mpsc};
use tracing::{info, warn};

use crate::cognitive::committee::agent::ProviderEvaluationAgent;
use crate::cognitive::common::types::*;
use crate::cognitive::mcts::CodeState;
use crate::cognitive::types::{CognitiveError, ImpactFactors};

/// Committee orchestrating consensus among provider agents with multi-round evaluation
#[derive(Debug)]
pub struct EvaluationCommittee {
    agents: Vec<ProviderEvaluationAgent>,
    config: CommitteeConfig,
    event_tx: mpsc::Sender<CommitteeEvent>,
    history: Arc<RwLock<Vec<EvaluationRound>>>,
}

impl EvaluationCommittee {
    pub async fn new(
        config: CommitteeConfig,
        event_tx: mpsc::Sender<CommitteeEvent>,
    ) -> Result<Self, CognitiveError> {
        let mut agents = Vec::new();
        for (perspective, model_name) in &config.agent_perspectives {
            let model_type = crate::cognitive::common::models::Model::available_types()
                .into_iter()
                .find(|m| m.display_name() == model_name)
                .ok_or_else(|| {
                    CognitiveError::ConfigError(format!("Model '{}' not found", model_name))
                })?;

            let agent = ProviderEvaluationAgent::new(model_type, perspective).await?;
            agents.push(agent);
        }

        Ok(Self {
            agents,
            config,
            event_tx,
            history: Arc::new(RwLock::new(Vec::new())),
        })
    }

    pub async fn evaluate_action(
        &self,
        state: &CodeState,
        action: &str,
        rubric: &EvaluationRubric,
    ) -> Result<ConsensusDecision, CognitiveError> {
        self.event_tx
            .send(CommitteeEvent::EvaluationStarted {
                action: action.to_string(),
                agent_count: self.agents.len(),
            })
            .await
            .ok();

        let mut current_round = 0;
        let mut all_evaluations: Vec<AgentEvaluation> = Vec::new();

        while current_round < self.config.rounds {
            let phase = self.determine_phase(current_round);
            let previous_evals = if current_round > 0 {
                Some(all_evaluations.as_slice())
            } else {
                None
            };

            let steering_feedback = if phase == EvaluationPhase::Refine {
                self.generate_steering_feedback(&all_evaluations).await
            } else {
                None
            };

            let evaluations = self
                .execute_evaluation_round(
                    state,
                    action,
                    rubric,
                    phase,
                    previous_evals,
                    steering_feedback.as_deref(),
                )
                .await?;

            all_evaluations.extend(evaluations.clone());
            let consensus = self.calculate_consensus(&all_evaluations);

            self.history.write().await.push(EvaluationRound {
                phase,
                evaluations,
                consensus: Some(consensus.clone()),
                steering_feedback,
            });

            self.event_tx
                .send(CommitteeEvent::RoundCompleted {
                    round: current_round,
                    consensus: Some(consensus.clone()),
                })
                .await
                .ok();

            if consensus.confidence >= self.config.consensus_threshold {
                let factors = ImpactFactors::from(consensus.clone());
                self.finalize_evaluation(action, consensus.clone(), factors, current_round + 1)
                    .await?;
                return Ok(consensus);
            }

            current_round += 1;
        }

        self.event_tx
            .send(CommitteeEvent::ConsensusFailed {
                action: action.to_string(),
                rounds: self.config.rounds,
            })
            .await
            .ok();

        Err(CognitiveError::ConsensusFailed(format!(
            "Action {} failed after {} rounds",
            action, self.config.rounds
        )))
    }

    fn determine_phase(&self, round: usize) -> EvaluationPhase {
        match round {
            0 => EvaluationPhase::Initial,
            1 => EvaluationPhase::Review,
            _ if round == self.config.rounds - 1 => EvaluationPhase::Finalize,
            _ => EvaluationPhase::Refine,
        }
    }

    async fn execute_evaluation_round(
        &self,
        state: &CodeState,
        action: &str,
        rubric: &EvaluationRubric,
        phase: EvaluationPhase,
        previous_evals: Option<&[AgentEvaluation]>,
        steering_feedback: Option<&str>,
    ) -> Result<Vec<AgentEvaluation>, CognitiveError> {
        let semaphore = Arc::new(Semaphore::new(self.config.concurrency));
        let mut tasks = FuturesUnordered::new();

        for agent in &self.agents {
            let permit = semaphore.clone().acquire_owned().await.map_err(|e| {
                CognitiveError::OptimizationError(format!(
                    "Failed to acquire semaphore permit for agent evaluation: {}",
                    e
                ))
            })?;
            let task = tokio::spawn({
                let agent = agent.clone();
                let state = state.clone();
                let action = action.to_string();
                let rubric = rubric.clone();
                let previous_evals = previous_evals.map(|v| v.to_vec());
                let steering_feedback = steering_feedback.map(|s| s.to_string());

                async move {
                    let _permit = permit;
                    agent
                        .evaluate_with_context(
                            &state,
                            &action,
                            &rubric,
                            phase,
                            previous_evals.as_deref(),
                            steering_feedback.as_deref(),
                        )
                        .await
                }
            });
            tasks.push(task);
        }

        let mut evaluations = Vec::new();
        while let Some(result) = tasks.next().await {
            match result {
                Ok(Ok(eval)) => evaluations.push(eval),
                Ok(Err(e)) => warn!("Agent evaluation failed: {}", e),
                Err(e) => warn!("Agent task failed: {}", e),
            }
        }

        Ok(evaluations)
    }

    async fn generate_steering_feedback(&self, evaluations: &[AgentEvaluation]) -> Option<String> {
        let consensus = self.calculate_consensus(evaluations);
        let mut feedback = Vec::new();

        if !consensus.dissenting_opinions.is_empty() {
            feedback.push("Address dissenting opinions:".to_string());
            for dissent in &consensus.dissenting_opinions {
                feedback.push(format!("- {}", dissent));
            }

            feedback.push("\nFocus on addressing these concerns to build consensus.".to_string());
        }

        if consensus.overall_score < 0.5 {
            feedback.push(format!(
                "\nLow scores indicate issues with: alignment ({:.2}), quality ({:.2}), or safety ({:.2})",
                consensus.overall_score * 2.0, // Rough estimates
                consensus.overall_score * 3.33,
                consensus.overall_score * 5.0
            ));
        }

        if feedback.is_empty() {
            None
        } else {
            Some(feedback.join("\n"))
        }
    }

    async fn finalize_evaluation(
        &self,
        action: &str,
        decision: ConsensusDecision,
        factors: ImpactFactors,
        rounds_taken: usize,
    ) -> Result<(), CognitiveError> {
        let confidence = factors.confidence;

        self.event_tx
            .send(CommitteeEvent::ConsensusReached {
                action: action.to_string(),
                decision,
                factors,
                rounds_taken,
            })
            .await
            .ok();

        info!(
            "Committee reached consensus on '{}' after {} rounds (confidence: {:.2})",
            action, rounds_taken, confidence
        );

        Ok(())
    }

    fn calculate_consensus(&self, evaluations: &[AgentEvaluation]) -> ConsensusDecision {
        let count = evaluations.len() as f64;

        // Count votes for progress
        let progress_votes = evaluations.iter().filter(|e| e.makes_progress).count();
        let makes_progress = progress_votes > evaluations.len() / 2;

        // Calculate average scores
        let avg_alignment = evaluations
            .iter()
            .map(|e| e.objective_alignment)
            .sum::<f64>()
            / count;

        let avg_quality = evaluations
            .iter()
            .map(|e| e.implementation_quality)
            .sum::<f64>()
            / count;

        let avg_risk = evaluations.iter().map(|e| e.risk_assessment).sum::<f64>() / count;

        // Weighted overall score (alignment matters most)
        let overall_score = avg_alignment * 0.5 + avg_quality * 0.3 + avg_risk * 0.2;

        // Collect all improvement suggestions
        let mut improvement_suggestions: Vec<String> = evaluations
            .iter()
            .flat_map(|e| e.suggested_improvements.iter().cloned())
            .collect();
        improvement_suggestions.sort();
        improvement_suggestions.dedup();

        // Collect dissenting opinions
        let dissenting_opinions: Vec<String> = evaluations
            .iter()
            .filter(|e| e.makes_progress != makes_progress)
            .map(|e| {
                format!(
                    "{}: {}",
                    e.agent_id,
                    e.reasoning.lines().next().unwrap_or("No reason given")
                )
            })
            .collect();

        // Calculate confidence based on agreement
        let alignment_std =
            self.calculate_std_dev(evaluations.iter().map(|e| e.objective_alignment));
        let quality_std =
            self.calculate_std_dev(evaluations.iter().map(|e| e.implementation_quality));
        let risk_std = self.calculate_std_dev(evaluations.iter().map(|e| e.risk_assessment));

        let avg_std = (alignment_std + quality_std + risk_std) / 3.0;
        let confidence = (progress_votes as f64 / count) * (1.0 / (1.0 + avg_std));

        ConsensusDecision {
            makes_progress,
            confidence,
            overall_score,
            improvement_suggestions,
            dissenting_opinions,
            latency_factor: 1.0,   // Default neutral factor
            memory_factor: 1.0,    // Default neutral factor
            relevance_factor: 1.0, // Default neutral factor
        }
    }

    #[inline]
    fn calculate_std_dev(&self, values: impl Iterator<Item = f64>) -> f64 {
        let values: Vec<f64> = values.collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }
}
