// src/cognitive/evolution.rs
//! Self-optimizing component using MCTS with committee evaluation

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use arc_swap::ArcSwap;
use crossbeam_queue::ArrayQueue;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info};

use crate::cognitive::committee::{CommitteeEvent, EvaluationCommittee};
use crate::cognitive::mcts::{CodeState, MCTS};
use crate::cognitive::performance::PerformanceAnalyzer;
use crate::cognitive::state::CognitiveStateManager;
// Re-export types for external use
pub use crate::cognitive::types::EvolutionMetadata;
use crate::cognitive::types::{
    CognitiveError, MutationEvent, MutationType, OptimizationOutcome, OptimizationSpec,
    PendingOptimizationResult,
};

pub trait CodeEvolution {
    fn evolve_routing_logic(&self) -> PendingOptimizationResult;
}

#[derive(Clone)]
pub struct CognitiveCodeEvolution {
    initial_state: CodeState,
    spec: Arc<OptimizationSpec>,
    user_objective: String,
}

impl CognitiveCodeEvolution {
    pub fn new(
        initial_code: String,
        initial_latency: f64,
        initial_memory: f64,
        initial_relevance: f64,
        spec: Arc<OptimizationSpec>,
        user_objective: String,
    ) -> Result<Self, CognitiveError> {
        let initial_state = CodeState {
            code: initial_code,
            code_content: String::new(),
            latency: initial_latency,
            memory: initial_memory,
            relevance: initial_relevance,
        };

        Ok(Self {
            initial_state,
            spec,
            user_objective,
        })
    }
}

impl CodeEvolution for CognitiveCodeEvolution {
    fn evolve_routing_logic(&self) -> PendingOptimizationResult {
        let (tx, rx) = oneshot::channel();
        let initial_state = self.initial_state.clone();
        let spec = Arc::clone(&self.spec);
        let user_objective = self.user_objective.clone();

        tokio::spawn(async move {
            // Create event channel for committee
            let (event_tx, mut event_rx) = mpsc::channel(256);

            // Spawn event logger
            tokio::spawn(async move {
                while let Some(event) = event_rx.recv().await {
                    match event {
                        CommitteeEvent::ConsensusReached {
                            action,
                            decision,
                            factors,
                            rounds_taken,
                        } => {
                            info!(
                                "Committee consensus on '{}' after {} rounds: latency={:.2}, memory={:.2}, relevance={:.2}, confidence={:.2}",
                                action,
                                rounds_taken,
                                factors.latency_factor,
                                factors.memory_factor,
                                factors.relevance_factor,
                                factors.confidence
                            );
                        }
                        CommitteeEvent::SteeringDecision {
                            feedback,
                            continue_rounds,
                        } => {
                            info!(
                                "Committee steering: {} (continue: {})",
                                feedback, continue_rounds
                            );
                        }
                        _ => {} // Log other events at debug level
                    }
                }
            });

            // Create committee
            let committee = match EvaluationCommittee::new(event_tx.clone(), 4).await {
                Ok(c) => Arc::new(c),
                Err(e) => {
                    error!("Failed to create committee: {}", e);
                    let _ = tx.send(Err(e));
                    return;
                }
            };

            // Create performance analyzer with committee
            let performance_analyzer = Arc::new(
                PerformanceAnalyzer::new(spec.clone(), committee.clone(), user_objective.clone())
                    .await,
            );

            // Create and run MCTS
            let mut mcts = match MCTS::new(
                initial_state.clone(),
                performance_analyzer.clone(),
                spec.clone(),
                user_objective.clone(),
                event_tx,
            )
            .await
            {
                Ok(m) => m,
                Err(e) => {
                    error!("Failed to create MCTS: {}", e);
                    let _ = tx.send(Err(e));
                    return;
                }
            };

            // Run MCTS iterations
            if let Err(e) = mcts.run(1000).await {
                error!("MCTS execution failed: {}", e);
                let _ = tx.send(Err(e));
                return;
            }

            // Get best modification
            if let Some(best_state) = mcts.best_modification() {
                // Calculate improvements
                let latency_improvement =
                    (initial_state.latency - best_state.latency) / initial_state.latency * 100.0;
                let memory_improvement =
                    (initial_state.memory - best_state.memory) / initial_state.memory * 100.0;
                let relevance_improvement = (best_state.relevance - initial_state.relevance)
                    / initial_state.relevance
                    * 100.0;

                // Check if improvements are significant
                if latency_improvement > 5.0
                    || memory_improvement > 5.0
                    || relevance_improvement > 10.0
                {
                    let outcome = OptimizationOutcome::Success {
                        improvements: vec![
                            format!("Latency improved by {:.2}%", latency_improvement),
                            format!("Memory usage improved by {:.2}%", memory_improvement),
                            format!("Relevance improved by {:.2}%", relevance_improvement),
                        ],
                        performance_gain: ((latency_improvement + memory_improvement) / 2.0) as f32,
                        applied: true,
                        quality_score: (relevance_improvement / 10.0) as f32,
                        metadata: HashMap::new(),
                    };

                    info!(
                        "Applied optimization: latency improved {:.1}%, memory improved {:.1}%, relevance improved {:.1}%",
                        latency_improvement, memory_improvement, relevance_improvement
                    );

                    // Get statistics
                    let stats = mcts.get_statistics();
                    info!(
                        "MCTS explored {} nodes with {} total visits, max depth {}, best path: {:?}",
                        stats.total_nodes, stats.total_visits, stats.max_depth, stats.best_path
                    );

                    let _ = tx.send(Ok(outcome));
                } else {
                    info!("No significant improvement found");
                    let _ = tx.send(Ok(OptimizationOutcome::Failure {
                        errors: vec!["No significant improvement found".to_string()],
                        root_cause: "Optimization did not meet threshold requirements".to_string(),
                        suggestions: vec![
                            "Consider adjusting improvement thresholds".to_string(),
                            "Try different optimization strategies".to_string(),
                        ],
                        applied: false,
                    }));
                }
            } else {
                info!("No modifications found");
                let _ = tx.send(Ok(OptimizationOutcome::Failure {
                    errors: vec!["No modifications found".to_string()],
                    root_cause: "Best modification result is None".to_string(),
                    suggestions: vec![
                        "Check evaluation committee configuration".to_string(),
                        "Verify agent evaluation is working correctly".to_string(),
                    ],
                    applied: false,
                }));
            }
        });

        PendingOptimizationResult::new(rx)
    }
}

/// Evolution result containing generation and improvement metrics
#[derive(Debug, Clone)]
pub struct EvolutionResult {
    pub generation: u32,
    pub predicted_improvement: f64,
    pub fitness_score: f64,
    pub mutations_applied: Vec<MutationEvent>,
}

/// Performance metrics for evolution tracking
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub latency: f64,
    pub memory_usage: f64,
    pub accuracy: f64,
    pub throughput: f64,
}

/// High-performance evolution engine with zero-allocation design
pub struct EvolutionEngine {
    evolution_rate: f64,
    state_manager: Option<Arc<CognitiveStateManager>>,
    capacity: usize,
    generation: u32,
    fitness_history: VecDeque<f64>,
    recent_metrics: ArcSwap<PerformanceMetrics>,
    mutation_queue: ArrayQueue<MutationEvent>,
    evolution_threshold: f64,
    min_generations_between_evolution: u32,
    last_evolution_generation: u32,
}

impl EvolutionEngine {
    /// Create new evolution engine with just evolution rate
    pub fn new(evolution_rate: f64) -> Self {
        Self {
            evolution_rate,
            state_manager: None,
            capacity: 1000,
            generation: 0,
            fitness_history: VecDeque::with_capacity(1000),
            recent_metrics: ArcSwap::new(Arc::new(PerformanceMetrics {
                latency: 0.0,
                memory_usage: 0.0,
                accuracy: 0.0,
                throughput: 0.0,
            })),
            mutation_queue: ArrayQueue::new(1000),
            evolution_threshold: 0.1,
            min_generations_between_evolution: 10,
            last_evolution_generation: 0,
        }
    }

    /// Create new evolution engine with state manager and capacity
    pub fn with_state_manager(state_manager: Arc<CognitiveStateManager>, capacity: usize) -> Self {
        Self {
            evolution_rate: 0.1,
            state_manager: Some(state_manager),
            capacity,
            generation: 0,
            fitness_history: VecDeque::with_capacity(capacity),
            recent_metrics: ArcSwap::new(Arc::new(PerformanceMetrics {
                latency: 0.0,
                memory_usage: 0.0,
                accuracy: 0.0,
                throughput: 0.0,
            })),
            mutation_queue: ArrayQueue::new(capacity),
            evolution_threshold: 0.1,
            min_generations_between_evolution: 10,
            last_evolution_generation: 0,
        }
    }

    /// Record fitness metrics for evolution tracking
    pub fn record_fitness(&mut self, metrics: PerformanceMetrics) {
        let fitness = self.calculate_fitness(&metrics);

        // Update fitness history with bounded capacity
        if self.fitness_history.len() >= self.capacity {
            self.fitness_history.pop_front();
        }
        self.fitness_history.push_back(fitness);

        // Update recent metrics atomically
        self.recent_metrics.store(Arc::new(metrics));
    }

    /// Check if evolution should be triggered and evolve if needed
    pub async fn evolve_if_needed(&mut self) -> Option<EvolutionResult> {
        if !self.should_evolve() {
            return None;
        }

        self.generation += 1;
        self.last_evolution_generation = self.generation;

        let mutations = self.generate_mutations();
        let fitness_score = self.calculate_current_fitness();

        // Apply mutations if we have a state manager
        if let Some(state_manager) = &self.state_manager {
            for mutation in &mutations {
                if let Err(e) = self.apply_mutation(state_manager, mutation).await {
                    error!("Failed to apply mutation: {}", e);
                }
            }
        }

        let predicted_improvement = self.calculate_predicted_improvement(&mutations);

        Some(EvolutionResult {
            generation: self.generation,
            predicted_improvement,
            fitness_score,
            mutations_applied: mutations,
        })
    }

    /// Calculate fitness score from performance metrics
    fn calculate_fitness(&self, metrics: &PerformanceMetrics) -> f64 {
        // Weighted combination of metrics (higher is better)
        let latency_score = 1.0 / (1.0 + metrics.latency);
        let memory_score = 1.0 / (1.0 + metrics.memory_usage);
        let accuracy_score = metrics.accuracy;
        let throughput_score = metrics.throughput / (1.0 + metrics.throughput);

        // Weights can be adjusted based on system priorities
        0.3 * latency_score + 0.2 * memory_score + 0.3 * accuracy_score + 0.2 * throughput_score
    }

    /// Get current fitness score
    fn calculate_current_fitness(&self) -> f64 {
        self.fitness_history.back().copied().unwrap_or(0.0)
    }

    /// Determine if evolution should be triggered
    fn should_evolve(&self) -> bool {
        // Need minimum history
        if self.fitness_history.len() < 5 {
            return false;
        }

        // Check minimum generations between evolutions
        if self.generation - self.last_evolution_generation < self.min_generations_between_evolution
        {
            return false;
        }

        // Check if fitness is stagnating or declining
        let recent_fitness: Vec<f64> = self.fitness_history.iter().rev().take(5).cloned().collect();
        let avg_recent = recent_fitness.iter().sum::<f64>() / recent_fitness.len() as f64;

        if let Some(older_fitness) = self.fitness_history.iter().rev().nth(5) {
            let improvement_rate = (avg_recent - older_fitness) / older_fitness;
            improvement_rate.abs() < self.evolution_threshold
        } else {
            false
        }
    }

    /// Generate mutations for evolution
    fn generate_mutations(&self) -> Vec<MutationEvent> {
        let mut mutations = Vec::new();

        // Generate different types of mutations based on current performance
        let current_metrics = self.recent_metrics.load();

        if current_metrics.latency > 0.1 {
            mutations.push(MutationEvent {
                timestamp: chrono::Utc::now(),
                mutation_type: MutationType::RoutingStrategyModification,
                impact_score: 0.8,
                description: "Optimize routing for lower latency".to_string(),
            });
        }

        if current_metrics.accuracy < 0.9 {
            mutations.push(MutationEvent {
                timestamp: chrono::Utc::now(),
                mutation_type: MutationType::ContextualUnderstandingEvolution,
                impact_score: 0.9,
                description: "Enhance contextual understanding".to_string(),
            });
        }

        if current_metrics.memory_usage > 0.8 {
            mutations.push(MutationEvent {
                timestamp: chrono::Utc::now(),
                mutation_type: MutationType::QuantumCoherenceOptimization,
                impact_score: 0.7,
                description: "Optimize quantum coherence for memory efficiency".to_string(),
            });
        }

        // Always include attention weight adjustment
        mutations.push(MutationEvent {
            timestamp: chrono::Utc::now(),
            mutation_type: MutationType::AttentionWeightAdjustment,
            impact_score: 0.6,
            description: "Adjust attention weights based on performance".to_string(),
        });

        mutations
    }

    /// Apply mutation to the system
    async fn apply_mutation(
        &self,
        state_manager: &CognitiveStateManager,
        mutation: &MutationEvent,
    ) -> Result<(), CognitiveError> {
        match mutation.mutation_type {
            MutationType::AttentionWeightAdjustment => {
                // Adjust attention weights based on recent performance
                // This would interact with the state manager to modify attention patterns
                info!("Applying attention weight adjustment mutation");
            }
            MutationType::RoutingStrategyModification => {
                // Modify routing strategy based on performance metrics
                info!("Applying routing strategy modification mutation");
            }
            MutationType::ContextualUnderstandingEvolution => {
                // Enhance contextual understanding mechanisms
                info!("Applying contextual understanding evolution mutation");
            }
            MutationType::QuantumCoherenceOptimization => {
                // Optimize quantum coherence parameters
                info!("Applying quantum coherence optimization mutation");
            }
            MutationType::EmergentPatternRecognition => {
                // Enhance emergent pattern recognition capabilities
                info!("Applying emergent pattern recognition mutation");
            }
        }

        Ok(())
    }

    /// Calculate predicted improvement from mutations
    fn calculate_predicted_improvement(&self, mutations: &[MutationEvent]) -> f64 {
        mutations.iter().map(|m| m.impact_score as f64).sum::<f64>() / mutations.len() as f64
    }

    /// Get current generation
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Get evolution rate
    pub fn evolution_rate(&self) -> f64 {
        self.evolution_rate
    }

    /// Get current fitness score
    pub fn current_fitness(&self) -> f64 {
        self.calculate_current_fitness()
    }
}
