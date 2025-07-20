// src/cognitive/mod.rs
//! Cognitive enhancement system for quantum memory optimization
//!
//! This module provides self-optimizing capabilities through committee-based
//! evaluation and Monte Carlo Tree Search (MCTS).

pub mod committee;

pub mod compiler;
pub mod evolution;
pub mod evolution_manager;
pub mod mcts;
pub mod mesh;
pub mod orchestrator;
pub mod performance;
pub mod types;

// Add the new common module
pub mod common;

// Core cognitive modules from existing implementation
pub mod attention;
pub mod llm;
pub mod manager;
pub mod state;

// Quantum-specific cognitive modules
pub mod quantum;
pub mod quantum_mcts;
pub mod quantum_orchestrator;

// Re-exports for convenience
// Re-export existing cognitive components
pub use attention::{AttentionMechanism, AttentionRouter};
// Re-export the refactored committee types
pub use committee::{CommitteeEvent, EvaluationCommittee};
pub use common::models::{Model, ModelType};
pub use common::types::{CommitteeConfig, ConsensusDecision};
pub use evolution::{
    CodeEvolution, CognitiveCodeEvolution, EvolutionEngine, EvolutionResult, Innovation,
    InnovationType, PerformanceMetrics,
};
pub use llm::{ModelConfig, PerformanceConfig, ProductionLLMProvider, ProviderConfig};
pub use manager::CognitiveMemoryManager;
pub use mcts::{CodeState, MCTS};
pub use orchestrator::InfiniteOrchestrator;
pub use quantum_mcts::{
    AtomicQuantumMetrics, QuantumMCTS, QuantumMCTSNode, QuantumNodeState, QuantumTreeStatistics,
};
pub use quantum_orchestrator::{QuantumOrchestrationConfig, QuantumOrchestrator, RecursiveState};
pub use state::CognitiveState;
pub use types::{
    CognitiveError, CognitiveMemoryNode, CognitiveSettings, EvolutionMetadata, ImpactFactors,
    OptimizationOutcome, OptimizationSpec, OptimizationType, PendingOptimizationResult,
    QuantumSignature,
};
