//! Core cognitive types and structures

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Cognitive state representing the current understanding and context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveState {
    pub activation_pattern: Vec<f32>,
    pub attention_weights: Vec<f32>,
    pub temporal_context: TemporalContext,
    pub uncertainty: f32,
    pub confidence: f32,
    pub meta_awareness: f32,
}

impl Default for CognitiveState {
    fn default() -> Self {
        Self {
            activation_pattern: Vec::new(),
            attention_weights: Vec::new(),
            temporal_context: TemporalContext::default(),
            uncertainty: 0.5,
            confidence: 0.5,
            meta_awareness: 0.5,
        }
    }
}

/// Temporal context and dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    pub history_embedding: Vec<f32>,
    pub prediction_horizon: Vec<f32>,
    pub causal_dependencies: Vec<CausalLink>,
    pub temporal_decay: f32,
}

impl Default for TemporalContext {
    fn default() -> Self {
        Self {
            history_embedding: Vec::new(),
            prediction_horizon: Vec::new(),
            causal_dependencies: Vec::new(),
            temporal_decay: 0.1,
        }
    }
}

/// Causal relationship between memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalLink {
    pub source_id: String,
    pub target_id: String,
    pub causal_strength: f32,
    pub temporal_distance: i64, // milliseconds
}

/// Quantum signature for superposition routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSignature {
    pub coherence_fingerprint: Vec<f32>,
    pub entanglement_bonds: Vec<EntanglementBond>,
    pub superposition_contexts: Vec<String>,
    pub collapse_probability: f32,
    pub entanglement_links: Vec<String>,
    pub quantum_entropy: f64,
    pub creation_time: chrono::DateTime<chrono::Utc>,
}

impl Default for QuantumSignature {
    fn default() -> Self {
        Self {
            coherence_fingerprint: vec![1.0], // Start with maximum coherence
            entanglement_bonds: Vec::new(),
            superposition_contexts: vec!["default".to_string()],
            collapse_probability: 0.0, // Start with 0% collapse probability
            entanglement_links: Vec::new(),
            quantum_entropy: 0.0, // Start with minimum entropy
            creation_time: chrono::Utc::now(),
        }
    }
}

/// Cognitive memory node with enhanced capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveMemoryNode {
    pub base_memory: crate::memory::MemoryNode,
    pub cognitive_state: CognitiveState,
    pub quantum_signature: Option<QuantumSignature>,
    pub evolution_metadata: Option<EvolutionMetadata>,
    pub attention_weights: Vec<f32>,
    pub semantic_relationships: Vec<String>,
}

impl CognitiveMemoryNode {
    /// Check if this memory node has enhanced cognitive capabilities
    pub fn is_enhanced(&self) -> bool {
        self.quantum_signature.is_some()
            || self.evolution_metadata.is_some()
            || !self.attention_weights.is_empty()
    }
}

impl From<crate::memory::MemoryNode> for CognitiveMemoryNode {
    fn from(memory_node: crate::memory::MemoryNode) -> Self {
        Self {
            base_memory: memory_node,
            cognitive_state: CognitiveState::default(),
            quantum_signature: None,
            evolution_metadata: None,
            attention_weights: Vec::new(),
            semantic_relationships: Vec::new(),
        }
    }
}

/// Configuration settings for cognitive memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveSettings {
    pub enable_quantum_routing: bool,
    pub enable_evolution: bool,
    pub enable_attention_mechanism: bool,
    pub max_cognitive_load: f32,
    pub quantum_coherence_threshold: f32,
    pub evolution_mutation_rate: f32,
    pub attention_decay_rate: f32,
    pub meta_awareness_level: f32,
    pub attention_heads: usize,
    pub quantum_coherence_time: f64,
    pub enabled: bool,
}

impl Default for CognitiveSettings {
    fn default() -> Self {
        Self {
            enable_quantum_routing: true,
            enable_evolution: true,
            enable_attention_mechanism: true,
            max_cognitive_load: 1.0,
            quantum_coherence_threshold: 0.8,
            evolution_mutation_rate: 0.1,
            attention_decay_rate: 0.95,
            meta_awareness_level: 0.7,
            attention_heads: 8,
            quantum_coherence_time: 0.1,
            enabled: true,
        }
    }
}

/// Quantum entanglement between memories or agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementBond {
    pub target_id: String,
    pub bond_strength: f32,
    pub entanglement_type: EntanglementType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntanglementType {
    Semantic,
    Temporal,
    Causal,
    Emergent,
    Werner,
    Weak,
    Bell,
    BellPair,
}

/// Evolution metadata tracking system development
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionMetadata {
    pub generation: u32,
    pub fitness_score: f32,
    pub mutation_history: Vec<MutationEvent>,
    pub specialization_domains: Vec<SpecializationDomain>,
    pub adaptation_rate: f32,
}

impl EvolutionMetadata {
    /// Create new evolution metadata
    pub fn new() -> Self {
        Self {
            generation: 0,
            fitness_score: 0.0,
            mutation_history: Vec::new(),
            specialization_domains: Vec::new(),
            adaptation_rate: 0.1,
        }
    }
}

/// A mutation event in system evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub mutation_type: MutationType,
    pub impact_score: f32,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutationType {
    AttentionWeightAdjustment,
    RoutingStrategyModification,
    ContextualUnderstandingEvolution,
    QuantumCoherenceOptimization,
    EmergentPatternRecognition,
}

/// Specialization domains for agent evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpecializationDomain {
    SemanticProcessing,
    TemporalAnalysis,
    CausalReasoning,
    PatternRecognition,
    ContextualUnderstanding,
    PredictiveModeling,
    MetaCognition,
}

/// Routing decision with confidence and alternatives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub strategy: RoutingStrategy,
    pub target_context: String,
    pub confidence: f32,
    pub alternatives: Vec<AlternativeRoute>,
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RoutingStrategy {
    Quantum,
    Attention,
    Causal,
    Emergent,
    Hybrid(Vec<RoutingStrategy>),
}

/// Alternative routing option
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeRoute {
    pub strategy: RoutingStrategy,
    pub confidence: f32,
    pub estimated_quality: f32,
}

/// Enhanced query with cognitive understanding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedQuery {
    pub original: String,
    pub intent: QueryIntent,
    pub context: String,
    pub priority: f32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub context_embedding: Vec<f32>,
    pub temporal_context: Option<TemporalContext>,
    pub cognitive_hints: Vec<String>,
    pub expected_complexity: f32,
}

impl Default for EnhancedQuery {
    fn default() -> Self {
        Self {
            original: String::new(),
            intent: QueryIntent::Retrieval, // Default to Retrieval intent
            context: String::new(),
            priority: 1.0, // Default normal priority
            timestamp: chrono::Utc::now(),
            context_embedding: Vec::new(),
            temporal_context: None,
            cognitive_hints: Vec::new(),
            expected_complexity: 0.5, // Medium complexity by default
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryIntent {
    Retrieval,
    Association,
    Prediction,
    Reasoning,
    Exploration,
    Creation,
}

/// Emergent pattern discovered by the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentPattern {
    pub id: Uuid,
    pub pattern_type: PatternType,
    pub strength: f32,
    pub affected_memories: Vec<String>,
    pub discovery_timestamp: chrono::DateTime<chrono::Utc>,
    pub description: String,
}

/// Impact factors for evaluation committee decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactFactors {
    pub alignment_score: f64,
    pub quality_score: f64,
    pub safety_score: f64,
    pub confidence: f64,
    pub improvement_suggestions: Vec<String>,
    pub potential_risks: Vec<String>,
    pub latency_factor: f64,
    pub memory_factor: f64,
    pub relevance_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    Temporal,
    Semantic,
    Causal,
    Behavioral,
    Structural,
}

/// Cognitive error types
#[derive(Debug, thiserror::Error)]
pub enum CognitiveError {
    #[error("Quantum routing error: {0}")]
    QuantumRoutingError(String),
    #[error("Quantum decoherence occurred: {0}")]
    QuantumDecoherence(String),

    #[error("Attention overflow: {0}")]
    AttentionOverflow(String),

    #[error("Evolution failure: {0}")]
    EvolutionFailure(String),

    #[error("Meta-consciousness error: {0}")]
    MetaConsciousnessError(String),

    #[error("Context processing error: {0}")]
    ContextProcessingError(String),

    #[error("Routing error: {0}")]
    RoutingError(String),

    #[error("Cognitive capacity exceeded: {0}")]
    CapacityExceeded(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Memory operation failed: {0}")]
    MemoryError(String),

    #[error("MCTS node not found")]
    NodeNotFound,

    #[error("MCTS child node not found")]
    NoChildFound,

    #[error("MCTS node has no untried actions")]
    NoUntriedActions,

    #[error("No result from MCTS search")]
    NoResult,

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Optimization error: {0}")]
    OptimizationError(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Orchestration error: {0}")]
    OrchestrationError(String),

    #[error("Specification error: {0}")]
    SpecError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Evaluation failed: {0}")]
    EvaluationFailed(String),

    #[error("Consensus failed: {0}")]
    ConsensusFailed(String),
}

impl From<crate::cognitive::quantum::router::QuantumRouterError> for CognitiveError {
    fn from(error: crate::cognitive::quantum::router::QuantumRouterError) -> Self {
        match error {
            crate::cognitive::quantum::router::QuantumRouterError::SuperpositionError(msg) => {
                CognitiveError::QuantumDecoherence(msg)
            }
            crate::cognitive::quantum::router::QuantumRouterError::EntanglementError(msg) => {
                CognitiveError::QuantumDecoherence(msg)
            }
            crate::cognitive::quantum::router::QuantumRouterError::MeasurementError(msg) => {
                CognitiveError::QuantumDecoherence(msg)
            }
            crate::cognitive::quantum::router::QuantumRouterError::ValidationError(msg) => {
                CognitiveError::ContextProcessingError(msg)
            }
            crate::cognitive::quantum::router::QuantumRouterError::IoError(e) => {
                CognitiveError::ContextProcessingError(e.to_string())
            }
        }
    }
}

impl From<serde_json::Error> for CognitiveError {
    fn from(error: serde_json::Error) -> Self {
        CognitiveError::ParseError(error.to_string())
    }
}

pub type CognitiveResult<T> = Result<T, CognitiveError>;

/// Specification for optimization operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSpec {
    pub objective: String,
    pub improvement_threshold: f32,
    pub constraints: Vec<String>,
    pub success_criteria: Vec<String>,
    pub optimization_type: OptimizationType,
    pub timeout_ms: Option<u64>,
    pub max_iterations: Option<u32>,
    pub target_quality: f32,
    pub baseline_metrics: BaselineMetrics,
    pub content_type: ContentType,
    pub evolution_rules: EvolutionRules,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    Performance,
    Quality,
    Efficiency,
    Accuracy,
    Custom(String),
}

/// Result of optimization operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationOutcome {
    Success {
        improvements: Vec<String>,
        performance_gain: f32,
        quality_score: f32,
        metadata: HashMap<String, serde_json::Value>,
        applied: bool,
    },
    PartialSuccess {
        improvements: Vec<String>,
        issues: Vec<String>,
        performance_gain: f32,
        quality_score: f32,
        applied: bool,
    },
    Failure {
        errors: Vec<String>,
        root_cause: String,
        suggestions: Vec<String>,
        applied: bool,
    },
}

impl OptimizationOutcome {
    /// Get whether this optimization was applied
    pub fn applied(&self) -> bool {
        match self {
            OptimizationOutcome::Success { applied, .. } => *applied,
            OptimizationOutcome::PartialSuccess { applied, .. } => *applied,
            OptimizationOutcome::Failure { applied, .. } => *applied,
        }
    }
}

/// Async optimization result wrapper
pub struct PendingOptimizationResult {
    rx: tokio::sync::oneshot::Receiver<CognitiveResult<OptimizationOutcome>>,
}

impl PendingOptimizationResult {
    pub fn new(rx: tokio::sync::oneshot::Receiver<CognitiveResult<OptimizationOutcome>>) -> Self {
        Self { rx }
    }
}

/// Content type classification for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentType {
    pub category: ContentCategory,
    pub complexity: f32,
    pub processing_hints: Vec<String>,
    pub format: String,
    pub restrictions: Restrictions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentCategory {
    Text,
    Code,
    Data,
    Media,
    Structured,
    Unstructured,
}

/// Restrictions for content processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Restrictions {
    pub max_memory_usage: Option<u64>,
    pub max_processing_time: Option<u64>,
    pub allowed_operations: Vec<String>,
    pub forbidden_operations: Vec<String>,
    pub security_level: SecurityLevel,
    pub compiler: String,
    pub max_latency_increase: f64,
    pub max_memory_increase: f64,
    pub min_relevance_improvement: f64,
}

impl Default for Restrictions {
    fn default() -> Self {
        Self {
            max_memory_usage: None,
            max_processing_time: None,
            allowed_operations: vec!["quantum_optimization".to_string()],
            forbidden_operations: vec![],
            security_level: SecurityLevel::Internal,
            compiler: "default".to_string(),
            max_latency_increase: 0.1,       // 10% max latency increase
            max_memory_increase: 0.1,        // 10% max memory increase
            min_relevance_improvement: 0.01, // 1% minimum relevance improvement
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityLevel {
    Public,
    Internal,
    Confidential,
    Restricted,
}

/// Constraints for optimization operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraints {
    pub memory_limit: Option<u64>,
    pub time_limit: Option<u64>,
    pub quality_threshold: f32,
    pub resource_constraints: Vec<ResourceConstraint>,
    pub size: usize,
    pub style: Vec<String>,
    pub schemas: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraint {
    pub resource_type: String,
    pub max_usage: f32,
    pub priority: f32,
}

/// Evolution rules for cognitive development
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionRules {
    pub mutation_rate: f32,
    pub selection_pressure: f32,
    pub crossover_rate: f32,
    pub elite_retention: f32,
    pub diversity_maintenance: f32,
    pub allowed_mutations: Vec<MutationType>,
    pub build_on_previous: bool,
    pub new_axis_per_iteration: bool,
    pub max_cumulative_latency_increase: f64,
    pub min_action_diversity: f64,
    pub validation_required: bool,
}

/// Performance metrics for cognitive operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub adaptation_rate: f64,
    pub memory_efficiency: f64,
    pub response_latency: f64,
    pub cost_savings: f64,
    pub accuracy: f64,
    pub relevance_gain: f64,
}

/// Baseline performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMetrics {
    pub response_time: f32,
    pub accuracy: f32,
    pub throughput: f32,
    pub resource_usage: f32,
    pub error_rate: f32,
    pub quality_score: f32,
    pub latency: f64,
    pub memory: f64,
    pub relevance: f64,
}

impl std::future::Future for PendingOptimizationResult {
    type Output = CognitiveResult<OptimizationOutcome>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        match std::pin::Pin::new(&mut self.rx).poll(cx) {
            std::task::Poll::Ready(Ok(result)) => std::task::Poll::Ready(result),
            std::task::Poll::Ready(Err(_)) => std::task::Poll::Ready(Err(
                CognitiveError::ContextProcessingError("Channel closed".to_string()),
            )),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}
