//! Quantum Monte Carlo Tree Search Implementation
//!
//! This module provides a high-performance, concurrent quantum MCTS implementation
//! using atomic operations and concurrent data structures for maximum performance.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrayvec::ArrayVec;
use crossbeam_skiplist::SkipMap;
use crossbeam_utils::CachePadded;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use smallvec::{SmallVec, smallvec};
use tracing::info;

use crate::cognitive::mcts::CodeState;
use crate::cognitive::quantum::QuantumConfig;
use crate::cognitive::types::CognitiveError;

/// Simple atomic wrapper for f64
#[derive(Debug)]
pub struct Atomic<T>(std::sync::atomic::AtomicU64, std::marker::PhantomData<T>);

impl Atomic<f64> {
    pub fn new(value: f64) -> Self {
        Self(
            std::sync::atomic::AtomicU64::new(value.to_bits()),
            std::marker::PhantomData,
        )
    }

    pub fn store(&self, value: f64, ordering: std::sync::atomic::Ordering) {
        self.0.store(value.to_bits(), ordering);
    }

    pub fn load(&self, ordering: std::sync::atomic::Ordering) -> f64 {
        f64::from_bits(self.0.load(ordering))
    }
}

/// Atomic wrapper for Complex64
#[derive(Debug)]
pub struct AtomicComplex64 {
    real: std::sync::atomic::AtomicU64,
    imag: std::sync::atomic::AtomicU64,
}

impl AtomicComplex64 {
    pub fn new(value: Complex64) -> Self {
        Self {
            real: std::sync::atomic::AtomicU64::new(value.re.to_bits()),
            imag: std::sync::atomic::AtomicU64::new(value.im.to_bits()),
        }
    }

    pub fn store(&self, value: Complex64, ordering: std::sync::atomic::Ordering) {
        self.real.store(value.re.to_bits(), ordering);
        self.imag.store(value.im.to_bits(), ordering);
    }

    pub fn load(&self, ordering: std::sync::atomic::Ordering) -> Complex64 {
        Complex64::new(
            f64::from_bits(self.real.load(ordering)),
            f64::from_bits(self.imag.load(ordering)),
        )
    }
}

/// Quantum node state for MCTS nodes
#[derive(Debug, Clone)]
pub struct QuantumNodeState {
    pub classical_state: CodeState,
    pub superposition_coefficients: Vec<Complex64>,
    pub entangled_nodes: Vec<String>,
    pub decoherence: f64,
    pub measurement_history: Vec<Complex64>,
}

/// Atomic quantum metrics for concurrent tracking
#[derive(Debug)]
pub struct AtomicQuantumMetrics {
    pub total_simulations: CachePadded<AtomicU64>,
    pub successful_expansions: CachePadded<AtomicU64>,
    pub quantum_measurements: CachePadded<AtomicU64>,
    pub entanglement_operations: CachePadded<AtomicU64>,
    pub decoherence_events: CachePadded<AtomicU64>,
}

impl AtomicQuantumMetrics {
    pub fn new() -> Self {
        Self {
            total_simulations: CachePadded::new(AtomicU64::new(0)),
            successful_expansions: CachePadded::new(AtomicU64::new(0)),
            quantum_measurements: CachePadded::new(AtomicU64::new(0)),
            entanglement_operations: CachePadded::new(AtomicU64::new(0)),
            decoherence_events: CachePadded::new(AtomicU64::new(0)),
        }
    }

    #[inline]
    pub fn inc_simulations(&self) {
        self.total_simulations.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_expansions(&self) {
        self.successful_expansions.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_measurements(&self) {
        self.quantum_measurements.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_entanglements(&self) {
        self.entanglement_operations.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_decoherence(&self) {
        self.decoherence_events.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> QuantumMetricsSnapshot {
        QuantumMetricsSnapshot {
            total_simulations: self.total_simulations.load(Ordering::Relaxed),
            successful_expansions: self.successful_expansions.load(Ordering::Relaxed),
            quantum_measurements: self.quantum_measurements.load(Ordering::Relaxed),
            entanglement_operations: self.entanglement_operations.load(Ordering::Relaxed),
            decoherence_events: self.decoherence_events.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of quantum metrics for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMetricsSnapshot {
    pub total_simulations: u64,
    pub successful_expansions: u64,
    pub quantum_measurements: u64,
    pub entanglement_operations: u64,
    pub decoherence_events: u64,
}

/// Quantum MCTS node with concurrent data structures
#[derive(Debug)]
pub struct QuantumMCTSNode {
    pub id: ArrayVec<u8, 64>,
    pub quantum_state: QuantumNodeState,
    pub visits: CachePadded<AtomicU64>,
    pub quantum_reward: AtomicComplex64,
    pub amplitude: AtomicComplex64,
    pub children: Arc<SkipMap<ArrayVec<u8, 64>, ArrayVec<u8, 64>>>,
    pub untried_actions: Arc<SkipMap<ArrayVec<u8, 128>, bool>>,
    pub is_terminal: bool,
}

impl QuantumMCTSNode {
    pub fn new(id: ArrayVec<u8, 64>, state: QuantumNodeState) -> Self {
        Self {
            id,
            quantum_state: state,
            visits: CachePadded::new(AtomicU64::new(0)),
            quantum_reward: AtomicComplex64::new(Complex64::new(0.0, 0.0)),
            amplitude: AtomicComplex64::new(Complex64::new(1.0, 0.0)),
            children: Arc::new(SkipMap::new()),
            untried_actions: Arc::new(SkipMap::new()),
            is_terminal: false,
        }
    }

    #[inline]
    pub fn add_visit(&self) {
        self.visits.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn update_reward(&self, reward: Complex64) {
        let current = self.quantum_reward.load(Ordering::Relaxed);
        self.quantum_reward
            .store(current + reward, Ordering::Relaxed);
    }

    #[inline]
    pub fn get_visit_count(&self) -> u64 {
        self.visits.load(Ordering::Relaxed)
    }
}

/// Main Quantum MCTS implementation
#[derive(Debug)]
pub struct QuantumMCTS {
    pub tree: Arc<SkipMap<ArrayVec<u8, 64>, Arc<QuantumMCTSNode>>>,
    pub root_id: ArrayVec<u8, 64>,
    pub user_objective: ArrayVec<u8, 256>,
    pub entanglement_graph: Arc<SkipMap<ArrayVec<u8, 64>, SmallVec<ArrayVec<u8, 64>, 4>>>,
    pub metrics: Arc<AtomicQuantumMetrics>,
    pub config: QuantumConfig,
    pub node_counter: CachePadded<AtomicU64>,
    pub action_buffer: SmallVec<ArrayVec<u8, 128>, 8>,
}

impl QuantumMCTS {
    /// Create new quantum MCTS instance
    pub fn new(
        initial_state: CodeState,
        user_objective: String,
        config: QuantumConfig,
    ) -> Result<Self, CognitiveError> {
        let root_id = Self::str_to_arrayvec("root");
        let user_obj_vec = Self::str_to_user_objective_arrayvec(&user_objective);

        let quantum_state = QuantumNodeState {
            classical_state: initial_state,
            superposition_coefficients: vec![Complex64::new(1.0, 0.0)],
            entangled_nodes: Vec::new(),
            decoherence: 0.0,
            measurement_history: Vec::new(),
        };

        let root_node = Arc::new(QuantumMCTSNode::new(root_id.clone(), quantum_state));
        let tree = Arc::new(SkipMap::new());
        tree.insert(root_id.clone(), root_node);

        Ok(Self {
            tree,
            root_id,
            user_objective: user_obj_vec,
            entanglement_graph: Arc::new(SkipMap::new()),
            metrics: Arc::new(AtomicQuantumMetrics::new()),
            config,
            node_counter: CachePadded::new(AtomicU64::new(1)),
            action_buffer: smallvec![],
        })
    }

    /// Run quantum MCTS for specified iterations
    pub fn run(&mut self, iterations: usize) -> Result<CodeState, CognitiveError> {
        info!("Starting quantum MCTS with {} iterations", iterations);

        for i in 0..iterations {
            // Selection phase
            let selected_node_id = self.select_node()?;

            // Expansion phase
            let expanded_node_id = self.expand_node(&selected_node_id)?;

            // Simulation phase
            let reward = self.simulate(&expanded_node_id)?;

            // Backpropagation phase
            self.backpropagate(&expanded_node_id, reward)?;

            // Apply quantum effects periodically
            if i % 10 == 0 {
                self.apply_quantum_effects()?;
            }
        }

        // Get best result
        let best_state =
            self.get_best_classical_state()
                .ok_or(CognitiveError::OptimizationError(
                    "No best state found".to_string(),
                ))?;

        info!(
            "Quantum MCTS completed with {} total nodes",
            self.tree.len()
        );
        Ok(best_state.classical_state)
    }

    /// Select node using quantum UCB
    #[inline]
    fn select_node(&self) -> Result<ArrayVec<u8, 64>, CognitiveError> {
        let mut current_id = self.root_id.clone();

        loop {
            let current_node = self
                .tree
                .get(&current_id)
                .ok_or(CognitiveError::OptimizationError(
                    "Node not found".to_string(),
                ))?
                .value()
                .clone();

            if current_node.is_terminal || !current_node.untried_actions.is_empty() {
                return Ok(current_id);
            }

            current_id = self.best_child(&current_id)?;
        }
    }

    /// Find best child using quantum UCB
    #[inline]
    fn best_child(&self, parent_id: &ArrayVec<u8, 64>) -> Result<ArrayVec<u8, 64>, CognitiveError> {
        let parent_node = self
            .tree
            .get(parent_id)
            .ok_or(CognitiveError::OptimizationError(
                "Parent node not found".to_string(),
            ))?
            .value()
            .clone();

        let parent_visits = parent_node.get_visit_count() as f64;
        let mut best_child_id = None;
        let mut best_ucb = f64::NEG_INFINITY;

        for entry in parent_node.children.iter() {
            let child_id = entry.key();
            if let Some(child_entry) = self.tree.get(child_id) {
                let child_node = child_entry.value();
                let ucb_value = self.quantum_ucb(child_node, parent_visits);

                if ucb_value > best_ucb {
                    best_ucb = ucb_value;
                    best_child_id = Some(child_id.clone());
                }
            }
        }

        best_child_id.ok_or(CognitiveError::OptimizationError(
            "No child found".to_string(),
        ))
    }

    /// Calculate quantum UCB value
    #[inline]
    fn quantum_ucb(&self, node: &QuantumMCTSNode, parent_visits: f64) -> f64 {
        let visits = node.get_visit_count();
        if visits == 0 {
            return f64::INFINITY;
        }

        let visits_f = visits as f64;
        let reward = node.quantum_reward.load(Ordering::Relaxed);
        let avg_reward = reward.re / visits_f;

        // Quantum amplitude factor
        let amplitude = node.amplitude.load(Ordering::Relaxed);
        let amplitude_factor = amplitude.norm();

        // Standard UCB with quantum enhancement
        let exploration = self.config.exploration_constant * (parent_visits.ln() / visits_f).sqrt();
        let quantum_bonus = amplitude_factor * 0.1;

        avg_reward + exploration + quantum_bonus
    }

    /// Expand node by adding new child
    #[inline]
    fn expand_node(&self, node_id: &ArrayVec<u8, 64>) -> Result<ArrayVec<u8, 64>, CognitiveError> {
        let node = self
            .tree
            .get(node_id)
            .ok_or(CognitiveError::OptimizationError(
                "Node not found for expansion".to_string(),
            ))?
            .value()
            .clone();

        if node.is_terminal {
            return Ok(node_id.clone());
        }

        // Get possible actions
        let actions = self.get_possible_actions(&node.quantum_state.classical_state);
        if actions.is_empty() {
            return Ok(node_id.clone());
        }

        // Select first untried action
        let action = &actions[0];

        // Generate new node ID
        let new_node_id = self.generate_node_id();

        // Create new quantum state
        let new_state = self.apply_action(&node.quantum_state.classical_state, action)?;
        let quantum_state = QuantumNodeState {
            classical_state: new_state,
            superposition_coefficients: vec![Complex64::new(0.8, 0.2)],
            entangled_nodes: Vec::new(),
            decoherence: 0.1,
            measurement_history: Vec::new(),
        };

        // Create and insert new node
        let new_node = Arc::new(QuantumMCTSNode::new(new_node_id.clone(), quantum_state));
        self.tree.insert(new_node_id.clone(), new_node);

        // Add child relationship
        node.children
            .insert(new_node_id.clone(), new_node_id.clone());

        self.metrics.inc_expansions();
        Ok(new_node_id)
    }

    /// Simulate from node to get reward
    #[inline]
    fn simulate(&self, node_id: &ArrayVec<u8, 64>) -> Result<Complex64, CognitiveError> {
        let node = self
            .tree
            .get(node_id)
            .ok_or(CognitiveError::OptimizationError(
                "Node not found for simulation".to_string(),
            ))?
            .value()
            .clone();

        let state = &node.quantum_state.classical_state;

        // Calculate quantum reward based on multiple factors
        let latency_reward = 1.0 / (1.0 + state.latency / 100.0);
        let memory_reward = 1.0 / (1.0 + state.memory / 1000.0);
        let relevance_reward = state.relevance;

        // Quantum interference effects
        let amplitude = node.amplitude.load(Ordering::Relaxed);
        let phase_factor = amplitude.arg().cos();

        // Decoherence penalty
        let decoherence_penalty = node.quantum_state.decoherence;

        let total_reward = (latency_reward + memory_reward + relevance_reward) * phase_factor
            - decoherence_penalty;
        let quantum_reward = Complex64::new(total_reward, phase_factor * 0.1);

        self.metrics.inc_simulations();
        Ok(quantum_reward)
    }

    /// Backpropagate reward through tree
    #[inline]
    fn backpropagate(
        &self,
        node_id: &ArrayVec<u8, 64>,
        reward: Complex64,
    ) -> Result<(), CognitiveError> {
        let current_id = node_id.clone();

        loop {
            if let Some(entry) = self.tree.get(&current_id) {
                let node = entry.value();
                node.add_visit();
                node.update_reward(reward);

                // Find parent (simplified - in practice would maintain parent links)
                if current_id == self.root_id {
                    break;
                }

                // For now, just update the current node and break
                // In a full implementation, we'd traverse up the tree
                break;
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Apply quantum effects like decoherence and entanglement
    #[inline]
    fn apply_quantum_effects(&self) -> Result<(), CognitiveError> {
        // Apply decoherence to all nodes
        for entry in self.tree.iter() {
            let node = entry.value();
            let current_amplitude = node.amplitude.load(Ordering::Relaxed);

            // Apply decoherence
            let decoherence_factor = 1.0 - self.config.decoherence_rate;
            let new_amplitude = current_amplitude * decoherence_factor;
            node.amplitude.store(new_amplitude, Ordering::Relaxed);
        }

        // Update entanglements
        self.update_entanglements()?;

        Ok(())
    }

    /// Update quantum entanglements between nodes
    #[inline]
    fn update_entanglements(&self) -> Result<(), CognitiveError> {
        let mut entanglement_count = 0;

        for entry1 in self.tree.iter() {
            let node1_id = entry1.key();
            let node1 = entry1.value();

            for entry2 in self.tree.iter() {
                let node2_id = entry2.key();
                if node1_id != node2_id {
                    let node2 = entry2.value();

                    // Calculate entanglement strength
                    let similarity = self.calculate_quantum_similarity(node1, node2);

                    if similarity > self.config.entanglement_probability {
                        // Create entanglement
                        let entangled_nodes = SmallVec::from_vec(vec![node2_id.clone()]);
                        self.entanglement_graph
                            .insert(node1_id.clone(), entangled_nodes);
                        entanglement_count += 1;
                    }
                }
            }
        }

        // Record the number of entanglements created
        for _ in 0..entanglement_count {
            self.metrics.inc_entanglements();
        }

        Ok(())
    }

    /// Calculate quantum similarity between nodes
    #[inline]
    fn calculate_quantum_similarity(
        &self,
        node1: &QuantumMCTSNode,
        node2: &QuantumMCTSNode,
    ) -> f64 {
        let amp1 = node1.amplitude.load(Ordering::Relaxed);
        let amp2 = node2.amplitude.load(Ordering::Relaxed);

        // Quantum fidelity calculation
        let fidelity = (amp1.conj() * amp2).norm();
        fidelity
    }

    /// Get best classical state from quantum tree
    fn get_best_classical_state(&self) -> Option<QuantumNodeState> {
        let mut best_node = None;
        let mut best_reward = f64::NEG_INFINITY;

        for entry in self.tree.iter() {
            let node = entry.value();
            let reward = node.quantum_reward.load(Ordering::Relaxed).re;
            let visits = node.get_visit_count();

            // Only consider nodes with sufficient visits
            if visits > 0 && reward > best_reward {
                best_reward = reward;
                best_node = Some(node.quantum_state.clone());
            }
        }

        best_node
    }

    /// Get possible actions for a given code state
    fn get_possible_actions(&self, state: &CodeState) -> SmallVec<ArrayVec<u8, 128>, 8> {
        let mut actions = smallvec![
            Self::str_to_action_arrayvec("refactor_complex_function"),
            Self::str_to_action_arrayvec("optimize_hot_path"),
            Self::str_to_action_arrayvec("add_logging"),
            Self::str_to_action_arrayvec("remove_dead_code"),
        ];

        // Add conditional actions based on state
        if state.latency > 100.0 {
            actions.push(Self::str_to_action_arrayvec("focus_latency_reduction"));
        }

        if state.memory > 1000.0 {
            actions.push(Self::str_to_action_arrayvec("focus_memory_reduction"));
        }

        actions
    }

    /// Apply action to code state
    fn apply_action(
        &self,
        state: &CodeState,
        action: &ArrayVec<u8, 128>,
    ) -> Result<CodeState, CognitiveError> {
        let action_str = std::str::from_utf8(action).map_err(|_| {
            CognitiveError::OptimizationError("Invalid action encoding".to_string())
        })?;

        let mut new_state = state.clone();

        match action_str {
            "refactor_complex_function" => {
                new_state.latency *= 0.9;
                new_state.relevance += 0.1;
            }
            "optimize_hot_path" => {
                new_state.latency *= 0.8;
                new_state.memory *= 0.95;
            }
            "add_logging" => {
                new_state.relevance += 0.05;
                new_state.memory *= 1.02;
            }
            "remove_dead_code" => {
                new_state.memory *= 0.9;
                new_state.relevance += 0.02;
            }
            "focus_latency_reduction" => {
                new_state.latency *= 0.7;
            }
            "focus_memory_reduction" => {
                new_state.memory *= 0.8;
            }
            _ => {
                // Default improvement
                new_state.relevance += 0.01;
            }
        }

        Ok(new_state)
    }

    /// Generate unique node ID
    #[inline]
    fn generate_node_id(&self) -> ArrayVec<u8, 64> {
        let id = self.node_counter.fetch_add(1, Ordering::Relaxed);
        Self::str_to_arrayvec(&format!("node_{}", id))
    }

    /// Convert string to ArrayVec for zero-allocation storage (node IDs)
    #[inline]
    fn str_to_arrayvec(s: &str) -> ArrayVec<u8, 64> {
        let mut vec = ArrayVec::new();
        for byte in s.bytes().take(64) {
            if vec.try_push(byte).is_err() {
                break;
            }
        }
        vec
    }

    /// Convert string to action ArrayVec for zero-allocation storage (actions)
    #[inline]
    fn str_to_action_arrayvec(s: &str) -> ArrayVec<u8, 128> {
        let mut vec = ArrayVec::new();
        for byte in s.bytes().take(128) {
            if vec.try_push(byte).is_err() {
                break;
            }
        }
        vec
    }

    /// Convert string to user objective ArrayVec for zero-allocation storage (user objectives)
    #[inline]
    fn str_to_user_objective_arrayvec(s: &str) -> ArrayVec<u8, 256> {
        let mut vec = ArrayVec::new();
        for byte in s.bytes().take(256) {
            if vec.try_push(byte).is_err() {
                break;
            }
        }
        vec
    }

    /// Get quantum statistics for analysis
    pub fn get_quantum_statistics(&self) -> QuantumTreeStatistics {
        let mut total_nodes = 0;
        let mut total_visits = 0;
        let mut max_amplitude = 0.0;
        let mut decoherence_sum = 0.0;

        for entry in self.tree.iter() {
            let node = entry.value();
            total_nodes += 1;
            total_visits += node.get_visit_count();

            let amplitude = node.amplitude.load(Ordering::Relaxed).norm();
            if amplitude > max_amplitude {
                max_amplitude = amplitude;
            }

            decoherence_sum += node.quantum_state.decoherence;
        }

        let total_entanglements = self.entanglement_graph.len();
        let avg_decoherence = if total_nodes > 0 {
            decoherence_sum / total_nodes as f64
        } else {
            0.0
        };

        QuantumTreeStatistics {
            total_nodes,
            total_visits,
            total_entanglements,
            avg_decoherence,
            max_amplitude,
            quantum_metrics: self.metrics.snapshot(),
        }
    }

    /// Recursively improve the quantum state through MCTS iterations
    pub fn recursive_improve(
        &mut self,
        max_iterations: usize,
    ) -> Result<CodeState, CognitiveError> {
        // Use the existing run method which implements the same logic
        self.run(max_iterations)
    }

    /// Get the best quantum modification found during search
    pub fn best_quantum_modification(&self) -> Result<CodeState, CognitiveError> {
        // Find the node with the highest quantum UCB score
        let mut best_state = CodeState::default();
        let mut best_score = 0.0;

        // Iterate through the tree to find the best node
        for entry in self.tree.iter() {
            let node = entry.value();
            let score = self.quantum_ucb(node, 1000.0); // Use a reasonable parent visit count
            if score > best_score {
                best_score = score;
                best_state = node.quantum_state.classical_state.clone();
            }
        }

        Ok(best_state)
    }
}

/// Statistics about the quantum tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumTreeStatistics {
    pub total_nodes: usize,
    pub total_visits: u64,
    pub total_entanglements: usize,
    pub avg_decoherence: f64,
    pub max_amplitude: f64,
    pub quantum_metrics: QuantumMetricsSnapshot,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_mcts_creation() {
        let initial_state = CodeState {
            code: "test code".to_string(),
            latency: 100.0,
            memory: 500.0,
            relevance: 0.8,
        };

        let config = QuantumConfig {
            exploration_constant: 1.414,
            decoherence_rate: 0.01,
            entanglement_probability: 0.1,
        };

        let mcts = QuantumMCTS::new(initial_state, "test objective".to_string(), config);
        assert!(mcts.is_ok());
    }

    #[test]
    fn test_atomic_metrics() {
        let metrics = AtomicQuantumMetrics::new();
        metrics.inc_simulations();
        metrics.inc_expansions();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_simulations, 1);
        assert_eq!(snapshot.successful_expansions, 1);
    }
}
