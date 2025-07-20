//! Quantum superposition state management

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::cognitive::quantum::complex::Complex64;
use crate::cognitive::quantum::entanglement::EntanglementLink;

/// Quantum superposition state with full quantum properties
#[derive(Debug, Clone)]
pub struct SuperpositionState {
    pub probability_amplitudes: BTreeMap<String, Complex64>,
    pub coherence_time: Duration,
    pub last_observation: Option<Instant>,
    pub entangled_memories: Vec<EntanglementLink>,
    pub phase_evolution: PhaseEvolution,
    pub decoherence_rate: f64,
    pub creation_time: Instant,
    pub observation_count: u64,
}

/// Phase evolution tracking for quantum states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseEvolution {
    pub initial_phase: f64,
    pub evolution_rate: f64,
    pub hamiltonian_coefficients: Vec<f64>,
    pub time_dependent_terms: Vec<TimeDependentTerm>,
}

/// Time-dependent term for Hamiltonian evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeDependentTerm {
    pub amplitude: f64,
    pub frequency: f64,
    pub phase_offset: f64,
}

impl SuperpositionState {
    /// Create a new superposition state
    pub fn new(coherence_time: Duration) -> Self {
        Self {
            probability_amplitudes: BTreeMap::new(),
            coherence_time,
            last_observation: None,
            entangled_memories: Vec::new(),
            phase_evolution: PhaseEvolution::default(),
            decoherence_rate: 0.01,
            creation_time: Instant::now(),
            observation_count: 0,
        }
    }

    /// Add a quantum state with given amplitude
    pub fn add_state(&mut self, label: String, amplitude: Complex64) {
        self.probability_amplitudes.insert(label, amplitude);
    }

    /// Map error for chaining operations
    pub fn map_err<F, E>(self, _f: F) -> Result<Self, E>
    where
        F: FnOnce(String) -> E,
    {
        Ok(self)
    }

    /// Rotate around X axis (quantum gate operation)
    pub fn rotate_x(&mut self, angle: f64) -> &mut Self {
        // Apply rotation to all amplitudes
        for amplitude in self.probability_amplitudes.values_mut() {
            let cos_half = (angle / 2.0).cos();
            let sin_half = (angle / 2.0).sin();
            let new_amplitude = Complex64::new(
                amplitude.real * cos_half - amplitude.imaginary * sin_half,
                amplitude.real * sin_half + amplitude.imaginary * cos_half,
            );
            *amplitude = new_amplitude;
        }
        self
    }

    /// Add a basis state to the superposition
    pub fn add_basis_state(&mut self, label: String, amplitude: Complex64) {
        self.probability_amplitudes.insert(label, amplitude);
    }

    /// Apply phase rotation to the superposition
    pub fn apply_phase(&mut self, phase: f64) -> &mut Self {
        let phase_factor = Complex64::new(phase.cos(), phase.sin());
        for amplitude in self.probability_amplitudes.values_mut() {
            *amplitude = *amplitude * phase_factor;
        }
        self
    }

    /// Rotate around Z axis (quantum gate operation)
    pub fn rotate_z(&mut self, angle: f64) -> &mut Self {
        let phase_factor = Complex64::new(0.0, angle / 2.0).exp();
        for amplitude in self.probability_amplitudes.values_mut() {
            *amplitude = *amplitude * phase_factor;
        }
        self
    }

    /// Entangle this superposition with another memory
    pub fn entangle(&mut self, memory_id: String) {
        use crate::cognitive::types::EntanglementType;

        // Create a basic EntanglementLink
        let link = crate::cognitive::quantum::entanglement::EntanglementLink::new(
            memory_id,
            EntanglementType::Bell,
        );
        self.entangled_memories.push(link);
    }

    /// Normalize the superposition to maintain quantum constraint
    pub fn normalize(&mut self) -> Result<(), String> {
        let total_probability: f64 = self
            .probability_amplitudes
            .values()
            .map(|amp| amp.magnitude().powi(2))
            .sum();

        if total_probability == 0.0 {
            return Err("Cannot normalize: zero total probability".to_string());
        }

        let normalization_factor = total_probability.sqrt();
        for amplitude in self.probability_amplitudes.values_mut() {
            *amplitude = *amplitude / normalization_factor;
        }

        Ok(())
    }

    /// Check if the state is still coherent
    pub fn is_coherent(&self) -> bool {
        let elapsed = self.creation_time.elapsed();
        elapsed < self.coherence_time
    }

    /// Calculate the von Neumann entropy of the state
    pub fn entropy(&self) -> f64 {
        let mut entropy = 0.0;

        for amplitude in self.probability_amplitudes.values() {
            let probability = amplitude.magnitude().powi(2);
            if probability > 0.0 {
                entropy -= probability * probability.ln();
            }
        }

        entropy
    }

    /// Apply decoherence based on elapsed time
    pub fn apply_decoherence(&mut self, elapsed: Duration) {
        let decay_factor = (-self.decoherence_rate * elapsed.as_secs_f64()).exp();

        for amplitude in self.probability_amplitudes.values_mut() {
            *amplitude = *amplitude * decay_factor;
        }
    }

    /// Mark state as observed
    pub fn observe(&mut self) {
        self.last_observation = Some(Instant::now());
        self.observation_count += 1;
    }

    /// Measure the quantum state, collapsing it to a single basis state
    /// Returns the measured state and its amplitude
    pub fn measure(&self) -> Result<(String, Complex64), String> {
        let total_probability: f64 = self
            .probability_amplitudes
            .values()
            .map(|amp| amp.magnitude().powi(2))
            .sum();

        if total_probability <= 0.0 {
            return Err("Cannot measure state with zero probability".to_string());
        }

        let mut rng = rand::rng();
        let random_value: f64 = rng.random_range(0.0..total_probability);

        let mut cumulative_prob = 0.0;
        for (state, amplitude) in &self.probability_amplitudes {
            let prob = amplitude.magnitude().powi(2);
            cumulative_prob += prob;

            if random_value <= cumulative_prob {
                return Ok((state.clone(), *amplitude));
            }
        }

        // This should theoretically never be reached due to floating point precision
        Err("Measurement failed due to floating point precision issues".to_string())
    }
}

impl Default for PhaseEvolution {
    fn default() -> Self {
        Self {
            initial_phase: 0.0,
            evolution_rate: 1.0,
            hamiltonian_coefficients: Vec::new(),
            time_dependent_terms: Vec::new(),
        }
    }
}

impl PhaseEvolution {
    /// Calculate the phase at a given time
    pub fn phase_at_time(&self, time: f64) -> f64 {
        let mut phase = self.initial_phase + self.evolution_rate * time;

        // Add time-dependent contributions
        for term in &self.time_dependent_terms {
            phase += term.amplitude * (term.frequency * time + term.phase_offset).sin();
        }

        phase
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_superposition_normalization() {
        let mut state = SuperpositionState::new(Duration::from_secs(1));
        state.add_state("state1".to_string(), Complex64::new(0.6, 0.0));
        state.add_state("state2".to_string(), Complex64::new(0.8, 0.0));

        state.normalize().unwrap();

        let total_prob: f64 = state
            .probability_amplitudes
            .values()
            .map(|amp| amp.magnitude().powi(2))
            .sum();

        assert!((total_prob - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_calculation() {
        let mut state = SuperpositionState::new(Duration::from_secs(1));
        state.add_state("state1".to_string(), Complex64::new(1.0, 0.0));
        state.normalize().unwrap();

        // Single state should have zero entropy
        assert_eq!(state.entropy(), 0.0);

        // Equal superposition should have maximum entropy
        state.add_state("state2".to_string(), Complex64::new(1.0, 0.0));
        state.normalize().unwrap();

        let entropy = state.entropy();
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_measure() {
        let mut state = SuperpositionState::new(Duration::from_secs(1));

        // Test with a single state
        state.add_state("state1".to_string(), Complex64::new(1.0, 0.0));
        state.normalize().unwrap();

        let (measured_state, amplitude) = state.measure().unwrap();
        assert_eq!(measured_state, "state1");
        assert!((amplitude.real - 1.0).abs() < f64::EPSILON);
        assert!((amplitude.imaginary - 0.0).abs() < f64::EPSILON);

        // Test with multiple states
        state.add_state("state2".to_string(), Complex64::new(1.0, 0.0));
        state.normalize().unwrap();

        // Test multiple measurements to ensure we get both states (with some probability)
        let mut state1_count = 0;
        let mut state2_count = 0;

        for _ in 0..1000 {
            let (measured_state, _) = state.measure().unwrap();
            if measured_state == "state1" {
                state1_count += 1;
            } else if measured_state == "state2" {
                state2_count += 1;
            } else {
                panic!("Unexpected state: {}", measured_state);
            }
        }

        // Both states should be measured with roughly equal probability
        assert!(state1_count > 0, "state1 was never measured");
        assert!(state2_count > 0, "state2 was never measured");
        let ratio = state1_count as f64 / (state1_count + state2_count) as f64;
        assert!(
            (ratio - 0.5).abs() < 0.1,
            "Measured ratio {} is not close to 0.5",
            ratio
        );

        // Test with zero probability state
        let zero_state = SuperpositionState::new(Duration::from_secs(1));
        assert!(zero_state.measure().is_err());
    }
}
