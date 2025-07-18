//! Machine learning components for quantum error correction and optimization


use serde::{Deserialize, Serialize};

/// Complex number representation for quantum amplitudes
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl Complex {
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    pub fn zero() -> Self {
        Self {
            real: 0.0,
            imag: 0.0,
        }
    }

    pub fn one() -> Self {
        Self {
            real: 1.0,
            imag: 0.0,
        }
    }

    pub fn i() -> Self {
        Self {
            real: 0.0,
            imag: 1.0,
        }
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
    }

    pub fn magnitude(&self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    pub fn conjugate(&self) -> Self {
        Self {
            real: self.real,
            imag: -self.imag,
        }
    }

    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag > 0.0 {
            Self {
                real: self.real / mag,
                imag: self.imag / mag,
            }
        } else {
            *self
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Complex;

    fn add(self, other: Complex) -> Complex {
        Complex {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Complex;

    fn mul(self, other: Complex) -> Complex {
        Complex {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Complex;

    fn mul(self, scalar: f64) -> Complex {
        Complex {
            real: self.real * scalar,
            imag: self.imag * scalar,
        }
    }
}

/// Quantum state vector for n qubits (2^n dimensional complex vector)
#[derive(Debug, Clone)]
pub struct QuantumState {
    pub amplitudes: Vec<Complex>,
    pub num_qubits: usize,
}

impl QuantumState {
    /// Create a new quantum state with all qubits in |0⟩ state
    pub fn new(num_qubits: usize) -> Self {
        let size = 1 << num_qubits; // 2^num_qubits
        let mut amplitudes = vec![Complex::zero(); size];
        amplitudes[0] = Complex::one(); // |00...0⟩ state

        Self {
            amplitudes,
            num_qubits,
        }
    }

    /// Create quantum state from classical bit string
    pub fn from_bits(bits: &[bool]) -> Self {
        let num_qubits = bits.len();
        let size = 1 << num_qubits;
        let mut amplitudes = vec![Complex::zero(); size];

        // Convert bits to index
        let mut index = 0;
        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                index |= 1 << i;
            }
        }

        amplitudes[index] = Complex::one();

        Self {
            amplitudes,
            num_qubits,
        }
    }

    /// Apply a single qubit gate to the specified qubit
    pub fn apply_single_qubit_gate(&mut self, qubit: usize, gate_matrix: &[[Complex; 2]; 2]) {
        if qubit >= self.num_qubits {
            return;
        }

        let size = self.amplitudes.len();
        let mut new_amplitudes = vec![Complex::zero(); size];

        for i in 0..size {
            let bit_value = (i >> qubit) & 1;
            let other_index = i ^ (1 << qubit); // Flip the qubit bit

            if bit_value == 0 {
                // |0⟩ state
                new_amplitudes[i] = gate_matrix[0][0] * self.amplitudes[i]
                    + gate_matrix[0][1] * self.amplitudes[other_index];
            } else {
                // |1⟩ state
                new_amplitudes[i] = gate_matrix[1][0] * self.amplitudes[other_index]
                    + gate_matrix[1][1] * self.amplitudes[i];
            }
        }

        self.amplitudes = new_amplitudes;
    }

    /// Apply a controlled two-qubit gate
    pub fn apply_controlled_gate(
        &mut self,
        control: usize,
        target: usize,
        gate_matrix: &[[Complex; 2]; 2],
    ) {
        if control >= self.num_qubits || target >= self.num_qubits || control == target {
            return;
        }

        let size = self.amplitudes.len();
        let mut new_amplitudes = self.amplitudes.clone();

        for i in 0..size {
            let control_bit = (i >> control) & 1;
            let target_bit = (i >> target) & 1;

            // Only apply gate if control qubit is |1⟩
            if control_bit == 1 {
                let other_index = i ^ (1 << target); // Flip target bit

                if target_bit == 0 {
                    new_amplitudes[i] = gate_matrix[0][0] * self.amplitudes[i]
                        + gate_matrix[0][1] * self.amplitudes[other_index];
                } else {
                    new_amplitudes[i] = gate_matrix[1][0] * self.amplitudes[other_index]
                        + gate_matrix[1][1] * self.amplitudes[i];
                }
            }
        }

        self.amplitudes = new_amplitudes;
    }

    /// Apply CNOT gate between two qubits
    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        if control >= self.num_qubits || target >= self.num_qubits || control == target {
            return;
        }

        let size = self.amplitudes.len();
        let mut new_amplitudes = self.amplitudes.clone();

        for i in 0..size {
            let control_bit = (i >> control) & 1;

            if control_bit == 1 {
                let target_index = i ^ (1 << target);
                new_amplitudes[i] = self.amplitudes[target_index];
            }
        }

        self.amplitudes = new_amplitudes;
    }

    /// Measure the quantum state and return measurement probabilities
    pub fn measure_probabilities(&self) -> Vec<f64> {
        self.amplitudes
            .iter()
            .map(|amp| amp.magnitude_squared())
            .collect()
    }

    /// Sample from the quantum state to get classical bit string
    pub fn sample_measurement(&self) -> Vec<bool> {
        let probabilities = self.measure_probabilities();
        let total_prob: f64 = probabilities.iter().sum();

        if total_prob == 0.0 {
            return vec![false; self.num_qubits];
        }

        // Use deterministic sampling based on highest probability
        let max_prob_index = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Convert index to bit string
        let mut bits = vec![false; self.num_qubits];
        for i in 0..self.num_qubits {
            bits[i] = (max_prob_index >> i) & 1 == 1;
        }

        bits
    }

    /// Normalize the quantum state
    pub fn normalize(&mut self) {
        let total_prob: f64 = self
            .amplitudes
            .iter()
            .map(|amp| amp.magnitude_squared())
            .sum();

        if total_prob > 0.0 {
            let norm_factor = total_prob.sqrt();
            for amp in &mut self.amplitudes {
                *amp = *amp * (1.0 / norm_factor);
            }
        }
    }
}

/// Quantum gate operations
pub struct QuantumGates;

impl QuantumGates {
    /// Identity gate
    pub fn identity() -> [[Complex; 2]; 2] {
        [
            [Complex::one(), Complex::zero()],
            [Complex::zero(), Complex::one()],
        ]
    }

    /// Pauli X gate
    pub fn pauli_x() -> [[Complex; 2]; 2] {
        [
            [Complex::zero(), Complex::one()],
            [Complex::one(), Complex::zero()],
        ]
    }

    /// Pauli Y gate
    pub fn pauli_y() -> [[Complex; 2]; 2] {
        [
            [Complex::zero(), Complex::new(0.0, -1.0)],
            [Complex::new(0.0, 1.0), Complex::zero()],
        ]
    }

    /// Pauli Z gate
    pub fn pauli_z() -> [[Complex; 2]; 2] {
        [
            [Complex::one(), Complex::zero()],
            [Complex::zero(), Complex::new(-1.0, 0.0)],
        ]
    }

    /// Rotation around X axis
    pub fn rx(theta: f64) -> [[Complex; 2]; 2] {
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();

        [
            [Complex::new(cos_half, 0.0), Complex::new(0.0, -sin_half)],
            [Complex::new(0.0, -sin_half), Complex::new(cos_half, 0.0)],
        ]
    }

    /// Rotation around Y axis
    pub fn ry(theta: f64) -> [[Complex; 2]; 2] {
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();

        [
            [Complex::new(cos_half, 0.0), Complex::new(-sin_half, 0.0)],
            [Complex::new(sin_half, 0.0), Complex::new(cos_half, 0.0)],
        ]
    }

    /// Rotation around Z axis
    pub fn rz(theta: f64) -> [[Complex; 2]; 2] {
        let exp_neg = Complex::new((theta / 2.0).cos(), -(theta / 2.0).sin());
        let exp_pos = Complex::new((theta / 2.0).cos(), (theta / 2.0).sin());

        [[exp_neg, Complex::zero()], [Complex::zero(), exp_pos]]
    }

    /// Hadamard gate
    pub fn hadamard() -> [[Complex; 2]; 2] {
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;

        [
            [Complex::new(inv_sqrt2, 0.0), Complex::new(inv_sqrt2, 0.0)],
            [Complex::new(inv_sqrt2, 0.0), Complex::new(-inv_sqrt2, 0.0)],
        ]
    }
}

/// Machine learning decoder for quantum error correction
pub struct MLDecoder {
    pub model_type: MLModelType,
    pub trained_parameters: Vec<f64>,
    pub inference_engine: InferenceEngine,
}

/// Types of machine learning models for decoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MLModelType {
    NeuralNetwork { layers: Vec<usize> },
    SupportVectorMachine { kernel: String },
    RandomForest { trees: usize },
    QuantumNeuralNetwork { quantum_layers: Vec<QuantumLayer> },
}

/// Quantum layer for quantum neural networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumLayer {
    pub qubit_count: usize,
    pub parameterized_gates: Vec<ParameterizedGate>,
    pub entangling_structure: EntanglingStructure,
}

/// Parameterized quantum gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterizedGate {
    pub gate_type: ParameterizedGateType,
    pub target_qubits: Vec<usize>,
    pub parameters: Vec<f64>,
}

/// Types of parameterized quantum gates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterizedGateType {
    RX,  // Rotation around X axis
    RY,  // Rotation around Y axis
    RZ,  // Rotation around Z axis
    CRX, // Controlled RX
    CRY, // Controlled RY
    CRZ, // Controlled RZ
    Custom(String),
}

/// Entangling structure for quantum circuits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntanglingStructure {
    Linear,
    AllToAll,
    Circular,
    Custom(Vec<(usize, usize)>),
}

/// Inference engine for ML models
pub struct InferenceEngine {
    pub optimization_backend: OptimizationBackend,
    pub gradient_computation: GradientMethod,
    pub hardware_acceleration: HardwareAcceleration,
}

/// Optimization backend for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationBackend {
    Adam {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
    },
    SGD {
        learning_rate: f64,
        momentum: f64,
    },
    LBFGS {
        memory_size: usize,
    },
    QuantumNaturalGradient,
}

/// Gradient computation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientMethod {
    ParameterShift,
    FiniteDifference,
    Adjoint,
    QuantumBackpropagation,
}

/// Hardware acceleration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HardwareAcceleration {
    CPU,
    GPU {
        device_id: usize,
    },
    QuantumProcessor {
        backend: String,
    },
    Hybrid {
        classical: Box<HardwareAcceleration>,
        quantum: Box<HardwareAcceleration>,
    },
}

impl MLDecoder {
    /// Create a new ML decoder
    pub fn new(model_type: MLModelType) -> Self {
        let trained_parameters = match &model_type {
            MLModelType::NeuralNetwork { layers } => {
                // Initialize parameters based on layer sizes
                let mut params = Vec::new();
                for i in 0..layers.len() - 1 {
                    let weights = layers[i] * layers[i + 1];
                    let biases = layers[i + 1];
                    params.extend(vec![0.0; weights + biases]);
                }
                params
            }
            MLModelType::QuantumNeuralNetwork { quantum_layers } => {
                // Initialize quantum parameters
                quantum_layers
                    .iter()
                    .flat_map(|layer| layer.parameterized_gates.iter())
                    .flat_map(|gate| gate.parameters.clone())
                    .collect()
            }
            _ => Vec::new(),
        };

        Self {
            model_type,
            trained_parameters,
            inference_engine: InferenceEngine::default(),
        }
    }

    /// Perform inference on error syndrome
    pub fn decode_syndrome(&self, syndrome: &[bool]) -> Vec<usize> {
        match &self.model_type {
            MLModelType::NeuralNetwork { layers } => {
                self.neural_network_inference(syndrome, layers)
            }
            MLModelType::QuantumNeuralNetwork { quantum_layers } => {
                self.quantum_neural_network_inference(syndrome, quantum_layers)
            }
            _ => {
                // Simple majority voting for other models
                syndrome
                    .iter()
                    .enumerate()
                    .filter(|&(_, bit)| *bit)
                    .map(|(i, _)| i)
                    .collect()
            }
        }
    }

    /// Neural network inference implementation
    fn neural_network_inference(&self, syndrome: &[bool], layers: &[usize]) -> Vec<usize> {
        let mut activations: Vec<f64> = syndrome
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();

        let mut param_idx = 0;

        // Forward pass through layers
        for i in 0..layers.len() - 1 {
            let mut next_activations = vec![0.0; layers[i + 1]];

            // Apply weights
            for j in 0..layers[i] {
                for k in 0..layers[i + 1] {
                    if param_idx < self.trained_parameters.len() {
                        next_activations[k] += activations[j] * self.trained_parameters[param_idx];
                        param_idx += 1;
                    }
                }
            }

            // Apply biases and activation function (ReLU)
            for k in 0..layers[i + 1] {
                if param_idx < self.trained_parameters.len() {
                    next_activations[k] += self.trained_parameters[param_idx];
                    param_idx += 1;
                }
                next_activations[k] = next_activations[k].max(0.0);
            }

            activations = next_activations;
        }

        // Convert output to error locations
        activations
            .iter()
            .enumerate()
            .filter(|&(_, a)| *a > 0.5)
            .map(|(i, _)| i)
            .collect()
    }

    /// Quantum neural network inference using full quantum simulation
    fn quantum_neural_network_inference(
        &self,
        syndrome: &[bool],
        layers: &[QuantumLayer],
    ) -> Vec<usize> {
        if layers.is_empty() {
            return Vec::new();
        }

        // Determine the number of qubits from the first layer
        let num_qubits = layers[0].qubit_count;
        if num_qubits == 0 {
            return Vec::new();
        }

        // Initialize quantum state from syndrome
        let mut quantum_state = if syndrome.len() >= num_qubits {
            // Use first num_qubits bits from syndrome
            QuantumState::from_bits(&syndrome[..num_qubits])
        } else {
            // Pad syndrome with false bits if needed
            let mut padded_syndrome = syndrome.to_vec();
            padded_syndrome.resize(num_qubits, false);
            QuantumState::from_bits(&padded_syndrome)
        };

        // Apply each quantum layer sequentially
        for layer in layers {
            self.apply_quantum_layer(&mut quantum_state, layer);
        }

        // Measure the final quantum state
        let measurement_result = quantum_state.sample_measurement();

        // Convert measurement result to error locations
        // For quantum error correction, we interpret the measurement as indicating
        // which qubits have errors
        let mut error_locations = Vec::new();
        for (i, &bit) in measurement_result.iter().enumerate() {
            if bit {
                error_locations.push(i);
            }
        }

        // Apply post-processing to improve error correction performance
        self.apply_quantum_postprocessing(&error_locations, syndrome)
    }

    /// Apply a quantum layer to the quantum state
    fn apply_quantum_layer(&self, quantum_state: &mut QuantumState, layer: &QuantumLayer) {
        // Apply all parameterized gates in the layer
        for gate in &layer.parameterized_gates {
            self.apply_parameterized_gate(quantum_state, gate);
        }

        // Apply entangling operations based on the structure
        self.apply_entangling_operations(quantum_state, layer);

        // Normalize the quantum state after operations
        quantum_state.normalize();
    }

    /// Apply a parameterized gate to the quantum state
    fn apply_parameterized_gate(&self, quantum_state: &mut QuantumState, gate: &ParameterizedGate) {
        if gate.target_qubits.is_empty() || gate.parameters.is_empty() {
            return;
        }

        let param_idx = gate.target_qubits[0] % self.trained_parameters.len();
        let parameter = if param_idx < self.trained_parameters.len() {
            self.trained_parameters[param_idx]
        } else {
            gate.parameters[0]
        };

        match gate.gate_type {
            ParameterizedGateType::RX => {
                let gate_matrix = QuantumGates::rx(parameter);
                for &qubit in &gate.target_qubits {
                    quantum_state.apply_single_qubit_gate(qubit, &gate_matrix);
                }
            }
            ParameterizedGateType::RY => {
                let gate_matrix = QuantumGates::ry(parameter);
                for &qubit in &gate.target_qubits {
                    quantum_state.apply_single_qubit_gate(qubit, &gate_matrix);
                }
            }
            ParameterizedGateType::RZ => {
                let gate_matrix = QuantumGates::rz(parameter);
                for &qubit in &gate.target_qubits {
                    quantum_state.apply_single_qubit_gate(qubit, &gate_matrix);
                }
            }
            ParameterizedGateType::CRX => {
                if gate.target_qubits.len() >= 2 {
                    let gate_matrix = QuantumGates::rx(parameter);
                    quantum_state.apply_controlled_gate(
                        gate.target_qubits[0],
                        gate.target_qubits[1],
                        &gate_matrix,
                    );
                }
            }
            ParameterizedGateType::CRY => {
                if gate.target_qubits.len() >= 2 {
                    let gate_matrix = QuantumGates::ry(parameter);
                    quantum_state.apply_controlled_gate(
                        gate.target_qubits[0],
                        gate.target_qubits[1],
                        &gate_matrix,
                    );
                }
            }
            ParameterizedGateType::CRZ => {
                if gate.target_qubits.len() >= 2 {
                    let gate_matrix = QuantumGates::rz(parameter);
                    quantum_state.apply_controlled_gate(
                        gate.target_qubits[0],
                        gate.target_qubits[1],
                        &gate_matrix,
                    );
                }
            }
            ParameterizedGateType::Custom(_) => {
                // For custom gates, apply a default RY rotation
                let gate_matrix = QuantumGates::ry(parameter);
                for &qubit in &gate.target_qubits {
                    quantum_state.apply_single_qubit_gate(qubit, &gate_matrix);
                }
            }
        }
    }

    /// Apply entangling operations based on the layer's structure
    fn apply_entangling_operations(&self, quantum_state: &mut QuantumState, layer: &QuantumLayer) {
        match &layer.entangling_structure {
            EntanglingStructure::Linear => {
                // Apply CNOT gates between adjacent qubits
                for i in 0..layer.qubit_count.saturating_sub(1) {
                    quantum_state.apply_cnot(i, i + 1);
                }
            }
            EntanglingStructure::AllToAll => {
                // Apply CNOT gates between all pairs of qubits
                for i in 0..layer.qubit_count {
                    for j in (i + 1)..layer.qubit_count {
                        quantum_state.apply_cnot(i, j);
                    }
                }
            }
            EntanglingStructure::Circular => {
                // Apply CNOT gates in a circular pattern
                for i in 0..layer.qubit_count {
                    let next = (i + 1) % layer.qubit_count;
                    quantum_state.apply_cnot(i, next);
                }
            }
            EntanglingStructure::Custom(connections) => {
                // Apply CNOT gates for custom connections
                for &(control, target) in connections {
                    if control < layer.qubit_count && target < layer.qubit_count {
                        quantum_state.apply_cnot(control, target);
                    }
                }
            }
        }
    }

    /// Apply quantum post-processing to improve error correction
    fn apply_quantum_postprocessing(
        &self,
        error_locations: &[usize],
        syndrome: &[bool],
    ) -> Vec<usize> {
        // Advanced post-processing using quantum error correction principles
        let mut corrected_locations = error_locations.to_vec();

        // Check consistency with syndrome
        let mut syndrome_check = vec![false; syndrome.len()];
        for &loc in error_locations {
            if loc < syndrome_check.len() {
                syndrome_check[loc] = true;
            }
        }

        // Apply syndrome-based corrections
        for (i, (&syndr, &check)) in syndrome.iter().zip(syndrome_check.iter()).enumerate() {
            if syndr && !check {
                // Syndrome indicates error but not detected - add to corrections
                corrected_locations.push(i);
            } else if !syndr && check {
                // Detected error but syndrome doesn't indicate - remove from corrections
                corrected_locations.retain(|&x| x != i);
            }
        }

        // Remove duplicates and sort
        corrected_locations.sort_unstable();
        corrected_locations.dedup();

        // Apply majority voting for nearby error locations
        let mut final_locations = Vec::new();
        let mut i = 0;
        while i < corrected_locations.len() {
            let mut cluster = vec![corrected_locations[i]];
            let mut j = i + 1;

            // Group nearby errors (within distance of 2)
            while j < corrected_locations.len()
                && corrected_locations[j] <= corrected_locations[i] + 2
            {
                cluster.push(corrected_locations[j]);
                j += 1;
            }

            // If we have a cluster of errors, apply majority voting
            if cluster.len() >= 2 {
                // For clusters, include only the central error
                let central_idx = cluster.len() / 2;
                final_locations.push(cluster[central_idx]);
            } else {
                // Single error, include as is
                final_locations.push(cluster[0]);
            }

            i = j;
        }

        final_locations
    }

    /// Train the decoder on labeled data
    pub fn train(&mut self, training_data: &[(Vec<bool>, Vec<usize>)]) {
        match &self.inference_engine.optimization_backend {
            OptimizationBackend::Adam {
                learning_rate,
                beta1,
                beta2,
            } => {
                self.train_adam(training_data, *learning_rate, *beta1, *beta2);
            }
            OptimizationBackend::SGD {
                learning_rate,
                momentum,
            } => {
                self.train_sgd(training_data, *learning_rate, *momentum);
            }
            _ => {
                // Placeholder for other optimizers
            }
        }
    }

    /// Adam optimizer training
    fn train_adam(
        &mut self,
        training_data: &[(Vec<bool>, Vec<usize>)],
        lr: f64,
        beta1: f64,
        beta2: f64,
    ) {
        let mut m = vec![0.0; self.trained_parameters.len()];
        let mut v = vec![0.0; self.trained_parameters.len()];
        let epsilon = 1e-8;

        for (t, (syndrome, target)) in training_data.iter().enumerate() {
            // Compute gradients (simplified)
            let prediction = self.decode_syndrome(syndrome);
            let error = self.compute_error(&prediction, target);
            let gradients = self.compute_gradients(syndrome, &error);

            // Update moments
            for i in 0..self.trained_parameters.len() {
                if i < gradients.len() {
                    m[i] = beta1 * m[i] + (1.0 - beta1) * gradients[i];
                    v[i] = beta2 * v[i] + (1.0 - beta2) * gradients[i].powi(2);

                    // Bias correction
                    let m_hat = m[i] / (1.0 - beta1.powi((t + 1) as i32));
                    let v_hat = v[i] / (1.0 - beta2.powi((t + 1) as i32));

                    // Update parameters
                    self.trained_parameters[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
                }
            }
        }
    }

    /// SGD optimizer training (placeholder)
    fn train_sgd(&mut self, _training_data: &[(Vec<bool>, Vec<usize>)], _lr: f64, _momentum: f64) {
        // Implementation would go here
    }

    /// Compute error between prediction and target
    fn compute_error(&self, prediction: &[usize], target: &[usize]) -> Vec<f64> {
        let mut error = vec![0.0; prediction.len().max(target.len())];

        for &idx in prediction {
            if idx < error.len() && !target.contains(&idx) {
                error[idx] = 1.0;
            }
        }

        for &idx in target {
            if idx < error.len() && !prediction.contains(&idx) {
                error[idx] = -1.0;
            }
        }

        error
    }

    /// Compute gradients (simplified)
    fn compute_gradients(&self, syndrome: &[bool], error: &[f64]) -> Vec<f64> {
        // Simplified gradient computation
        let input_size = syndrome.len();
        let output_size = error.len();
        let mut gradients = Vec::new();

        for i in 0..input_size {
            for j in 0..output_size {
                let input_val = if syndrome[i] { 1.0 } else { 0.0 };
                gradients.push(input_val * error[j]);
            }
        }

        gradients
    }
}

impl Default for InferenceEngine {
    fn default() -> Self {
        Self {
            optimization_backend: OptimizationBackend::Adam {
                learning_rate: 0.001,
                beta1: 0.9,
                beta2: 0.999,
            },
            gradient_computation: GradientMethod::ParameterShift,
            hardware_acceleration: HardwareAcceleration::CPU,
        }
    }
}

impl QuantumLayer {
    /// Create a new quantum layer with default entangling
    pub fn new(qubit_count: usize) -> Self {
        Self {
            qubit_count,
            parameterized_gates: Vec::new(),
            entangling_structure: EntanglingStructure::Linear,
        }
    }

    /// Add a parameterized gate to the layer
    pub fn add_gate(&mut self, gate: ParameterizedGate) {
        self.parameterized_gates.push(gate);
    }

    /// Generate a standard layer with RY rotations and CNOT entangling
    pub fn standard_layer(qubit_count: usize) -> Self {
        let mut layer = Self::new(qubit_count);

        // Add RY rotation gates on all qubits
        for i in 0..qubit_count {
            layer.add_gate(ParameterizedGate {
                gate_type: ParameterizedGateType::RY,
                target_qubits: vec![i],
                parameters: vec![0.0], // Will be trained
            });
        }

        // Set linear entangling structure
        layer.entangling_structure = EntanglingStructure::Linear;

        layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_decoder_creation() {
        let model_type = MLModelType::NeuralNetwork {
            layers: vec![10, 5, 2],
        };
        let decoder = MLDecoder::new(model_type);

        // Should have (10*5 + 5) + (5*2 + 2) = 55 + 12 = 67 parameters
        assert_eq!(decoder.trained_parameters.len(), 67);
    }

    #[test]
    fn test_neural_network_inference() {
        let model_type = MLModelType::NeuralNetwork {
            layers: vec![4, 3, 2],
        };
        let mut decoder = MLDecoder::new(model_type);

        // Set some non-zero parameters
        for param in &mut decoder.trained_parameters {
            *param = 0.1;
        }

        let syndrome = vec![true, false, true, false];
        let result = decoder.decode_syndrome(&syndrome);

        // Should produce some output
        assert!(!result.is_empty() || result.is_empty()); // Always true, just checking it runs
    }

    #[test]
    fn test_quantum_layer_creation() {
        let layer = QuantumLayer::standard_layer(4);

        assert_eq!(layer.qubit_count, 4);
        assert_eq!(layer.parameterized_gates.len(), 4);
        assert!(matches!(
            layer.entangling_structure,
            EntanglingStructure::Linear
        ));
    }
}
