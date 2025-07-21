//! Advanced sampling strategies for transformer model inference
//!
//! This module provides production-grade sampling implementations with:
//! - Zero allocation where possible
//! - Numerically stable algorithms  
//! - Composable processor chains
//! - Comprehensive error handling
//! - High-performance tensor operations

use candle_core::{Result as CandleResult, Tensor};

pub mod temperature;
pub mod nucleus;
pub mod topk;
pub mod repetition;
pub mod composite;
pub mod gumbel;
pub mod typical;
pub mod mirostat;
pub mod simd;
// pub mod mirostat;
// pub mod simd;

pub use temperature::TemperatureProcessor;
pub use nucleus::TopPProcessor;
pub use topk::TopKProcessor;
pub use repetition::RepetitionPenaltyProcessor;
pub use composite::CompositeProcessor;
pub use gumbel::GumbelSoftmaxProcessor;
pub use typical::TypicalSamplingProcessor;
// TODO: Export remaining processors when implemented
// pub use mirostat::MirostatProcessor;
// pub use simd::SimdOptimizedProcessor;

// CompositeProcessor is imported from composite module

/// Errors that can occur during logits processing
#[derive(Debug, Clone, thiserror::Error)]
pub enum SamplingError {
    #[error("Invalid temperature: {0} (must be > 0.0)")]
    InvalidTemperature(f64),
    
    #[error("Invalid top-p value: {0} (must be in [0.0, 1.0])")]
    InvalidTopP(f64),
    
    #[error("Invalid top-k value: {0} (must be > 0)")]
    InvalidTopK(usize),
    
    #[error("Invalid repetition penalty: {0} (must be >= 1.0)")]
    InvalidRepetitionPenalty(f64),
    
    #[error("Logits tensor error: {0}")]
    TensorError(String),
    
    #[error("Numerical instability detected: {0}")]
    NumericalInstability(String),
    
    #[error("Empty vocabulary: cannot sample from zero-length logits")]
    EmptyVocabulary,
    
    #[error("Empty logits: no valid logits found for processing")]
    EmptyLogits,
    
    #[error("Processing failed: {0}")]
    ProcessingFailed(String),
    
    #[error("Processor chain error: {0}")]
    ProcessorChainError(String),
}

impl From<candle_core::Error> for SamplingError {
    fn from(err: candle_core::Error) -> Self {
        SamplingError::TensorError(err.to_string())
    }
}

// Import the main LogitsProcessor trait from logits.rs
pub use crate::logits::LogitsProcessor;

/// High-performance sampling configuration with validation
#[derive(Debug, Clone)]
pub struct Sampling {
    /// Temperature for probability scaling (> 0.0)
    pub temperature: f64,
    /// Top-p nucleus sampling threshold [0.0, 1.0]
    pub top_p: Option<f64>,
    /// Top-k token limit (> 0)
    pub top_k: Option<usize>,
    /// Repetition penalty factor (>= 1.0)
    pub repetition_penalty: Option<f64>,
    /// Maximum context length for repetition tracking
    pub repetition_context_size: usize,
    /// Typical sampling probability mass [0.0, 1.0]
    pub typical_p: Option<f64>,
    /// Gumbel-Softmax temperature
    pub gumbel_temperature: Option<f32>,
    /// Gumbel-Softmax hard sampling mode
    pub gumbel_hard: bool,
    /// Random seed for reproducible sampling
    pub random_seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            repetition_context_size: 64,
            typical_p: None,
            gumbel_temperature: None,
            gumbel_hard: false,
            random_seed: 42,
        }
    }
}

impl Sampling {
    /// Create new sampling configuration with validation
    pub fn new() -> Self {
        Self::default()
    }

    /// Set temperature with validation
    pub fn temperature(mut self, temperature: f64) -> Result<Self, SamplingError> {
        if temperature <= 0.0 || !temperature.is_finite() {
            return Err(SamplingError::InvalidTemperature(temperature));
        }
        self.temperature = temperature;
        Ok(self)
    }

    /// Set top-p with validation
    pub fn top_p(mut self, top_p: f64) -> Result<Self, SamplingError> {
        if !(0.0..=1.0).contains(&top_p) || !top_p.is_finite() {
            return Err(SamplingError::InvalidTopP(top_p));
        }
        self.top_p = Some(top_p);
        Ok(self)
    }

    /// Set top-k with validation
    pub fn top_k(mut self, top_k: usize) -> Result<Self, SamplingError> {
        if top_k == 0 {
            return Err(SamplingError::InvalidTopK(top_k));
        }
        self.top_k = Some(top_k);
        Ok(self)
    }

    /// Set repetition penalty with validation
    pub fn repetition_penalty(mut self, penalty: f64) -> Result<Self, SamplingError> {
        if penalty < 1.0 || !penalty.is_finite() {
            return Err(SamplingError::InvalidRepetitionPenalty(penalty));
        }
        self.repetition_penalty = Some(penalty);
        Ok(self)
    }

    /// Set repetition context size
    pub fn repetition_context_size(mut self, size: usize) -> Self {
        self.repetition_context_size = size;
        self
    }

    /// Set typical sampling with validation
    pub fn typical_p(mut self, typical_p: f64) -> Result<Self, SamplingError> {
        if !(0.0..=1.0).contains(&typical_p) || !typical_p.is_finite() {
            return Err(SamplingError::InvalidTopP(typical_p));
        }
        self.typical_p = Some(typical_p);
        Ok(self)
    }

    /// Set Gumbel-Softmax temperature with validation
    pub fn gumbel_temperature(mut self, temperature: f32) -> Result<Self, SamplingError> {
        if temperature <= 0.0 || !temperature.is_finite() {
            return Err(SamplingError::InvalidTemperature(temperature as f64));
        }
        self.gumbel_temperature = Some(temperature);
        Ok(self)
    }

    /// Enable hard Gumbel-Softmax sampling
    pub fn gumbel_hard(mut self, hard: bool) -> Self {
        self.gumbel_hard = hard;
        self
    }

    /// Set random seed for reproducible sampling
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = seed;
        self
    }

    /// Build composite processor from configuration
    pub fn build_processor(&self) -> Result<CompositeProcessor, SamplingError> {
        let mut builder = LogitsProcessorBuilder::new();

        // Add repetition penalty first (context-dependent)
        if let Some(penalty) = self.repetition_penalty {
            builder = builder.repetition_penalty(penalty, self.repetition_context_size)?;
        }

        // Add temperature scaling
        builder = builder.temperature(self.temperature)?;

        // Add top-k filtering (before top-p for efficiency)
        if let Some(k) = self.top_k {
            builder = builder.top_k(k)?;
        }

        // Add top-p nucleus sampling
        if let Some(p) = self.top_p {
            builder = builder.top_p(p)?;
        }

        // Add typical sampling
        if let Some(typical_p) = self.typical_p {
            builder = builder.typical_sampling(typical_p)?;
        }

        // Add Gumbel-Softmax sampling (usually mutually exclusive with other sampling)
        if let Some(gumbel_temp) = self.gumbel_temperature {
            let device = candle_core::Device::Cpu; // Default device, should be configurable
            builder = builder.gumbel_softmax(gumbel_temp, self.gumbel_hard, self.random_seed, device)?;
        }

        builder.build()
    }

    /// Build composite processor with device specification
    pub fn build_processor_with_device(&self, device: candle_core::Device) -> Result<CompositeProcessor, SamplingError> {
        let mut builder = LogitsProcessorBuilder::new();

        // Add repetition penalty first (context-dependent)
        if let Some(penalty) = self.repetition_penalty {
            builder = builder.repetition_penalty(penalty, self.repetition_context_size)?;
        }

        // Add temperature scaling
        builder = builder.temperature(self.temperature)?;

        // Add top-k filtering (before top-p for efficiency)
        if let Some(k) = self.top_k {
            builder = builder.top_k(k)?;
        }

        // Add top-p nucleus sampling
        if let Some(p) = self.top_p {
            builder = builder.top_p(p)?;
        }

        // Add typical sampling
        if let Some(typical_p) = self.typical_p {
            builder = builder.typical_sampling(typical_p)?;
        }

        // Add Gumbel-Softmax sampling with specified device
        if let Some(gumbel_temp) = self.gumbel_temperature {
            builder = builder.gumbel_softmax(gumbel_temp, self.gumbel_hard, self.random_seed, device)?;
        }

        builder.build()
    }
}

/// Builder for constructing logits processor chains
pub struct LogitsProcessorBuilder {
    processors: Vec<Box<dyn LogitsProcessor>>,
}

impl LogitsProcessorBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    /// Add temperature processor
    pub fn temperature(mut self, temperature: f64) -> Result<Self, SamplingError> {
        let processor = TemperatureProcessor::new(temperature)?;
        self.processors.push(Box::new(processor));
        Ok(self)
    }

    /// Add top-p processor
    pub fn top_p(mut self, top_p: f64) -> Result<Self, SamplingError> {
        let processor = TopPProcessor::new(top_p)?;
        self.processors.push(Box::new(processor));
        Ok(self)
    }

    /// Add top-k processor
    pub fn top_k(mut self, top_k: usize) -> Result<Self, SamplingError> {
        let processor = TopKProcessor::new(top_k)?;
        self.processors.push(Box::new(processor));
        Ok(self)
    }

    /// Add repetition penalty processor
    pub fn repetition_penalty(
        mut self,
        penalty: f64,
        context_size: usize,
    ) -> Result<Self, SamplingError> {
        let processor = RepetitionPenaltyProcessor::new(penalty, context_size)?;
        self.processors.push(Box::new(processor));
        Ok(self)
    }

    /// Add typical sampling processor
    pub fn typical_sampling(mut self, typical_p: f64) -> Result<Self, SamplingError> {
        let processor = TypicalSamplingProcessor::new(typical_p)?;
        self.processors.push(Box::new(processor));
        Ok(self)
    }

    /// Add Gumbel-Softmax processor
    pub fn gumbel_softmax(
        mut self, 
        temperature: f32, 
        hard: bool, 
        seed: u64, 
        device: candle_core::Device
    ) -> Result<Self, SamplingError> {
        let processor = GumbelSoftmaxProcessor::new(temperature, hard, seed, device)?;
        self.processors.push(Box::new(processor));
        Ok(self)
    }

    /// Add custom processor
    pub fn custom(mut self, processor: Box<dyn LogitsProcessor>) -> Self {
        self.processors.push(processor);
        self
    }

    /// Build composite processor
    pub fn build(self) -> Result<CompositeProcessor, SamplingError> {
        CompositeProcessor::new(self.processors)
    }
}

impl Default for LogitsProcessorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for logits processing
pub mod utils {
    use super::*;
    use candle_core::Device;

    /// Apply softmax with temperature scaling and numerical stability
    #[inline(always)]
    pub fn stable_softmax(
        logits: &Tensor,
        temperature: f64,
        device: &Device,
    ) -> CandleResult<Tensor> {
        // Scale by temperature first
        let scaled = if (temperature - 1.0).abs() < f64::EPSILON {
            logits.clone()
        } else {
            logits.affine(1.0 / temperature, 0.0)?
        };

        // Find maximum for numerical stability
        let max_logit = scaled.max_keepdim(candle_core::D::Minus1)?;
        let shifted = scaled.broadcast_sub(&max_logit)?;

        // Apply softmax
        let exp_logits = shifted.exp()?;
        let sum_exp = exp_logits.sum_keepdim(candle_core::D::Minus1)?;
        exp_logits.broadcast_div(&sum_exp)
    }

    /// Sample from categorical distribution using efficient algorithms
    #[inline(always)]
    pub fn categorical_sample(
        probs: &Tensor,
        rng: &mut impl rand::Rng,
    ) -> CandleResult<u32> {
        let probs_vec = probs.to_vec1::<f32>()?;
        
        // Validate probabilities
        let sum: f32 = probs_vec.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(candle_core::Error::Msg(
                "Invalid probability distribution".to_string()
            ));
        }

        // Generate random value
        let random_val: f32 = rng.gen_range(0.0..sum);
        let mut cumulative = 0.0;

        // Find sample using cumulative distribution
        for (idx, &prob) in probs_vec.iter().enumerate() {
            cumulative += prob;
            if random_val <= cumulative {
                return Ok(idx as u32);
            }
        }

        // Fallback to last token (handles floating point precision issues)
        Ok((probs_vec.len() - 1) as u32)
    }

    /// Check for numerical instabilities in logits
    #[inline(always)]
    pub fn validate_logits(logits: &Tensor) -> Result<(), SamplingError> {
        // This is a simplified check - in production, you might want more thorough validation
        let shape = logits.shape();
        if shape.dims().is_empty() || shape.dims().iter().any(|&d| d == 0) {
            return Err(SamplingError::EmptyVocabulary);
        }
        Ok(())
    }

    /// Efficient tensor sorting for top-k and top-p operations
    #[inline(always)]
    pub fn argsort_descending(values: &[f32]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..values.len()).collect();
        indices.sort_by(|&a, &b| {
            values[b].partial_cmp(&values[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn test_sampling_configuration_validation() {
        // Valid configuration
        let config = Sampling::new()
            .temperature(0.8).expect("valid temperature")
            .top_p(0.9).expect("valid top-p")
            .top_k(50).expect("valid top-k")
            .repetition_penalty(1.1).expect("valid repetition penalty");
        
        assert!((config.temperature - 0.8).abs() < f64::EPSILON);
        assert!((config.top_p.unwrap() - 0.9).abs() < f64::EPSILON);
        assert_eq!(config.top_k.unwrap(), 50);
        assert!((config.repetition_penalty.unwrap() - 1.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_invalid_parameters() {
        // Invalid temperature
        assert!(matches!(
            Sampling::new().temperature(0.0),
            Err(SamplingError::InvalidTemperature(0.0))
        ));

        // Invalid top-p
        assert!(matches!(
            Sampling::new().top_p(1.5),
            Err(SamplingError::InvalidTopP(1.5))
        ));

        // Invalid top-k
        assert!(matches!(
            Sampling::new().top_k(0),
            Err(SamplingError::InvalidTopK(0))
        ));

        // Invalid repetition penalty
        assert!(matches!(
            Sampling::new().repetition_penalty(0.5),
            Err(SamplingError::InvalidRepetitionPenalty(0.5))
        ));
    }

    #[test]
    fn test_builder_pattern() {
        let builder = LogitsProcessorBuilder::new();
        let _processor = builder
            .temperature(0.8).expect("valid temperature")
            .top_k(40).expect("valid top-k")
            .build().expect("build succeeds");
    }

    #[test]
    fn test_utils_argsort() {
        let values = vec![0.1, 0.8, 0.3, 0.9, 0.2];
        let sorted_indices = utils::argsort_descending(&values);
        
        // Should be sorted in descending order: 0.9, 0.8, 0.3, 0.2, 0.1
        assert_eq!(sorted_indices, vec![3, 1, 2, 4, 0]);
    }

    #[test]
    fn test_validate_logits() {
        let device = Device::Cpu;
        
        // Valid logits
        let logits = Tensor::ones((5,), DType::F32, &device).expect("tensor creation");
        assert!(utils::validate_logits(&logits).is_ok());
        
        // Empty logits should fail
        let empty_logits = Tensor::zeros((0,), DType::F32, &device).expect("tensor creation");
        assert!(matches!(
            utils::validate_logits(&empty_logits),
            Err(SamplingError::EmptyVocabulary)
        ));
    }
}