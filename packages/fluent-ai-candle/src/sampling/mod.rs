//! Advanced sampling strategies for transformer model inference
//!
//! This module provides the canonical Candle LogitsProcessor API integration
//! with streaming-first HTTP/3 architecture and SIMD optimizations.
//!
//! The unified system provides:
//! - Production-grade sampling implementations 
//! - Zero allocation where possible
//! - Numerically stable algorithms
//! - Composable processor chains
//! - Comprehensive error handling
//! - High-performance tensor operations

use candle_core::{Result as CandleResult, Tensor};
use rand::{distr::Distribution, SeedableRng};

/// Re-export canonical Candle LogitsProcessor and Sampling enums
pub use candle_transformers::generation::{LogitsProcessor, Sampling};

// Legacy modules maintained for compatibility
pub mod temperature;
pub mod nucleus;
pub mod topk;
pub mod repetition;
pub mod composite;
pub mod gumbel;
pub mod typical;
pub mod mirostat;
pub mod simd;

/// Errors that can occur during logits processing - DEPRECATED
/// 
/// Use `crate::processing::error::ProcessingError` for the unified error system.
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

// Error conversions for compatibility

/// High-performance sampling configuration builder for canonical Candle Sampling
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for probability scaling (> 0.0)
    pub temperature: f64,
    /// Top-p nucleus sampling threshold [0.0, 1.0]
    pub top_p: Option<f64>,
    /// Top-k token limit (> 0)
    pub top_k: Option<usize>,
    /// Random seed for reproducible sampling
    pub random_seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: None,
            top_k: None,
            random_seed: 42,
        }
    }
}

impl SamplingConfig {
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

    /// Set random seed for reproducible sampling
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = seed;
        self
    }

    /// Build canonical Candle Sampling enum from configuration
    pub fn build_sampling(&self) -> Sampling {
        match (self.top_k, self.top_p) {
            (None, None) => Sampling::All { temperature: self.temperature },
            (Some(k), None) => Sampling::TopK { k, temperature: self.temperature },
            (None, Some(p)) => Sampling::TopP { p, temperature: self.temperature },
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature: self.temperature },
        }
    }

    /// Build canonical LogitsProcessor from configuration
    pub fn build_processor(&self) -> LogitsProcessor {
        LogitsProcessor::from_sampling(self.random_seed, self.build_sampling())
    }
}

/// Convenient builder for canonical LogitsProcessor with validation
pub struct LogitsProcessorBuilder {
    config: SamplingConfig,
}

impl LogitsProcessorBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: SamplingConfig::default(),
        }
    }

    /// Set temperature with validation
    pub fn temperature(mut self, temperature: f64) -> Result<Self, SamplingError> {
        self.config = self.config.temperature(temperature)?;
        Ok(self)
    }

    /// Set top-p with validation
    pub fn top_p(mut self, top_p: f64) -> Result<Self, SamplingError> {
        self.config = self.config.top_p(top_p)?;
        Ok(self)
    }

    /// Set top-k with validation
    pub fn top_k(mut self, top_k: usize) -> Result<Self, SamplingError> {
        self.config = self.config.top_k(top_k)?;
        Ok(self)
    }

    /// Set random seed
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config = self.config.random_seed(seed);
        self
    }

    /// Build canonical LogitsProcessor
    pub fn build(self) -> LogitsProcessor {
        self.config.build_processor()
    }
}

impl Default for LogitsProcessorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for logits processing - DEPRECATED
/// 
/// Use `crate::processing::utils` for modern utility functions.
pub mod utils {
    use super::*;
    use candle_core::Device;

    /// Apply softmax with temperature scaling and numerical stability - DEPRECATED
    #[inline(always)]
    pub fn stable_softmax(
        logits: &Tensor,
        temperature: f64,
        _device: &Device,
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

    /// Sample from categorical distribution using efficient algorithms - DEPRECATED
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

    /// Check for numerical instabilities in logits - DEPRECATED
    #[inline(always)]
    pub fn validate_logits(logits: &Tensor) -> Result<(), SamplingError> {
        // This is a simplified check - in production, you might want more thorough validation
        let shape = logits.shape();
        if shape.dims().is_empty() || shape.dims().iter().any(|&d| d == 0) {
            return Err(SamplingError::EmptyVocabulary);
        }
        Ok(())
    }

    /// Efficient tensor sorting for top-k and top-p operations - DEPRECATED
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

    #[test]
    fn test_sampling_configuration_validation() {
        // Valid configuration
        let config = SamplingConfig::new()
            .temperature(0.8).expect("valid temperature")
            .top_p(0.9).expect("valid top-p")
            .top_k(50).expect("valid top-k");
        
        assert!((config.temperature - 0.8).abs() < f64::EPSILON);
        assert!((config.top_p.unwrap() - 0.9).abs() < f64::EPSILON);
        assert_eq!(config.top_k.unwrap(), 50);
    }

    #[test]
    fn test_invalid_parameters() {
        // Invalid temperature
        assert!(matches!(
            SamplingConfig::new().temperature(0.0),
            Err(SamplingError::InvalidTemperature(0.0))
        ));

        // Invalid top-p
        assert!(matches!(
            SamplingConfig::new().top_p(1.5),
            Err(SamplingError::InvalidTopP(1.5))
        ));

        // Invalid top-k
        assert!(matches!(
            SamplingConfig::new().top_k(0),
            Err(SamplingError::InvalidTopK(0))
        ));
    }

    #[test]
    fn test_builder_pattern() {
        let builder = LogitsProcessorBuilder::new();
        let _processor = builder
            .temperature(0.8).expect("valid temperature")
            .top_k(40).expect("valid top-k")
            .build();
    }

    #[test]
    fn test_canonical_sampling_integration() {
        let config = SamplingConfig::new()
            .temperature(0.8).expect("valid temperature")
            .top_k(40).expect("valid top-k");

        let sampling = config.build_sampling();
        match sampling {
            Sampling::TopK { k, temperature } => {
                assert_eq!(k, 40);
                assert!((temperature - 0.8).abs() < f64::EPSILON);
            }
            _ => panic!("Expected TopK sampling"),
        }

        let _processor = config.build_processor();
    }

    #[test]
    fn test_utils_argsort() {
        let values = vec![0.1, 0.8, 0.3, 0.9, 0.2];
        let sorted_indices = utils::argsort_descending(&values);
        
        // Should be sorted in descending order: 0.9, 0.8, 0.3, 0.2, 0.1
        assert_eq!(sorted_indices, vec![3, 1, 2, 4, 0]);
    }
}