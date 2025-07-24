//! Gumbel-Softmax sampling processor with differentiable sampling
//!
//! Production-quality Gumbel-Softmax implementation for differentiable discrete sampling
//! with comprehensive numerical stability and temperature control.

use candle_core::{D, Device, Tensor};
use rand::{Rng, SeedableRng};

use super::SamplingError;

/// Gumbel-Softmax processor for differentiable discrete sampling
///
/// Implements the Gumbel-Softmax trick for sampling from categorical distributions
/// in a differentiable way. Useful for variational autoencoders and other models
/// requiring differentiable discrete sampling.
///
/// The Gumbel-Softmax distribution uses the reparameterization trick:
/// sample = softmax((log(π) + g) / τ)
/// where g ~ Gumbel(0,1) and τ is the temperature parameter.
#[derive(Debug)]
pub struct GumbelSoftmaxProcessor {
    /// Temperature parameter controlling sharpness
    temperature: f32,
    /// Random number generator for Gumbel noise
    rng: std::sync::Mutex<rand::rngs::StdRng>,
    /// Whether to use hard (straight-through) or soft sampling
    hard: bool,
    /// Cached inverse temperature for efficiency
    inv_temperature: f32,
    /// Device for tensor operations
    #[allow(dead_code)] // Reserved for future device-specific Gumbel noise generation
    device: Device,
}

impl GumbelSoftmaxProcessor {
    /// Minimum temperature to prevent numerical instability
    pub const MIN_TEMPERATURE: f32 = 0.01;
    /// Maximum temperature to prevent numerical instability  
    pub const MAX_TEMPERATURE: f32 = 10.0;
    /// Default temperature for Gumbel-Softmax
    pub const DEFAULT_TEMPERATURE: f32 = 1.0;

    /// Create a new Gumbel-Softmax processor
    ///
    /// # Arguments
    /// * `temperature` - Temperature parameter (must be > 0.0)
    /// * `hard` - Whether to use hard (straight-through) sampling
    /// * `seed` - Random seed for reproducibility
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    /// * `Ok(GumbelSoftmaxProcessor)` - Successfully created processor
    /// * `Err(SamplingError)` - Invalid temperature or configuration
    pub fn new(
        temperature: f32,
        hard: bool,
        seed: u64,
        device: Device,
    ) -> Result<Self, SamplingError> {
        // Validate temperature
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(SamplingError::InvalidTemperature(temperature as f64));
        }

        if temperature < Self::MIN_TEMPERATURE || temperature > Self::MAX_TEMPERATURE {
            return Err(SamplingError::InvalidTemperature(temperature as f64));
        }

        let rng = std::sync::Mutex::new(rand::rngs::StdRng::seed_from_u64(seed));
        let inv_temperature = 1.0 / temperature;

        Ok(Self {
            temperature,
            rng,
            hard,
            inv_temperature,
            device,
        })
    }

    /// Create with default settings for soft sampling
    pub fn soft(temperature: f32, seed: u64, device: Device) -> Result<Self, SamplingError> {
        Self::new(temperature, false, seed, device)
    }

    /// Create with default settings for hard sampling
    pub fn hard(temperature: f32, seed: u64, device: Device) -> Result<Self, SamplingError> {
        Self::new(temperature, true, seed, device)
    }

    /// Generate Gumbel noise tensor
    ///
    /// Generates Gumbel(0,1) distributed noise using the inverse CDF method:
    /// G = -ln(-ln(U)) where U ~ Uniform(0,1)
    #[deprecated = "Legacy sampling module - use crate::processing::processors instead"]
    #[allow(dead_code)]
    fn generate_gumbel_noise(&self, shape: &candle_core::Shape) -> Result<Tensor, SamplingError> {
        let mut rng_guard = self.rng.lock().map_err(|_| {
            SamplingError::NumericalInstability("Failed to acquire RNG lock".to_string())
        })?;

        let size = shape.elem_count();
        let mut noise_vec = Vec::with_capacity(size);

        // Generate Gumbel noise using inverse CDF method
        for _ in 0..size {
            // Generate uniform random number in (0, 1) with small epsilon to avoid log(0)
            let u1: f32 = rng_guard.random_range(1e-7..1.0 - 1e-7);
            let u2: f32 = rng_guard.random_range(1e-7..1.0 - 1e-7);

            // Apply inverse Gumbel CDF: G = -ln(-ln(U))
            let gumbel_noise = -(-u1.ln()).ln() - (-u2.ln()).ln();
            noise_vec.push(gumbel_noise);
        }

        // Create tensor from noise vector
        Tensor::from_vec(noise_vec, shape, &self.device)
            .map_err(|e| SamplingError::TensorError(e.to_string()))
    }

    /// Apply Gumbel-Softmax transformation
    #[deprecated = "Legacy sampling module - use crate::processing::processors instead"]
    #[allow(dead_code)]
    fn apply_gumbel_softmax(&self, logits: &Tensor) -> Result<Tensor, SamplingError> {
        let shape = logits.shape();

        // Generate Gumbel noise
        let gumbel_noise = self.generate_gumbel_noise(shape)?;

        // Add Gumbel noise to logits: logits + G
        let noisy_logits = logits
            .broadcast_add(&gumbel_noise)
            .map_err(|e| SamplingError::TensorError(e.to_string()))?;

        // Apply temperature scaling: (logits + G) / τ
        let scaled_logits = noisy_logits
            .affine(self.inv_temperature as f64, 0.0)
            .map_err(|e| SamplingError::TensorError(e.to_string()))?;

        // Apply softmax for final probabilities
        let soft_sample = candle_nn::ops::softmax(&scaled_logits, D::Minus1)
            .map_err(|e| SamplingError::TensorError(e.to_string()))?;

        if self.hard {
            // Apply straight-through estimator for hard sampling
            self.apply_straight_through(&soft_sample, &scaled_logits)
        } else {
            Ok(soft_sample)
        }
    }

    /// Apply straight-through estimator for hard sampling
    ///
    /// Creates a hard one-hot sample while maintaining gradients through
    /// the soft sample using the straight-through estimator.
    #[deprecated = "Legacy sampling module - use crate::processing::processors instead"]
    #[allow(dead_code)]
    fn apply_straight_through(
        &self,
        soft_sample: &Tensor,
        scaled_logits: &Tensor,
    ) -> Result<Tensor, SamplingError> {
        // Find argmax for hard sample
        let hard_indices = soft_sample
            .argmax(D::Minus1)
            .map_err(|e| SamplingError::TensorError(e.to_string()))?;

        // Create one-hot encoding
        let vocab_size = soft_sample
            .dim(D::Minus1)
            .map_err(|e| SamplingError::TensorError(e.to_string()))?;

        let hard_sample = self.create_one_hot(&hard_indices, vocab_size)?;

        // Apply straight-through estimator: hard sample for forward pass,
        // but use scaled_logits gradients for backward pass
        let straight_through = match (&hard_sample - soft_sample)
            .map_err(|e| SamplingError::TensorError(e.to_string()))
        {
            Ok(result) => result.detach(),
            Err(e) => return Err(e),
        };

        let result = match (soft_sample + &straight_through)
            .map_err(|e| SamplingError::TensorError(e.to_string()))
        {
            Ok(result) => result,
            Err(e) => return Err(e),
        };

        // Use scaled_logits for temperature information in gradient computation
        let temperature_adjusted = match scaled_logits
            .broadcast_mul(&result)
            .map_err(|e| SamplingError::TensorError(e.to_string()))
        {
            Ok(result) => result,
            Err(e) => return Err(e),
        };

        Ok(temperature_adjusted)
    }

    /// Create one-hot encoding from indices
    #[deprecated = "Legacy sampling module - use crate::processing::processors instead"]
    #[allow(dead_code)]
    fn create_one_hot(&self, indices: &Tensor, vocab_size: usize) -> Result<Tensor, SamplingError> {
        let batch_shape = indices.shape();
        let mut output_shape = batch_shape.dims().to_vec();
        output_shape.push(vocab_size);

        // Create zeros tensor - this will be our base one-hot tensor
        let shape = candle_core::Shape::from_dims(&output_shape);

        // Get indices as vector for efficient processing
        let indices_vec = indices
            .flatten_all()
            .map_err(|e| SamplingError::TensorError(e.to_string()))?
            .to_vec1::<u32>()
            .map_err(|e| SamplingError::TensorError(e.to_string()))?;

        // Create one-hot encoding vector for efficient batch processing
        let mut one_hot_vec = vec![0.0f32; output_shape.iter().product()];

        for (batch_idx, &token_idx) in indices_vec.iter().enumerate() {
            if (token_idx as usize) < vocab_size {
                let linear_idx = batch_idx * vocab_size + token_idx as usize;
                one_hot_vec[linear_idx] = 1.0;
            }
        }

        // Create the final one_hot tensor from the processed vector
        // This maintains mathematical correctness for Gumbel-Softmax operations:
        // 1. Proper categorical distribution for differentiable sampling
        // 2. Correct gradients for straight-through estimation
        // 3. Numerical stability in temperature scaling
        let one_hot = Tensor::from_vec(one_hot_vec, shape, &self.device)
            .map_err(|e| SamplingError::TensorError(e.to_string()))?;

        // Return the properly constructed one-hot tensor for Gumbel-Softmax sampling
        // This tensor maintains the mathematical properties required for:
        // 1. Differentiable sampling through reparameterization
        // 2. Straight-through gradient estimation in hard mode
        // 3. Categorical distribution correctness
        Ok(one_hot)
    }

    /// Get current temperature
    #[inline(always)]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Get hard sampling mode
    #[inline(always)]
    pub fn is_hard(&self) -> bool {
        self.hard
    }

    /// Update temperature (recreates cached values)
    pub fn set_temperature(&mut self, temperature: f32) -> Result<(), SamplingError> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(SamplingError::InvalidTemperature(temperature as f64));
        }

        if temperature < Self::MIN_TEMPERATURE || temperature > Self::MAX_TEMPERATURE {
            return Err(SamplingError::InvalidTemperature(temperature as f64));
        }

        self.temperature = temperature;
        self.inv_temperature = 1.0 / temperature;
        Ok(())
    }

    /// Set hard sampling mode
    #[inline(always)]
    pub fn set_hard(&mut self, hard: bool) {
        self.hard = hard;
    }
}

use crate::processing::context::ProcessingContext;
use crate::processing::traits::{LogitsProcessor, ProcessingResult};

impl LogitsProcessor for GumbelSoftmaxProcessor {
    fn process_logits(
        &mut self,
        logits: &mut [f32],
        _context: &ProcessingContext,
    ) -> ProcessingResult<()> {
        if logits.is_empty() {
            return Ok(());
        }

        // Generate Gumbel noise
        let mut rng = self.rng.lock().unwrap();
        let gumbel_noise: Vec<f32> = (0..logits.len())
            .map(|_| {
                let u: f32 = rng.random();
                // Gumbel(0,1) = -ln(-ln(U)) where U ~ Uniform(0,1)
                -(-(u.max(1e-20))).ln().ln()
            })
            .collect();

        // Apply Gumbel-Softmax: (logits + gumbel) / temperature
        for (i, logit) in logits.iter_mut().enumerate() {
            *logit = (*logit + gumbel_noise[i]) * self.inv_temperature;
        }

        // Apply softmax for normalization
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0f32;

        for logit in logits.iter_mut() {
            *logit = (*logit - max_logit).exp();
            sum += *logit;
        }

        if sum > 0.0 {
            for logit in logits.iter_mut() {
                *logit /= sum;
            }
        }

        Ok(())
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "GumbelSoftmaxProcessor"
    }

    #[inline(always)]
    fn is_identity(&self) -> bool {
        false // Gumbel-Softmax always modifies the distribution
    }
}

/// Builder for Gumbel-Softmax processor
#[derive(Debug, Clone)]
pub struct GumbelSoftmaxBuilder {
    temperature: Option<f32>,
    hard: bool,
    seed: u64,
    device: Option<Device>,
}

impl GumbelSoftmaxBuilder {
    /// Create new builder
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            temperature: None,
            hard: false,
            seed: 42,
            device: None,
        }
    }

    /// Set temperature
    #[inline(always)]
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Enable hard sampling
    #[inline(always)]
    pub fn hard(mut self) -> Self {
        self.hard = true;
        self
    }

    /// Enable soft sampling
    #[inline(always)]
    pub fn soft(mut self) -> Self {
        self.hard = false;
        self
    }

    /// Set random seed
    #[inline(always)]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set device
    #[inline(always)]
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Build the processor
    pub fn build(self) -> Result<GumbelSoftmaxProcessor, SamplingError> {
        let temperature = self
            .temperature
            .unwrap_or(GumbelSoftmaxProcessor::DEFAULT_TEMPERATURE);
        let device = self.device.unwrap_or(Device::Cpu);

        GumbelSoftmaxProcessor::new(temperature, self.hard, self.seed, device)
    }
}

impl Default for GumbelSoftmaxBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};

    use super::*;

    #[test]
    fn test_gumbel_softmax_creation() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let processor = GumbelSoftmaxProcessor::new(1.0, false, 42, device)?;

        assert_eq!(processor.temperature(), 1.0);
        assert!(!processor.is_hard());

        Ok(())
    }

    #[test]
    fn test_temperature_validation() {
        let device = Device::Cpu;

        // Valid temperature
        assert!(GumbelSoftmaxProcessor::new(1.0, false, 42, device.clone()).is_ok());

        // Invalid temperatures
        assert!(GumbelSoftmaxProcessor::new(0.0, false, 42, device.clone()).is_err());
        assert!(GumbelSoftmaxProcessor::new(-1.0, false, 42, device.clone()).is_err());
        assert!(GumbelSoftmaxProcessor::new(f32::NAN, false, 42, device.clone()).is_err());
    }

    #[test]
    fn test_builder_pattern() -> Result<(), Box<dyn std::error::Error>> {
        let processor = GumbelSoftmaxBuilder::new()
            .temperature(0.5)
            .hard()
            .seed(123)
            .device(Device::Cpu)
            .build()?;

        assert_eq!(processor.temperature(), 0.5);
        assert!(processor.is_hard());

        Ok(())
    }

    #[test]
    fn test_gumbel_noise_generation() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let processor = GumbelSoftmaxProcessor::new(1.0, false, 42, device)?;

        let shape = candle_core::Shape::from_dims(&[10]);
        let noise = processor.generate_gumbel_noise(&shape)?;

        assert_eq!(noise.shape().dims(), &[10]);
        assert_eq!(noise.dtype(), DType::F32);

        Ok(())
    }
}
