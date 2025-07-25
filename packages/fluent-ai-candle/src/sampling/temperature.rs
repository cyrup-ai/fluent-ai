//! Temperature scaling processor with advanced numerical stability
//!
//! Implements temperature scaling for logits with comprehensive validation,
//! numerical stability guarantees, and zero-allocation tensor operations.

use candle_core::{D, Tensor};
use candle_nn::ops;

use super::SamplingError;
// Removed unused import: crate::processing::traits::LogitsProcessor

/// Temperature scaling processor for controlling generation randomness
///
/// Temperature scaling modifies the sharpness of the probability distribution:
/// - temperature < 1.0: More focused, less random (sharper distribution)
/// - temperature = 1.0: No change (identity)
/// - temperature > 1.0: More random, less focused (flatter distribution)
///
/// Implementation uses numerically stable operations to prevent overflow/underflow.
#[derive(Debug, Clone)]
pub struct TemperatureProcessor {
    temperature: f32,
    inv_temperature: f32, // Pre-computed 1/temperature for efficiency
    is_identity: bool,    // Optimization for temperature == 1.0
}

impl TemperatureProcessor {
    /// Minimum allowed temperature to prevent numerical instability
    pub const MIN_TEMPERATURE: f32 = 0.001;

    /// Maximum allowed temperature to prevent numerical instability
    pub const MAX_TEMPERATURE: f32 = 100.0;

    /// Temperature threshold below which we consider it effectively zero
    pub const ZERO_THRESHOLD: f32 = 1e-6;

    /// Create a new temperature processor
    ///
    /// # Arguments
    /// * `temperature` - Temperature value (must be > 0.0)
    ///
    /// # Returns
    /// * `Ok(TemperatureProcessor)` - Successfully created processor
    /// * `Err(SamplingError::InvalidTemperature)` - Invalid temperature value
    ///
    /// # Examples
    /// ```
    /// use fluent_ai_candle::sampling::TemperatureProcessor;
    ///
    /// let processor = TemperatureProcessor::new(0.8)?;
    /// ```
    #[inline(always)]
    pub fn new(temperature: f32) -> Result<Self, SamplingError> {
        // Validate temperature range
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(SamplingError::InvalidTemperature(temperature.into()));
        }

        if temperature < Self::MIN_TEMPERATURE {
            return Err(SamplingError::InvalidTemperature(temperature.into()));
        }

        if temperature > Self::MAX_TEMPERATURE {
            return Err(SamplingError::InvalidTemperature(temperature.into()));
        }

        let is_identity = (temperature - 1.0).abs() < Self::ZERO_THRESHOLD;
        let inv_temperature = if is_identity { 1.0 } else { 1.0 / temperature };

        Ok(Self {
            temperature,
            inv_temperature,
            is_identity})
    }

    /// Get the temperature value
    #[inline(always)]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Check if this processor is effectively an identity operation
    #[inline(always)]
    pub fn is_identity(&self) -> bool {
        self.is_identity
    }

    /// Process logits with temperature scaling (public interface)
    ///
    /// This is the main public method for applying temperature scaling.
    #[inline(always)]
    pub fn process(&self, logits: &Tensor) -> Result<Tensor, SamplingError> {
        self.apply_temperature_scaling(logits)
    }

    /// Apply temperature scaling with numerical stability
    ///
    /// Uses the formula: scaled_logits = logits / temperature
    /// with special handling for numerical edge cases.
    fn apply_temperature_scaling(&self, logits: &Tensor) -> Result<Tensor, SamplingError> {
        if self.is_identity {
            // Identity case - no scaling needed
            return Ok(logits.clone());
        }

        // Check for numerical issues in input logits
        let logits_vec = logits.to_vec1::<f32>().map_err(SamplingError::from)?;

        // Detect and handle numerical instabilities
        let has_nan = logits_vec.iter().any(|&x| x.is_nan());
        let has_inf = logits_vec.iter().any(|&x| x.is_infinite());

        if has_nan {
            return Err(SamplingError::NumericalInstability(
                "NaN values detected in logits".to_string(),
            ));
        }

        // Handle infinite values by clamping to prevent overflow after scaling
        let processed_logits = if has_inf {
            let max_safe_logit = f32::MAX.ln() * self.temperature;
            let min_safe_logit = -max_safe_logit;

            let clamped: Vec<f32> = logits_vec
                .iter()
                .map(|&x| {
                    if x.is_infinite() && x > 0.0 {
                        max_safe_logit
                    } else if x.is_infinite() && x < 0.0 {
                        min_safe_logit
                    } else {
                        x
                    }
                })
                .collect();

            Tensor::from_vec(clamped, logits.shape(), logits.device())
                .map_err(SamplingError::from)?
        } else {
            logits.clone()
        };

        // Apply temperature scaling: logits / temperature
        // Using multiplication by inverse for better numerical properties
        let scaled = processed_logits
            .affine(self.inv_temperature as f64, 0.0)
            .map_err(SamplingError::from)?;

        // Validate output for numerical stability
        let output_vec = scaled.to_vec1::<f32>().map_err(SamplingError::from)?;
        let output_has_nan = output_vec.iter().any(|&x| x.is_nan());

        if output_has_nan {
            return Err(SamplingError::NumericalInstability(
                "NaN values detected in temperature scaling output".to_string(),
            ));
        }

        Ok(scaled)
    }
}

// TODO: Update to new LogitsProcessor API that uses process_logits() instead of process()
// impl LogitsProcessor for TemperatureProcessor {
//     #[inline(always)]
//     fn process_logits(&mut self, logits: &mut [f32], context: &ProcessingContext) -> ProcessingResult<()> {
//         // Implementation needed for new API
//     }
//
//     #[inline(always)]
//     fn name(&self) -> &'static str {
//         "TemperatureProcessor"
//     }
// }

/// Builder for temperature processor with validation and presets
#[derive(Debug, Clone)]
pub struct TemperatureBuilder {
    temperature: Option<f32>}

impl TemperatureBuilder {
    /// Create a new temperature builder
    #[inline(always)]
    pub fn new() -> Self {
        Self { temperature: None }
    }

    /// Set temperature value
    #[inline(always)]
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Use creative temperature (higher randomness)
    #[inline(always)]
    pub fn creative(mut self) -> Self {
        self.temperature = Some(0.8);
        self
    }

    /// Use balanced temperature (moderate randomness)
    #[inline(always)]
    pub fn balanced(mut self) -> Self {
        self.temperature = Some(0.7);
        self
    }

    /// Use focused temperature (lower randomness)
    #[inline(always)]
    pub fn focused(mut self) -> Self {
        self.temperature = Some(0.3);
        self
    }

    /// Use deterministic temperature (very low randomness)
    #[inline(always)]
    pub fn deterministic(mut self) -> Self {
        self.temperature = Some(0.1);
        self
    }

    /// Build the temperature processor
    pub fn build(self) -> Result<TemperatureProcessor, SamplingError> {
        let temperature = self.temperature.unwrap_or(1.0);
        TemperatureProcessor::new(temperature)
    }
}

impl Default for TemperatureBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for temperature-related operations
pub mod utils {
    use super::*;

    /// Calculate effective temperature based on context
    ///
    /// Adjusts temperature based on generation context to prevent
    /// repetition and maintain coherence.
    #[inline(always)]
    pub fn adaptive_temperature(
        base_temperature: f32,
        repetition_factor: f32,
        context_length: usize,
    ) -> f32 {
        // Increase temperature slightly with repetition to encourage diversity
        let repetition_adjustment = 1.0 + (repetition_factor - 1.0) * 0.1;

        // Decrease temperature slightly for longer contexts to maintain coherence
        let context_adjustment = 1.0 - (context_length as f32 * 0.001).min(0.2);

        let adjusted = base_temperature * repetition_adjustment * context_adjustment;

        // Clamp to valid range
        adjusted.clamp(
            TemperatureProcessor::MIN_TEMPERATURE,
            TemperatureProcessor::MAX_TEMPERATURE,
        )
    }

    /// Calculate temperature from desired entropy level
    ///
    /// Maps desired entropy (0.0 to 1.0) to appropriate temperature value.
    #[inline(always)]
    pub fn temperature_from_entropy(desired_entropy: f32) -> f32 {
        if desired_entropy <= 0.0 {
            return TemperatureProcessor::MIN_TEMPERATURE;
        }
        if desired_entropy >= 1.0 {
            return 1.0;
        }

        // Exponential mapping for more intuitive entropy control
        let mapped = 0.1 + (desired_entropy * desired_entropy) * 0.9;
        mapped.clamp(TemperatureProcessor::MIN_TEMPERATURE, 1.0)
    }

    /// Estimate entropy of probability distribution after temperature scaling
    pub fn estimate_entropy_after_scaling(
        logits: &Tensor,
        temperature: f32,
    ) -> Result<f32, SamplingError> {
        // Apply temperature scaling
        let processor = TemperatureProcessor::new(temperature)?;
        let scaled_logits = processor.process(logits)?;

        // Convert to probabilities
        let probs = ops::softmax(&scaled_logits, D::Minus1).map_err(SamplingError::from)?;
        let prob_vec = probs.to_vec1::<f32>().map_err(SamplingError::from)?;

        // Calculate entropy: -Σ(p * log(p))
        let entropy = prob_vec
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum::<f32>();

        Ok(entropy)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};

    use super::*;

    fn create_test_logits() -> CandleResult<Tensor> {
        let device = Device::Cpu;
        let logits = vec![1.0, 2.0, 3.0, 0.5, 1.5];
        Tensor::from_vec(logits, (1, 5), &device)
    }

    #[test]
    fn test_temperature_validation() {
        // Valid temperatures
        assert!(TemperatureProcessor::new(0.1).is_ok());
        assert!(TemperatureProcessor::new(1.0).is_ok());
        assert!(TemperatureProcessor::new(2.0).is_ok());

        // Invalid temperatures
        assert!(TemperatureProcessor::new(0.0).is_err());
        assert!(TemperatureProcessor::new(-1.0).is_err());
        assert!(TemperatureProcessor::new(f32::NAN).is_err());
        assert!(TemperatureProcessor::new(f32::INFINITY).is_err());
    }

    #[test]
    fn test_identity_optimization() -> Result<(), Box<dyn std::error::Error>> {
        let processor = TemperatureProcessor::new(1.0)?;
        assert!(processor.is_identity());

        let processor = TemperatureProcessor::new(0.8)?;
        assert!(!processor.is_identity());

        Ok(())
    }

    #[test]
    fn test_temperature_builder() -> Result<(), Box<dyn std::error::Error>> {
        let processor = TemperatureBuilder::new().creative().build()?;
        assert_eq!(processor.temperature(), 0.8);

        let processor = TemperatureBuilder::new().focused().build()?;
        assert_eq!(processor.temperature(), 0.3);

        Ok(())
    }

    #[test]
    fn test_adaptive_temperature() {
        let base_temp = 0.7;
        let adjusted = utils::adaptive_temperature(base_temp, 1.2, 100);
        assert!(adjusted > base_temp); // Should increase due to repetition

        let adjusted = utils::adaptive_temperature(base_temp, 1.0, 1000);
        assert!(adjusted < base_temp); // Should decrease due to long context
    }
}
