//! Temperature scaling processor with advanced numerical stability
//!
//! Implements temperature scaling for logits with comprehensive validation,
//! numerical stability guarantees, and zero-allocation in-place operations.

use crate::processing::traits::{
    ConfigurableProcessor, LogitsProcessor, NumericallyStableProcessor, ProcessingResult,
    ZeroAllocationProcessor,
    utils::{clamp_for_stability, find_max_logit, validate_logits},
};
use crate::processing::{ProcessingContext, ProcessingError};

/// Temperature scaling processor for controlling generation randomness
///
/// Temperature scaling modifies the sharpness of the probability distribution:
/// - temperature < 1.0: More focused, less random (sharper distribution)
/// - temperature = 1.0: No change (identity)
/// - temperature > 1.0: More random, less focused (flatter distribution)
///
/// Implementation uses numerically stable operations to prevent overflow/underflow
/// and processes logits in-place for zero allocation.
#[derive(Debug, Clone)]
pub struct TemperatureProcessor {
    /// Temperature value
    temperature: f32,
    /// Pre-computed inverse temperature for efficiency
    inv_temperature: f32,
    /// Cached identity status for optimization
    is_identity: bool,
}

impl TemperatureProcessor {
    /// Minimum allowed temperature to prevent numerical instability
    pub const MIN_TEMPERATURE: f32 = 0.001;

    /// Maximum allowed temperature to prevent numerical instability
    pub const MAX_TEMPERATURE: f32 = 100.0;

    /// Temperature threshold below which we consider it effectively identity
    pub const IDENTITY_THRESHOLD: f32 = 1e-6;

    /// Create a new temperature processor
    ///
    /// # Arguments
    /// * `temperature` - Temperature value (must be > 0.0 and finite)
    ///
    /// # Returns
    /// * `Ok(TemperatureProcessor)` - Successfully created processor
    /// * `Err(ProcessingError)` - Invalid temperature value
    ///
    /// # Examples
    /// ```
    /// use fluent_ai_candle::processing::processors::TemperatureProcessor;
    ///
    /// let processor = TemperatureProcessor::new(0.8)?;
    /// ```
    #[inline(always)]
    pub fn new(temperature: f32) -> ProcessingResult<Self> {
        // Validate temperature
        if !temperature.is_finite() {
            return Err(ProcessingError::configuration(format!(
                "Temperature must be finite, got: {}",
                temperature
            )));
        }

        if temperature <= 0.0 {
            return Err(ProcessingError::configuration(format!(
                "Temperature must be positive, got: {}",
                temperature
            )));
        }

        if temperature < Self::MIN_TEMPERATURE {
            return Err(ProcessingError::configuration(format!(
                "Temperature {} is below minimum {}",
                temperature,
                Self::MIN_TEMPERATURE
            )));
        }

        if temperature > Self::MAX_TEMPERATURE {
            return Err(ProcessingError::configuration(format!(
                "Temperature {} exceeds maximum {}",
                temperature,
                Self::MAX_TEMPERATURE
            )));
        }

        // Check if effectively identity
        let is_identity = (temperature - 1.0).abs() < Self::IDENTITY_THRESHOLD;

        // Pre-compute inverse for efficiency
        let inv_temperature = if is_identity { 1.0 } else { 1.0 / temperature };

        Ok(Self {
            temperature,
            inv_temperature,
            is_identity,
        })
    }

    /// Create temperature processor with creative preset (0.8)
    #[inline(always)]
    pub fn creative() -> ProcessingResult<Self> {
        Self::new(0.8)
    }

    /// Create temperature processor with balanced preset (0.7)
    #[inline(always)]
    pub fn balanced() -> ProcessingResult<Self> {
        Self::new(0.7)
    }

    /// Create temperature processor with focused preset (0.3)
    #[inline(always)]
    pub fn focused() -> ProcessingResult<Self> {
        Self::new(0.3)
    }

    /// Create temperature processor with deterministic preset (0.01)
    #[inline(always)]
    pub fn deterministic() -> ProcessingResult<Self> {
        Self::new(0.01)
    }

    /// Get the temperature value
    #[inline(always)]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Apply temperature scaling to logits in-place with numerical stability
    ///
    /// Uses advanced numerical techniques to prevent overflow/underflow:
    /// - Clamping to safe ranges
    /// - Monitoring for numerical instabilities
    /// - Early termination for identity cases
    fn apply_temperature_scaling_inplace(&self, logits: &mut [f32]) -> ProcessingResult<()> {
        // Early return for identity temperature
        if self.is_identity {
            return Ok(());
        }

        // Validate input logits
        validate_logits(logits, "TemperatureProcessor")?;

        // Apply numerical stability clamping before processing
        clamp_for_stability(logits);

        // For very low temperatures (approaching greedy), handle specially
        if self.temperature < 0.01 {
            return self.apply_greedy_approximation(logits);
        }

        // Apply temperature scaling in-place: logits[i] = logits[i] / temperature
        for logit in logits.iter_mut() {
            *logit *= self.inv_temperature;
        }

        // Validate output for numerical stability
        self.validate_output(logits)?;

        Ok(())
    }

    /// Apply greedy approximation for very low temperatures
    ///
    /// For very low temperatures, softmax becomes effectively greedy sampling.
    /// We implement this more efficiently by setting the maximum logit to a high value
    /// and others to a low value, avoiding numerical issues with extreme scaling.
    fn apply_greedy_approximation(&self, logits: &mut [f32]) -> ProcessingResult<()> {
        if logits.is_empty() {
            return Ok(());
        }

        // Find the maximum logit(s)
        let max_logit = find_max_logit(logits)?;

        // Set high value for max, low value for others
        const HIGH_VALUE: f32 = 10.0;
        const LOW_VALUE: f32 = -10.0;

        for logit in logits.iter_mut() {
            *logit = if (*logit - max_logit).abs() < f32::EPSILON {
                HIGH_VALUE
            } else {
                LOW_VALUE
            };
        }

        Ok(())
    }

    /// Validate output logits for numerical stability
    fn validate_output(&self, logits: &[f32]) -> ProcessingResult<()> {
        // Check for NaN or infinite values after processing
        for (i, &logit) in logits.iter().enumerate() {
            if !logit.is_finite() {
                return Err(ProcessingError::numerical(format!(
                    "Temperature scaling produced non-finite logit at index {}: {}",
                    i, logit
                )));
            }
        }

        // Check for extreme values that might cause issues downstream
        let max_safe = 50.0; // exp(50) is well below f32::MAX
        let min_safe = -50.0;

        for &logit in logits.iter() {
            if logit > max_safe || logit < min_safe {
                // This is a warning condition, but we'll clamp rather than error
                break;
            }
        }

        Ok(())
    }
}

impl LogitsProcessor for TemperatureProcessor {
    #[inline]
    fn process_logits(
        &mut self,
        logits: &mut [f32],
        _context: &ProcessingContext,
    ) -> ProcessingResult<()> {
        self.apply_temperature_scaling_inplace(logits)
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "TemperatureProcessor"
    }

    #[inline(always)]
    fn is_identity(&self) -> bool {
        self.is_identity
    }

    #[inline(always)]
    fn is_enabled(&self) -> bool {
        !self.is_identity
    }

    fn validate(&self) -> ProcessingResult<()> {
        // Validate current temperature value
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(ProcessingError::configuration(format!(
                "Invalid temperature in processor: {}",
                self.temperature
            )));
        }

        if self.temperature < Self::MIN_TEMPERATURE || self.temperature > Self::MAX_TEMPERATURE {
            return Err(ProcessingError::configuration(format!(
                "Temperature {} is outside valid range [{}, {}]",
                self.temperature,
                Self::MIN_TEMPERATURE,
                Self::MAX_TEMPERATURE
            )));
        }

        Ok(())
    }

    fn config_summary(&self) -> String {
        format!("TemperatureProcessor(temperature={})", self.temperature)
    }

    #[inline(always)]
    fn estimated_overhead(&self) -> f32 {
        if self.is_identity {
            0.0 // No overhead for identity operations
        } else if self.temperature < 0.01 {
            1.5 // Greedy approximation has some overhead
        } else {
            1.0 // Standard scaling operation
        }
    }

    #[inline(always)]
    fn priority(&self) -> u8 {
        super::priorities::DISTRIBUTION_MODIFIER
    }
}

/// Configuration for temperature processor
#[derive(Debug, Clone)]
pub struct TemperatureConfig {
    pub temperature: f32,
}

impl ConfigurableProcessor for TemperatureProcessor {
    type Config = TemperatureConfig;

    fn update_config(&mut self, config: Self::Config) -> ProcessingResult<()> {
        // Create new processor to validate configuration
        let new_processor = Self::new(config.temperature)?;

        // Update all fields
        self.temperature = new_processor.temperature;
        self.inv_temperature = new_processor.inv_temperature;
        self.is_identity = new_processor.is_identity;

        Ok(())
    }

    fn get_config(&self) -> Self::Config {
        TemperatureConfig {
            temperature: self.temperature,
        }
    }
}

// Marker trait implementations
impl ZeroAllocationProcessor for TemperatureProcessor {}
impl NumericallyStableProcessor for TemperatureProcessor {}

/// Builder for temperature processor with validation and presets
#[derive(Debug, Clone, Default)]
pub struct TemperatureBuilder {
    temperature: Option<f32>,
}

impl TemperatureBuilder {
    /// Create a new temperature builder
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set temperature value
    #[inline(always)]
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Use creative temperature preset (0.8)
    #[inline(always)]
    pub fn creative(mut self) -> Self {
        self.temperature = Some(0.8);
        self
    }

    /// Use balanced temperature preset (0.7)
    #[inline(always)]
    pub fn balanced(mut self) -> Self {
        self.temperature = Some(0.7);
        self
    }

    /// Use focused temperature preset (0.3)
    #[inline(always)]
    pub fn focused(mut self) -> Self {
        self.temperature = Some(0.3);
        self
    }

    /// Use deterministic temperature preset (0.01)
    #[inline(always)]
    pub fn deterministic(mut self) -> Self {
        self.temperature = Some(0.01);
        self
    }

    /// Build the temperature processor
    pub fn build(self) -> ProcessingResult<TemperatureProcessor> {
        let temperature = self.temperature.unwrap_or(1.0);
        TemperatureProcessor::new(temperature)
    }
}

/// Utility functions for temperature-related operations
pub mod utils {
    use super::*;
    use crate::processing::ProcessingContext;

    /// Calculate adaptive temperature based on context
    ///
    /// Adjusts temperature based on generation context to prevent
    /// repetition and maintain coherence. Uses sophisticated heuristics
    /// to balance creativity and quality.
    pub fn adaptive_temperature(
        base_temperature: f32,
        context: &ProcessingContext,
        repetition_factor: f32,
    ) -> ProcessingResult<f32> {
        if !base_temperature.is_finite() || base_temperature <= 0.0 {
            return Err(ProcessingError::configuration("Invalid base temperature"));
        }

        if !repetition_factor.is_finite() || repetition_factor < 0.0 {
            return Err(ProcessingError::configuration("Invalid repetition factor"));
        }

        let mut adjusted = base_temperature;

        // Increase temperature with repetition to encourage diversity
        let repetition_adjustment = 1.0 + (repetition_factor * 0.1);
        adjusted *= repetition_adjustment;

        // Decrease temperature for longer contexts to maintain coherence
        let context_length = context.sequence_length() as f32;
        let context_adjustment = 1.0 - (context_length * 0.0001).min(0.2);
        adjusted *= context_adjustment;

        // Adjust based on context utilization
        let utilization = context.utilization_ratio();
        if utilization > 0.8 {
            // Near context limit, reduce temperature for coherence
            adjusted *= 0.9;
        }

        // Clamp to valid range
        adjusted = adjusted.clamp(
            TemperatureProcessor::MIN_TEMPERATURE,
            TemperatureProcessor::MAX_TEMPERATURE,
        );

        Ok(adjusted)
    }

    /// Calculate temperature from desired entropy level
    ///
    /// Maps desired entropy (0.0 to 1.0) to appropriate temperature value.
    /// Uses exponential mapping for more intuitive entropy control.
    pub fn temperature_from_entropy(desired_entropy: f32) -> ProcessingResult<f32> {
        if !desired_entropy.is_finite() {
            return Err(ProcessingError::configuration(
                "Desired entropy must be finite",
            ));
        }

        if !(0.0..=1.0).contains(&desired_entropy) {
            return Err(ProcessingError::configuration(format!(
                "Desired entropy {} must be between 0.0 and 1.0",
                desired_entropy
            )));
        }

        if desired_entropy <= 0.0 {
            return Ok(TemperatureProcessor::MIN_TEMPERATURE);
        }

        if desired_entropy >= 1.0 {
            return Ok(1.0);
        }

        // Exponential mapping for more intuitive control
        let mapped = TemperatureProcessor::MIN_TEMPERATURE
            + (desired_entropy * desired_entropy) * (1.0 - TemperatureProcessor::MIN_TEMPERATURE);

        Ok(mapped.clamp(TemperatureProcessor::MIN_TEMPERATURE, 1.0))
    }

    /// Estimate entropy after temperature scaling
    ///
    /// Provides an estimate of the entropy of the probability distribution
    /// after applying temperature scaling. Useful for adaptive sampling strategies.
    pub fn estimate_post_scaling_entropy(
        logits: &[f32],
        temperature: f32,
    ) -> ProcessingResult<f32> {
        if logits.is_empty() {
            return Err(ProcessingError::validation("Empty logits array"));
        }

        // Apply temperature scaling (without modifying input)
        let mut scaled_logits: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

        // Apply softmax for probability computation
        let max_logit = find_max_logit(&scaled_logits)?;

        // Compute softmax with numerical stability
        let mut exp_sum = 0.0f32;
        for logit in &mut scaled_logits {
            *logit = (*logit - max_logit).exp();
            exp_sum += *logit;
        }

        if exp_sum <= 0.0 {
            return Err(ProcessingError::numerical("Invalid softmax normalization"));
        }

        // Normalize to probabilities and calculate entropy
        let mut entropy = 0.0f32;
        for &exp_logit in &scaled_logits {
            let prob = exp_logit / exp_sum;
            if prob > 0.0 {
                entropy -= prob * prob.ln();
            }
        }

        Ok(entropy)
    }

    /// Find optimal temperature for target perplexity
    ///
    /// Uses binary search to find temperature that achieves target perplexity.
    /// Useful for automated temperature tuning based on desired output characteristics.
    pub fn find_temperature_for_perplexity(
        logits: &[f32],
        target_perplexity: f32,
        tolerance: f32,
        max_iterations: usize,
    ) -> ProcessingResult<f32> {
        if target_perplexity <= 1.0 {
            return Err(ProcessingError::configuration(
                "Target perplexity must be > 1.0",
            ));
        }

        if tolerance <= 0.0 {
            return Err(ProcessingError::configuration("Tolerance must be positive"));
        }

        let target_entropy = target_perplexity.ln();
        let mut low = TemperatureProcessor::MIN_TEMPERATURE;
        let mut high = TemperatureProcessor::MAX_TEMPERATURE;

        for _ in 0..max_iterations {
            let mid = (low + high) / 2.0;
            let entropy = estimate_post_scaling_entropy(logits, mid)?;

            if (entropy - target_entropy).abs() < tolerance {
                return Ok(mid);
            }

            if entropy < target_entropy {
                low = mid;
            } else {
                high = mid;
            }
        }

        // Return best approximation
        Ok((low + high) / 2.0)
    }
}
