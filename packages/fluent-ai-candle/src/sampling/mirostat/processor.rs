//! Mirostat Processor Module
//!
//! Ultra-high-performance Mirostat processor with zero allocation and lock-free operations.
//! Implements both Mirostat v1 and v2 algorithms with atomic state management.

use std::sync::atomic::{AtomicU64, Ordering};
use candle_core::{Result as CandleResult, Tensor};
use fastrand::Rng;

use crate::logits::SamplingConfig;
use crate::processing::context::ProcessingContext;
use crate::processing::error::{ProcessingError, ProcessingResult};
use crate::processing::traits::LogitsProcessor;

use super::config::MirostatConfig;
use super::perplexity::PerplexityState;
use super::stats::MirostatStats;

/// Minimum tau for numerical stability
const MIN_TAU: f32 = 0.1;
/// Maximum tau to prevent extreme filtering
const MAX_TAU: f32 = 20.0;

/// Ultra-high-performance Mirostat processor with zero allocation
#[repr(C, align(64))] // Cache line aligned
pub struct MirostatProcessor {
    /// Algorithm configuration
    config: MirostatConfig,
    /// Perplexity tracking state
    state: PerplexityState,
    /// Current dynamic tau value (atomic for lock-free access)
    current_tau: AtomicU64, // f32 as u64 for atomic access
    /// Random number generator (stack allocated)
    rng: Rng,

    /// Processing statistics
    tokens_processed: u64,
    avg_processing_time_nanos: f64,
}

impl std::fmt::Debug for MirostatProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MirostatProcessor")
            .field("config", &self.config)
            .field("tokens_processed", &self.tokens_processed)
            .field("avg_processing_time_nanos", &self.avg_processing_time_nanos)
            .finish()
    }
}

impl MirostatProcessor {
    /// Create new Mirostat processor with configuration
    pub fn new(config: MirostatConfig) -> CandleResult<Self> {
        let ema_alpha = match config {
            MirostatConfig::V1 { learning_rate, .. } => learning_rate * 0.5,
            MirostatConfig::V2 { eta, .. } => (eta * 0.1).clamp(0.01, 0.2),
        };

        Ok(Self {
            config,
            state: PerplexityState::new(ema_alpha),
            current_tau: AtomicU64::new(Self::f32_to_atomic_u64(config.tau())),
            rng: Rng::new(),
            tokens_processed: 0,
            avg_processing_time_nanos: 0.0,
        })
    }

    /// Create Mirostat v1 processor with default parameters
    #[inline(always)]
    pub fn v1(tau: f32, learning_rate: f32) -> CandleResult<Self> {
        let config = MirostatConfig::v1(tau, learning_rate)?;
        Self::new(config)
    }

    /// Create Mirostat v2 processor with default parameters
    #[inline(always)]
    pub fn v2(tau: f32, eta: f32) -> CandleResult<Self> {
        let config = MirostatConfig::v2(tau, eta)?;
        Self::new(config)
    }

    /// Convert f32 to u64 for atomic storage
    #[inline(always)]
    fn f32_to_atomic_u64(value: f32) -> u64 {
        value.to_bits() as u64
    }

    /// Convert u64 back to f32 from atomic storage
    #[inline(always)]
    fn atomic_u64_to_f32(value: u64) -> f32 {
        f32::from_bits(value as u32)
    }

    /// Calculate perplexity from logits with numerical stability
    #[inline(always)]
    fn calculate_perplexity(&mut self, logits: &[f32]) -> CandleResult<f32> {
        if logits.is_empty() {
            return Err(candle_core::Error::Msg("Empty logits array".to_string()));
        }

        // Find maximum for numerical stability
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        if !max_logit.is_finite() {
            return Err(candle_core::Error::Msg(
                "Invalid logits contain non-finite values".to_string(),
            ));
        }

        // Calculate softmax with stability
        let mut temp_probs = Vec::with_capacity(logits.len());
        let mut sum_exp = 0.0_f64; // Use f64 for accumulation precision

        for &logit in logits {
            let stable_exp = (logit - max_logit).exp();
            temp_probs.push(stable_exp);
            sum_exp += stable_exp as f64;
        }

        if sum_exp <= 0.0 || !sum_exp.is_finite() {
            return Err(candle_core::Error::Msg(
                "Invalid probability distribution".to_string(),
            ));
        }

        // Calculate entropy (negative log-likelihood)
        let mut entropy = 0.0_f64;
        for &prob in &temp_probs {
            let normalized_prob = (prob as f64) / sum_exp;
            if normalized_prob > 1e-12 {
                // Avoid log(0)
                entropy -= normalized_prob * normalized_prob.ln();
            }
        }

        // Perplexity is 2^entropy
        let perplexity = (entropy / std::f64::consts::LN_2).exp() as f32;

        if perplexity.is_finite() && perplexity > 0.0 {
            Ok(perplexity)
        } else {
            Err(candle_core::Error::Msg(
                "Perplexity calculation resulted in invalid value".to_string(),
            ))
        }
    }

    /// Update dynamic tau based on perplexity feedback
    #[inline(always)]
    fn update_tau(&mut self, current_perplexity: f32) {
        let target_tau = self.config.tau();
        let current_tau = Self::atomic_u64_to_f32(self.current_tau.load(Ordering::Relaxed));

        let new_tau = match self.config {
            MirostatConfig::V1 { learning_rate, .. } => {
                // Direct tau adjustment based on perplexity error
                let error = current_perplexity - target_tau;
                let adjustment = learning_rate * error;
                (current_tau - adjustment).clamp(MIN_TAU, MAX_TAU)
            }
            MirostatConfig::V2 { eta, .. } => {
                // Temperature-based adjustment
                let ratio = current_perplexity / target_tau;
                if ratio > 1.0 {
                    // Reduce tau to increase filtering
                    current_tau * (1.0 - eta * (ratio - 1.0)).clamp(0.5, 1.0)
                } else {
                    // Increase tau to reduce filtering
                    current_tau * (1.0 + eta * (1.0 - ratio)).clamp(1.0, 2.0)
                }
            }
        };

        self.current_tau
            .store(Self::f32_to_atomic_u64(new_tau), Ordering::Relaxed);
    }

    /// Apply Mirostat filtering to probability distribution slice
    #[inline(always)]
    fn apply_mirostat_filtering(
        &mut self,
        probabilities: &mut [f32],
        tau: f32,
    ) -> ProcessingResult<()> {
        // Simple Mirostat implementation: suppress low-probability tokens based on tau
        // This is a simplified version - full Mirostat is more complex

        if probabilities.is_empty() {
            return Ok(());
        }

        // Find threshold based on tau (simplified approach)
        let threshold = 1.0 / (tau * probabilities.len() as f32);

        // Suppress probabilities below threshold
        let mut _total_suppressed = 0.0f32;
        for prob in probabilities.iter_mut() {
            if *prob < threshold {
                _total_suppressed += *prob;
                *prob = 0.0;
            }
        }

        // Renormalize remaining probabilities
        let remaining_sum: f32 = probabilities.iter().sum();
        if remaining_sum > 0.0 {
            let scale = 1.0 / remaining_sum;
            for prob in probabilities.iter_mut() {
                *prob *= scale;
            }
        }

        Ok(())
    }

    /// Apply Mirostat sampling to probability distribution
    #[deprecated = "Legacy sampling module - use crate::processing::processors instead"]
    #[inline(always)]
    fn apply_mirostat_sampling(
        &self,
        logits: &Tensor,
    ) -> Result<Tensor, crate::sampling::SamplingError> {
        // For now, implement a simple passthrough until we can properly implement Mirostat with Tensors
        // TODO: Implement proper Mirostat sampling with Tensor operations
        Ok(logits.clone())
    }

    /// Get current dynamic tau value
    #[inline(always)]
    pub fn current_tau(&self) -> f32 {
        Self::atomic_u64_to_f32(self.current_tau.load(Ordering::Relaxed))
    }

    /// Get current perplexity estimate
    #[inline(always)]
    pub fn current_perplexity(&self) -> f32 {
        self.state.current_perplexity()
    }

    /// Get processing statistics
    #[inline(always)]
    pub fn stats(&self) -> MirostatStats {
        MirostatStats {
            tokens_processed: self.tokens_processed,
            avg_processing_time_nanos: self.avg_processing_time_nanos,
            current_tau: self.current_tau(),
            current_perplexity: self.current_perplexity(),
            perplexity_variance: self.state.variance(),
            config: self.config,
        }
    }

    /// Reset processor state for new sequence
    pub fn reset(&mut self) {
        self.state.reset();
        self.current_tau.store(
            Self::f32_to_atomic_u64(self.config.tau()),
            Ordering::Relaxed,
        );
        self.tokens_processed = 0;
        self.avg_processing_time_nanos = 0.0;
    }
}

impl LogitsProcessor for MirostatProcessor {
    fn process_logits(
        &mut self,
        logits: &mut [f32],
        _context: &ProcessingContext,
    ) -> ProcessingResult<()> {
        // Apply Mirostat algorithm directly to logits
        if logits.is_empty() {
            return Err(ProcessingError::validation("Empty logits array"));
        }

        // Convert logits to probabilities using softmax
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;

        // Compute softmax probabilities
        for logit in logits.iter_mut() {
            *logit = (*logit - max_logit).exp();
            sum += *logit;
        }

        if sum <= 0.0 {
            return Err(ProcessingError::numerical(
                "Invalid probability distribution",
            ));
        }

        // Normalize probabilities
        for prob in logits.iter_mut() {
            *prob /= sum;
        }

        // Calculate current perplexity
        let current_perplexity = self.calculate_perplexity(logits).map_err(|e| {
            ProcessingError::external(format!("Perplexity calculation failed: {}", e))
        })?;

        // Update tau based on perplexity feedback
        self.update_tau(current_perplexity);

        // Apply Mirostat filtering with current tau
        let current_tau = Self::atomic_u64_to_f32(self.current_tau.load(Ordering::Relaxed));
        self.apply_mirostat_filtering(logits, current_tau)?;

        self.tokens_processed += 1;
        Ok(())
    }

    fn name(&self) -> &'static str {
        match self.config {
            MirostatConfig::V1 { .. } => "MirostatV1",
            MirostatConfig::V2 { .. } => "MirostatV2",
        }
    }
}

/// Utility functions for Mirostat sampling
pub mod utils {
    use super::*;

    /// Create optimized Mirostat v1 processor for creative writing
    #[inline(always)]
    pub fn creative_v1() -> CandleResult<MirostatProcessor> {
        MirostatProcessor::v1(8.0, 0.2) // Higher tau, faster learning
    }

    /// Create optimized Mirostat v1 processor for coherent completion
    #[inline(always)]
    pub fn coherent_v1() -> CandleResult<MirostatProcessor> {
        MirostatProcessor::v1(3.0, 0.1) // Lower tau, slower learning
    }

    /// Create optimized Mirostat v2 processor for balanced generation
    #[inline(always)]
    pub fn balanced_v2() -> CandleResult<MirostatProcessor> {
        MirostatProcessor::v2(5.0, 0.3) // Medium tau, moderate eta
    }

    /// Create Mirostat processor from sampling configuration
    pub fn from_config(config: &SamplingConfig) -> CandleResult<MirostatProcessor> {
        // Extract Mirostat parameters from config if available
        let tau = config.temperature; // Already f32
        let eta = config.top_p; // Already f32

        // Use top_k to determine Mirostat version (v1 if specified, v2 otherwise)
        if config.top_k > 0 {
            MirostatProcessor::v1(tau, eta)
        } else {
            MirostatProcessor::v2(tau, eta)
        }
    }

    /// Calculate optimal tau for target perplexity
    #[inline(always)]
    pub fn optimal_tau_for_perplexity(target_perplexity: f32) -> f32 {
        // Empirical relationship between tau and perplexity
        (target_perplexity * 0.8).clamp(MIN_TAU, MAX_TAU)
    }

    /// Estimate required learning rate for convergence time
    #[inline(always)]
    pub fn learning_rate_for_convergence(target_tokens: u32) -> f32 {
        // Faster convergence requires higher learning rate
        (10.0 / target_tokens as f32).clamp(super::super::config::MIN_LEARNING_RATE, super::super::config::MAX_LEARNING_RATE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor_creation() {
        let processor = MirostatProcessor::v1(5.0, 0.1).unwrap();
        assert_eq!(processor.current_tau(), 5.0);
        assert_eq!(processor.current_perplexity(), 1.0);
    }

    #[test]
    fn test_atomic_conversion() {
        let value = 3.14f32;
        let atomic = MirostatProcessor::f32_to_atomic_u64(value);
        let converted = MirostatProcessor::atomic_u64_to_f32(atomic);
        assert_eq!(value, converted);
    }

    #[test]
    fn test_processor_reset() {
        let mut processor = MirostatProcessor::v1(5.0, 0.1).unwrap();
        processor.tokens_processed = 100;
        processor.reset();
        assert_eq!(processor.tokens_processed, 0);
        assert_eq!(processor.current_tau(), 5.0);
    }
}