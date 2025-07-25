//! Logits processing module for SIMD-accelerated operations

mod nucleus;
mod penalties;
pub mod processing;
pub mod processor;
pub mod simd;
pub mod topk;

pub use nucleus::*;
pub use penalties::*;
pub use processing::*;
// processor module exports DefaultLogitsProcessor
// simd module provides compatibility aliases
pub use topk::topk_filtering_simd;

use crate::config::ProcessorConfig;
use crate::context::ProcessingContext;

/// Error type for logits processing operations
#[derive(Debug, thiserror::Error)]
pub enum LogitsError {
    /// Invalid input length for logits processing
    #[error("Invalid input length: {0}")]
    InvalidInputLength(usize),

    /// Numerical computation error during processing
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Unsupported operation requested
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// Error during sampling process
    #[error("Sampling error: {0}")]
    SamplingError(String),

    /// Configuration validation error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// SIMD processing error
    #[error("SIMD error: {0}")]
    SimdError(#[from] crate::error::SimdError),
}

/// Result type for logits processing operations
pub type LogitsResult<T> = Result<T, LogitsError>;

/// Trait for logits processing operations
pub trait LogitsProcessor: Send + Sync {
    /// Process logits in-place
    fn process(&mut self, logits: &mut [f32], context: &ProcessingContext) -> LogitsResult<()>;

    /// Get the current configuration
    fn config(&self) -> &ProcessorConfig;

    /// Get a mutable reference to the configuration
    fn config_mut(&mut self) -> &mut ProcessorConfig;
}

// DefaultLogitsProcessor is implemented in the processor module

/// Process logits using scalar operations (fallback when SIMD is not available)
pub fn process_logits_scalar(
    logits: &mut [f32],
    context: &ProcessingContext,
    config: &ProcessorConfig,
) -> LogitsResult<()> {
    // Apply temperature scaling
    let temp = context.temperature;
    if temp != 1.0 && temp > 0.0 {
        let inv_temp = 1.0 / temp;
        for x in logits.iter_mut() {
            *x *= inv_temp;
        }
    }

    // Apply top-k filtering if enabled
    if let Some(k) = context.top_k.filter(|_| config.top_k.is_some()) {
        if k < logits.len() {
            // Find the k-th largest element
            let mut sorted: Vec<f32> = logits.iter().copied().collect();
            sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

            // Set all elements below the k-th to negative infinity
            let threshold = sorted[k];
            for x in logits.iter_mut() {
                if *x < threshold {
                    *x = f32::NEG_INFINITY;
                }
            }
        }
    }

    // Apply nucleus sampling if enabled
    if let Some(p) = context.top_p.filter(|_| config.top_p.is_some()) {
        if p > 0.0 && p < 1.0 {
            // Find max logit for numerical stability (zero allocation)
            let max_logit = logits.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));

            // Create SmallVec of (index, logit) pairs
            let mut sorted: SmallVec<[(usize, f32); 512]> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();

            // Sort in descending order by logit value
            sorted.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Compute total sum of exp(shifted) without allocation
            let mut total_sum = 0.0f64;
            for &logit in logits.iter() {
                total_sum += ((logit - max_logit) as f64).exp();
            }

            // Now find cutoff using cumulative normalized prob
            let mut cumsum = 0.0f64;
            let mut cutoff = sorted.len();

            for (i, &(_, logit)) in sorted.iter().enumerate() {
                let prob = ((logit - max_logit) as f64).exp() / total_sum;
                cumsum += prob;
                if cumsum >= p as f64 {
                    cutoff = i + 1;
                    break;
                }
            }

            // Collect indices to keep (using SmallVec to avoid alloc if small)
            let mut keep_indices: SmallVec<[usize; 512]> =
                sorted[..cutoff].iter().map(|&(idx, _)| idx).collect();

            // Mask logits not in keep (in-place, zero alloc)
            for (i, logit) in logits.iter_mut().enumerate() {
                if !keep_indices.contains(&i) {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_temperature_scaling() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let context = ProcessingContext::new().with_temperature(0.5);
        let config = ProcessorConfig::default();

        process_logits_scalar(&mut logits, &context, &config).unwrap();
        assert_relative_eq!(logits[0], 2.0);
        assert_relative_eq!(logits[1], 4.0);
        assert_relative_eq!(logits[2], 6.0);
    }

    #[test]
    fn test_topk_filtering() {
        let mut logits = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let context = ProcessingContext::new().with_top_k(Some(2));
        let config = ProcessorConfig::default();

        process_logits_scalar(&mut logits, &context, &config).unwrap();

        // Only top 2 values should remain non-negative infinity
        let non_inf = logits.iter().filter(|&&x| x > f32::NEG_INFINITY).count();
        assert_eq!(non_inf, 2);
        // Check specific values
        let mut sorted = logits.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!(logits.contains(&sorted[0]));
        assert!(logits.contains(&sorted[1]));
    }

    #[test]
    fn test_nucleus_sampling() {
        let mut logits = vec![0.1, 0.2, 0.3, 0.4];
        let context = ProcessingContext::new().with_top_p(Some(0.5));
        let config = ProcessorConfig::default();

        process_logits_scalar(&mut logits, &context, &config).unwrap();

        // Verify correct masking (indices 0 and 1 masked for top_p=0.5)
        let has_inf = logits.iter().filter(|&&x| x == f32::NEG_INFINITY).count() == 2;
        assert!(has_inf);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert!(logits[2] > f32::NEG_INFINITY);
        assert!(logits[3] > f32::NEG_INFINITY);
    }
}
