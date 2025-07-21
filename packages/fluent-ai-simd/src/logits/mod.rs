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
    fn process(
        &mut self,
        logits: &mut [f32],
        context: &ProcessingContext,
    ) -> LogitsResult<()>;
    
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
            // Sort logits in descending order
            let mut sorted_indices: Vec<usize> = (0..logits.len()).collect();
            sorted_indices.sort_unstable_by(|&i, &j| {
                logits[j].partial_cmp(&logits[i])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            
            // Calculate cumulative probability
            let mut cumulative = 0.0;
            let mut cutoff = logits.len();
            
            // First, get the logits values for the sorted indices
            let sorted_values: Vec<f32> = sorted_indices.iter().map(|&i| logits[i]).collect();
            
            // Find the cutoff point where cumulative probability exceeds p
            for (i, &value) in sorted_values.iter().enumerate() {
                cumulative += value.exp();
                if cumulative > p {
                    cutoff = i + 1;
                    break;
                }
            }
            
            // Set all elements after cutoff to negative infinity
            for &idx in &sorted_indices[cutoff..] {
                logits[idx] = f32::NEG_INFINITY;
            }
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
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
        let mut non_inf = logits.iter().filter(|&&x| x > f32::NEG_INFINITY).count();
        assert_eq!(non_inf, 2);
    }
    
    #[test]
    fn test_nucleus_sampling() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let context = ProcessingContext::new().with_top_p(Some(0.6));
        let config = ProcessorConfig::default();
        
        process_logits_scalar(&mut logits, &context, &config).unwrap();
        
        // At least one value should be set to negative infinity
        let has_inf = logits.iter().any(|&x| x == f32::NEG_INFINITY);
        assert!(has_inf);
    }
}
