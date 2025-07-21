//! Logits processing module for SIMD-accelerated operations

mod nucleus;
mod penalties;
mod processor;
mod topk;

pub use nucleus::*;
pub use penalties::*;
pub use processor::*;
pub use topk::*;

use crate::config::ProcessorConfig;
use crate::context::ProcessingContext;

/// Error type for logits processing operations
#[derive(Debug, thiserror::Error)]
pub enum LogitsError {
    #[error("Invalid input length: {0}")]
    InvalidInputLength(usize),
    
    #[error("Numerical error: {0}")]
    NumericalError(String),
    
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    
    #[error("Sampling error: {0}")]
    SamplingError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(#[from] crate::config::ConfigError),
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

/// Default implementation of LogitsProcessor
pub struct DefaultLogitsProcessor {
    config: ProcessorConfig,
}

impl Default for DefaultLogitsProcessor {
    fn default() -> Self {
        Self {
            config: ProcessorConfig::default(),
        }
    }
}

impl LogitsProcessor for DefaultLogitsProcessor {
    fn process(
        &mut self,
        logits: &mut [f32],
        context: &ProcessingContext,
    ) -> LogitsResult<()> {
        // Use SIMD implementation if available, otherwise fall back to scalar
        if cfg!(target_feature = "avx2") || cfg!(target_feature = "avx") || cfg!(target_feature = "sse4.1") {
            // Use SIMD-accelerated implementation
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                return process_logits_simd(logits, context, &self.config);
            }
        }
        
        // Fall back to scalar implementation
        process_logits_scalar(logits, context, &self.config)
    }
    
    fn config(&self) -> &ProcessorConfig {
        &self.config
    }
    
    fn config_mut(&mut self) -> &mut ProcessorConfig {
        &mut self.config
    }
}

/// Process logits using SIMD instructions
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn process_logits_simd(
    logits: &mut [f32],
    context: &ProcessingContext,
    config: &ProcessorConfig,
) -> LogitsResult<()> {
    // Import the required SIMD intrinsics
    use std::arch::x86_64::*;
    
    // Ensure the slice is properly aligned for SIMD operations
    let len = logits.len();
    if len == 0 {
        return Ok(());
    }
    
    // Process in chunks of 8 floats (256 bits)
    let chunks = len / 8;
    let remainder = len % 8;
    
    // Process full chunks
    for i in 0..chunks {
        let offset = i * 8;
        let ptr = logits.as_mut_ptr().add(offset) as *mut __m256;
        
        // Load 8 floats into a SIMD register
        let mut values = unsafe { _mm256_loadu_ps(logits.as_ptr().add(offset)) };
        
        // Apply temperature scaling
        let temp = _mm256_set1_ps(context.temperature);
        values = unsafe { _mm256_div_ps(values, temp) };
        
        // Store the result back
        unsafe { _mm256_storeu_ps(ptr, values) };
    }
    
    // Process remaining elements with scalar operations
    for i in (chunks * 8)..len {
        logits[i] /= context.temperature;
    }
    
    // Apply top-k filtering if needed
    if let Some(k) = context.top_k {
        topk_filtering_simd(logits, k)?;
    }
    
    // Apply nucleus sampling if needed
    if let Some(top_p) = context.top_p {
        prepare_nucleus_sampling_simd(logits, top_p as f64)?;
    }
    
    // Apply penalties if needed
    if config.repetition_penalty != 1.0 || config.frequency_penalty != 0.0 || config.presence_penalty != 0.0 {
        apply_penalties_simd(logits, context, config)?;
    }
    
    // Normalize probabilities
    normalize_probabilities_simd(logits)?;
    
    Ok(())
}

/// Process logits using scalar operations (fallback)
fn process_logits_scalar(
    logits: &mut [f32],
    context: &ProcessingContext,
    config: &ProcessorConfig,
) -> LogitsResult<()> {
    // Temperature scaling
    if context.temperature != 1.0 {
        let inv_temp = 1.0 / context.temperature;
        for x in logits.iter_mut() {
            *x *= inv_temp;
        }
    }
    
    // Top-k filtering
    if let Some(k) = context.top_k {
        topk_filtering_scalar(logits, k)?;
    }
    
    // Nucleus sampling
    if let Some(top_p) = context.top_p {
        prepare_nucleus_sampling_scalar(logits, top_p as f64)?;
    }
    
    // Apply penalties
    if config.repetition_penalty != 1.0 || config.frequency_penalty != 0.0 || config.presence_penalty != 0.0 {
        apply_penalties_scalar(logits, context, config)?;
    }
    
    // Normalize probabilities
    normalize_probabilities_scalar(logits)?;
    
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
