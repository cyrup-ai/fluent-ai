//! SIMD Acceleration Bridge for Sampling Operations
//!
//! This module provides compatibility between the candle package and the shared
//! fluent-ai-simd crate. All actual SIMD implementations have been moved to 
//! the shared crate to eliminate duplication across packages.

// Re-export everything from shared SIMD crate for backward compatibility  
pub use fluent_ai_simd::logits_simd::*;

use candle_core::{Result as CandleResult, Tensor};
use crate::error::CandleError;
use super::LogitsProcessor;

/// Convert SimdError to CandleError for compatibility
impl From<fluent_ai_simd::SimdError> for CandleError {
    fn from(err: fluent_ai_simd::SimdError) -> Self {
        match err {
            fluent_ai_simd::SimdError::InvalidConfiguration(msg) => {
                CandleError::InvalidConfiguration(&msg)
            }
            fluent_ai_simd::SimdError::InvalidInput(msg) => {
                CandleError::InvalidInput(&msg)
            }
            fluent_ai_simd::SimdError::ProcessingError(msg) => {
                CandleError::ProcessingError(&msg)
            }
            fluent_ai_simd::SimdError::NotSupported(msg) => {
                CandleError::Other(&msg)
            }
        }
    }
}

/// Bridge processor that implements LogitsProcessor for SIMD softmax
pub struct CandleSimdSoftmaxProcessor {
    inner: fluent_ai_simd::logits_simd::SimdSoftmaxProcessor,
}

impl CandleSimdSoftmaxProcessor {
    /// Create new SIMD softmax processor
    pub fn new(temperature: f32) -> CandleResult<Self> {
        let inner = fluent_ai_simd::logits_simd::SimdSoftmaxProcessor::new(temperature)
            .map_err(CandleError::from)?;
        
        Ok(Self { inner })
    }
    
    /// Get processing statistics
    pub fn stats(&self) -> fluent_ai_simd::logits_simd::SimdStats {
        self.inner.stats()
    }
    
    /// Process logits with SIMD acceleration
    pub fn process_logits(&mut self, logits: &mut [f32]) -> CandleResult<()> {
        self.inner.simd_softmax(logits).map_err(CandleError::from)
    }
}

impl LogitsProcessor for CandleSimdSoftmaxProcessor {
    fn sample(&mut self, logits: &mut [f32]) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
        // Process logits with SIMD softmax
        self.process_logits(logits)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
        
        // Sample from processed probabilities using simple multinomial
        let mut cumsum = 0.0_f64;
        let rand_val: f32 = fastrand::f32();
        
        for (i, &prob) in logits.iter().enumerate() {
            cumsum += prob as f64;
            if rand_val <= cumsum as f32 {
                return Ok(i as u32);
            }
        }
        
        // Fallback to last token if no match (numerical precision issue)
        Ok((logits.len() - 1) as u32)
    }
}

/// Bridge processor that implements LogitsProcessor for SIMD top-k
pub struct CandleSimdTopKProcessor {
    inner: fluent_ai_simd::logits_simd::SimdTopKProcessor,
}

impl CandleSimdTopKProcessor {
    /// Create new SIMD top-k processor
    pub fn new(k: usize) -> CandleResult<Self> {
        let inner = fluent_ai_simd::logits_simd::SimdTopKProcessor::new(k)
            .map_err(CandleError::from)?;
        
        Ok(Self { inner })
    }
    
    /// Process logits with SIMD top-k selection
    pub fn process_logits(&mut self, logits: &mut [f32]) -> CandleResult<()> {
        use fluent_ai_simd::SimdProcessor;
        self.inner.process_logits_simd(logits).map_err(CandleError::from)
    }
}

impl LogitsProcessor for CandleSimdTopKProcessor {
    fn sample(&mut self, logits: &mut [f32]) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
        // Process logits with SIMD top-k filtering
        self.process_logits(logits)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
        
        // Sample from filtered probabilities
        let mut cumsum = 0.0_f64;
        let rand_val: f32 = fastrand::f32();
        
        for (i, &prob) in logits.iter().enumerate() {
            if prob > f32::NEG_INFINITY {
                cumsum += prob as f64;
                if rand_val <= cumsum as f32 {
                    return Ok(i as u32);
                }
            }
        }
        
        // Fallback to first non-filtered token
        for (i, &prob) in logits.iter().enumerate() {
            if prob > f32::NEG_INFINITY {
                return Ok(i as u32);
            }
        }
        
        Ok(0)
    }
}

/// Utility functions for SIMD operations (compatibility layer)
pub mod utils {
    use super::*;
    
    /// Check if current platform supports SIMD operations
    #[inline(always)]
    pub fn simd_supported() -> bool {
        fluent_ai_simd::logits_simd::utils::simd_supported()
    }
    
    /// Create optimized SIMD softmax processor for inference
    #[inline(always)]
    pub fn create_simd_softmax(temperature: f32) -> CandleResult<CandleSimdSoftmaxProcessor> {
        CandleSimdSoftmaxProcessor::new(temperature)
    }
    
    /// Create optimized SIMD top-k processor
    #[inline(always)]
    pub fn create_simd_top_k(k: usize) -> CandleResult<CandleSimdTopKProcessor> {
        CandleSimdTopKProcessor::new(k)
    }
    
    /// Benchmark SIMD vs scalar performance
    pub fn benchmark_simd_performance(size: usize, iterations: u32) -> fluent_ai_simd::logits_simd::SimdBenchmarkResult {
        fluent_ai_simd::logits_simd::utils::benchmark_simd_performance(size, iterations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_candle_simd_softmax_processor() {
        let mut processor = CandleSimdSoftmaxProcessor::new(1.0).expect("Failed to create processor");
        
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        processor.process_logits(&mut logits).expect("SIMD softmax failed");
        
        // Check that probabilities sum to 1.0
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Probabilities do not sum to 1.0: {}", sum);
        
        // Check that probabilities are in descending order (since input was ascending)
        for i in 1..logits.len() {
            assert!(logits[i-1] < logits[i], "Probabilities not in expected order");
        }
    }
    
    #[test]
    fn test_candle_simd_top_k_processor() {
        let mut processor = CandleSimdTopKProcessor::new(3).expect("Failed to create processor");
        
        let mut logits = vec![5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0, 4.0];
        processor.process_logits(&mut logits).expect("Top-k failed");
        
        // Check that only 3 values remain non-negative-infinity
        let non_inf_count = logits.iter().filter(|&&x| x > f32::NEG_INFINITY).count();
        assert_eq!(non_inf_count, 3);
    }
    
    #[test]
    fn test_error_conversion() {
        let simd_err = fluent_ai_simd::SimdError::InvalidConfiguration("test".to_string());
        let _candle_err: CandleError = simd_err.into();
        // Just ensure conversion compiles and runs
    }
}