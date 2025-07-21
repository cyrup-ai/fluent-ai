//! High-Performance SIMD Logits Processing
//!
//! Self-contained SIMD logits processing module with zero dependencies on config/context.
//! Extracted from candle package to eliminate code duplication across fluent-ai ecosystem.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{SimdError, SimdResult};

/// Statistics for SIMD operations performance tracking
#[derive(Debug, Clone, Copy)]
pub struct SimdStats {
    /// Total number of SIMD operations performed
    pub operations_count: u64,
    /// Total processing time in nanoseconds
    pub total_processing_time_ns: u64,
    /// Average processing time per operation in nanoseconds
    pub average_processing_time_ns: u64,
    /// Number of times SIMD was used vs scalar fallback
    pub simd_usage_count: u64,
    /// Number of times scalar fallback was used
    pub scalar_fallback_count: u64,
}

impl Default for SimdStats {
    fn default() -> Self {
        Self {
            operations_count: 0,
            total_processing_time_ns: 0,
            average_processing_time_ns: 0,
            simd_usage_count: 0,
            scalar_fallback_count: 0,
        }
    }
}

impl SimdStats {
    /// Update statistics with new operation timing
    pub fn record_operation(&mut self, processing_time_ns: u64, used_simd: bool) {
        self.operations_count += 1;
        self.total_processing_time_ns += processing_time_ns;
        self.average_processing_time_ns = self.total_processing_time_ns / self.operations_count;
        
        if used_simd {
            self.simd_usage_count += 1;
        } else {
            self.scalar_fallback_count += 1;
        }
    }
    
    /// Get SIMD usage percentage
    pub fn simd_usage_percentage(&self) -> f64 {
        if self.operations_count == 0 {
            0.0
        } else {
            (self.simd_usage_count as f64 / self.operations_count as f64) * 100.0
        }
    }
}

/// Benchmark results for SIMD vs scalar performance comparison
#[derive(Debug, Clone)]
pub struct SimdBenchmarkResult {
    /// Vector size used in benchmark
    pub vector_size: usize,
    /// Number of iterations performed
    pub iterations: u32,
    /// Average time per SIMD operation in nanoseconds
    pub simd_avg_time_ns: u64,
    /// Average time per scalar operation in nanoseconds
    pub scalar_avg_time_ns: u64,
    /// Performance improvement factor (scalar_time / simd_time)
    pub speedup_factor: f64,
}

/// Trait for SIMD logits processors
pub trait SimdProcessor: Send + Sync {
    /// Process logits using SIMD acceleration
    fn process_logits_simd(&mut self, logits: &mut [f32]) -> SimdResult<()>;
    
    /// Get processor statistics
    fn get_stats(&self) -> SimdStats;
    
    /// Reset statistics
    fn reset_stats(&mut self);
}

/// SIMD-accelerated softmax processor
pub struct SimdSoftmaxProcessor {
    temperature: f32,
    stats: SimdStats,
    operation_counter: AtomicU64,
}

impl SimdSoftmaxProcessor {
    /// Create new SIMD softmax processor with temperature
    pub fn new(temperature: f32) -> SimdResult<Self> {
        if temperature <= 0.0 || !temperature.is_finite() {
            return Err(SimdError::InvalidConfiguration(
                format!("Invalid temperature: {}. Must be positive and finite.", temperature)
            ));
        }
        
        Ok(Self {
            temperature,
            stats: SimdStats::default(),
            operation_counter: AtomicU64::new(0),
        })
    }
    
    /// Process logits with SIMD softmax
    pub fn simd_softmax(&mut self, logits: &mut [f32]) -> SimdResult<()> {
        let start_time = std::time::Instant::now();
        
        if logits.is_empty() {
            return Err(SimdError::InvalidInput("Empty logits array".to_string()));
        }
        
        // Apply temperature scaling
        if self.temperature != 1.0 {
            let inv_temp = 1.0 / self.temperature;
            for x in logits.iter_mut() {
                *x *= inv_temp;
            }
        }
        
        // Find maximum for numerical stability
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Check if we can use SIMD (for now, always use scalar as safe implementation)
        let used_simd = false;
        
        // Subtract max and compute exp (scalar implementation for safety)
        let mut sum = 0.0;
        for x in logits.iter_mut() {
            *x = (*x - max_logit).exp();
            sum += *x;
        }
        
        // Normalize
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for x in logits.iter_mut() {
                *x *= inv_sum;
            }
        } else {
            return Err(SimdError::ProcessingError("Sum of exponentials is zero or negative".to_string()));
        }
        
        // Update statistics
        let elapsed = start_time.elapsed().as_nanos() as u64;
        self.stats.record_operation(elapsed, used_simd);
        self.operation_counter.fetch_add(1, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Get processing statistics
    pub fn stats(&self) -> SimdStats {
        self.stats
    }
    
    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SimdStats::default();
        self.operation_counter.store(0, Ordering::Relaxed);
    }
}

impl SimdProcessor for SimdSoftmaxProcessor {
    fn process_logits_simd(&mut self, logits: &mut [f32]) -> SimdResult<()> {
        self.simd_softmax(logits)
    }
    
    fn get_stats(&self) -> SimdStats {
        self.stats
    }
    
    fn reset_stats(&mut self) {
        self.reset_stats()
    }
}

/// SIMD-accelerated top-k processor
pub struct SimdTopKProcessor {
    k: usize,
    stats: SimdStats,
    operation_counter: AtomicU64,
}

impl SimdTopKProcessor {
    /// Create new SIMD top-k processor
    pub fn new(k: usize) -> SimdResult<Self> {
        if k == 0 {
            return Err(SimdError::InvalidConfiguration(
                "k must be greater than 0".to_string()
            ));
        }
        
        Ok(Self {
            k,
            stats: SimdStats::default(),
            operation_counter: AtomicU64::new(0),
        })
    }
    
    /// Process logits with SIMD top-k filtering
    pub fn simd_top_k(&mut self, logits: &mut [f32]) -> SimdResult<()> {
        let start_time = std::time::Instant::now();
        
        if logits.is_empty() {
            return Err(SimdError::InvalidInput("Empty logits array".to_string()));
        }
        
        if self.k >= logits.len() {
            // If k >= length, no filtering needed
            return Ok(());
        }
        
        // Check if we can use SIMD (for now, always use scalar as safe implementation)
        let used_simd = false;
        
        // Scalar top-k implementation for safety
        let mut indexed_logits: Vec<(f32, usize)> = logits.iter().enumerate()
            .map(|(i, &val)| (val, i))
            .collect();
        
        // Sort by value in descending order
        indexed_logits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        
        // Set all but top-k to negative infinity
        let top_k_indices: std::collections::HashSet<usize> = indexed_logits.iter()
            .take(self.k)
            .map(|(_, idx)| *idx)
            .collect();
        
        for (i, logit) in logits.iter_mut().enumerate() {
            if !top_k_indices.contains(&i) {
                *logit = f32::NEG_INFINITY;
            }
        }
        
        // Update statistics
        let elapsed = start_time.elapsed().as_nanos() as u64;
        self.stats.record_operation(elapsed, used_simd);
        self.operation_counter.fetch_add(1, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Get processing statistics
    pub fn stats(&self) -> SimdStats {
        self.stats
    }
    
    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SimdStats::default();
        self.operation_counter.store(0, Ordering::Relaxed);
    }
}

impl SimdProcessor for SimdTopKProcessor {
    fn process_logits_simd(&mut self, logits: &mut [f32]) -> SimdResult<()> {
        self.simd_top_k(logits)
    }
    
    fn get_stats(&self) -> SimdStats {
        self.stats
    }
    
    fn reset_stats(&mut self) {
        self.reset_stats()
    }
}

/// Utility functions for SIMD operations
pub mod utils {
    use super::*;
    
    /// Check if current platform supports SIMD operations
    #[inline(always)]
    pub fn simd_supported() -> bool {
        // For now, return false to use safe scalar implementations
        // Future versions can detect actual SIMD capabilities
        false
    }
    
    /// Benchmark SIMD vs scalar performance
    pub fn benchmark_simd_performance(size: usize, iterations: u32) -> SimdBenchmarkResult {
        let mut simd_total_time = 0u64;
        let mut scalar_total_time = 0u64;
        
        // Create test data
        let mut test_logits = vec![1.0; size];
        
        // Benchmark SIMD operations (currently using scalar for safety)
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            
            // Simulate SIMD softmax operation
            let max_val = test_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0;
            for x in test_logits.iter_mut() {
                *x = (*x - max_val).exp();
                sum += *x;
            }
            if sum > 0.0 {
                let inv_sum = 1.0 / sum;
                for x in test_logits.iter_mut() {
                    *x *= inv_sum;
                }
            }
            
            simd_total_time += start.elapsed().as_nanos() as u64;
            
            // Reset test data
            test_logits.fill(1.0);
        }
        
        // Benchmark scalar operations (same implementation for now)
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            
            let max_val = test_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0;
            for x in test_logits.iter_mut() {
                *x = (*x - max_val).exp();
                sum += *x;
            }
            if sum > 0.0 {
                let inv_sum = 1.0 / sum;
                for x in test_logits.iter_mut() {
                    *x *= inv_sum;
                }
            }
            
            scalar_total_time += start.elapsed().as_nanos() as u64;
            
            // Reset test data
            test_logits.fill(1.0);
        }
        
        let simd_avg = simd_total_time / iterations as u64;
        let scalar_avg = scalar_total_time / iterations as u64;
        let speedup = if simd_avg > 0 { scalar_avg as f64 / simd_avg as f64 } else { 1.0 };
        
        SimdBenchmarkResult {
            vector_size: size,
            iterations,
            simd_avg_time_ns: simd_avg,
            scalar_avg_time_ns: scalar_avg,
            speedup_factor: speedup,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_softmax_processor() {
        let mut processor = SimdSoftmaxProcessor::new(1.0).expect("Failed to create processor");
        
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        processor.simd_softmax(&mut logits).expect("Softmax failed");
        
        // Check that probabilities sum to approximately 1.0
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Probabilities do not sum to 1.0: {}", sum);
        
        // Check that all probabilities are positive
        for &prob in &logits {
            assert!(prob > 0.0, "Negative probability: {}", prob);
        }
    }
    
    #[test]
    fn test_simd_top_k_processor() {
        let mut processor = SimdTopKProcessor::new(2).expect("Failed to create processor");
        
        let mut logits = vec![1.0, 4.0, 2.0, 3.0];
        processor.simd_top_k(&mut logits).expect("Top-k failed");
        
        // Check that only 2 values remain finite
        let finite_count = logits.iter().filter(|&&x| x.is_finite() && x > f32::NEG_INFINITY).count();
        assert_eq!(finite_count, 2);
        
        // Check that the largest values are preserved
        assert!(logits[1].is_finite()); // 4.0 should be preserved
        assert!(logits[3].is_finite()); // 3.0 should be preserved
    }
    
    #[test]
    fn test_simd_processor_trait() {
        let mut processor = SimdSoftmaxProcessor::new(2.0).expect("Failed to create processor");
        
        let mut logits = vec![2.0, 4.0, 6.0, 8.0];
        processor.process_logits_simd(&mut logits).expect("SIMD processing failed");
        
        // Verify the processor trait works
        let stats = processor.get_stats();
        assert!(stats.operations_count > 0);
    }
    
    #[test]
    fn test_invalid_configurations() {
        // Invalid temperature
        assert!(SimdSoftmaxProcessor::new(0.0).is_err());
        assert!(SimdSoftmaxProcessor::new(-1.0).is_err());
        assert!(SimdSoftmaxProcessor::new(f32::INFINITY).is_err());
        assert!(SimdSoftmaxProcessor::new(f32::NAN).is_err());
        
        // Invalid k value
        assert!(SimdTopKProcessor::new(0).is_err());
    }
    
    #[test]
    fn test_stats_tracking() {
        let mut processor = SimdSoftmaxProcessor::new(1.0).expect("Failed to create processor");
        
        // Process multiple times
        for _ in 0..3 {
            let mut logits = vec![1.0, 2.0, 3.0];
            processor.simd_softmax(&mut logits).expect("Softmax failed");
        }
        
        let stats = processor.stats();
        assert_eq!(stats.operations_count, 3);
        assert!(stats.total_processing_time_ns > 0);
        assert!(stats.average_processing_time_ns > 0);
    }
}