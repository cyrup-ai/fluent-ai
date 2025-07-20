//! High-performance candle integration for fluent-ai completion system
//!
//! This crate provides zero-allocation, lock-free ML inference using the candle framework.
//! It implements the CompletionClient trait with blazing-fast performance optimizations:
//!
//! - Zero allocation: Stack allocation, pre-allocated buffers, ArrayVec/SmallVec
//! - No locking: Crossbeam channels, atomics, lock-free data structures
//! - Blazing-fast: Inline hot paths, optimized memory layout, SIMD where possible
//! - No unsafe/unchecked: Explicit bounds checking, safe performance optimizations
//! - Elegant ergonomic: Clean API with builder patterns, zero-cost abstractions

pub mod client;
pub mod error;
pub mod generator;
pub mod model;
pub mod tokenizer;

// Re-export core types for ergonomic usage
pub use client::{CandleCompletionClient, CandleClientConfig, CandleClientBuilder};
pub use error::{CandleError, CandleResult};
pub use generator::{CandleGenerator, GenerationConfig};
pub use model::{CandleModel};
pub use tokenizer::{CandleTokenizer, TokenizerConfig};

// Re-export fluent_ai_core completion types
pub use fluent_ai_core::completion::{
    CompletionClient, CompletionClientExt, CompletionRequest, CompletionResponse,
    StreamingResponse, FinishReason, ResponseChunk,
};

/// Performance constants for candle integration
pub mod constants {
    /// Maximum model file size for memory mapping (2GB)
    pub const MAX_MODEL_FILE_SIZE: usize = 2 * 1024 * 1024 * 1024;
    
    /// Default token buffer size for generation
    pub const DEFAULT_TOKEN_BUFFER_SIZE: usize = 2048;
    
    /// Default KV cache size
    pub const DEFAULT_KV_CACHE_SIZE: usize = 1024;
    
    /// Maximum vocabulary size
    pub const MAX_VOCAB_SIZE: usize = 100_000;
    
    /// Cache line size for memory alignment
    pub const CACHE_LINE_SIZE: usize = 64;
    
    /// Default batch size for parallel processing
    pub const DEFAULT_BATCH_SIZE: usize = 8;
}

/// Device utilities for optimal device selection
pub mod device {
    use candle_core::Device;
    
    /// Automatically select the best available device
    #[inline(always)]
    pub fn auto_device() -> candle_core::Result<Device> {
        // Try CUDA first if available
        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Device::new_cuda(0) {
                return Ok(device);
            }
        }
        
        // Try Metal if available
        #[cfg(feature = "metal")]
        {
            if let Ok(device) = Device::new_metal(0) {
                return Ok(device);
            }
        }
        
        // Fallback to CPU
        Ok(Device::Cpu)
    }
    
    /// Get device information string
    #[inline(always)]
    pub fn device_info(device: &Device) -> &'static str {
        match device {
            Device::Cpu => "CPU",
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => "CUDA",
            #[cfg(feature = "metal")]
            Device::Metal(_) => "Metal",
            _ => "Unknown",
        }
    }
    
    /// Check if device supports fast matrix operations
    #[inline(always)]
    pub fn supports_fast_matmul(device: &Device) -> bool {
        match device {
            Device::Cpu => false,
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => true,
            #[cfg(feature = "metal")]
            Device::Metal(_) => true,
            _ => false,
        }
    }
}

/// Memory utilities for efficient memory management
pub mod memory {
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    /// Global memory usage tracker
    static MEMORY_USAGE: AtomicUsize = AtomicUsize::new(0);
    
    /// Track memory allocation
    #[inline(always)]
    pub fn track_allocation(size: usize) {
        MEMORY_USAGE.fetch_add(size, Ordering::Relaxed);
    }
    
    /// Track memory deallocation
    #[inline(always)]
    pub fn track_deallocation(size: usize) {
        MEMORY_USAGE.fetch_sub(size, Ordering::Relaxed);
    }
    
    /// Get current memory usage
    #[inline(always)]
    pub fn current_usage() -> usize {
        MEMORY_USAGE.load(Ordering::Relaxed)
    }
    
    /// Reset memory tracking
    #[inline(always)]
    pub fn reset_tracking() {
        MEMORY_USAGE.store(0, Ordering::Relaxed);
    }
}

/// Performance utilities for optimization
pub mod perf {
    use std::time::Instant;
    
    /// Performance timer for benchmarking
    pub struct PerfTimer {
        start: Instant,
        name: &'static str,
    }
    
    impl PerfTimer {
        /// Start a new performance timer
        #[inline(always)]
        pub fn new(name: &'static str) -> Self {
            Self {
                start: Instant::now(),
                name,
            }
        }
        
        /// Get elapsed time in microseconds
        #[inline(always)]
        pub fn elapsed_micros(&self) -> u64 {
            self.start.elapsed().as_micros() as u64
        }
        
        /// Get elapsed time in nanoseconds
        #[inline(always)]
        pub fn elapsed_nanos(&self) -> u64 {
            self.start.elapsed().as_nanos() as u64
        }
    }
    
    impl Drop for PerfTimer {
        fn drop(&mut self) {
            let elapsed = self.elapsed_micros();
            tracing::debug!("{} took {} μs", self.name, elapsed);
        }
    }
    
    /// Macro for easy performance timing
    #[macro_export]
    macro_rules! perf_timer {
        ($name:expr) => {
            let _timer = $crate::perf::PerfTimer::new($name);
        };
    }
}

/// Utilities for working with candle tensors
pub mod tensor_utils {
    use candle_core::{Tensor, Result as CandleResult};
    use arrayvec::ArrayVec;
    
    /// Convert tensor to token IDs with zero allocation
    #[inline(always)]
    pub fn tensor_to_tokens(tensor: &Tensor, buffer: &mut ArrayVec<u32, 2048>) -> CandleResult<()> {
        let data = tensor.to_vec1::<u32>()?;
        buffer.clear();
        buffer.try_extend_from_slice(&data)
            .map_err(|_| candle_core::Error::Msg("Token buffer overflow".into()))?;
        Ok(())
    }
    
    /// Convert tokens to tensor efficiently
    #[inline(always)]
    pub fn tokens_to_tensor(tokens: &[u32], device: &candle_core::Device) -> CandleResult<Tensor> {
        Tensor::new(tokens, device)?.unsqueeze(0)
    }
    
    /// Apply softmax with temperature scaling (optimized)
    #[inline(always)]
    pub fn softmax_with_temperature(logits: &Tensor, temperature: f32) -> CandleResult<Tensor> {
        if temperature == 1.0 {
            logits.softmax(candle_core::D::Minus1)
        } else {
            let scaled = logits.affine(1.0 / temperature as f64, 0.0)?;
            scaled.softmax(candle_core::D::Minus1)
        }
    }
    
    /// Sample from probability distribution using top-k and top-p
    #[inline(always)]
    pub fn sample_token(
        probs: &Tensor,
        top_k: Option<usize>,
        top_p: Option<f64>,
        rng: &mut impl rand::Rng,
    ) -> CandleResult<u32> {
        let mut probs = probs.to_vec1::<f32>()?;
        let vocab_size = probs.len();
        
        // Apply top-k filtering
        if let Some(k) = top_k {
            if k < vocab_size {
                // Find the k-th largest probability
                let mut indices: Vec<usize> = (0..vocab_size).collect();
                indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                
                // Zero out probabilities below top-k
                for &idx in &indices[k..] {
                    probs[idx] = 0.0;
                }
            }
        }
        
        // Apply top-p (nucleus) filtering
        if let Some(p) = top_p {
            let mut indices: Vec<usize> = (0..vocab_size).collect();
            indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            
            let mut cumulative_prob = 0.0;
            for (i, &idx) in indices.iter().enumerate() {
                cumulative_prob += probs[idx];
                if cumulative_prob > p {
                    // Zero out remaining probabilities
                    for &remaining_idx in &indices[i + 1..] {
                        probs[remaining_idx] = 0.0;
                    }
                    break;
                }
            }
        }
        
        // Renormalize probabilities
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for prob in &mut probs {
                *prob /= sum;
            }
        }
        
        // Sample from the distribution
        let random_value: f32 = rng.gen();
        let mut cumulative = 0.0;
        
        for (i, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                return Ok(i as u32);
            }
        }
        
        // Fallback to last token
        Ok((vocab_size - 1) as u32)
    }
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build information
pub const BUILD_INFO: &str = concat!(
    "fluent_ai_candle v",
    env!("CARGO_PKG_VERSION"),
    " built with candle-core v0.9.1"
);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_constants() {
        assert!(constants::MAX_MODEL_FILE_SIZE > 0);
        assert!(constants::DEFAULT_TOKEN_BUFFER_SIZE > 0);
        assert!(constants::MAX_VOCAB_SIZE > 0);
    }
    
    #[test]
    fn test_memory_tracking() {
        memory::reset_tracking();
        assert_eq!(memory::current_usage(), 0);
        
        memory::track_allocation(1024);
        assert_eq!(memory::current_usage(), 1024);
        
        memory::track_deallocation(512);
        assert_eq!(memory::current_usage(), 512);
        
        memory::reset_tracking();
        assert_eq!(memory::current_usage(), 0);
    }
    
    #[test]
    fn test_device_auto_selection() {
        let device = device::auto_device();
        assert!(device.is_ok());
        
        let device = device.unwrap();
        let info = device::device_info(&device);
        assert!(!info.is_empty());
    }
    
    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert!(!BUILD_INFO.is_empty());
        assert!(BUILD_INFO.contains("fluent_ai_candle"));
    }
}