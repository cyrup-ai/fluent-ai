//! SIMD Acceleration Bridge for Sampling Operations
//!
//! This module provides zero-allocation compatibility between the candle package and the shared
//! fluent-ai-simd crate. All actual SIMD implementations have been moved to the shared crate
//! to eliminate duplication across packages and achieve blazing-fast performance.

use candle_core::{Result as CandleResult, Tensor};
// Import real SIMD types from shared crate
use fluent_ai_simd::{
    config::ProcessorConfig,
    context::ProcessingContext,
    error::SimdError,
    logits::{
        processing::{apply_temperature_scaling_simd, DefaultLogitsProcessor},
        topk::topk_filtering_simd,
        LogitsProcessor, LogitsResult,
    },
    ops::{compute_softmax_inplace, SoftmaxProcessor, TemperatureProcessor},
};

use crate::error::CandleError;

/// Convert SimdError to CandleError for compatibility
impl From<SimdError> for CandleError {
    #[inline(always)]
    fn from(err: SimdError) -> Self {
        match err {
            SimdError::InvalidConfiguration(msg) => {
                CandleError::InvalidConfiguration("Invalid SIMD configuration")
            }
            SimdError::InvalidInput(msg) => CandleError::InvalidInput("Invalid SIMD input"),
            SimdError::ProcessingError(msg) => {
                CandleError::ProcessingError("SIMD processing failed")
            }
            SimdError::NotSupported(msg) => CandleError::Other("SIMD operation not supported"),
        }
    }
}

/// Bridge processor that implements LogitsProcessor using shared SIMD operations
#[repr(C, align(64))]
pub struct CandleSimdProcessor {
    inner: DefaultLogitsProcessor,
    config: ProcessorConfig,
}

impl CandleSimdProcessor {
    /// Create new SIMD-accelerated processor with zero allocation
    #[inline(always)]
    pub fn new() -> CandleResult<Self> {
        Ok(Self {
            inner: DefaultLogitsProcessor::new(),
            config: ProcessorConfig::default(),
        })
    }

    /// Create processor with custom configuration
    #[inline(always)]
    pub fn with_config(config: ProcessorConfig) -> CandleResult<Self> {
        Ok(Self {
            inner: DefaultLogitsProcessor::with_config(config.clone()),
            config,
        })
    }

    /// Process logits with SIMD acceleration (zero allocation)
    #[inline(always)]
    pub fn process_logits(
        &mut self,
        logits: &mut [f32],
        context: &ProcessingContext,
    ) -> CandleResult<()> {
        self.inner
            .process(logits, context)
            .map_err(|e| CandleError::ProcessingError("SIMD logits processing failed"))
    }

    /// Apply temperature scaling with SIMD optimization
    #[inline(always)]
    pub fn apply_temperature(&self, logits: &mut [f32], temperature: f32) -> CandleResult<()> {
        apply_temperature_scaling_simd(logits, temperature).map_err(CandleError::from)
    }

    /// Apply top-k filtering with SIMD optimization
    #[inline(always)]
    pub fn apply_topk(&self, logits: &mut [f32], k: usize) -> CandleResult<()> {
        topk_filtering_simd(logits, k).map_err(CandleError::from)
    }

    /// Get current configuration
    #[inline(always)]
    pub const fn config(&self) -> &ProcessorConfig {
        &self.config
    }
}

impl Default for CandleSimdProcessor {
    #[inline(always)]
    fn default() -> Self {
        Self::new().expect("Failed to create default SIMD processor")
    }
}

/// Zero-allocation softmax processor using shared SIMD operations
#[repr(C, align(64))]
pub struct CandleSoftmaxProcessor {
    inner: SoftmaxProcessor,
}

impl CandleSoftmaxProcessor {
    /// Create new SIMD softmax processor with zero allocation
    #[inline(always)]
    pub fn new(temperature: f32) -> CandleResult<Self> {
        let inner = SoftmaxProcessor::new(temperature).map_err(CandleError::from)?;

        Ok(Self { inner })
    }

    /// Compute softmax in-place with SIMD acceleration
    #[inline(always)]
    pub fn softmax_inplace(&mut self, logits: &mut [f32]) -> CandleResult<()> {
        compute_softmax_inplace(logits).map_err(CandleError::from)
    }

    /// Get processing statistics (zero allocation)
    #[inline(always)]
    pub fn stats(&self) -> fluent_ai_simd::ops::SoftmaxStats {
        self.inner.stats()
    }
}

/// Zero-allocation temperature processor using shared SIMD operations
#[repr(C, align(64))]
pub struct CandleTemperatureProcessor {
    inner: TemperatureProcessor,
}

impl CandleTemperatureProcessor {
    /// Create new SIMD temperature processor with zero allocation
    #[inline(always)]
    pub fn new(temperature: f32) -> CandleResult<Self> {
        let inner = TemperatureProcessor::new(temperature).map_err(CandleError::from)?;

        Ok(Self { inner })
    }

    /// Apply temperature scaling with SIMD optimization
    #[inline(always)]
    pub fn apply_temperature(&mut self, logits: &mut [f32]) -> CandleResult<()> {
        self.inner
            .apply_temperature_inplace(logits)
            .map_err(CandleError::from)
    }

    /// Get processing statistics (zero allocation)
    #[inline(always)]
    pub fn stats(&self) -> fluent_ai_simd::ops::TemperatureStats {
        self.inner.stats()
    }
}

/// Utility functions for SIMD operations (compatibility layer)
pub mod utils {
    use super::*;

    /// Check if current platform supports SIMD operations
    #[inline(always)]
    pub fn simd_supported() -> bool {
        fluent_ai_simd::simd_available()
    }

    /// Create optimized SIMD processor for inference (zero allocation)
    #[inline(always)]
    pub fn create_simd_processor() -> CandleResult<CandleSimdProcessor> {
        CandleSimdProcessor::new()
    }

    /// Create optimized SIMD softmax processor (zero allocation)
    #[inline(always)]
    pub fn create_simd_softmax(temperature: f32) -> CandleResult<CandleSoftmaxProcessor> {
        CandleSoftmaxProcessor::new(temperature)
    }

    /// Create optimized SIMD temperature processor (zero allocation)
    #[inline(always)]
    pub fn create_simd_temperature(temperature: f32) -> CandleResult<CandleTemperatureProcessor> {
        CandleTemperatureProcessor::new(temperature)
    }

    /// Benchmark SIMD vs scalar performance (zero allocation)
    pub fn benchmark_simd_performance(
        size: usize,
        iterations: u32,
    ) -> fluent_ai_simd::benchmark::BenchmarkResult {
        fluent_ai_simd::benchmark::run_benchmark(size, iterations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_simd_processor() {
        let mut processor = CandleSimdProcessor::new().expect("Failed to create processor");
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let context = ProcessingContext::new().with_temperature(1.0);

        processor
            .process_logits(&mut logits, &context)
            .expect("SIMD processing failed");

        // Verify logits were processed (values should be different from input)
        assert_ne!(logits, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_candle_softmax_processor() {
        let mut processor = CandleSoftmaxProcessor::new(1.0).expect("Failed to create processor");

        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        processor
            .softmax_inplace(&mut logits)
            .expect("SIMD softmax failed");

        // Check that probabilities sum to approximately 1.0
        let sum: f32 = logits.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Probabilities do not sum to 1.0: {}",
            sum
        );

        // Check that all probabilities are positive
        for &prob in &logits {
            assert!(prob > 0.0, "Negative probability found: {}", prob);
        }
    }

    #[test]
    fn test_candle_temperature_processor() {
        let mut processor =
            CandleTemperatureProcessor::new(0.5).expect("Failed to create processor");

        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();

        processor
            .apply_temperature(&mut logits)
            .expect("Temperature scaling failed");

        // Values should be scaled by 1/temperature = 2.0
        for (original_val, scaled_val) in original.iter().zip(logits.iter()) {
            let expected = original_val * 2.0;
            assert!(
                (scaled_val - expected).abs() < 1e-6,
                "Expected {}, got {}",
                expected,
                scaled_val
            );
        }
    }

    #[test]
    fn test_error_conversion() {
        let simd_err = SimdError::InvalidConfiguration("test".to_string());
        let _candle_err: CandleError = simd_err.into();
        // Just ensure conversion compiles and runs
    }

    #[test]
    fn test_utils() {
        // Just verify these functions exist and can be called
        let _supported = utils::simd_supported();
        let _processor = utils::create_simd_processor();
        let _softmax = utils::create_simd_softmax(1.0);
        let _temperature = utils::create_simd_temperature(1.0);
    }
}
