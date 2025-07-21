//! Unified LogitsProcessor trait system for sophisticated sampling strategies
//!
//! This module provides the core trait definitions for the processing system with:
//! - Context-aware processing with token history
//! - Zero allocation patterns with in-place operations  
//! - Comprehensive error handling without unwrap/expect
//! - Composable processor architecture
//! - Production-ready numerical stability

use crate::processing::error::ProcessingError;
use crate::processing::context::ProcessingContext;

/// Result type for processing operations
pub type ProcessingResult<T> = Result<T, ProcessingError>;

/// Core trait for logits processing strategies with context awareness
/// 
/// This trait defines the interface for all logits processors, providing:
/// - In-place logits modification for zero allocation
/// - Context-aware processing with token history
/// - Comprehensive error handling
/// - Composition support through trait objects
/// 
/// # Implementation Requirements
/// 
/// Implementations must:
/// - Never use unwrap() or expect() in production code
/// - Handle all numerical edge cases gracefully
/// - Maintain numerical stability
/// - Support zero-allocation processing patterns
/// - Provide semantic error information
/// 
/// # Examples
/// 
/// ```rust
/// use fluent_ai_candle::processing::{LogitsProcessor, ProcessingContext, ProcessingResult};
/// 
/// struct CustomProcessor;
/// 
/// impl LogitsProcessor for CustomProcessor {
///     fn process_logits(
///         &mut self, 
///         logits: &mut [f32], 
///         context: &ProcessingContext
///     ) -> ProcessingResult<()> {
///         // Process logits in-place
///         for logit in logits.iter_mut() {
///             *logit *= 0.9; // Example scaling
///         }
///         Ok(())
///     }
///     
///     fn name(&self) -> &'static str {
///         "CustomProcessor"
///     }
/// }
/// ```
pub trait LogitsProcessor: Send + Sync + std::fmt::Debug {
    /// Process logits in-place with context awareness
    /// 
    /// This method modifies the logits array directly to avoid allocations.
    /// The processing context provides access to token history, position,
    /// and other relevant information for sophisticated processing strategies.
    /// 
    /// # Arguments
    /// 
    /// * `logits` - Mutable slice of logits to process in-place
    /// * `context` - Processing context with token history and metadata
    /// 
    /// # Returns
    /// 
    /// * `Ok(())` - Processing completed successfully
    /// * `Err(ProcessingError)` - Processing failed with semantic error
    /// 
    /// # Implementation Notes
    /// 
    /// - Must validate input parameters and handle edge cases
    /// - Should use numerical stability techniques for float operations
    /// - Must not panic or use unwrap/expect in production code
    /// - Should be efficient for repeated calls in generation loops
    fn process_logits(&mut self, logits: &mut [f32], context: &ProcessingContext) -> ProcessingResult<()>;

    /// Get processor name for debugging and metrics
    /// 
    /// Used for logging, debugging, and performance profiling.
    /// Should return a static string identifying the processor type.
    fn name(&self) -> &'static str;

    /// Check if processor is effectively an identity operation
    /// 
    /// Returns true if the processor will not modify the logits.
    /// Used for optimization in processor chains to skip no-op processors.
    /// 
    /// Default implementation returns false (assumes processor modifies logits).
    #[inline(always)]
    fn is_identity(&self) -> bool {
        false
    }

    /// Check if processor is enabled and should be applied
    /// 
    /// Returns true if the processor should be applied to logits.
    /// Used for conditional processing based on configuration.
    /// 
    /// Default implementation returns true (processor is always enabled).
    #[inline(always)]
    fn is_enabled(&self) -> bool {
        true
    }

    /// Validate processor configuration
    /// 
    /// Checks if the processor is properly configured and can be used.
    /// Called during processor chain construction to catch configuration errors early.
    /// 
    /// Default implementation returns Ok (no validation required).
    fn validate(&self) -> ProcessingResult<()> {
        Ok(())
    }

    /// Reset processor state for new sequence
    /// 
    /// Called when starting a new generation sequence to reset any internal state.
    /// Most processors don't need to maintain state, but some might cache information
    /// across multiple process_logits calls.
    /// 
    /// Default implementation does nothing (no state to reset).
    fn reset(&mut self) {
        // Default implementation - no state to reset
    }

    /// Get processor configuration summary for debugging
    /// 
    /// Returns a human-readable description of the processor configuration.
    /// Used for debugging and configuration validation.
    /// 
    /// Default implementation returns the processor name.
    fn config_summary(&self) -> String {
        self.name().to_string()
    }

    /// Estimate processing overhead for performance profiling
    /// 
    /// Returns an estimate of the processing time overhead in relative units.
    /// Used for processor chain optimization and performance profiling.
    /// 
    /// Default implementation returns 1.0 (average overhead).
    fn estimated_overhead(&self) -> f32 {
        1.0
    }

    /// Check if processor supports parallel execution
    /// 
    /// Returns true if the processor can be safely executed in parallel
    /// with other processors. Most processors modify logits in-place and
    /// cannot be parallelized, but some read-only processors might support it.
    /// 
    /// Default implementation returns false (no parallel execution support).
    fn supports_parallel_execution(&self) -> bool {
        false
    }

    /// Get processor priority for ordering in chains
    /// 
    /// Returns a priority value used for automatic ordering in processor chains.
    /// Lower values indicate higher priority (should be applied first).
    /// 
    /// Standard priorities:
    /// - 0-10: Context-dependent processors (repetition penalty)
    /// - 10-20: Distribution modifiers (temperature)
    /// - 20-30: Filtering processors (top-k, top-p)
    /// - 30+: Post-processing and normalization
    /// 
    /// Default implementation returns 50 (low priority).
    fn priority(&self) -> u8 {
        50
    }
}

/// Trait for processors that support cloning (for composition)
/// 
/// This trait extends LogitsProcessor with cloning capability, enabling
/// processors to be stored in collections and cloned as needed.
pub trait CloneableLogitsProcessor: LogitsProcessor + Clone {
    /// Clone the processor into a Box for dynamic dispatch
    fn box_clone(&self) -> Box<dyn LogitsProcessor>;
}

/// Blanket implementation for all Clone + LogitsProcessor types
impl<T: LogitsProcessor + Clone + 'static> CloneableLogitsProcessor for T {
    fn box_clone(&self) -> Box<dyn LogitsProcessor> {
        Box::new(self.clone())
    }
}

/// Trait for processors that require context updates
/// 
/// Some processors need to update the processing context based on generated tokens
/// or maintain internal state. This trait provides methods for context interaction.
pub trait ContextAwareProcessor: LogitsProcessor {
    /// Update processor state after token generation
    /// 
    /// Called after a token is generated to update processor-specific state.
    /// Used by processors that track token frequency, sequences, or other
    /// context-dependent information.
    /// 
    /// # Arguments
    /// 
    /// * `token_id` - The generated token ID
    /// * `context` - Mutable reference to processing context
    fn update_for_token(&mut self, token_id: u32, context: &mut ProcessingContext) -> ProcessingResult<()>;

    /// Prepare processor for specific context
    /// 
    /// Called before processing to allow the processor to prepare for
    /// the current context. Used for processors that need to analyze
    /// the context before applying transformations.
    /// 
    /// # Arguments
    /// 
    /// * `context` - Processing context to prepare for
    fn prepare_for_context(&mut self, context: &ProcessingContext) -> ProcessingResult<()>;
}

/// Trait for processors that can be optimized with SIMD operations
/// 
/// Processors implementing this trait can leverage SIMD optimizations
/// for better performance with large vocabulary sizes or long logits arrays.
pub trait SimdOptimizedProcessor: LogitsProcessor {
    /// Check if SIMD optimizations are available
    /// 
    /// Returns true if SIMD optimizations are available for the current
    /// hardware and array size. Used to choose between SIMD and scalar
    /// implementations at runtime.
    fn simd_available(&self, array_size: usize) -> bool;

    /// Process logits using SIMD optimizations
    /// 
    /// SIMD-optimized version of process_logits for better performance.
    /// Should only be called if simd_available returns true.
    /// 
    /// # Safety
    /// 
    /// This method may use SIMD intrinsics. Implementations must ensure
    /// proper alignment and array bounds checking.
    fn process_logits_simd(&mut self, logits: &mut [f32], context: &ProcessingContext) -> ProcessingResult<()>;
}

/// Trait for processors that can be configured dynamically
/// 
/// Allows processors to be reconfigured without reconstruction,
/// useful for dynamic sampling parameter adjustment during generation.
pub trait ConfigurableProcessor: LogitsProcessor {
    /// Configuration type for this processor
    type Config: Clone + Send + Sync + std::fmt::Debug;

    /// Update processor configuration
    /// 
    /// Updates the processor configuration and validates the new settings.
    /// Returns an error if the configuration is invalid.
    /// 
    /// # Arguments
    /// 
    /// * `config` - New configuration to apply
    fn update_config(&mut self, config: Self::Config) -> ProcessingResult<()>;

    /// Get current processor configuration
    /// 
    /// Returns a copy of the current processor configuration.
    /// Used for configuration persistence and debugging.
    fn get_config(&self) -> Self::Config;
}

/// Trait for processors that support batched operations
/// 
/// Some processors can process multiple logits arrays more efficiently
/// in batch mode. This trait provides methods for batch processing.
pub trait BatchProcessor: LogitsProcessor {
    /// Process multiple logits arrays in batch
    /// 
    /// Processes multiple logits arrays with shared context for better efficiency.
    /// Used in beam search and other multi-hypothesis generation scenarios.
    /// 
    /// # Arguments
    /// 
    /// * `logits_batch` - Slice of mutable logits arrays to process
    /// * `contexts` - Processing contexts for each array
    fn process_batch(
        &mut self, 
        logits_batch: &mut [&mut [f32]], 
        contexts: &[ProcessingContext]
    ) -> ProcessingResult<()>;

    /// Get optimal batch size for this processor
    /// 
    /// Returns the optimal batch size for maximum performance.
    /// Used for batch size optimization in generation systems.
    fn optimal_batch_size(&self) -> usize {
        4 // Default batch size
    }
}

/// Helper trait for processor chain composition
/// 
/// Provides utilities for building and managing processor chains
/// with automatic optimization and error handling.
pub trait ProcessorChain: LogitsProcessor {
    /// Add processor to the chain
    fn add_processor(&mut self, processor: Box<dyn LogitsProcessor>) -> ProcessingResult<()>;

    /// Remove processor from the chain by name
    fn remove_processor(&mut self, name: &str) -> ProcessingResult<bool>;

    /// Get processor count in chain
    fn processor_count(&self) -> usize;

    /// Check if chain is empty
    fn is_empty(&self) -> bool;

    /// Clear all processors from chain
    fn clear(&mut self);

    /// Get processor names in execution order
    fn processor_names(&self) -> Vec<&str>;

    /// Optimize processor chain for performance
    /// 
    /// Reorders processors, removes identity processors, and applies
    /// other optimizations to improve performance.
    fn optimize(&mut self) -> ProcessingResult<()>;
}

/// Marker trait for processors that guarantee no allocations
/// 
/// Processors implementing this trait guarantee that they will not
/// perform any heap allocations during processing. Used for
/// real-time and low-latency applications.
pub trait ZeroAllocationProcessor: LogitsProcessor {}

/// Marker trait for processors that are numerically stable
/// 
/// Processors implementing this trait guarantee numerical stability
/// across all supported input ranges. Used for validation and
/// testing in production systems.
pub trait NumericallyStableProcessor: LogitsProcessor {}

/// Utility functions for trait implementations
pub mod utils {
    use super::*;

    /// Validate logits array for common issues
    /// 
    /// Performs basic validation that is common to most processors:
    /// - Non-empty array
    /// - No NaN or infinite values
    /// - Reasonable value ranges
    /// 
    /// # Arguments
    /// 
    /// * `logits` - Logits array to validate
    /// * `processor_name` - Name of the processor for error context
    pub fn validate_logits(logits: &[f32], processor_name: &str) -> ProcessingResult<()> {
        if logits.is_empty() {
            return Err(ProcessingError::InvalidConfiguration(
                format!("{}: Empty logits array", processor_name)
            ));
        }

        for (i, &logit) in logits.iter().enumerate() {
            if !logit.is_finite() {
                return Err(ProcessingError::NumericalError(
                    format!("{}: Non-finite logit at index {}: {}", processor_name, i, logit)
                ));
            }
        }

        Ok(())
    }

    /// Apply numerical stability clamping to logits
    /// 
    /// Clamps logits to prevent overflow/underflow in subsequent operations.
    /// Uses conservative bounds that work with most mathematical operations.
    /// 
    /// # Arguments
    /// 
    /// * `logits` - Mutable logits array to clamp
    pub fn clamp_for_stability(logits: &mut [f32]) {
        const MAX_LOGIT: f32 = 50.0;  // exp(50) is well below f32::MAX
        const MIN_LOGIT: f32 = -50.0; // exp(-50) is well above f32::MIN_POSITIVE

        for logit in logits.iter_mut() {
            *logit = logit.clamp(MIN_LOGIT, MAX_LOGIT);
        }
    }

    /// Find maximum logit value for numerical stability
    /// 
    /// Finds the maximum logit value in a numerically stable way.
    /// Returns an error if no valid maximum is found.
    /// 
    /// # Arguments
    /// 
    /// * `logits` - Logits array to analyze
    pub fn find_max_logit(logits: &[f32]) -> ProcessingResult<f32> {
        logits.iter()
            .copied()
            .filter(|x| x.is_finite())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| ProcessingError::NumericalError(
                "No finite logits found for max calculation".to_string()
            ))
    }

    /// Apply softmax normalization in-place with numerical stability
    /// 
    /// Applies softmax normalization to logits array in-place using
    /// the log-sum-exp trick for numerical stability.
    /// 
    /// # Arguments
    /// 
    /// * `logits` - Mutable logits array to normalize
    pub fn apply_stable_softmax(logits: &mut [f32]) -> ProcessingResult<()> {
        if logits.is_empty() {
            return Ok(());
        }

        // Find maximum for numerical stability
        let max_logit = find_max_logit(logits)?;

        // Subtract max and compute exp
        for logit in logits.iter_mut() {
            *logit = (*logit - max_logit).exp();
        }

        // Compute sum
        let sum: f32 = logits.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            return Err(ProcessingError::NumericalError(
                "Invalid softmax normalization sum".to_string()
            ));
        }

        // Normalize
        for logit in logits.iter_mut() {
            *logit /= sum;
        }

        Ok(())
    }

    /// Check if two processor names are equivalent
    /// 
    /// Handles name comparison with case insensitivity and common variations.
    /// Used for processor lookup and replacement in chains.
    /// 
    /// # Arguments
    /// 
    /// * `name1` - First processor name
    /// * `name2` - Second processor name
    pub fn processor_names_equivalent(name1: &str, name2: &str) -> bool {
        name1.eq_ignore_ascii_case(name2) ||
        name1.replace("_", "").eq_ignore_ascii_case(&name2.replace("_", "")) ||
        name1.replace("-", "").eq_ignore_ascii_case(&name2.replace("-", ""))
    }

    /// Calculate processing priority order
    /// 
    /// Returns the optimal order for processors based on their priorities
    /// and dependencies. Used for automatic processor chain optimization.
    /// 
    /// # Arguments
    /// 
    /// * `processors` - Slice of processor references
    pub fn calculate_optimal_order(processors: &[&dyn LogitsProcessor]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..processors.len()).collect();
        
        // Sort by priority (lower values first)
        indices.sort_by_key(|&i| processors[i].priority());
        
        indices
    }
}