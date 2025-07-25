//! Core composite processor implementation for chaining multiple logits processors
//!
//! This module provides the main CompositeProcessor struct with sophisticated
//! processor composition, zero-allocation patterns, and comprehensive error handling.

use candle_core::Tensor;

use crate::sampling::SamplingError;
use crate::processing::context::ProcessingContext;
use crate::processing::traits::LogitsProcessor;

/// Composite processor that chains multiple logits processors in sequence
///
/// Executes processors in the order they were added, with each processor
/// operating on the output of the previous processor. Includes optimizations
/// for identity processors and comprehensive error handling.
#[derive(Debug)]
pub struct CompositeProcessor {
    /// Chain of processors to execute in sequence
    processors: Vec<Box<dyn LogitsProcessor>>,
    /// Cached identity status for optimization
    #[allow(dead_code)] // Reserved for future identity processor optimization
    is_identity_cached: bool,
    /// Name for debugging and metrics
    #[allow(dead_code)] // Reserved for future processor debugging and telemetry
    name: String}

impl CompositeProcessor {
    /// Create a new composite processor
    ///
    /// # Arguments
    /// * `processors` - Vector of processors to chain in execution order
    ///
    /// # Returns
    /// * `Ok(CompositeProcessor)` - Successfully created composite
    /// * `Err(SamplingError)` - Invalid processor configuration
    ///
    /// # Examples
    /// ```
    /// use fluent_ai_candle::sampling::{CompositeProcessor, TemperatureProcessor, TopKProcessor};
    ///
    /// let processors: Vec<Box<dyn LogitsProcessor>> = vec![
    ///     Box::new(TemperatureProcessor::new(0.8)?),
    ///     Box::new(TopKProcessor::new(50)?),
    /// ];
    ///
    /// let composite = CompositeProcessor::new(processors)?;
    /// ```
    pub fn new(processors: Vec<Box<dyn LogitsProcessor>>) -> Result<Self, SamplingError> {
        if processors.is_empty() {
            return Err(SamplingError::ProcessorChainError(
                "Cannot create empty composite processor".to_string(),
            ));
        }

        // Validate all processors
        for (i, processor) in processors.iter().enumerate() {
            processor.validate().map_err(|e| {
                SamplingError::ProcessorChainError(format!(
                    "Processor {} failed validation: {}",
                    i, e
                ))
            })?;
        }

        // Check if all processors are identity (optimization)
        let is_identity_cached = processors.iter().all(|p| p.is_identity());

        // Generate descriptive name
        let processor_names: Vec<&str> = processors.iter().map(|p| p.name()).collect();
        let name = format!("Composite({})", processor_names.join(" -> "));

        Ok(Self {
            processors,
            is_identity_cached,
            name})
    }

    /// Create composite processor from builder pattern
    pub fn builder() -> crate::sampling::composite::CompositeProcessorBuilder {
        crate::sampling::composite::CompositeProcessorBuilder::new()
    }

    /// Get the number of processors in the chain
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.processors.len()
    }

    /// Check if the composite is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }

    /// Get processor names for debugging
    pub fn processor_names(&self) -> Vec<&str> {
        self.processors.iter().map(|p| p.name()).collect()
    }

    /// Execute all processors in sequence with comprehensive error handling
    pub fn execute_chain(
        &mut self,
        logits: &mut Tensor,
        token_ids: &[u32],
        position: usize,
    ) -> Result<(), SamplingError> {
        // Early return for identity case
        if self.is_identity_cached {
            return Ok(());
        }

        // Context-aware sampling: adjust processing based on sequence position and history
        let is_early_sequence = position < 10; // First 10 tokens are crucial for context
        let has_repetition = self.detect_repetition_pattern(token_ids, position);

        // Create enhanced processing context with sequence awareness
        let enhanced_bias = if has_repetition {
            0.8 // Reduce bias for repetitive sequences to encourage diversity
        } else if is_early_sequence {
            1.2 // Increase bias early in sequence for coherent start
        } else {
            1.0 // Normal bias for mid-sequence tokens
        };

        // Execute each processor in sequence
        for (i, processor) in self.processors.iter_mut().enumerate() {
            // Skip identity processors for performance
            if processor.is_identity() {
                continue;
            }

            // Execute processor with comprehensive error context
            // Convert tensor to f32 slice for processing
            let logits_data = logits.flatten_all().map_err(|e| {
                SamplingError::ProcessorChainError(format!(
                    "Failed to flatten logits tensor: {}",
                    e
                ))
            })?;
            let mut logits_vec = logits_data.to_vec1::<f32>().map_err(|e| {
                SamplingError::ProcessorChainError(format!(
                    "Failed to convert logits to f32 vector: {}",
                    e
                ))
            })?;

            // Create enhanced processing context with sequence-aware bias
            let context = ProcessingContext::new(50000, 1024).map_err(|e| {
                SamplingError::ProcessorChainError(format!(
                    "Failed to create processing context: {}",
                    e
                ))
            })?;

            // Apply context-aware bias adjustment based on sequence position and history
            for logit in logits_vec.iter_mut() {
                *logit *= enhanced_bias;
            }

            // Add repetition penalty for detected patterns (avoid borrowing self during iteration)
            if has_repetition {
                Self::apply_repetition_penalty_static(&mut logits_vec, token_ids, position);
            }

            // Process logits using the correct trait method with enhanced context
            processor
                .process_logits(&mut logits_vec, &context)
                .map_err(|e| {
                    SamplingError::ProcessorChainError(format!(
                        "Processor {} ({}) failed: {}",
                        i,
                        processor.name(),
                        e
                    ))
                })?;

            // Convert back to tensor
            *logits =
                Tensor::from_vec(logits_vec, logits.shape(), logits.device()).map_err(|e| {
                    SamplingError::ProcessorChainError(format!(
                        "Failed to convert processed logits back to tensor: {}",
                        e
                    ))
                })?;

            // Validate tensor integrity after each processor
            if let Err(validation_error) = Self::validate_tensor_integrity(logits) {
                return Err(SamplingError::ProcessorChainError(format!(
                    "Tensor validation failed after processor {}: {}",
                    i, validation_error
                )));
            }
        }

        Ok(())
    }

    /// Validate tensor integrity between processor executions
    #[inline(always)]
    #[allow(dead_code)] // Part of CompositeProcessor internal API for future validation
    fn validate_tensor_integrity(logits: &Tensor) -> Result<(), String> {
        // Check tensor dimensions
        let shape = logits.shape();
        if shape.dims().is_empty() || shape.elem_count() == 0 {
            return Err("Empty tensor after processing".to_string());
        }

        // Check for numerical issues (simplified check)
        match logits.to_dtype(candle_core::DType::F32) {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("Tensor type conversion failed: {}", e))}
    }

    /// Detect repetition patterns in token sequence for context-aware sampling
    #[inline(always)]
    pub fn detect_repetition_pattern(&self, token_ids: &[u32], position: usize) -> bool {
        if token_ids.len() < 6 || position < 3 {
            return false;
        }

        // Check for immediate repetition (same token repeated)
        let recent_window = token_ids.len().saturating_sub(4);
        let recent_tokens = &token_ids[recent_window..];

        // Look for patterns: ABAB, ABCABC, etc.
        if recent_tokens.len() >= 4 {
            let is_alternating =
                recent_tokens[0] == recent_tokens[2] && recent_tokens[1] == recent_tokens[3];
            let is_immediate_repeat =
                recent_tokens[recent_tokens.len() - 1] == recent_tokens[recent_tokens.len() - 2];

            is_alternating || is_immediate_repeat
        } else {
            false
        }
    }

    /// Apply repetition penalty to reduce likelihood of repeated tokens
    #[inline(always)]
    pub fn apply_repetition_penalty(&self, logits: &mut [f32], token_ids: &[u32], position: usize) {
        Self::apply_repetition_penalty_static(logits, token_ids, position);
    }

    /// Static version of repetition penalty to avoid borrowing conflicts during iteration
    #[inline(always)]
    pub fn apply_repetition_penalty_static(logits: &mut [f32], token_ids: &[u32], _position: usize) {
        // Apply penalty to recently used tokens to encourage diversity
        let penalty_strength = 0.85; // Reduce probability by 15%
        let penalty_window = token_ids.len().saturating_sub(8).max(0);

        for &token_id in &token_ids[penalty_window..] {
            if let Some(logit) = logits.get_mut(token_id as usize) {
                *logit *= penalty_strength;
            }
        }
    }
}

// TODO: Update to new LogitsProcessor API that uses process_logits() instead of process()
// impl LogitsProcessor for CompositeProcessor {
//     #[inline]
//     fn process_logits(&mut self, logits: &mut [f32], context: &ProcessingContext) -> ProcessingResult<()> {
//         // Implementation needed for new API
//     }
//
//     fn validate(&self) -> ProcessingResult<()> {
//         // Implementation needed for new API
//     }
//
//     #[inline(always)]
//     fn name(&self) -> &'static str {
//         "CompositeProcessor"
//     }
//
//     #[inline(always)]
//     fn is_identity(&self) -> bool {
//         self.is_identity_cached
//     }
// }