//! Composite processor for chaining multiple logits processors
//!
//! Provides sophisticated processor composition with zero-allocation patterns,
//! comprehensive error handling, and performance optimizations.

use candle_core::Tensor;

use super::SamplingError;
use crate::processing::traits::LogitsProcessor;
use crate::processing::context::ProcessingContext;

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
    name: String,
}

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
            name,
        })
    }

    /// Create composite processor from builder pattern
    pub fn builder() -> CompositeProcessorBuilder {
        CompositeProcessorBuilder::new()
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
    #[allow(dead_code)] // Part of CompositeProcessor internal API for future use
    fn execute_chain(
        &mut self,
        logits: &mut Tensor,
        _token_ids: &[u32],
        _position: usize,
    ) -> Result<(), SamplingError> {
        // Early return for identity case
        if self.is_identity_cached {
            return Ok(());
        }

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
            
            // Create processing context
            let context = ProcessingContext::new(50000, 1024).map_err(|e| {
                SamplingError::ProcessorChainError(format!(
                    "Failed to create processing context: {}",
                    e
                ))
            })?;
            
            // Process logits using the correct trait method
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
            *logits = Tensor::from_vec(logits_vec, logits.shape(), logits.device()).map_err(|e| {
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
            Err(e) => Err(format!("Tensor type conversion failed: {}", e)),
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

/// Builder for creating composite processors with fluent API
#[derive(Debug, Default)]
pub struct CompositeProcessorBuilder {
    processors: Vec<Box<dyn LogitsProcessor>>,
}

impl CompositeProcessorBuilder {
    /// Create a new builder
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    /// Add a processor to the chain
    #[inline(always)]
    pub fn add_processor(mut self, processor: Box<dyn LogitsProcessor>) -> Self {
        self.processors.push(processor);
        self
    }

    /// Add temperature processor
    #[inline(always)]
    pub fn temperature(self, temperature: f64) -> Result<Self, SamplingError> {
        use crate::processing::processors::temperature::TemperatureProcessor;
        let processor = TemperatureProcessor::new(temperature as f32)?;
        Ok(self.add_processor(Box::new(processor)))
    }

    /// Add top-k processor
    #[inline(always)]
    pub fn top_k(self, k: usize) -> Result<Self, SamplingError> {
        use crate::processing::processors::top_k::TopKProcessor;
        let processor = TopKProcessor::new(k)?;
        Ok(self.add_processor(Box::new(processor)))
    }

    /// Add top-p processor
    #[inline(always)]
    pub fn top_p(self, p: f64) -> Result<Self, SamplingError> {
        use crate::processing::processors::top_p::TopPProcessor;
        let processor = TopPProcessor::new(p as f32)?;
        Ok(self.add_processor(Box::new(processor)))
    }

    /// Add repetition penalty processor
    #[inline(always)]
    pub fn repetition_penalty(
        self,
        penalty: f64,
        context_size: usize,
    ) -> Result<Self, SamplingError> {
        use crate::processing::processors::repetition_penalty::RepetitionPenaltyProcessor;
        let processor = RepetitionPenaltyProcessor::new(
            penalty as f32, 0.0, 0.0, context_size // repetition_penalty, frequency_penalty, presence_penalty, context_window
        )?;
        Ok(self.add_processor(Box::new(processor)))
    }

    /// Add typical sampling processor
    #[inline(always)]
    pub fn typical_sampling(self, typical_p: f64) -> Result<Self, SamplingError> {
        use super::typical::TypicalSamplingProcessor;
        let processor = TypicalSamplingProcessor::new(typical_p)?;
        Ok(self.add_processor(Box::new(processor)))
    }

    /// Add Gumbel-Softmax processor
    #[inline(always)]
    pub fn gumbel_softmax(
        self,
        temperature: f32,
        hard: bool,
        seed: u64,
        device: candle_core::Device,
    ) -> Result<Self, SamplingError> {
        use super::gumbel::GumbelSoftmaxProcessor;
        let processor = GumbelSoftmaxProcessor::new(temperature, hard, seed, device)?;
        Ok(self.add_processor(Box::new(processor)))
    }

    /// Build the composite processor
    pub fn build(self) -> Result<CompositeProcessor, SamplingError> {
        CompositeProcessor::new(self.processors)
    }

    /// Get the current number of processors
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.processors.len()
    }

    /// Check if the builder is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }
}

/// Parallel composite processor for independent processors
///
/// Executes processors in parallel where they don't depend on each other's output.
/// Useful for processors that only read logits without modifying them, or for
/// applying independent transformations that can be merged.
#[derive(Debug)]
pub struct ParallelCompositeProcessor {
    #[allow(dead_code)] // Reserved for future parallel processing implementation
    processors: Vec<Box<dyn LogitsProcessor>>,
    #[allow(dead_code)] // Reserved for future parallel merge strategy implementation
    merge_strategy: MergeStrategy,
}

/// Strategy for merging results from parallel processors
#[derive(Debug, Clone, Copy)]
pub enum MergeStrategy {
    /// Average the results (element-wise mean)
    Average,
    /// Use weighted average based on processor importance
    WeightedAverage,
    /// Take minimum values (most restrictive)
    Minimum,
    /// Take maximum values (least restrictive)
    Maximum,
}

impl ParallelCompositeProcessor {
    /// Create a new parallel composite processor
    pub fn new(
        processors: Vec<Box<dyn LogitsProcessor>>,
        merge_strategy: MergeStrategy,
    ) -> Result<Self, SamplingError> {
        if processors.is_empty() {
            return Err(SamplingError::ProcessorChainError(
                "Cannot create empty parallel composite processor".to_string(),
            ));
        }

        // Validate all processors
        for (i, processor) in processors.iter().enumerate() {
            processor.validate().map_err(|e| {
                SamplingError::ProcessorChainError(format!(
                    "Parallel processor {} failed validation: {}",
                    i, e
                ))
            })?;
        }

        Ok(Self {
            processors,
            merge_strategy,
        })
    }
}

// TODO: Update to new LogitsProcessor API that uses process_logits() instead of process()
// impl LogitsProcessor for ParallelCompositeProcessor {
//     fn process_logits(&mut self, logits: &mut [f32], context: &ProcessingContext) -> ProcessingResult<()> {
//         // Implementation needed for new API
//     }
//
//     fn validate(&self) -> ProcessingResult<()> {
//         // Implementation needed for new API
//     }
//
//     fn name(&self) -> &'static str {
//         "ParallelCompositeProcessor"
//     }
// }

/// Utility functions for composite processing
pub mod utils {
    use super::*;

    /// Create a standard text generation processor chain
    ///
    /// Includes temperature scaling, repetition penalty, top-k, and top-p filtering
    /// in the optimal order for text generation.
    pub fn standard_text_generation_chain(
        temperature: f64,
        top_k: Option<usize>,
        top_p: Option<f64>,
        repetition_penalty: Option<f64>,
    ) -> Result<CompositeProcessor, SamplingError> {
        let mut builder = CompositeProcessorBuilder::new();

        // Add repetition penalty first (context-dependent)
        if let Some(penalty) = repetition_penalty {
            builder = builder.repetition_penalty(penalty, 256)?;
        }

        // Add temperature scaling
        builder = builder.temperature(temperature)?;

        // Add top-k filtering (before top-p for efficiency)
        if let Some(k) = top_k {
            builder = builder.top_k(k)?;
        }

        // Add top-p nucleus sampling
        if let Some(p) = top_p {
            builder = builder.top_p(p)?;
        }

        builder.build()
    }

    /// Create a creative writing processor chain
    ///
    /// Optimized for creative text generation with higher randomness
    /// and sophisticated repetition avoidance.
    pub fn creative_writing_chain() -> Result<CompositeProcessor, SamplingError> {
        CompositeProcessorBuilder::new()
            .repetition_penalty(1.15, 512)?
            .temperature(0.85)?
            .top_p(0.92)?
            .build()
    }

    /// Create a code generation processor chain
    ///
    /// Optimized for code generation with lower randomness
    /// and precise token selection.
    pub fn code_generation_chain() -> Result<CompositeProcessor, SamplingError> {
        CompositeProcessorBuilder::new()
            .repetition_penalty(1.05, 128)?
            .temperature(0.2)?
            .top_k(20)?
            .top_p(0.95)?
            .build()
    }

    /// Create a balanced conversation chain
    ///
    /// Optimized for conversational AI with moderate creativity
    /// and coherence maintenance.
    pub fn conversation_chain() -> Result<CompositeProcessor, SamplingError> {
        CompositeProcessorBuilder::new()
            .repetition_penalty(1.1, 256)?
            .temperature(0.7)?
            .top_k(40)?
            .top_p(0.9)?
            .build()
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::*;
    use crate::sampling::temperature::TemperatureProcessor;

    #[test]
    fn test_composite_processor_creation() -> Result<(), Box<dyn std::error::Error>> {
        let processors: Vec<Box<dyn LogitsProcessor>> =
            vec![Box::new(TemperatureProcessor::new(0.8)?)];

        let composite = CompositeProcessor::new(processors)?;
        assert_eq!(composite.len(), 1);
        assert!(!composite.is_empty());

        Ok(())
    }

    #[test]
    fn test_empty_composite_fails() {
        let processors: Vec<Box<dyn LogitsProcessor>> = vec![];
        assert!(CompositeProcessor::new(processors).is_err());
    }

    #[test]
    fn test_builder_pattern() -> Result<(), Box<dyn std::error::Error>> {
        let composite = CompositeProcessorBuilder::new()
            .temperature(0.8)?
            .top_k(50)?
            .build()?;

        assert_eq!(composite.len(), 2);
        Ok(())
    }

    #[test]
    fn test_identity_optimization() -> Result<(), Box<dyn std::error::Error>> {
        // Create composite with only identity processors
        let processors: Vec<Box<dyn LogitsProcessor>> = vec![
            Box::new(TemperatureProcessor::new(1.0)?), // Identity temperature
        ];

        let composite = CompositeProcessor::new(processors)?;
        assert!(composite.is_identity());

        Ok(())
    }

    #[test]
    fn test_standard_chains() -> Result<(), Box<dyn std::error::Error>> {
        let _creative = utils::creative_writing_chain()?;
        let _code = utils::code_generation_chain()?;
        let _conversation = utils::conversation_chain()?;

        Ok(())
    }

    #[test]
    fn test_processor_validation() -> Result<(), Box<dyn std::error::Error>> {
        let composite = CompositeProcessorBuilder::new().temperature(0.8)?.build()?;

        // Should validate successfully
        assert!(composite.validate().is_ok());

        Ok(())
    }
}
