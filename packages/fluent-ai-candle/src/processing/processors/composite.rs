//! Composite Processor for Chaining Multiple LogitsProcessors
//!
//! Implements a sophisticated processor composition system that allows chaining multiple
//! LogitsProcessor instances into a single processing pipeline. Provides intelligent
//! optimization, error handling, and performance monitoring for complex processing chains.

use crate::processing::traits::{LogitsProcessor, ProcessingResult};
use crate::processing::{ProcessingContext, ProcessingError};

/// Composite processor that chains multiple LogitsProcessor instances
///
/// The CompositeProcessor applies a sequence of LogitsProcessor implementations
/// in order, allowing for complex sampling strategies that combine multiple techniques.
/// It provides intelligent optimizations:
/// - Skips identity processors automatically
/// - Early termination on errors
/// - Efficient memory management with zero-allocation patterns
/// - Validation of the entire processing chain
#[derive(Debug)]
pub struct CompositeProcessor {
    /// Chain of processors to apply in order
    processors: Vec<Box<dyn LogitsProcessor>>,

    /// Cached identity state (optimization)
    is_identity: bool,

    /// Processing chain name for debugging
    name: String,
}

impl CompositeProcessor {
    /// Create new composite processor with no processors
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
            is_identity: true,
            name: "CompositeProcessor".to_string(),
        }
    }

    /// Create composite processor with initial processors
    pub fn with_processors(processors: Vec<Box<dyn LogitsProcessor>>) -> ProcessingResult<Self> {
        let mut composite = Self::new();

        for processor in processors {
            composite.add_processor(processor)?;
        }

        Ok(composite)
    }

    /// Add processor to the end of the chain
    pub fn add_processor(&mut self, processor: Box<dyn LogitsProcessor>) -> ProcessingResult<()> {
        // Validate processor before adding
        processor.validate()?;

        // Update identity cache
        if !processor.is_identity() {
            self.is_identity = false;
        }

        // Add processor to chain
        self.processors.push(processor);

        // Update composite name for debugging
        self.update_name();

        Ok(())
    }

    /// Insert processor at specific position in the chain
    pub fn insert_processor(
        &mut self,
        index: usize,
        processor: Box<dyn LogitsProcessor>,
    ) -> ProcessingResult<()> {
        if index > self.processors.len() {
            return Err(ProcessingError::InvalidConfiguration(format!(
                "Insert index {} exceeds chain length {}",
                index,
                self.processors.len()
            )));
        }

        // Validate processor before inserting
        processor.validate()?;

        // Update identity cache
        if !processor.is_identity() {
            self.is_identity = false;
        }

        // Insert processor at specified position
        self.processors.insert(index, processor);

        // Update composite name
        self.update_name();

        Ok(())
    }

    /// Remove processor at specific index
    pub fn remove_processor(&mut self, index: usize) -> ProcessingResult<Box<dyn LogitsProcessor>> {
        if index >= self.processors.len() {
            return Err(ProcessingError::InvalidConfiguration(format!(
                "Remove index {} exceeds chain length {}",
                index,
                self.processors.len()
            )));
        }

        let removed_processor = self.processors.remove(index);

        // Recalculate identity state
        self.recalculate_identity();

        // Update composite name
        self.update_name();

        Ok(removed_processor)
    }

    /// Clear all processors from the chain
    pub fn clear(&mut self) {
        self.processors.clear();
        self.is_identity = true;
        self.name = "CompositeProcessor".to_string();
    }

    /// Get number of processors in the chain
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.processors.len()
    }

    /// Check if the composite processor is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }

    /// Get reference to processor at index
    pub fn get_processor(&self, index: usize) -> Option<&dyn LogitsProcessor> {
        self.processors.get(index).map(|p| p.as_ref())
    }

    /// Get processor names in order
    pub fn processor_names(&self) -> Vec<&'static str> {
        self.processors.iter().map(|p| p.name()).collect()
    }

    /// Create optimized processing chain by removing identity processors
    pub fn optimize(&mut self) -> usize {
        let original_len = self.processors.len();

        // Remove identity processors
        self.processors.retain(|p| !p.is_identity());

        // Recalculate identity state
        self.recalculate_identity();

        // Update name
        self.update_name();

        original_len - self.processors.len() // Return number removed
    }

    /// Validate entire processing chain
    pub fn validate_chain(&self) -> ProcessingResult<()> {
        for (i, processor) in self.processors.iter().enumerate() {
            processor.validate().map_err(|e| {
                ProcessingError::ProcessorChainError(format!(
                    "Processor {} ({}) validation failed: {}",
                    i,
                    processor.name(),
                    e
                ))
            })?;
        }
        Ok(())
    }

    /// Apply all processors in sequence with error handling
    #[inline]
    fn apply_processor_chain(
        &mut self,
        logits: &mut [f32],
        context: &ProcessingContext,
    ) -> ProcessingResult<()> {
        if self.is_identity {
            return Ok(()); // Fast path for identity chains
        }

        // Apply each processor in sequence
        for (i, processor) in self.processors.iter_mut().enumerate() {
            // Skip identity processors for performance
            if processor.is_identity() {
                continue;
            }

            // Apply processor with detailed error context
            processor.process_logits(logits, context).map_err(|e| {
                ProcessingError::ProcessorChainError(format!(
                    "Processor {} ({}) failed: {}",
                    i,
                    processor.name(),
                    e
                ))
            })?;
        }

        Ok(())
    }

    /// Update composite name based on contained processors
    fn update_name(&mut self) {
        if self.processors.is_empty() {
            self.name = "CompositeProcessor".to_string();
        } else if self.processors.len() == 1 {
            self.name = format!("CompositeProcessor[{}]", self.processors[0].name());
        } else {
            let names: Vec<_> = self.processors.iter().map(|p| p.name()).collect();
            self.name = format!("CompositeProcessor[{}]", names.join(" -> "));
        }
    }

    /// Recalculate identity state after modifications
    fn recalculate_identity(&mut self) {
        self.is_identity =
            self.processors.is_empty() || self.processors.iter().all(|p| p.is_identity());
    }

    /// Create a builder for composing processors fluently
    pub fn builder() -> CompositeProcessorBuilder {
        CompositeProcessorBuilder::new()
    }
}

impl Default for CompositeProcessor {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl LogitsProcessor for CompositeProcessor {
    #[inline]
    fn process_logits(
        &mut self,
        logits: &mut [f32],
        context: &ProcessingContext,
    ) -> ProcessingResult<()> {
        self.apply_processor_chain(logits, context)
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        // Return static name, detailed name is available via debug formatting
        "CompositeProcessor"
    }

    #[inline(always)]
    fn is_identity(&self) -> bool {
        self.is_identity
    }

    fn validate(&self) -> ProcessingResult<()> {
        self.validate_chain()
    }
}

/// Builder pattern for creating composite processors fluently
#[derive(Debug, Default)]
pub struct CompositeProcessorBuilder {
    processors: Vec<Box<dyn LogitsProcessor>>,
}

impl CompositeProcessorBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    /// Add processor to the chain
    pub fn add<P: LogitsProcessor + 'static>(mut self, processor: P) -> Self {
        self.processors.push(Box::new(processor));
        self
    }

    /// Add boxed processor to the chain
    pub fn add_boxed(mut self, processor: Box<dyn LogitsProcessor>) -> Self {
        self.processors.push(processor);
        self
    }

    /// Build the composite processor
    pub fn build(self) -> ProcessingResult<CompositeProcessor> {
        CompositeProcessor::with_processors(self.processors)
    }

    /// Build and optimize the composite processor
    pub fn build_optimized(self) -> ProcessingResult<CompositeProcessor> {
        let mut composite = CompositeProcessor::with_processors(self.processors)?;
        composite.optimize();
        Ok(composite)
    }
}

/// Convenience macro for creating composite processors
#[macro_export]
macro_rules! composite_processor {
    ($($processor:expr),* $(,)?) => {{
        let mut builder = $crate::processing::processors::composite::CompositeProcessorBuilder::new();
        $(
            builder = builder.add($processor);
        )*
        builder.build()
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing::processors::{
        repetition_penalty::RepetitionPenaltyProcessor, temperature::TemperatureProcessor,
        top_k::TopKProcessor, top_p::TopPProcessor,
    };

    /// Mock identity processor for testing
    #[derive(Debug)]
    struct IdentityProcessor;

    impl LogitsProcessor for IdentityProcessor {
        fn process_logits(
            &mut self,
            _logits: &mut [f32],
            _context: &ProcessingContext,
        ) -> ProcessingResult<()> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "IdentityProcessor"
        }

        fn is_identity(&self) -> bool {
            true
        }
    }

    /// Mock failing processor for error testing
    #[derive(Debug)]
    struct FailingProcessor;

    impl LogitsProcessor for FailingProcessor {
        fn process_logits(
            &mut self,
            _logits: &mut [f32],
            _context: &ProcessingContext,
        ) -> ProcessingResult<()> {
            Err(ProcessingError::ProcessingFailed(
                "Mock failure".to_string(),
            ))
        }

        fn name(&self) -> &'static str {
            "FailingProcessor"
        }

        fn validate(&self) -> ProcessingResult<()> {
            Err(ProcessingError::InvalidConfiguration(
                "Mock validation failure".to_string(),
            ))
        }
    }

    #[test]
    fn test_empty_composite() {
        let mut processor = CompositeProcessor::new();
        assert!(processor.is_empty());
        assert_eq!(processor.len(), 0);
        assert!(processor.is_identity());

        let context = ProcessingContext::default();
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();

        assert!(processor.process_logits(&mut logits, &context).is_ok());
        assert_eq!(logits, original); // Should be unchanged
    }

    #[test]
    fn test_single_processor() {
        let mut processor = CompositeProcessor::new();
        let temp_processor = TemperatureProcessor::new(0.5).unwrap();

        assert!(processor.add_processor(Box::new(temp_processor)).is_ok());
        assert_eq!(processor.len(), 1);
        assert!(!processor.is_identity());

        let names = processor.processor_names();
        assert_eq!(names, vec!["TemperatureProcessor"]);
    }

    #[test]
    fn test_multiple_processors() {
        let mut builder = CompositeProcessorBuilder::new();
        builder = builder.add(TemperatureProcessor::new(0.8).unwrap());
        builder = builder.add(TopKProcessor::new(40).unwrap());
        builder = builder.add(TopPProcessor::new(0.9).unwrap());

        let mut processor = builder.build().unwrap();
        assert_eq!(processor.len(), 3);
        assert!(!processor.is_identity());

        let names = processor.processor_names();
        assert_eq!(
            names,
            vec!["TemperatureProcessor", "TopKProcessor", "TopPProcessor"]
        );

        // Test processing
        let context = ProcessingContext::default();
        let mut logits = vec![1.0, 2.0, 3.0, 0.5, 0.1];
        assert!(processor.process_logits(&mut logits, &context).is_ok());
    }

    #[test]
    fn test_identity_optimization() {
        let mut processor = CompositeProcessor::new();

        // Add mix of identity and non-identity processors
        processor
            .add_processor(Box::new(TemperatureProcessor::new(0.8).unwrap()))
            .unwrap();
        processor
            .add_processor(Box::new(IdentityProcessor))
            .unwrap();
        processor
            .add_processor(Box::new(TopKProcessor::new(40).unwrap()))
            .unwrap();
        processor
            .add_processor(Box::new(IdentityProcessor))
            .unwrap();

        assert_eq!(processor.len(), 4);

        let removed_count = processor.optimize();
        assert_eq!(removed_count, 2); // Should remove 2 identity processors
        assert_eq!(processor.len(), 2);

        let names = processor.processor_names();
        assert_eq!(names, vec!["TemperatureProcessor", "TopKProcessor"]);
    }

    #[test]
    fn test_processor_insertion_removal() {
        let mut processor = CompositeProcessor::new();

        // Add initial processors
        processor
            .add_processor(Box::new(TemperatureProcessor::new(0.8).unwrap()))
            .unwrap();
        processor
            .add_processor(Box::new(TopKProcessor::new(40).unwrap()))
            .unwrap();

        // Insert at beginning
        processor
            .insert_processor(
                0,
                Box::new(RepetitionPenaltyProcessor::with_repetition_penalty(1.2).unwrap()),
            )
            .unwrap();
        assert_eq!(processor.len(), 3);
        assert_eq!(processor.processor_names()[0], "RepetitionPenaltyProcessor");

        // Insert in middle
        processor
            .insert_processor(2, Box::new(TopPProcessor::new(0.9).unwrap()))
            .unwrap();
        assert_eq!(processor.len(), 4);
        assert_eq!(processor.processor_names()[2], "TopPProcessor");

        // Remove from middle
        let removed = processor.remove_processor(1).unwrap();
        assert_eq!(removed.name(), "TemperatureProcessor");
        assert_eq!(processor.len(), 3);

        // Test invalid indices
        assert!(processor
            .insert_processor(10, Box::new(IdentityProcessor))
            .is_err());
        assert!(processor.remove_processor(10).is_err());
    }

    #[test]
    fn test_error_handling() {
        // Test validation failure during add
        let mut processor = CompositeProcessor::new();
        assert!(processor.add_processor(Box::new(FailingProcessor)).is_err());

        // Test chain validation
        let mut processor = CompositeProcessor::new();
        processor
            .add_processor(Box::new(TemperatureProcessor::new(0.8).unwrap()))
            .unwrap();
        // Manually add invalid processor (bypassing validation)
        processor.processors.push(Box::new(FailingProcessor));
        processor.recalculate_identity();

        assert!(processor.validate_chain().is_err());

        // Test processing error propagation
        let context = ProcessingContext::default();
        let mut logits = vec![1.0, 2.0, 3.0];
        assert!(processor.process_logits(&mut logits, &context).is_err());
    }

    #[test]
    fn test_clear() {
        let mut processor = CompositeProcessor::new();
        processor
            .add_processor(Box::new(TemperatureProcessor::new(0.8).unwrap()))
            .unwrap();
        processor
            .add_processor(Box::new(TopKProcessor::new(40).unwrap()))
            .unwrap();

        assert_eq!(processor.len(), 2);
        assert!(!processor.is_identity());

        processor.clear();
        assert_eq!(processor.len(), 0);
        assert!(processor.is_identity());
        assert!(processor.is_empty());
    }

    #[test]
    fn test_builder_pattern() {
        let processor = CompositeProcessorBuilder::new()
            .add(TemperatureProcessor::new(0.8).unwrap())
            .add(TopKProcessor::new(40).unwrap())
            .add(TopPProcessor::new(0.9).unwrap())
            .build()
            .unwrap();

        assert_eq!(processor.len(), 3);

        let names = processor.processor_names();
        assert_eq!(
            names,
            vec!["TemperatureProcessor", "TopKProcessor", "TopPProcessor"]
        );
    }

    #[test]
    fn test_optimized_builder() {
        let mut processor = CompositeProcessorBuilder::new()
            .add(TemperatureProcessor::new(0.8).unwrap())
            .add(IdentityProcessor)
            .add(TopKProcessor::new(40).unwrap())
            .add(IdentityProcessor)
            .build_optimized()
            .unwrap();

        assert_eq!(processor.len(), 2); // Identity processors should be removed

        let names = processor.processor_names();
        assert_eq!(names, vec!["TemperatureProcessor", "TopKProcessor"]);
    }

    #[test]
    fn test_complex_processing_chain() {
        let mut processor = CompositeProcessorBuilder::new()
            .add(RepetitionPenaltyProcessor::with_repetition_penalty(1.1).unwrap())
            .add(TemperatureProcessor::new(0.8).unwrap())
            .add(TopKProcessor::new(40).unwrap())
            .add(TopPProcessor::new(0.9).unwrap())
            .build()
            .unwrap();

        let mut context = ProcessingContext::default();
        // Some repeated tokens
        context.add_token(1).unwrap();
        context.add_token(2).unwrap();
        context.add_token(3).unwrap();
        context.add_token(1).unwrap();
        context.add_token(2).unwrap();

        let mut logits = vec![2.0, 1.5, 1.0, 0.5, 0.1, 0.05, 0.01];
        let original = logits.clone();

        assert!(processor.process_logits(&mut logits, &context).is_ok());

        // Verify that processing occurred (logits should be modified)
        assert_ne!(logits, original);

        // Verify that the chain applied all processors
        assert_eq!(processor.len(), 4);
    }
}
