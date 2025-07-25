//! Parallel composite processor for independent processor execution
//!
//! This module provides parallel processing capabilities for logits processors
//! that can run independently and have their results merged using various strategies.

use crate::sampling::SamplingError;
use crate::processing::traits::LogitsProcessor;

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

    /// Get the number of parallel processors
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.processors.len()
    }

    /// Check if the parallel processor is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }

    /// Get the merge strategy
    #[inline(always)]
    pub fn merge_strategy(&self) -> MergeStrategy {
        self.merge_strategy
    }

    /// Get processor names for debugging
    pub fn processor_names(&self) -> Vec<&str> {
        self.processors.iter().map(|p| p.name()).collect()
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

/// Builder for parallel composite processors
#[derive(Debug, Default)]
pub struct ParallelCompositeProcessorBuilder {
    processors: Vec<Box<dyn LogitsProcessor>>,
    merge_strategy: Option<MergeStrategy>,
}

impl ParallelCompositeProcessorBuilder {
    /// Create a new parallel builder
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
            merge_strategy: None,
        }
    }

    /// Add a processor to parallel execution
    #[inline(always)]
    pub fn add_processor(mut self, processor: Box<dyn LogitsProcessor>) -> Self {
        self.processors.push(processor);
        self
    }

    /// Set the merge strategy
    #[inline(always)]
    pub fn merge_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.merge_strategy = Some(strategy);
        self
    }

    /// Use averaging merge strategy
    #[inline(always)]
    pub fn average_merge(self) -> Self {
        self.merge_strategy(MergeStrategy::Average)
    }

    /// Use weighted averaging merge strategy
    #[inline(always)]
    pub fn weighted_average_merge(self) -> Self {
        self.merge_strategy(MergeStrategy::WeightedAverage)
    }

    /// Use minimum merge strategy (most restrictive)
    #[inline(always)]
    pub fn minimum_merge(self) -> Self {
        self.merge_strategy(MergeStrategy::Minimum)
    }

    /// Use maximum merge strategy (least restrictive)
    #[inline(always)]
    pub fn maximum_merge(self) -> Self {
        self.merge_strategy(MergeStrategy::Maximum)
    }

    /// Build the parallel composite processor
    pub fn build(self) -> Result<ParallelCompositeProcessor, SamplingError> {
        let merge_strategy = self.merge_strategy.unwrap_or(MergeStrategy::Average);
        ParallelCompositeProcessor::new(self.processors, merge_strategy)
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