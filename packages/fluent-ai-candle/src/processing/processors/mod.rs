//! Unified processor implementations for sophisticated sampling strategies
//!
//! This module provides production-ready processor implementations with:
//! - Zero allocation patterns and lock-free operations
//! - Comprehensive error handling without unwrap/expect
//! - Context-aware processing with SIMD optimizations
//! - Numerical stability guarantees
//! - Composable architecture for complex sampling strategies

pub mod composite;
pub mod repetition_penalty;
pub mod temperature;
pub mod top_k;
pub mod top_p;

// Re-export all processors for convenience
pub use composite::{CompositeProcessor, CompositeProcessorBuilder};
pub use repetition_penalty::RepetitionPenaltyProcessor;
pub use temperature::TemperatureProcessor;
pub use top_k::TopKProcessor;
pub use top_p::TopPProcessor;

/// Processor priority constants for automatic ordering
pub mod priorities {
    /// Context-dependent processors (highest priority)
    pub const CONTEXT_DEPENDENT: u8 = 10;

    /// Distribution modifiers (high priority)
    pub const DISTRIBUTION_MODIFIER: u8 = 20;

    /// Filtering processors (medium priority)
    pub const FILTERING: u8 = 30;

    /// Post-processing (low priority)
    pub const POST_PROCESSING: u8 = 40;

    /// Custom processors (lowest priority)
    pub const CUSTOM: u8 = 50;
}

/// Common processor configurations for different use cases
pub mod presets {
    use crate::processing::{traits::LogitsProcessor, ProcessingResult};
    use super::{
        CompositeProcessor,
        RepetitionPenaltyProcessor,
        TemperatureProcessor,
        TopKProcessor,
        TopPProcessor,
    };

    /// Create standard text generation processor chain
    pub fn text_generation(
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
        repetition_penalty: Option<f32>,
    ) -> ProcessingResult<CompositeProcessor> {
        let mut processors: Vec<Box<dyn LogitsProcessor>> = Vec::new();

        // Add repetition penalty first (context-dependent)
        if let Some(penalty) = repetition_penalty {
            processors.push(Box::new(RepetitionPenaltyProcessor::new(
                penalty, 0.0, 0.0, 0, // repetition_penalty, frequency_penalty, presence_penalty, context_window
            )?));
        }

        // Add temperature scaling
        processors.push(Box::new(TemperatureProcessor::new(temperature)?));

        // Add top-k filtering
        if let Some(k) = top_k {
            processors.push(Box::new(TopKProcessor::new(k)?));
        }

        // Add top-p nucleus sampling
        if let Some(p) = top_p {
            processors.push(Box::new(TopPProcessor::new(p)?));
        }

        CompositeProcessor::with_processors(processors)
    }

    /// Create creative writing processor chain
    pub fn creative_writing() -> ProcessingResult<CompositeProcessor> {
        text_generation(
            0.85,       // Higher temperature for creativity
            None,       // No top-k limit
            Some(0.92), // Nucleus sampling
            Some(1.15), // Moderate repetition penalty
        )
    }

    /// Create code generation processor chain
    pub fn code_generation() -> ProcessingResult<CompositeProcessor> {
        text_generation(
            0.2,        // Low temperature for precision
            Some(20),   // Focused vocabulary
            Some(0.95), // High nucleus threshold
            Some(1.05), // Minimal repetition penalty
        )
    }

    /// Create conversational processor chain
    pub fn conversation() -> ProcessingResult<CompositeProcessor> {
        text_generation(
            0.7,       // Balanced temperature
            Some(40),  // Moderate vocabulary focus
            Some(0.9), // Standard nucleus sampling
            Some(1.1), // Standard repetition penalty
        )
    }

    /// Create greedy sampling processor (deterministic)
    pub fn greedy_sampling() -> ProcessingResult<CompositeProcessor> {
        let processors: Vec<Box<dyn LogitsProcessor>> = vec![
            Box::new(TemperatureProcessor::new(0.01)?), // Near-zero temperature
        ];
        CompositeProcessor::with_processors(processors)
    }

    /// Create high-entropy sampling processor (maximum randomness)
    pub fn high_entropy_sampling() -> ProcessingResult<CompositeProcessor> {
        let processors: Vec<Box<dyn LogitsProcessor>> = vec![
            Box::new(TemperatureProcessor::new(2.0)?), // High temperature
        ];
        CompositeProcessor::with_processors(processors)
    }
}

/// Utility functions for processor management
pub mod utils {
    use crate::processing::{traits::LogitsProcessor, ProcessingError, ProcessingResult};

    /// Validate processor compatibility
    ///
    /// Checks if processors can be safely composed together.
    /// Some processors may have conflicting behavior or requirements.
    pub fn validate_processor_compatibility(
        processors: &[&dyn LogitsProcessor],
    ) -> ProcessingResult<()> {
        if processors.is_empty() {
            return Err(ProcessingError::InvalidConfiguration(
                "Empty processor list".to_string(),
            ));
        }

        // Check for duplicate processor types
        let mut processor_names: std::collections::HashSet<&str> = std::collections::HashSet::new();

        for processor in processors {
            let name = processor.name();
            if processor_names.contains(name) {
                return Err(ProcessingError::InvalidConfiguration(format!(
                    "Duplicate processor type: {}",
                    name
                )));
            }
            processor_names.insert(name);
        }

        // Validate individual processors
        for processor in processors {
            processor.validate()?;
        }

        Ok(())
    }

    /// Find processor by name in chain
    ///
    /// Returns the index of the first processor matching the given name.
    pub fn find_processor_by_name(
        processors: &[&dyn LogitsProcessor],
        name: &str,
    ) -> Option<usize> {
        processors.iter().position(|p| p.name() == name)
    }

    /// Check if processor chain is identity (no-op)
    ///
    /// Returns true if all processors in the chain are identity operations.
    pub fn is_identity_chain(processors: &[&dyn LogitsProcessor]) -> bool {
        processors.iter().all(|p| p.is_identity())
    }

    /// Generate processor chain summary
    ///
    /// Creates a human-readable summary of the processor chain configuration.
    pub fn chain_summary(processors: &[&dyn LogitsProcessor]) -> String {
        if processors.is_empty() {
            return "Empty processor chain".to_string();
        }

        let names: Vec<&str> = processors.iter().map(|p| p.name()).collect();
        format!("Processor chain: {}", names.join(" -> "))
    }

    /// Validate logits for basic correctness
    ///
    /// Performs basic validation to ensure logits are in a valid state.
    pub fn validate_logits(logits: &[f32], context: &str) -> ProcessingResult<()> {
        if logits.is_empty() {
            return Err(ProcessingError::invalid_input(format!(
                "Empty logits array in context: {}",
                context
            )));
        }

        // Check for invalid values
        for (i, &logit) in logits.iter().enumerate() {
            if !logit.is_finite() {
                return Err(ProcessingError::NumericalError(format!(
                    "Non-finite logit at index {} in context {}: {}",
                    i, context, logit
                )));
            }
        }

        Ok(())
    }
}
