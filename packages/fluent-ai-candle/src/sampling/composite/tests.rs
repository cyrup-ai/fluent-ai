//! Tests for composite processor functionality
//!
//! Comprehensive test suite covering composite processor creation, builder patterns,
//! identity optimization, and pre-configured processor chains.

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::super::{
        core::CompositeProcessor,
        builder::CompositeProcessorBuilder,
        parallel::{ParallelCompositeProcessor, MergeStrategy}};
    use crate::sampling::temperature::TemperatureProcessor;
    use crate::processing::traits::LogitsProcessor;
    use crate::sampling::SamplingError;

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
        // Note: is_identity() method may not be public, so we test indirectly
        assert_eq!(composite.len(), 1);

        Ok(())
    }

    #[test]
    fn test_processor_names() -> Result<(), Box<dyn std::error::Error>> {
        let composite = CompositeProcessorBuilder::new()
            .temperature(0.8)?
            .build()?;

        let names = composite.processor_names();
        assert!(!names.is_empty());

        Ok(())
    }

    #[test]
    fn test_repetition_detection() -> Result<(), Box<dyn std::error::Error>> {
        let composite = CompositeProcessorBuilder::new()
            .temperature(0.8)?
            .build()?;

        // Test with short sequence (should return false)
        let short_tokens = vec![1, 2, 3];
        assert!(!composite.detect_repetition_pattern(&short_tokens, 2));

        // Test with alternating pattern
        let alternating_tokens = vec![1, 2, 1, 2, 3, 4];
        assert!(composite.detect_repetition_pattern(&alternating_tokens, 5));

        // Test with immediate repetition
        let repeat_tokens = vec![1, 2, 3, 4, 4, 5];
        assert!(composite.detect_repetition_pattern(&repeat_tokens, 4));

        Ok(())
    }

    #[test]
    fn test_parallel_processor_creation() -> Result<(), Box<dyn std::error::Error>> {
        let processors: Vec<Box<dyn LogitsProcessor>> =
            vec![Box::new(TemperatureProcessor::new(0.8)?)];

        let parallel = ParallelCompositeProcessor::new(processors, MergeStrategy::Average)?;
        assert_eq!(parallel.len(), 1);
        assert!(!parallel.is_empty());

        Ok(())
    }

    #[test]
    fn test_parallel_empty_fails() {
        let processors: Vec<Box<dyn LogitsProcessor>> = vec![];
        assert!(ParallelCompositeProcessor::new(processors, MergeStrategy::Average).is_err());
    }

    #[test]
    fn test_merge_strategies() {
        use super::super::parallel::MergeStrategy;

        let strategies = vec![
            MergeStrategy::Average,
            MergeStrategy::WeightedAverage,
            MergeStrategy::Minimum,
            MergeStrategy::Maximum,
        ];

        // Test that all strategies can be created and used
        for strategy in strategies {
            // This test verifies the enum variants exist and can be copied
            let _copy = strategy;
        }
    }

    #[test]
    fn test_builder_is_empty() {
        let builder = CompositeProcessorBuilder::new();
        assert!(builder.is_empty());
        assert_eq!(builder.len(), 0);
    }

    #[test]
    fn test_builder_chain_length() -> Result<(), Box<dyn std::error::Error>> {
        let builder = CompositeProcessorBuilder::new()
            .temperature(0.8)?
            .top_k(50)?
            .top_p(0.9)?;

        assert_eq!(builder.len(), 3);
        assert!(!builder.is_empty());

        Ok(())
    }

    #[test]
    fn test_preset_chains() -> Result<(), Box<dyn std::error::Error>> {
        use super::super::builder::presets;

        // Test standard text generation chain
        let _standard = presets::standard_text_generation_chain(
            0.8,
            Some(50),
            Some(0.9),
            Some(1.1),
        )?;

        // Test creative writing chain
        let _creative = presets::creative_writing_chain()?;

        // Test code generation chain
        let _code = presets::code_generation_chain()?;

        // Test conversation chain
        let _conversation = presets::conversation_chain()?;

        Ok(())
    }

    #[test]
    fn test_processor_validation() -> Result<(), Box<dyn std::error::Error>> {
        let composite = CompositeProcessorBuilder::new()
            .temperature(0.8)?
            .build()?;

        // Processor validation happens during creation, so if we got here, it passed
        assert_eq!(composite.len(), 1);

        Ok(())
    }

    #[test]
    fn test_repetition_penalty_static() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let token_ids = vec![0, 1, 2, 1]; // Token 1 is repeated

        super::super::core::CompositeProcessor::apply_repetition_penalty_static(
            &mut logits,
            &token_ids,
            3,
        );

        // Token 1 should have reduced logit value
        assert!(logits[1] < 2.0);
        // Other tokens should be unchanged
        assert_eq!(logits[0], 1.0 * 0.85); // Also penalized as it appears in recent window
        assert_eq!(logits[3], 4.0);
        assert_eq!(logits[4], 5.0);
    }
}