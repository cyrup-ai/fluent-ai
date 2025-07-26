use fluent_ai_candle::processing::processors::composite::*;
use fluent_ai_candle::processing::processors::*;

use crate::processing::processors::{
        repetition_penalty::RepetitionPenaltyProcessor, temperature::TemperatureProcessor,
        top_k::TopKProcessor, top_p::TopPProcessor};

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
        assert!(
            processor
                .insert_processor(10, Box::new(IdentityProcessor))
                .is_err()
        );
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
