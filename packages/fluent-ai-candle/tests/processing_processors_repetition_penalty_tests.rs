use fluent_ai_candle::processing::processors::repetition_penalty::*;
use fluent_ai_candle::processing::processors::*;

#[test]
    fn test_repetition_penalty_creation() {
        // Valid configurations
        let processor = RepetitionPenaltyProcessor::new(1.2, 0.0, 0.0, 0);
        assert!(processor.is_ok());

        let processor = RepetitionPenaltyProcessor::with_repetition_penalty(1.5);
        assert!(processor.is_ok());
        assert_eq!(processor.unwrap().repetition_penalty(), 1.5);

        // Invalid configurations
        assert!(RepetitionPenaltyProcessor::new(0.8, 0.0, 0.0, 0).is_err()); // penalty < 1.0
        assert!(RepetitionPenaltyProcessor::new(1.0, -0.1, 0.0, 0).is_err()); // negative frequency
        assert!(RepetitionPenaltyProcessor::new(1.0, 0.0, -0.1, 0).is_err()); // negative presence

        // Identity case
        let identity = RepetitionPenaltyProcessor::new(1.0, 0.0, 0.0, 0).unwrap();
        assert!(identity.is_identity());
    }

    #[test]
    fn test_token_frequency_tracking() {
        let mut processor = RepetitionPenaltyProcessor::new(1.2, 0.0, 0.0, 0).unwrap();

        // Create context with repeated tokens
        let mut context = ProcessingContext::default();
        // Token 1 appears 3 times, token 2 twice, token 3 once
        context.add_token(1).unwrap();
        context.add_token(2).unwrap();
        context.add_token(3).unwrap();
        context.add_token(1).unwrap();
        context.add_token(2).unwrap();
        context.add_token(1).unwrap();

        processor.update_token_frequencies(&context);

        assert_eq!(processor.get_token_frequency(1), 3);
        assert_eq!(processor.get_token_frequency(2), 2);
        assert_eq!(processor.get_token_frequency(3), 1);
        assert_eq!(processor.get_token_frequency(99), 0); // Not present
    }

    #[test]
    fn test_context_window() {
        let mut processor = RepetitionPenaltyProcessor::new(1.2, 0.0, 0.0, 3).unwrap(); // Window of 3

        let mut context = ProcessingContext::default();
        // Only last 3 tokens: [4, 5, 1]
        context.add_token(1).unwrap();
        context.add_token(2).unwrap();
        context.add_token(3).unwrap();
        context.add_token(4).unwrap();
        context.add_token(5).unwrap();
        context.add_token(1).unwrap();

        processor.update_token_frequencies(&context);

        assert_eq!(processor.get_token_frequency(1), 1); // Only count the last occurrence
        assert_eq!(processor.get_token_frequency(4), 1);
        assert_eq!(processor.get_token_frequency(5), 1);
        assert_eq!(processor.get_token_frequency(2), 0); // Outside window
        assert_eq!(processor.get_token_frequency(3), 0); // Outside window
    }

    #[test]
    fn test_penalty_application() {
        let mut processor = RepetitionPenaltyProcessor::new(2.0, 0.1, 0.05, 0).unwrap();

        let mut context = ProcessingContext::default();
        // Token 0 appears twice, token 1 once
        context.add_token(0).unwrap();
        context.add_token(1).unwrap();
        context.add_token(0).unwrap();

        let mut logits = vec![1.0, 0.5, 2.0]; // 3 tokens
        let original = logits.clone();

        processor.process_logits(&mut logits, &context).unwrap();

        // Token 0 and 1 should have penalties applied
        assert!(logits[0] < original[0], "Token 0 should be penalized");
        assert!(logits[1] < original[1], "Token 1 should be penalized");
        assert_eq!(logits[2], original[2]); // Token 2 unchanged (not in history)
    }

    #[test]
    fn test_frequency_vs_presence_penalty() {
        // Test frequency penalty (scales with frequency)
        let mut freq_processor = RepetitionPenaltyProcessor::new(1.0, 0.1, 0.0, 0).unwrap();
        let mut context = ProcessingContext::default();
        // Token 0 appears 3 times, token 1 once
        context.add_token(0).unwrap();
        context.add_token(0).unwrap();
        context.add_token(0).unwrap();
        context.add_token(1).unwrap();

        let mut logits = vec![1.0, 1.0];
        freq_processor
            .process_logits(&mut logits, &context)
            .unwrap();

        // Token 0 should have larger penalty due to higher frequency
        assert!(
            logits[0] < logits[1],
            "Higher frequency should result in larger penalty"
        );

        // Test presence penalty (fixed regardless of frequency)
        let mut pres_processor = RepetitionPenaltyProcessor::new(1.0, 0.0, 0.1, 0).unwrap();
        let mut logits2 = vec![1.0, 1.0];
        pres_processor
            .process_logits(&mut logits2, &context)
            .unwrap();

        // Both tokens should have similar penalties (only presence matters)
        let diff = (logits2[0] - logits2[1]).abs();
        assert!(
            diff < 0.01,
            "Presence penalty should not depend on frequency"
        );
    }

    #[test]
    fn test_identity_optimization() {
        let mut processor = RepetitionPenaltyProcessor::new(1.0, 0.0, 0.0, 0).unwrap();
        assert!(processor.is_identity());

        let context = ProcessingContext::default();
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();

        processor.process_logits(&mut logits, &context).unwrap();

        // Should be unchanged
        assert_eq!(logits, original);
    }

    #[test]
    fn test_parameter_updates() {
        let mut processor = RepetitionPenaltyProcessor::new(1.0, 0.0, 0.0, 0).unwrap();

        // Test valid updates
        assert!(processor.set_repetition_penalty(1.5).is_ok());
        assert_eq!(processor.repetition_penalty(), 1.5);
        assert!(!processor.is_identity());

        assert!(processor.set_frequency_penalty(0.1).is_ok());
        assert_eq!(processor.frequency_penalty(), 0.1);

        assert!(processor.set_presence_penalty(0.05).is_ok());
        assert_eq!(processor.presence_penalty(), 0.05);

        // Test invalid updates
        assert!(processor.set_repetition_penalty(0.5).is_err());
        assert!(processor.set_frequency_penalty(-0.1).is_err());
        assert!(processor.set_presence_penalty(-0.1).is_err());
    }

    #[test]
    fn test_edge_cases() {
        let mut processor = RepetitionPenaltyProcessor::new(1.2, 0.0, 0.0, 0).unwrap();
        let context = ProcessingContext::default();

        // Empty logits
        let mut empty: Vec<f32> = vec![];
        assert!(processor.process_logits(&mut empty, &context).is_ok());

        // Single token
        let mut single = vec![1.0];
        assert!(processor.process_logits(&mut single, &context).is_ok());

        // Empty context
        let empty_context = ProcessingContext::default();
        let mut logits = vec![1.0, 2.0, 3.0];
        assert!(
            processor
                .process_logits(&mut logits, &empty_context)
                .is_ok()
        );
    }

    #[test]
    fn test_validation() {
        let processor = RepetitionPenaltyProcessor::new(1.2, 0.1, 0.05, 10).unwrap();
        assert!(processor.validate().is_ok());

        // Test invalid configurations
        let mut invalid = RepetitionPenaltyProcessor::new(1.0, 0.0, 0.0, 0).unwrap();
        invalid.repetition_penalty = 0.5; // Invalid
        assert!(invalid.validate().is_err());

        invalid.repetition_penalty = 1.2;
        invalid.frequency_penalty = -0.1; // Invalid
        assert!(invalid.validate().is_err());

        invalid.frequency_penalty = 0.1;
        invalid.presence_penalty = -0.1; // Invalid
        assert!(invalid.validate().is_err());
    }
