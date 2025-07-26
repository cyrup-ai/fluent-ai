use fluent_ai_candle::processing::processors::top_p::*;
use fluent_ai_candle::processing::processors::*;

use approx::assert_relative_eq;

    

    #[test]
    fn test_top_p_creation() {
        let processor = TopPProcessor::new(0.9);
        assert!(processor.is_ok());
        assert_eq!(processor.unwrap().p(), 0.9);

        // Test invalid values
        assert!(TopPProcessor::new(0.0).is_err());
        assert!(TopPProcessor::new(1.1).is_err());
        assert!(TopPProcessor::new(-0.1).is_err());

        // Test edge case: p = 1.0 (identity)
        let identity = TopPProcessor::new(1.0).unwrap();
        assert!(identity.is_identity());
    }

    #[test]
    fn test_nucleus_filtering() {
        let mut processor = TopPProcessor::new(0.8).unwrap();
        let context = ProcessingContext::default();

        // Test basic nucleus sampling
        let mut logits = vec![1.0, 2.0, 0.5, 0.1, 0.05]; // Probabilities: high to low
        let original = logits.clone();

        processor.process_logits(&mut logits, &context).unwrap();

        // Verify that some tokens were filtered (set to very low values)
        let filtered_count = logits.iter().filter(|&&x| x < -1e9).count();
        assert!(filtered_count > 0, "Some tokens should be filtered");

        // Test identity case
        let mut identity_processor = TopPProcessor::new(1.0).unwrap();
        let mut identity_logits = original;
        let expected = identity_logits.clone();

        identity_processor
            .process_logits(&mut identity_logits, &context)
            .unwrap();

        // Should be unchanged
        for (actual, expected) in identity_logits.iter().zip(expected.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_numerical_stability() {
        let mut processor = TopPProcessor::new(0.5).unwrap();
        let context = ProcessingContext::default();

        // Test with large logit values
        let mut large_logits = vec![100.0, 99.0, 98.0, 1.0, 0.0];
        assert!(
            processor
                .process_logits(&mut large_logits, &context)
                .is_ok()
        );

        // Test with small logit values
        let mut small_logits = vec![-100.0, -101.0, -102.0, -103.0, -104.0];
        assert!(
            processor
                .process_logits(&mut small_logits, &context)
                .is_ok()
        );
    }

    #[test]
    fn test_edge_cases() {
        let mut processor = TopPProcessor::new(0.9).unwrap();
        let context = ProcessingContext::default();

        // Test empty logits
        let mut empty: Vec<f32> = vec![];
        assert!(processor.process_logits(&mut empty, &context).is_ok());

        // Test single logit
        let mut single = vec![1.0];
        assert!(processor.process_logits(&mut single, &context).is_ok());

        // Test uniform distribution
        let mut uniform = vec![1.0, 1.0, 1.0, 1.0];
        assert!(processor.process_logits(&mut uniform, &context).is_ok());
    }

    #[test]
    fn test_configuration_updates() {
        let mut processor = TopPProcessor::new(0.9).unwrap();

        // Test valid update
        assert!(processor.set_p(0.5).is_ok());
        assert_eq!(processor.p(), 0.5);
        assert!(!processor.is_identity());

        // Test invalid updates
        assert!(processor.set_p(0.0).is_err());
        assert!(processor.set_p(1.1).is_err());

        // Processor should maintain previous valid state
        assert_eq!(processor.p(), 0.5);

        // Test identity update
        assert!(processor.set_p(1.0).is_ok());
        assert!(processor.is_identity());
    }

    #[test]
    fn test_validation() {
        let processor = TopPProcessor::new(0.9).unwrap();
        assert!(processor.validate().is_ok());

        let mut invalid_processor = TopPProcessor::new(0.9).unwrap();
        invalid_processor.p = 1.5; // Manually set invalid value
        assert!(invalid_processor.validate().is_err());
    }
