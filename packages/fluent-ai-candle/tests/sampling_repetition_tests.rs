use fluent_ai_candle::sampling::repetition::*;
use fluent_ai_candle::sampling::*;

use candle_core::{DType, Device};

    

    fn create_test_logits() -> Tensor {
        let device = Device::Cpu;
        // Create logits for 5 tokens
        Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], (5,), &device).expect("tensor creation")
    }

    #[test]
    fn test_repetition_penalty_validation() {
        // Valid penalties
        assert!(RepetitionPenaltyProcessor::new(1.0, 0.0, 0.0, 64).is_ok());
        assert!(RepetitionPenaltyProcessor::new(1.1, 0.0, 0.0, 64).is_ok());
        assert!(RepetitionPenaltyProcessor::new(2.0, 0.0, 0.0, 64).is_ok());

        // Invalid penalties
        assert!(RepetitionPenaltyProcessor::new(0.9, 0.0, 0.0, 64).is_err());
        assert!(RepetitionPenaltyProcessor::new(-1.0, 0.0, 0.0, 64).is_err());
    }

    #[test]
    fn test_identity_penalty() {
        let processor = RepetitionPenaltyProcessor::new(1.0, 0.0, 0.0, 64).expect("valid penalty");
        assert!(processor.is_identity());

        let mut logits = create_test_logits();
        let original_vec = logits.to_vec1::<f32>().expect("conversion");

        let token_ids = vec![0, 1, 2];
        processor
            .process(&mut logits, &token_ids, 3)
            .expect("processing succeeds");

        let processed_vec = logits.to_vec1::<f32>().expect("conversion");

        // Logits should be unchanged for penalty = 1.0
        for (orig, proc) in original_vec.iter().zip(processed_vec.iter()) {
            assert!((orig - proc).abs() < 1e-6);
        }
    }

    #[test]
    fn test_repetition_penalty_application() {
        let processor = RepetitionPenaltyProcessor::new(2.0, 0.0, 0.0, 64).expect("valid penalty");

        let mut logits = create_test_logits();
        let original_vec = logits.to_vec1::<f32>().expect("conversion");

        // Token history with repetitions: token 1 appears twice
        let token_ids = vec![0, 1, 2, 1, 3];
        processor
            .process(&mut logits, &token_ids, 5)
            .expect("processing succeeds");

        let processed_vec = logits.to_vec1::<f32>().expect("conversion");

        // Token 1 should be penalized more than others
        let penalty_factor = 2.0_f64.powi(2).ln() as f32; // penalty^count
        let expected_logit_1 = original_vec[1] - penalty_factor;

        assert!((processed_vec[1] - expected_logit_1).abs() < f32::EPSILON * 10.0);

        // Tokens that appeared once should have smaller penalties
        let single_penalty = 2.0_f64.ln() as f32;
        assert!(
            (processed_vec[0] - (original_vec[0] - single_penalty)).abs() < f32::EPSILON * 10.0
        );
    }

    #[test]
    fn test_context_window() {
        let processor = RepetitionPenaltyProcessor::new(1.5, 0.0, 0.0, 3).expect("valid penalty");

        // Test with long token history, only last 3 should matter
        let long_history = vec![0, 0, 0, 1, 2, 3]; // Only [1, 2, 3] should be considered
        let context = processor.get_context_window(&long_history, 6);

        assert_eq!(context, &[1, 2, 3]);
    }

    #[test]
    fn test_processor_trait_methods() {
        let processor = RepetitionPenaltyProcessor::new(1.3, 0.0, 0.0, 128).expect("valid penalty");

        assert_eq!(processor.name(), "RepetitionPenaltyProcessor");
        assert!(processor.validate().is_ok());
        assert!(!processor.is_identity());
    }
