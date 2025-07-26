use fluent_ai_candle::sampling::simd::*;
use fluent_ai_candle::sampling::*;

#[test]
    fn test_candle_simd_processor() {
        let mut processor = CandleSimdProcessor::new().expect("Failed to create processor");
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let context = ProcessingContext::new().with_temperature(1.0);

        processor
            .process_logits(&mut logits, &context)
            .expect("SIMD processing failed");

        // Verify logits were processed (values should be different from input)
        assert_ne!(logits, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_candle_softmax_processor() {
        let mut processor = CandleSoftmaxProcessor::new(1.0).expect("Failed to create processor");

        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        processor
            .softmax_inplace(&mut logits)
            .expect("SIMD softmax failed");

        // Check that probabilities sum to approximately 1.0
        let sum: f32 = logits.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Probabilities do not sum to 1.0: {}",
            sum
        );

        // Check that all probabilities are positive
        for &prob in &logits {
            assert!(prob > 0.0, "Negative probability found: {}", prob);
        }
    }

    #[test]
    fn test_candle_temperature_processor() {
        let mut processor =
            CandleTemperatureProcessor::new(0.5).expect("Failed to create processor");

        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();

        processor
            .apply_temperature(&mut logits)
            .expect("Temperature scaling failed");

        // Values should be scaled by 1/temperature = 2.0
        for (original_val, scaled_val) in original.iter().zip(logits.iter()) {
            let expected = original_val * 2.0;
            assert!(
                (scaled_val - expected).abs() < 1e-6,
                "Expected {}, got {}",
                expected,
                scaled_val
            );
        }
    }

    #[test]
    fn test_error_conversion() {
        let simd_err = SimdError::InvalidConfiguration("test".to_string());
        let _candle_err: CandleError = simd_err.into();
        // Just ensure conversion compiles and runs
    }

    #[test]
    fn test_utils() {
        // Just verify these functions exist and can be called
        let _supported = utils::simd_supported();
        let _processor = utils::create_simd_processor();
        let _softmax = utils::create_simd_softmax(1.0);
        let _temperature = utils::create_simd_temperature(1.0);
    }
