use fluent_ai_candle::sampling::typical::*;
use fluent_ai_candle::sampling::*;

use candle_core::{DType, Device};

    

    #[test]
    fn test_typical_sampling_creation() -> Result<(), Box<dyn std::error::Error>> {
        let processor = TypicalSamplingProcessor::new(0.9)?;
        assert_eq!(processor.typical_p(), 0.9);
        Ok(())
    }

    #[test]
    fn test_typical_p_validation() {
        // Valid typical_p
        assert!(TypicalSamplingProcessor::new(0.9).is_ok());
        assert!(TypicalSamplingProcessor::new(0.1).is_ok());

        // Invalid typical_p
        assert!(TypicalSamplingProcessor::new(0.0).is_err());
        assert!(TypicalSamplingProcessor::new(1.5).is_err());
        assert!(TypicalSamplingProcessor::new(f64::NAN).is_err());
    }

    #[test]
    fn test_entropy_calculation() -> Result<(), Box<dyn std::error::Error>> {
        let processor = TypicalSamplingProcessor::new(0.9)?;

        // Uniform distribution entropy
        let uniform_probs = vec![0.25, 0.25, 0.25, 0.25];
        let entropy = processor.calculate_entropy(&uniform_probs);
        let expected_entropy = 4.0 * 0.25 * (0.25_f64).ln().abs();
        assert!((entropy - expected_entropy).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_builder_pattern() -> Result<(), Box<dyn std::error::Error>> {
        let processor = TypicalSamplingBuilder::new()
            .typical_p(0.8)
            .min_entropy(1e-6)
            .use_approximation()
            .build()?;

        assert_eq!(processor.typical_p(), 0.8);

        Ok(())
    }

    #[test]
    fn test_identity_check() -> Result<(), Box<dyn std::error::Error>> {
        let identity_processor = TypicalSamplingProcessor::new(1.0)?;
        assert!(identity_processor.is_identity());

        let non_identity_processor = TypicalSamplingProcessor::new(0.9)?;
        assert!(!non_identity_processor.is_identity());

        Ok(())
    }
