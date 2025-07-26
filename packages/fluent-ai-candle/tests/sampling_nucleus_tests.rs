use fluent_ai_candle::sampling::nucleus::*;
use fluent_ai_candle::sampling::*;

use candle_core::{DType, Device};

    

    fn create_test_logits() -> CandleResult<Tensor> {
        let device = Device::Cpu;
        let logits = vec![1.0, 2.0, 3.0, 0.5, 1.5];
        Tensor::from_vec(logits, (1, 5), &device)
    }

    #[test]
    fn test_top_p_validation() {
        // Valid top_p values
        assert!(TopPProcessor::new(0.1).is_ok());
        assert!(TopPProcessor::new(0.9).is_ok());
        assert!(TopPProcessor::new(1.0).is_ok());

        // Invalid top_p values
        assert!(TopPProcessor::new(-0.1).is_err());
        assert!(TopPProcessor::new(1.1).is_err());
        assert!(TopPProcessor::new(f32::NAN).is_err());
        assert!(TopPProcessor::new(f32::INFINITY).is_err());
    }

    #[test]
    fn test_identity_optimization() -> Result<(), Box<dyn std::error::Error>> {
        let processor = TopPProcessor::new(1.0)?;
        assert!(processor.is_identity());

        let processor = TopPProcessor::new(0.9)?;
        assert!(!processor.is_identity());

        Ok(())
    }

    #[test]
    fn test_top_p_builder() -> Result<(), Box<dyn std::error::Error>> {
        let processor = TopPBuilder::new().balanced().build()?;
        assert_eq!(processor.top_p(), 0.9);

        let processor = TopPBuilder::new().conservative().build()?;
        assert_eq!(processor.top_p(), 0.7);

        Ok(())
    }

    #[test]
    fn test_nucleus_size_calculation() -> Result<(), Box<dyn std::error::Error>> {
        let logits = create_test_logits()?;
        let nucleus_size = utils::calculate_nucleus_size(&logits, 0.9)?;
        assert!(nucleus_size > 0);
        assert!(nucleus_size <= 5); // Should not exceed vocab size

        Ok(())
    }

    #[test]
    fn test_adaptive_top_p() {
        let base_top_p = 0.9;
        let low_entropy = 1.0;
        let high_entropy = 3.0;
        let max_entropy = 4.0;

        let adaptive_low = utils::adaptive_top_p(base_top_p, low_entropy, max_entropy);
        let adaptive_high = utils::adaptive_top_p(base_top_p, high_entropy, max_entropy);

        assert!(adaptive_high > adaptive_low); // Higher entropy should increase top_p
    }
