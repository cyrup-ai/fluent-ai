use fluent_ai_candle::sampling::temperature::*;
use fluent_ai_candle::sampling::*;

use candle_core::{DType, Device};

    

    fn create_test_logits() -> CandleResult<Tensor> {
        let device = Device::Cpu;
        let logits = vec![1.0, 2.0, 3.0, 0.5, 1.5];
        Tensor::from_vec(logits, (1, 5), &device)
    }

    #[test]
    fn test_temperature_validation() {
        // Valid temperatures
        assert!(TemperatureProcessor::new(0.1).is_ok());
        assert!(TemperatureProcessor::new(1.0).is_ok());
        assert!(TemperatureProcessor::new(2.0).is_ok());

        // Invalid temperatures
        assert!(TemperatureProcessor::new(0.0).is_err());
        assert!(TemperatureProcessor::new(-1.0).is_err());
        assert!(TemperatureProcessor::new(f32::NAN).is_err());
        assert!(TemperatureProcessor::new(f32::INFINITY).is_err());
    }

    #[test]
    fn test_identity_optimization() -> Result<(), Box<dyn std::error::Error>> {
        let processor = TemperatureProcessor::new(1.0)?;
        assert!(processor.is_identity());

        let processor = TemperatureProcessor::new(0.8)?;
        assert!(!processor.is_identity());

        Ok(())
    }

    #[test]
    fn test_temperature_builder() -> Result<(), Box<dyn std::error::Error>> {
        let processor = TemperatureBuilder::new().creative().build()?;
        assert_eq!(processor.temperature(), 0.8);

        let processor = TemperatureBuilder::new().focused().build()?;
        assert_eq!(processor.temperature(), 0.3);

        Ok(())
    }

    #[test]
    fn test_adaptive_temperature() {
        let base_temp = 0.7;
        let adjusted = utils::adaptive_temperature(base_temp, 1.2, 100);
        assert!(adjusted > base_temp); // Should increase due to repetition

        let adjusted = utils::adaptive_temperature(base_temp, 1.0, 1000);
        assert!(adjusted < base_temp); // Should decrease due to long context
    }
