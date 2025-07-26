use fluent_ai_candle::sampling::gumbel::*;
use fluent_ai_candle::sampling::*;

use candle_core::{DType, Device};

    

    #[test]
    fn test_gumbel_softmax_creation() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let processor = GumbelSoftmaxProcessor::new(1.0, false, 42, device)?;

        assert_eq!(processor.temperature(), 1.0);
        assert!(!processor.is_hard());

        Ok(())
    }

    #[test]
    fn test_temperature_validation() {
        let device = Device::Cpu;

        // Valid temperature
        assert!(GumbelSoftmaxProcessor::new(1.0, false, 42, device.clone()).is_ok());

        // Invalid temperatures
        assert!(GumbelSoftmaxProcessor::new(0.0, false, 42, device.clone()).is_err());
        assert!(GumbelSoftmaxProcessor::new(-1.0, false, 42, device.clone()).is_err());
        assert!(GumbelSoftmaxProcessor::new(f32::NAN, false, 42, device.clone()).is_err());
    }

    #[test]
    fn test_builder_pattern() -> Result<(), Box<dyn std::error::Error>> {
        let processor = GumbelSoftmaxBuilder::new()
            .temperature(0.5)
            .hard()
            .seed(123)
            .device(Device::Cpu)
            .build()?;

        assert_eq!(processor.temperature(), 0.5);
        assert!(processor.is_hard());

        Ok(())
    }

    #[test]
    fn test_gumbel_noise_generation() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let processor = GumbelSoftmaxProcessor::new(1.0, false, 42, device)?;

        let shape = candle_core::Shape::from_dims(&[10]);
        let noise = processor.generate_gumbel_noise(&shape)?;

        assert_eq!(noise.shape().dims(), &[10]);
        assert_eq!(noise.dtype(), DType::F32);

        Ok(())
    }
