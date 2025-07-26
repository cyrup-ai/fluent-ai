use fluent_ai_candle::model::loading::quantization::*;
use fluent_ai_candle::model::loading::*;

use candle_core::Device;
    
    #[test]
    fn test_int8_quantization() -> CandleResult<()> {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[0.0f32, 1.0, 2.0, 3.0, 4.0], &device)?;
        let config = QuantizationConfig {
            quant_type: QuantizationType::Int8,
            ..Default::default()
        };
        
        let quantized = quantize_tensor(&tensor, &config)?;
        let dequantized = dequantize_tensor(&quantized, &device)?;
        
        // Check that dequantization is approximately correct
        let original = tensor.to_vec1::<f32>()?;
        let recovered = dequantized.to_vec1::<f32>()?;
        
        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 0.1, "Original: {}, Recovered: {}", o, r);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_no_quantization() -> CandleResult<()> {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[0.0f32, 1.0, 2.0, 3.0, 4.0], &device)?;
        let config = QuantizationConfig::default();
        
        let quantized = quantize_tensor(&tensor, &config)?;
        let dequantized = dequantize_tensor(&quantized, &device)?;
        
        // With no quantization, the tensors should be identical
        assert_eq!(tensor.to_vec1::<f32>()?, dequantized.to_vec1::<f32>()?);
        
        Ok(())
    }
