use fluent_ai_candle::sampling::mirostat::processor::*;
use fluent_ai_candle::sampling::mirostat::*;

#[test]
    fn test_processor_creation() {
        let processor = MirostatProcessor::v1(5.0, 0.1).unwrap();
        assert_eq!(processor.current_tau(), 5.0);
        assert_eq!(processor.current_perplexity(), 1.0);
    }

    #[test]
    fn test_atomic_conversion() {
        let value = 3.14f32;
        let atomic = MirostatProcessor::f32_to_atomic_u64(value);
        let converted = MirostatProcessor::atomic_u64_to_f32(atomic);
        assert_eq!(value, converted);
    }

    #[test]
    fn test_processor_reset() {
        let mut processor = MirostatProcessor::v1(5.0, 0.1).unwrap();
        processor.tokens_processed = 100;
        processor.reset();
        assert_eq!(processor.tokens_processed, 0);
        assert_eq!(processor.current_tau(), 5.0);
    }
