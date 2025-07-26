use fluent_ai_candle::processing::processors::top_k::core::*;
use fluent_ai_candle::processing::processors::top_k::*;

use crate::processing::ProcessingContext;

    #[test]
    fn test_new_processor() {
        let processor = TopKProcessor::new(50).unwrap();
        assert_eq!(processor.k(), 50);
        assert!(!processor.is_identity());
    }

    #[test]
    fn test_identity_processor() {
        let processor = TopKProcessor::new(0).unwrap();
        assert_eq!(processor.k(), 0);
        assert!(processor.is_identity());
    }

    #[test]
    fn test_presets() {
        assert_eq!(TopKProcessor::small().unwrap().k(), 20);
        assert_eq!(TopKProcessor::medium().unwrap().k(), 50);
        assert_eq!(TopKProcessor::large().unwrap().k(), 100);
    }

    #[test]
    fn test_max_k_validation() {
        let result = TopKProcessor::new(MAX_TOP_K + 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_top_k_filtering() {
        let mut processor = TopKProcessor::new(3).unwrap();
        let mut logits = vec![1.0, 3.0, 2.0, 0.0, -1.0];
        
        processor.apply_top_k_filtering(&mut logits).unwrap();
        
        // Top 3 values (3.0, 2.0, 1.0) should remain, others become -inf
        assert!(logits[1] > f32::NEG_INFINITY); // 3.0
        assert!(logits[2] > f32::NEG_INFINITY); // 2.0
        assert!(logits[0] > f32::NEG_INFINITY); // 1.0
        assert_eq!(logits[3], f32::NEG_INFINITY); // 0.0
        assert_eq!(logits[4], f32::NEG_INFINITY); // -1.0
    }

    #[test]
    fn test_identity_no_filtering() {
        let mut processor = TopKProcessor::new(0).unwrap();
        let original_logits = vec![1.0, 3.0, 2.0, 0.0, -1.0];
        let mut logits = original_logits.clone();
        
        processor.apply_top_k_filtering(&mut logits).unwrap();
        
        // Identity processor should not modify logits
        assert_eq!(logits, original_logits);
    }

    #[test] 
    fn test_logits_processor_trait() {
        let mut processor = TopKProcessor::new(2).unwrap();
        let mut logits = vec![1.0, 3.0, 2.0, 0.0];
        let context = ProcessingContext::default();
        
        processor.process_logits(&mut logits, &context).unwrap();
        
        assert_eq!(processor.name(), "TopK");
        assert!(!processor.is_identity());
        assert!(processor.is_enabled());
        assert!(processor.validate().is_ok());
        assert!(processor.estimated_overhead() > 0.0);
    }
