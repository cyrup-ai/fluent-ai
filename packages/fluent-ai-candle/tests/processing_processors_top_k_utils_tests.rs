use fluent_ai_candle::processing::processors::top_k::utils::*;
use fluent_ai_candle::processing::processors::top_k::*;

#[test]
    fn test_adaptive_top_k() {
        let context = ProcessingContext::default();
        let result = adaptive_top_k(50, &context, 0.5).unwrap();
        assert!(result <= MAX_TOP_K);
        assert!(result > 0);
    }

    #[test]
    fn test_perplexity_based_k() {
        // High perplexity should increase k
        let high_perplexity_k = perplexity_based_k(50, 10.0, 5.0).unwrap();
        assert!(high_perplexity_k >= 50);
        
        // Low perplexity should decrease k
        let low_perplexity_k = perplexity_based_k(50, 2.0, 5.0).unwrap();
        assert!(low_perplexity_k <= 50);
    }

    #[test]
    fn test_temperature_adjusted_k() {
        // High temperature should decrease k
        let high_temp_k = temperature_adjusted_k(50, 2.0, 1.0).unwrap();
        assert!(high_temp_k <= 50);
        
        // Low temperature should increase k
        let low_temp_k = temperature_adjusted_k(50, 0.3, 1.0).unwrap();
        assert!(low_temp_k >= 50);
    }

    #[test]
    fn test_k_for_coverage() {
        let logits = vec![2.0, 1.0, 0.0, -1.0, -2.0];
        let k = k_for_coverage(&logits, 0.8).unwrap();
        assert!(k <= logits.len());
        assert!(k >= 1);
    }

    #[test]
    fn test_estimate_effective_vocab_size() {
        // Small k should return approximately k
        assert_eq!(estimate_effective_vocab_size(&[1.0; 100], 5).unwrap(), 5);
        
        // Large k should apply ratio
        let result = estimate_effective_vocab_size(&[1.0; 100], 60).unwrap();
        assert!(result < 60);
        assert!(result > 30);
        
        // k=0 should return full size
        assert_eq!(estimate_effective_vocab_size(&[1.0; 100], 0).unwrap(), 100);
        
        // k >= vocab_size should return full size
        assert_eq!(estimate_effective_vocab_size(&[1.0; 50], 100).unwrap(), 50);
    }

    #[test]
    fn test_validate_top_k_config() {
        // Valid configurations
        assert!(validate_top_k_config(50, 1000).is_ok());
        assert!(validate_top_k_config(0, 1000).is_ok());
        assert!(validate_top_k_config(MAX_TOP_K, 1000).is_ok());
        
        // Invalid: k exceeds maximum
        assert!(validate_top_k_config(MAX_TOP_K + 1, 1000).is_err());
        
        // Valid but unusual: k > vocab_size (should not error)
        assert!(validate_top_k_config(100, 50).is_ok());
    }

    #[test]
    fn test_invalid_inputs() {
        let context = ProcessingContext::default();
        
        // Invalid base_k
        assert!(perplexity_based_k(MAX_TOP_K + 1, 5.0, 5.0).is_err());
        assert!(temperature_adjusted_k(MAX_TOP_K + 1, 1.0, 1.0).is_err());
        assert!(adaptive_top_k(MAX_TOP_K + 1, &context, 0.5).is_err());
        
        // Invalid perplexity
        assert!(perplexity_based_k(50, -1.0, 5.0).is_err());
        assert!(perplexity_based_k(50, f32::NAN, 5.0).is_err());
        assert!(perplexity_based_k(50, 5.0, 0.0).is_err());
        
        // Invalid temperature  
        assert!(temperature_adjusted_k(50, -1.0, 1.0).is_err());
        assert!(temperature_adjusted_k(50, f32::INFINITY, 1.0).is_err());
        assert!(temperature_adjusted_k(50, 1.0, 0.0).is_err());
        
        // Invalid diversity factor
        assert!(adaptive_top_k(50, &context, -0.1).is_err());
        assert!(adaptive_top_k(50, &context, f32::NAN).is_err());
        
        // Invalid coverage target
        assert!(k_for_coverage(&[1.0; 5], -0.1).is_err());
        assert!(k_for_coverage(&[1.0; 5], 1.1).is_err());
    }
