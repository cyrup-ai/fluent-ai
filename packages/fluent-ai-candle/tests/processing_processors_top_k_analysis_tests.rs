use fluent_ai_candle::processing::processors::top_k::analysis::*;
use fluent_ai_candle::processing::processors::top_k::*;

#[test]
    fn test_k_for_coverage() {
        let logits = vec![2.0, 1.0, 0.0, -1.0, -2.0];
        let k = k_for_coverage(&logits, 0.8).unwrap();
        assert!(k <= 5);
        assert!(k >= 1);
    }

    #[test]
    fn test_estimate_effective_vocab_size() {
        assert_eq!(estimate_effective_vocab_size(&[1.0; 100], 5).unwrap(), 5);
        assert_eq!(estimate_effective_vocab_size(&[1.0; 100], 30).unwrap(), 24);
        assert_eq!(estimate_effective_vocab_size(&[1.0; 100], 0).unwrap(), 100);
    }

    #[test]
    fn test_entropy_based_coverage() {
        let logits = vec![1.0, 0.5, 0.0, -0.5, -1.0];
        let k = entropy_based_coverage(&logits, 0.9).unwrap();
        assert!(k <= 5);
        assert!(k >= 1);
    }
