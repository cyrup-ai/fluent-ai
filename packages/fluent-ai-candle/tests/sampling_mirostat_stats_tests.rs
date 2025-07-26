use fluent_ai_candle::sampling::mirostat::stats::*;
use fluent_ai_candle::sampling::mirostat::*;

#[test]
    fn test_stats_stability() {
        let mut stats = MirostatStats::default();
        assert!(!stats.is_stable()); // Not enough tokens
        
        stats.tokens_processed = 20;
        stats.perplexity_variance = 0.5;
        assert!(stats.is_stable()); // Low variance + enough tokens
        
        stats.perplexity_variance = 2.0;
        assert!(!stats.is_stable()); // High variance
    }

    #[test]
    fn test_convergence_ratio() {
        let mut stats = MirostatStats::default();
        stats.current_tau = 5.0; // Same as default target
        assert_eq!(stats.convergence_ratio(), 1.0);
        
        stats.current_tau = 2.5; // Half of target
        assert_eq!(stats.convergence_ratio(), 0.5);
        
        stats.current_tau = 10.0; // Double target
        assert_eq!(stats.convergence_ratio(), 0.5);
    }

    #[test]
    fn test_tau_deviation() {
        let mut stats = MirostatStats::default();
        stats.current_tau = 5.0; // Same as target
        assert_eq!(stats.tau_deviation_percent(), 0.0);
        
        stats.current_tau = 5.25; // 5% above
        assert!((stats.tau_deviation_percent() - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_quality_score() {
        let mut stats = MirostatStats::default();
        stats.tokens_processed = 20;
        stats.perplexity_variance = 0.5;
        stats.avg_processing_time_nanos = 5000.0;
        
        let score = stats.quality_score();
        assert!(score > 0.8 && score <= 1.0);
    }

    #[test]
    fn test_tokens_per_second() {
        let mut stats = MirostatStats::default();
        stats.avg_processing_time_nanos = 1_000_000.0; // 1ms per token
        assert_eq!(stats.tokens_per_second(), 1000.0);
        
        stats.avg_processing_time_nanos = 0.0;
        assert_eq!(stats.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_summary_format() {
        let stats = MirostatStats::default();
        let summary = stats.summary();
        assert!(summary.contains("Mirostat v1"));
        assert!(summary.contains("τ="));
        assert!(summary.contains("perplexity="));
        assert!(summary.contains("tokens"));
    }
