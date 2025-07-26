use fluent_ai_candle::sampling::mirostat::perplexity::*;
use fluent_ai_candle::sampling::mirostat::*;

#[test]
    fn test_perplexity_state_creation() {
        let state = PerplexityState::new(0.1);
        assert_eq!(state.current_perplexity(), 1.0);
        assert_eq!(state.sample_count, 0);
        assert_eq!(state.history.len(), 0);
        assert_eq!(state.ema_alpha, 0.1);
    }

    #[test]
    fn test_perplexity_variance() {
        let mut state = PerplexityState::new(0.1);
        assert_eq!(state.variance(), 0.0);
        
        #[allow(deprecated)]
        {
            state.add_sample(1.0);
            state.add_sample(2.0);
        }
        
        assert!(state.variance() > 0.0);
    }

    #[test]
    fn test_perplexity_reset() {
        let mut state = PerplexityState::new(0.1);
        
        #[allow(deprecated)]
        {
            state.add_sample(1.5);
        }
        
        assert!(state.sample_count() > 0);
        
        state.reset();
        assert_eq!(state.sample_count(), 0);
        assert_eq!(state.current_perplexity(), 1.0);
    }
