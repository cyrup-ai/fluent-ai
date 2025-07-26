use fluent_ai_candle::model::loading::recovery::*;
use fluent_ai_candle::model::loading::*;

use candle_core::TensorError;
    
    #[test]
    fn test_fail_fast_strategy() {
        let mut context = RecoveryContext::new(RecoveryStrategy::FailFast);
        let error = CandleError::TensorError(TensorError::UnexpectedDType);
        
        assert!(!context.should_recover(&error));
        let action = context.recovery_action(error);
        
        if let RecoveryAction::Fail(_) = action {
            // Expected
        } else {
            panic!("Expected Fail action");
        }
    }
    
    #[test]
    fn test_retry_strategy() {
        let mut context = RecoveryContext::new(RecoveryStrategy::Retry(3));
        let error = CandleError::TensorError(TensorError::UnexpectedDType);
        
        assert!(context.should_recover(&error));
        
        // First attempt
        let action = context.recovery_action(error);
        assert!(matches!(action, RecoveryAction::Retry));
        
        // Second attempt
        let error = CandleError::TensorError(TensorError::UnexpectedDType);
        let action = context.recovery_action(error);
        assert!(matches!(action, RecoveryAction::Retry));
        
        // Third attempt (last one)
        let error = CandleError::TensorError(TensorError::UnexpectedDType);
        let action = context.recovery_action(error);
        assert!(matches!(action, RecoveryAction::Retry));
        
        // Fourth attempt should fail
        let error = CandleError::TensorError(TensorError::UnexpectedDType);
        let action = context.recovery_action(error);
        assert!(matches!(action, RecoveryAction::Fail(_)));
    }
    
    #[test]
    fn test_cancellation() {
        let context = RecoveryContext::new(RecoveryStrategy::Retry(3));
        context.cancel();
        
        let error = CandleError::TensorError(TensorError::UnexpectedDType);
        assert!(!context.should_recover(&error));
        assert!(context.is_cancelled());
    }
