use fluent_ai_candle::error::macros::*;
use fluent_ai_candle::error::*;

use crate::error::CandleError;

    #[test]
    fn test_candle_error_macro() {
        let error = CandleError::ModelLoadingError("test".to_string());
        let error_with_context = candle_error!(error, "test_operation");
        assert_eq!(error_with_context.context.operation, "test_operation");
    }
