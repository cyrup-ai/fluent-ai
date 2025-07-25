//! Error handling macros for convenient error creation with context

use super::error_context::{CandleErrorWithContext, ErrorContext};

/// Macro for creating errors with context
#[macro_export]
macro_rules! candle_error {
    ($error:expr, $operation:expr) => {
        CandleErrorWithContext {
            error: $error,
            context: ErrorContext::new($operation),
        }
    };
    ($error:expr, $operation:expr, model = $model:expr) => {
        CandleErrorWithContext {
            error: $error,
            context: ErrorContext::new($operation).with_model_name($model),
        }
    };
    ($error:expr, $operation:expr, device = $device:expr) => {
        CandleErrorWithContext {
            error: $error,
            context: ErrorContext::new($operation).with_device($device),
        }
    };
    ($error:expr, $operation:expr, context = $context:expr) => {
        CandleErrorWithContext {
            error: $error,
            context: ErrorContext::new($operation).with_context($context),
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CandleError;

    #[test]
    fn test_candle_error_macro() {
        let error = CandleError::ModelLoadingError("test".to_string());
        let error_with_context = candle_error!(error, "test_operation");
        assert_eq!(error_with_context.context.operation, "test_operation");
    }
}