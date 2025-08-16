//! Standard trait implementations for JsonPathError

use super::core::JsonPathError;

impl std::error::Error for JsonPathError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // No nested errors in our current implementation
        None
    }
}
