//! Integration tests for the function evaluator
//!
//! This module contains comprehensive integration tests that verify the complete
//! function evaluation pipeline, including function dispatch, error handling,
//! and value conversion across different scenarios.

pub mod complex_data;
pub mod edge_cases;
pub mod error_handling;
pub mod function_dispatch;
pub mod mock_evaluator;
pub mod performance;
pub mod unicode_special;
pub mod value_conversion;

// Re-export test utilities for convenience
pub use mock_evaluator::mock_evaluator;

// Test module declarations
#[cfg(test)]
mod tests {
    // Import all test modules to ensure they are compiled and run
    use super::complex_data;
    use super::edge_cases;
    use super::error_handling;
    use super::function_dispatch;
    use super::performance;
    use super::unicode_special;
    use super::value_conversion;
}
