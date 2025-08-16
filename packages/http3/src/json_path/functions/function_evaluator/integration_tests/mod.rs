//! Integration tests for the function evaluator
//!
//! This module contains comprehensive integration tests that verify the complete
//! function evaluation pipeline, including function dispatch, error handling,
//! and value conversion across different scenarios.

pub mod mock_evaluator;
pub mod function_dispatch;
pub mod error_handling;
pub mod value_conversion;
pub mod complex_data;
pub mod unicode_special;
pub mod edge_cases;
pub mod performance;

// Re-export test utilities for convenience
pub use mock_evaluator::mock_evaluator;

// Test module declarations
#[cfg(test)]
mod tests {
    // Import all test modules to ensure they are compiled and run
    use super::function_dispatch;
    use super::error_handling;
    use super::value_conversion;
    use super::complex_data;
    use super::unicode_special;
    use super::edge_cases;
    use super::performance;
}