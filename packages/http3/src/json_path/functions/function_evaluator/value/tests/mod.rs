//! Test modules for value() function implementation
//!
//! Comprehensive test coverage organized by logical concerns:
//! - Mock evaluator utilities
//! - Argument validation tests
//! - Property access and nested object tests
//! - Current context expression tests
//! - Literal value and expression delegation tests
//! - Type conversion tests for different JSON types
//! - Edge cases including Unicode handling

pub mod mock_evaluator;
pub mod argument_validation;
pub mod property_access;
pub mod current_context;
pub mod literal_values;
pub mod type_conversion;
pub mod edge_cases;

// Re-export mock evaluator for use across test modules
pub use mock_evaluator::mock_evaluator;