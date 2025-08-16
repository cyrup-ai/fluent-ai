//! Comprehensive test suite for JSONPath core evaluator
//!
//! Modular organization of tests covering RFC 9535 compliance,
//! basic operations, advanced features, and edge cases.

pub mod basic_selectors;
pub mod array_operations;
pub mod recursive_descent;
pub mod filter_expressions;
pub mod edge_cases_debug;
pub mod rfc_compliance;

// Re-export all test modules for convenience
pub use basic_selectors::*;
pub use array_operations::*;
pub use recursive_descent::*;
pub use filter_expressions::*;
pub use edge_cases_debug::*;
pub use rfc_compliance::*;