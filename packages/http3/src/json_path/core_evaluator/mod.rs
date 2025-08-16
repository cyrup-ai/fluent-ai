//! Core JSONPath evaluator module
//!
//! This module provides the main JSONPath evaluation functionality decomposed into
//! logical submodules for maintainability and clarity.

pub mod core_evaluator;
pub mod timeout_evaluation;
pub mod selector_application;
pub mod filter_evaluation;
pub mod property_operations;
pub mod descendant_operations;

// Re-export the main types and functions
pub use core_evaluator::{CoreJsonPathEvaluator, JsonPathResult};