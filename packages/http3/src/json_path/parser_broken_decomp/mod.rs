//! JSONPath Expression Compilation and Core Parsing Logic
//!
//! This module provides compile-time optimization of JSONPath expressions into
//! efficient runtime selectors. Focuses on tokenization, parsing, and selector
//! compilation with zero-allocation execution paths.
//! 
//! # Modules
//! 
//! - [`types`] - Core data structures and enums for JSONPath expressions
//! - [`tokenizer`] - Tokenization logic for JSONPath syntax
//! - [`parser`] - Expression parsing logic using recursive descent
//! - [`compiler`] - Main compilation entry point and public API

pub mod types;
pub mod tokenizer;
pub mod parser;
pub mod compiler;

// Re-export the main types and functions for backward compatibility
pub use types::{JsonPathExpression, ComplexityMetrics, JsonSelector, FilterExpression, FilterValue, ComparisonOp, LogicalOp};
pub use compiler::compile;