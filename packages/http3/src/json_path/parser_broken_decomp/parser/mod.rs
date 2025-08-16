//! Expression parsing logic for JSONPath
//!
//! Decomposed recursive descent parser for JSONPath expressions and filter predicates
//! with proper operator precedence and selector handling.

pub mod core;
pub mod selectors;
pub mod filters;
pub mod primary;

// Re-export main parser
pub use core::JsonPathParser;