//! High-performance JSONPath expression parser and compiler
//!
//! This module provides compile-time optimization of JSONPath expressions into
//! efficient runtime selectors. Supports the full JSONPath specification with
//! zero-allocation execution paths.

// Re-export main types for backward compatibility
pub use crate::json_path::ast::{
    ComparisonOp, ComplexityMetrics, FilterExpression, FilterValue, JsonSelector, LogicalOp,
};
pub use crate::json_path::compiler::JsonPathParser;
pub use crate::json_path::expression::JsonPathExpression;
pub use crate::json_path::tokenizer::ExpressionParser;
pub use crate::json_path::tokens::{Token, TokenMatcher};
