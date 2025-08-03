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

// Re-export new RFC 9535 implementation modules
pub use crate::json_path::type_system::{FunctionType, TypedValue, FunctionSignature, TypeSystem};
pub use crate::json_path::normalized_paths::{NormalizedPath, PathSegment, NormalizedPathProcessor};
pub use crate::json_path::null_semantics::{PropertyAccessResult, NullSemantics};
pub use crate::json_path::safe_parsing::{SafeParsingContext, Utf8Handler, Utf8RecoveryStrategy, SafeStringBuffer};
