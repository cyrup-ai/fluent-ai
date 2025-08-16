//! Core types and structures for JSONPath evaluation
//!
//! Defines the main CoreJsonPathEvaluator struct and associated types.

use serde_json::Value;
use crate::json_path::error::JsonPathError;
use crate::json_path::parser::{JsonPathParser, JsonSelector};

/// Result type for JSONPath operations
pub type JsonPathResult<T> = Result<T, JsonPathError>;

/// Core JSONPath evaluator that works with parsed JSON according to RFC 9535
///
/// This evaluator supports the complete JSONPath specification with optimized performance
/// and protection against pathological inputs.
#[derive(Debug, Clone)]
pub struct CoreJsonPathEvaluator {
    /// The parsed selectors from the JSONPath expression
    pub(crate) selectors: Vec<JsonSelector>,
    /// The original expression string for debugging and error reporting
    pub(crate) expression: String,
}

impl CoreJsonPathEvaluator {
    /// Create new evaluator with JSONPath expression
    pub fn new(expression: &str) -> JsonPathResult<Self> {
        // Compile the expression to get the parsed selectors
        let compiled = JsonPathParser::compile(expression)?;
        let selectors = compiled.selectors().to_vec();

        Ok(Self {
            selectors,
            expression: expression.to_string(),
        })
    }

    /// Get the original expression string
    pub fn expression(&self) -> &str {
        &self.expression
    }

    /// Get the parsed selectors
    pub fn selectors(&self) -> &[JsonSelector] {
        &self.selectors
    }

    /// Create a temporary evaluator instance for internal use
    pub(crate) fn create_temp_evaluator(expression: &str) -> JsonPathResult<Self> {
        let compiled = JsonPathParser::compile(expression)?;
        Ok(Self {
            selectors: compiled.selectors().to_vec(),
            expression: expression.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluator_creation() {
        let evaluator = CoreJsonPathEvaluator::new("$.store.book[0]").unwrap();
        assert_eq!(evaluator.expression(), "$.store.book[0]");
        assert!(!evaluator.selectors().is_empty());
    }

    #[test]
    fn test_evaluator_invalid_expression() {
        let result = CoreJsonPathEvaluator::new("$.[invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_evaluator_clone() {
        let evaluator = CoreJsonPathEvaluator::new("$.test").unwrap();
        let cloned = evaluator.clone();
        assert_eq!(evaluator.expression(), cloned.expression());
        assert_eq!(evaluator.selectors().len(), cloned.selectors().len());
    }

    #[test]
    fn test_temp_evaluator_creation() {
        let temp = CoreJsonPathEvaluator::create_temp_evaluator("$.temp").unwrap();
        assert_eq!(temp.expression(), "$.temp");
    }

    #[test]
    fn test_evaluator_debug() {
        let evaluator = CoreJsonPathEvaluator::new("$.debug").unwrap();
        let debug_str = format!("{:?}", evaluator);
        assert!(debug_str.contains("CoreJsonPathEvaluator"));
        assert!(debug_str.contains("$.debug"));
    }
}