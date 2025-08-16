//! Core JsonPathExpression structure and basic methods
//!
//! Provides the main JSONPath expression struct with basic accessors
//! and utility methods for compiled JSONPath expressions.

use crate::json_path::ast::JsonSelector;

/// Compiled JSONPath expression optimized for streaming evaluation
#[derive(Debug, Clone)]
pub struct JsonPathExpression {
    /// Optimized selector chain for runtime execution
    selectors: Vec<JsonSelector>,
    /// Original expression string for debugging
    original: String,
    /// Whether this expression targets an array for streaming
    is_array_stream: bool,
}

impl JsonPathExpression {
    /// Create new JsonPathExpression
    #[inline]
    pub fn new(selectors: Vec<JsonSelector>, original: String, is_array_stream: bool) -> Self {
        Self {
            selectors,
            original,
            is_array_stream,
        }
    }

    /// Get original JSONPath expression string
    #[inline]
    pub fn original(&self) -> &str {
        &self.original
    }

    /// Get original JSONPath expression string (alias for original)
    #[inline]
    pub fn as_string(&self) -> String {
        self.original.clone()
    }

    /// Check if this expression targets array elements for streaming
    #[inline]
    pub fn is_array_stream(&self) -> bool {
        self.is_array_stream
    }

    /// Get compiled selector chain
    #[inline]
    pub fn selectors(&self) -> &[JsonSelector] {
        &self.selectors
    }

    /// Check if expression has recursive descent
    #[inline]
    pub fn has_recursive_descent(&self) -> bool {
        self.selectors
            .iter()
            .any(|s| matches!(s, JsonSelector::RecursiveDescent))
    }

    /// Get the starting position of recursive descent in selector chain
    #[inline]
    pub fn recursive_descent_start(&self) -> Option<usize> {
        self.selectors
            .iter()
            .position(|s| matches!(s, JsonSelector::RecursiveDescent))
    }

    /// Get root selector (first non-root selector in expression)
    ///
    /// Returns the first meaningful selector after the root ($) identifier.
    /// This is commonly used to determine the root navigation behavior.
    #[inline]
    pub fn root_selector(&self) -> Option<&JsonSelector> {
        self.selectors
            .iter()
            .find(|selector| !matches!(selector, JsonSelector::Root))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_expression() -> JsonPathExpression {
        JsonPathExpression::new(
            vec![JsonSelector::Root, JsonSelector::Wildcard],
            "$..*".to_string(),
            false,
        )
    }

    #[test]
    fn test_expression_creation() {
        let expr = create_test_expression();
        assert_eq!(expr.original(), "$..*");
        assert_eq!(expr.as_string(), "$..*");
        assert!(!expr.is_array_stream());
        assert_eq!(expr.selectors().len(), 2);
    }

    #[test]
    fn test_has_recursive_descent() {
        let expr_with_recursive = JsonPathExpression::new(
            vec![JsonSelector::Root, JsonSelector::RecursiveDescent],
            "$..".to_string(),
            false,
        );
        assert!(expr_with_recursive.has_recursive_descent());

        let expr_without_recursive = JsonPathExpression::new(
            vec![JsonSelector::Root, JsonSelector::Wildcard],
            "$.*".to_string(),
            false,
        );
        assert!(!expr_without_recursive.has_recursive_descent());
    }

    #[test]
    fn test_recursive_descent_start() {
        let expr = JsonPathExpression::new(
            vec![JsonSelector::Root, JsonSelector::RecursiveDescent],
            "$..".to_string(),
            false,
        );
        assert_eq!(expr.recursive_descent_start(), Some(1));
    }

    #[test]
    fn test_root_selector() {
        let expr = create_test_expression();
        match expr.root_selector() {
            Some(JsonSelector::Wildcard) => {
                // Expected
            }
            _ => panic!("Expected Wildcard selector"),
        }
    }
}
