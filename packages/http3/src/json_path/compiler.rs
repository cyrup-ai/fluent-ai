//! JSONPath expression compiler and entry point
//!
//! Provides the main parser interface for compiling JSONPath expressions
//! into optimized AST structures.

use crate::json_path::{
    ast::JsonSelector,
    error::{JsonPathResult, invalid_expression_error},
    expression::JsonPathExpression,
    tokenizer::ExpressionParser,
};

/// JSONPath expression parser and compiler
pub struct JsonPathParser;

impl JsonPathParser {
    /// Compile JSONPath expression into optimized selector chain
    ///
    /// # Arguments
    ///
    /// * `expression` - JSONPath expression string (e.g., "$.data[*]", "$.items[?(@.active)]")
    ///
    /// # Returns
    ///
    /// Compiled `JsonPathExpression` optimized for streaming evaluation.
    ///
    /// # Errors
    ///
    /// Returns `JsonPathError::InvalidExpression` for syntax errors or unsupported features.
    ///
    /// # Performance
    ///
    /// Expression compilation is performed once at construction time. Runtime evaluation
    /// uses pre-compiled selectors for maximum performance.
    pub fn compile(expression: &str) -> JsonPathResult<JsonPathExpression> {
        if expression.is_empty() {
            return Err(invalid_expression_error(
                expression,
                "empty expression not allowed",
                Some(0),
            ));
        }

        // RFC 9535 Compliance: JSONPath expressions must start with '$'
        if !expression.starts_with('$') {
            // Special case: provide specific error for @ outside filter context
            if expression.starts_with('@') {
                return Err(invalid_expression_error(
                    expression,
                    "current node identifier '@' is only valid within filter expressions [?...]",
                    Some(0),
                ));
            }
            return Err(invalid_expression_error(
                expression,
                "JSONPath expressions must start with '$'",
                Some(0),
            ));
        }

        // RFC 9535 Compliance: JSONPath expressions cannot end with '.' unless it's recursive descent
        // '$.' is invalid (incomplete property access)
        // '$..' is valid (recursive descent)
        if expression.ends_with('.') && !expression.ends_with("..") {
            return Err(invalid_expression_error(
                expression,
                "incomplete property access (ends with '.')",
                Some(expression.len() - 1),
            ));
        }

        let mut parser = ExpressionParser::new(expression);
        let selectors = parser.parse()?;

        // RFC 9535 Compliance: Root-only expressions "$" are NOT valid per RFC 9535
        // ABNF: jsonpath-query = root-identifier segments
        //       segments = *(S segment)  ; zero or more segments per RFC 9535
        // Therefore "$" (root-only) is VALID per RFC 9535

        // Determine if this is an array streaming expression
        let is_array_stream = selectors.iter().any(|s| {
            matches!(
                s,
                JsonSelector::Wildcard | JsonSelector::Slice { .. } | JsonSelector::Filter { .. }
            )
        });

        Ok(JsonPathExpression::new(
            selectors,
            expression.to_string(),
            is_array_stream,
        ))
    }

    /// Validate JSONPath expression syntax without compilation
    ///
    /// Faster than full compilation when only validation is needed.
    pub fn validate(expression: &str) -> JsonPathResult<()> {
        Self::compile(expression).map(|_| ())
    }
}
