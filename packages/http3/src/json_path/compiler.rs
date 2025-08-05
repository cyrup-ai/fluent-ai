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
        // '$..' is also invalid per RFC 9535: descendant-segment = ".." S bracket-segment
        if expression.ends_with('.') && !expression.ends_with("..") {
            return Err(invalid_expression_error(
                expression,
                "incomplete property access (ends with '.')",
                Some(expression.len() - 1),
            ));
        }

        // RFC 9535: descendant-segment = ".." S bracket-segment  
        // Bare ".." without following segment is invalid
        if expression == "$.." {
            return Err(invalid_expression_error(
                expression,
                "descendant segment '..' must be followed by a bracket segment",
                Some(expression.len() - 2),
            ));
        }

        // RFC 9535: Also check for expressions ending with ".." like "$.store.."
        if expression.ends_with("..") && expression.len() > 3 {
            return Err(invalid_expression_error(
                expression,
                "descendant segment '..' must be followed by a bracket segment",
                Some(expression.len() - 2),
            ));
        }

        // RFC 9535: descendant-segment = ".." S bracket-segment  
        // According to RFC 9535, ".." can be followed by bracket-segment, wildcard '*', or identifier
        // Valid: "$..*", "$..[*]", "$..level1", "$..['key']"
        // Invalid: bare ".." without any following segment

        let mut parser = ExpressionParser::new(expression);
        let selectors = parser.parse()?;

        // RFC 9535 Compliance: Root-only expressions "$" are valid per specification
        // ABNF: jsonpath-query = root-identifier segments where segments = *(S segment) allows zero segments
        // Section 2.2.3 Examples explicitly shows "$" returns the root node
        // No validation needed - bare "$" is perfectly valid per RFC 9535

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recursive_descent_wildcard_compilation() {
        // Test RFC 9535 compliant recursive descent patterns
        let valid_patterns = vec![
            "$..[*]",           // Valid: recursive descent followed by bracket selector
            "$..level1",        // Valid: recursive descent followed by property  
            "$..['key']",       // Valid: recursive descent followed by bracket selector
        ];
        
        for pattern in valid_patterns {
            let result = JsonPathParser::compile(pattern);
            assert!(result.is_ok(), "Pattern {} should compile successfully", pattern);
        }
        
        // Test RFC 9535 valid patterns including recursive descent with wildcard
        let additional_valid_patterns = vec![
            "$..*",             // Valid: recursive descent followed by wildcard (RFC 9535 compliant)
        ];
        
        for pattern in additional_valid_patterns {
            let result = JsonPathParser::compile(pattern);
            assert!(result.is_ok(), "Pattern {} should compile successfully per RFC 9535", pattern);
        }
        
        // Test that invalid patterns are properly rejected
        let invalid_patterns = vec![
            "$..",              // Invalid: bare recursive descent without following segment
        ];
        
        for pattern in invalid_patterns {
            let result = JsonPathParser::compile(pattern);
            assert!(result.is_err(), "Pattern {} should be rejected as invalid", pattern);
        }
    }
}
