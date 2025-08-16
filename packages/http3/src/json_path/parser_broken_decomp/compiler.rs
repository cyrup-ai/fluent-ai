//! Main compilation entry point for JSONPath expressions
//! 
//! Provides the public API for compiling JSONPath expressions into optimized selectors.

use super::types::JsonPathExpression;
use super::tokenizer::JsonPathTokenizer;
use super::parser::JsonPathParser;
use crate::json_path::error::JsonPathResult;

/// Compile a JSONPath expression string into an optimized expression
pub fn compile(expression: &str) -> JsonPathResult<JsonPathExpression> {
    // Tokenize the input
    let mut tokenizer = JsonPathTokenizer::new(expression.to_string());
    tokenizer.tokenize()?;
    
    // Parse the tokens into selectors
    let mut parser = JsonPathParser::from_tokenizer(tokenizer);
    let selectors = parser.parse()?;
    
    // Determine if this is an array stream expression
    let is_array_stream = selectors.len() >= 2 && 
        selectors.iter().any(|s| matches!(s, super::types::JsonSelector::Wildcard)) ||
        selectors.iter().any(|s| matches!(s, super::types::JsonSelector::Index { .. })) ||
        selectors.iter().any(|s| matches!(s, super::types::JsonSelector::Slice { .. }));
    
    Ok(JsonPathExpression {
        selectors,
        original: expression.to_string(),
        is_array_stream,
    })
}