//! Selector parsing for dot and bracket notation
//!
//! Handles parsing of property selectors, index selectors, slice selectors,
//! wildcard selectors, and filter selectors in JSONPath expressions.

use super::super::types::{JsonSelector, FilterExpression};
use super::super::tokenizer::Token;
use crate::json_path::error::{invalid_expression_error, JsonPathResult};
use super::core::JsonPathParser;

/// Parse dot notation selector (.property, .*, etc.)
pub(crate) fn parse_dot_selector(parser: &mut JsonPathParser) -> JsonPathResult<JsonSelector> {
    match parser.tokens.front() {
        Some(Token::Identifier(name)) => {
            let name = name.clone();
            parser.tokens.pop_front();
            Ok(JsonSelector::Property { name })
        }
        Some(Token::Star) => {
            parser.tokens.pop_front();
            Ok(JsonSelector::Wildcard)
        }
        _ => Err(invalid_expression_error(
            &parser.input,
            "expected property name or '*' after '.'",
            Some(parser.position),
        )),
    }
}

/// Parse bracket notation selector ([index], [start:end], ['property'], etc.)
pub(crate) fn parse_bracket_selector(parser: &mut JsonPathParser) -> JsonPathResult<JsonSelector> {
    match parser.tokens.front() {
        Some(Token::Integer(index)) => {
            let index = *index as i32;
            parser.tokens.pop_front();
            
            // Check for slice notation
            if matches!(parser.tokens.front(), Some(Token::Colon)) {
                parser.tokens.pop_front();
                let end = if let Some(Token::Integer(end_val)) = parser.tokens.front() {
                    let end_val = *end_val as i32;
                    parser.tokens.pop_front();
                    Some(end_val)
                } else {
                    None
                };
                
                let step = if matches!(parser.tokens.front(), Some(Token::Colon)) {
                    parser.tokens.pop_front();
                    if let Some(Token::Integer(step_val)) = parser.tokens.front() {
                        let step_val = *step_val as i32;
                        parser.tokens.pop_front();
                        Some(step_val)
                    } else {
                        None
                    }
                } else {
                    None
                };
                
                parser.expect_token(Token::RightBracket)?;
                Ok(JsonSelector::Slice {
                    start: Some(index),
                    end,
                    step,
                })
            } else {
                parser.expect_token(Token::RightBracket)?;
                Ok(JsonSelector::Index {
                    index,
                    from_end: index < 0,
                })
            }
        }
        Some(Token::Star) => {
            parser.tokens.pop_front();
            parser.expect_token(Token::RightBracket)?;
            Ok(JsonSelector::Wildcard)
        }
        
        Some(Token::Question) => {
            parser.tokens.pop_front();
            use super::filters;
            let filter_expr = filters::parse_filter_expression(parser)?;
            parser.expect_token(Token::RightBracket)?;
            Ok(JsonSelector::Filter { expression: filter_expr })
        }
        
        Some(Token::String(name)) => {
            let name = name.clone();
            parser.tokens.pop_front();
            parser.expect_token(Token::RightBracket)?;
            Ok(JsonSelector::Property { name })
        }
        
        _ => Err(invalid_expression_error(
            &parser.input,
            "expected index, string, or filter in bracket expression",
            Some(parser.position),
        )),
    }
}