//! Primary expression parsing for literals and function calls
//!
//! Handles parsing of primary expressions including literals, current context (@),
//! property paths, function calls, and parenthesized expressions.

use super::super::types::{FilterExpression, FilterValue};
use super::super::tokenizer::Token;
use crate::json_path::error::{invalid_expression_error, JsonPathResult};
use super::core::JsonPathParser;

/// Parse primary expression (literals, @, function calls, etc.)
pub(crate) fn parse_primary_expression(parser: &mut JsonPathParser) -> JsonPathResult<FilterExpression> {
    match parser.tokens.front() {
        Some(Token::At) => {
            parser.tokens.pop_front();
            if matches!(parser.tokens.front(), Some(Token::Dot)) {
                parser.tokens.pop_front();
                parse_property_path(parser)
            } else {
                Ok(FilterExpression::Current)
            }
        }
        
        Some(Token::String(value)) => {
            let value = value.clone();
            parser.tokens.pop_front();
            Ok(FilterExpression::Literal {
                value: FilterValue::String(value),
            })
        }
        
        Some(Token::Integer(value)) => {
            let value = *value;
            parser.tokens.pop_front();
            Ok(FilterExpression::Literal {
                value: FilterValue::Integer(value),
            })
        }
        
        Some(Token::Number(value)) => {
            let value = *value;
            parser.tokens.pop_front();
            Ok(FilterExpression::Literal {
                value: FilterValue::Number(value),
            })
        }
        
        Some(Token::Boolean(value)) => {
            let value = *value;
            parser.tokens.pop_front();
            Ok(FilterExpression::Literal {
                value: FilterValue::Boolean(value),
            })
        }
        
        Some(Token::Null) => {
            parser.tokens.pop_front();
            Ok(FilterExpression::Literal {
                value: FilterValue::Null,
            })
        }
        
        Some(Token::Identifier(name)) => {
            let name = name.clone();
            parser.tokens.pop_front();
            
            if matches!(parser.tokens.front(), Some(Token::LeftParen)) {
                parser.tokens.pop_front();
                let args = parse_function_arguments(parser)?;
                parser.expect_token(Token::RightParen)?;
                Ok(FilterExpression::Function { name, args })
            } else {
                Err(invalid_expression_error(
                    &parser.input,
                    &format!("unexpected identifier: {}", name),
                    Some(parser.position),
                ))
            }
        }
        
        Some(Token::LeftParen) => {
            parser.tokens.pop_front();
            use super::filters;
            let expr = filters::parse_filter_expression(parser)?;
            parser.expect_token(Token::RightParen)?;
            Ok(expr)
        }
        
        _ => Err(invalid_expression_error(
            &parser.input,
            "expected literal, @, function call, or parenthesized expression",
            Some(parser.position),
        )),
    }
}

/// Parse property path (@.property.nested)
fn parse_property_path(parser: &mut JsonPathParser) -> JsonPathResult<FilterExpression> {
    let mut path = Vec::new();
    
    if let Some(Token::Identifier(name)) = parser.tokens.front() {
        path.push(name.clone());
        parser.tokens.pop_front();
        
        while matches!(parser.tokens.front(), Some(Token::Dot)) {
            parser.tokens.pop_front();
            if let Some(Token::Identifier(name)) = parser.tokens.front() {
                path.push(name.clone());
                parser.tokens.pop_front();
            } else {
                break;
            }
        }
    }
    
    if path.is_empty() {
        return Err(invalid_expression_error(
            &parser.input,
            "expected property name after '@.'",
            Some(parser.position),
        ));
    }
    
    Ok(FilterExpression::Property { path })
}

/// Parse function arguments
fn parse_function_arguments(parser: &mut JsonPathParser) -> JsonPathResult<Vec<FilterExpression>> {
    let mut args = Vec::new();
    
    // Check for empty argument list
    if matches!(parser.tokens.front(), Some(Token::RightParen)) {
        return Ok(args);
    }
    
    // Parse first argument
    use super::filters;
    args.push(filters::parse_filter_expression(parser)?);
    
    // Parse remaining arguments
    while matches!(parser.tokens.front(), Some(Token::Comma)) {
        parser.tokens.pop_front();
        args.push(filters::parse_filter_expression(parser)?);
    }
    
    Ok(args)
}