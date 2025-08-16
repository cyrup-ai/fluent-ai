//! Filter expression parsing with logical and comparison operators
//!
//! Handles parsing of filter expressions with proper operator precedence
//! including logical OR/AND and comparison operators.

use super::super::types::{FilterExpression, ComparisonOp, LogicalOp};
use super::super::tokenizer::Token;
use crate::json_path::error::JsonPathResult;
use super::core::JsonPathParser;

/// Parse filter expression (?(...))
pub(crate) fn parse_filter_expression(parser: &mut JsonPathParser) -> JsonPathResult<FilterExpression> {
    parse_logical_or(parser)
}

/// Parse logical OR expression
fn parse_logical_or(parser: &mut JsonPathParser) -> JsonPathResult<FilterExpression> {
    let mut left = parse_logical_and(parser)?;
    
    while matches!(parser.tokens.front(), Some(Token::LogicalOr)) {
        parser.tokens.pop_front();
        let right = parse_logical_and(parser)?;
        left = FilterExpression::Logical {
            left: Box::new(left),
            operator: LogicalOp::Or,
            right: Box::new(right),
        };
    }
    
    Ok(left)
}

/// Parse logical AND expression
fn parse_logical_and(parser: &mut JsonPathParser) -> JsonPathResult<FilterExpression> {
    let mut left = parse_equality(parser)?;
    
    while matches!(parser.tokens.front(), Some(Token::LogicalAnd)) {
        parser.tokens.pop_front();
        let right = parse_equality(parser)?;
        left = FilterExpression::Logical {
            left: Box::new(left),
            operator: LogicalOp::And,
            right: Box::new(right),
        };
    }
    
    Ok(left)
}

/// Parse equality expression
fn parse_equality(parser: &mut JsonPathParser) -> JsonPathResult<FilterExpression> {
    let mut left = parse_relational(parser)?;
    
    while let Some(token) = parser.tokens.front() {
        let op = match token {
            Token::Equal => ComparisonOp::Equal,
            Token::NotEqual => ComparisonOp::NotEqual,
            _ => break,
        };
        
        parser.tokens.pop_front();
        let right = parse_relational(parser)?;
        left = FilterExpression::Comparison {
            left: Box::new(left),
            operator: op,
            right: Box::new(right),
        };
    }
    
    Ok(left)
}

/// Parse relational expression
fn parse_relational(parser: &mut JsonPathParser) -> JsonPathResult<FilterExpression> {
    let mut left = parse_primary(parser)?;
    
    while let Some(token) = parser.tokens.front() {
        let op = match token {
            Token::Less => ComparisonOp::Less,
            Token::LessEq => ComparisonOp::LessEq,
            Token::Greater => ComparisonOp::Greater,
            Token::GreaterEq => ComparisonOp::GreaterEq,
            _ => break,
        };
        
        parser.tokens.pop_front();
        let right = parse_primary(parser)?;
        left = FilterExpression::Comparison {
            left: Box::new(left),
            operator: op,
            right: Box::new(right),
        };
    }
    
    Ok(left)
}

/// Parse primary expression (literals, @, function calls, etc.)
fn parse_primary(parser: &mut JsonPathParser) -> JsonPathResult<FilterExpression> {
    use super::primary;
    primary::parse_primary_expression(parser)
}