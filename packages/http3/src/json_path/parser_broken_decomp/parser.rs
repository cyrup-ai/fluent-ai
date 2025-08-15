//! Expression parsing logic for JSONPath
//! 
//! Provides recursive descent parsing for JSONPath expressions and filter predicates.

use super::types::{JsonSelector, FilterExpression, FilterValue, ComparisonOp, LogicalOp};
use super::tokenizer::{Token, JsonPathTokenizer};
use crate::json_path::error::{invalid_expression_error, JsonPathResult};
use std::collections::VecDeque;

/// JSONPath expression parser
#[derive(Debug)]
pub struct JsonPathParser {
    input: String,
    position: usize,
    pub(super) tokens: VecDeque<Token>,
}

impl JsonPathParser {
    /// Create a new parser from tokenizer
    pub fn from_tokenizer(tokenizer: JsonPathTokenizer) -> Self {
        Self {
            input: tokenizer.input,
            position: tokenizer.position,
            tokens: tokenizer.tokens,
        }
    }
    
    /// Parse the complete JSONPath expression
    pub fn parse(&mut self) -> JsonPathResult<Vec<JsonSelector>> {
        let mut selectors = Vec::new();
        
        // Expect root selector ($)
        if !matches!(self.tokens.front(), Some(Token::Dollar)) {
            return Err(invalid_expression_error(
                &self.input,
                "JSONPath expression must start with '$'",
                Some(self.position),
            ));
        }
        self.tokens.pop_front();
        selectors.push(JsonSelector::Root);
        
        // Parse remaining selectors
        while !self.tokens.is_empty() {
            selectors.push(self.parse_selector()?);
        }
        
        Ok(selectors)
    }    /// Parse a single selector
    fn parse_selector(&mut self) -> JsonPathResult<JsonSelector> {
        match self.tokens.front() {
            Some(Token::Dot) => {
                self.tokens.pop_front();
                self.parse_dot_selector()
            }
            Some(Token::DotDot) => {
                self.tokens.pop_front();
                Ok(JsonSelector::RecursiveDescent)
            }
            Some(Token::LeftBracket) => {
                self.tokens.pop_front();
                self.parse_bracket_selector()
            }
            _ => Err(invalid_expression_error(
                &self.input,
                "expected '.', '..' or '[' after selector",
                Some(self.position),
            )),
        }
    }
    
    /// Parse dot notation selector (.property, .*, etc.)
    fn parse_dot_selector(&mut self) -> JsonPathResult<JsonSelector> {
        match self.tokens.front() {
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                self.tokens.pop_front();
                Ok(JsonSelector::Property { name })
            }
            Some(Token::Star) => {
                self.tokens.pop_front();
                Ok(JsonSelector::Wildcard)
            }
            _ => Err(invalid_expression_error(
                &self.input,
                "expected property name or '*' after '.'",
                Some(self.position),
            )),
        }
    }
    
    /// Parse bracket notation selector ([index], [start:end], ['property'], etc.)
    fn parse_bracket_selector(&mut self) -> JsonPathResult<JsonSelector> {
        match self.tokens.front() {
            Some(Token::Integer(index)) => {
                let index = *index as i32;
                self.tokens.pop_front();
                
                // Check for slice notation
                if matches!(self.tokens.front(), Some(Token::Colon)) {
                    self.tokens.pop_front();
                    let end = if let Some(Token::Integer(end_val)) = self.tokens.front() {
                        let end_val = *end_val as i32;
                        self.tokens.pop_front();
                        Some(end_val)
                    } else {
                        None
                    };
                    
                    let step = if matches!(self.tokens.front(), Some(Token::Colon)) {
                        self.tokens.pop_front();
                        if let Some(Token::Integer(step_val)) = self.tokens.front() {
                            let step_val = *step_val as i32;
                            self.tokens.pop_front();
                            Some(step_val)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    
                    self.expect_token(Token::RightBracket)?;
                    Ok(JsonSelector::Slice {
                        start: Some(index),
                        end,
                        step,
                    })
                } else {
                    self.expect_token(Token::RightBracket)?;
                    Ok(JsonSelector::Index {
                        index,
                        from_end: index < 0,
                    })
                }
            }            Some(Token::Star) => {
                self.tokens.pop_front();
                self.expect_token(Token::RightBracket)?;
                Ok(JsonSelector::Wildcard)
            }
            
            Some(Token::Question) => {
                self.tokens.pop_front();
                let filter_expr = self.parse_filter_expression()?;
                self.expect_token(Token::RightBracket)?;
                Ok(JsonSelector::Filter { expression: filter_expr })
            }
            
            Some(Token::String(name)) => {
                let name = name.clone();
                self.tokens.pop_front();
                self.expect_token(Token::RightBracket)?;
                Ok(JsonSelector::Property { name })
            }
            
            _ => Err(invalid_expression_error(
                &self.input,
                "expected index, string, or filter in bracket expression",
                Some(self.position),
            )),
        }
    }
    
    /// Parse filter expression (?(...))
    fn parse_filter_expression(&mut self) -> JsonPathResult<FilterExpression> {
        self.parse_logical_or()
    }
    
    /// Parse logical OR expression
    fn parse_logical_or(&mut self) -> JsonPathResult<FilterExpression> {
        let mut left = self.parse_logical_and()?;
        
        while matches!(self.tokens.front(), Some(Token::LogicalOr)) {
            self.tokens.pop_front();
            let right = self.parse_logical_and()?;
            left = FilterExpression::Logical {
                left: Box::new(left),
                operator: LogicalOp::Or,
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }    /// Parse logical AND expression
    fn parse_logical_and(&mut self) -> JsonPathResult<FilterExpression> {
        let mut left = self.parse_equality()?;
        
        while matches!(self.tokens.front(), Some(Token::LogicalAnd)) {
            self.tokens.pop_front();
            let right = self.parse_equality()?;
            left = FilterExpression::Logical {
                left: Box::new(left),
                operator: LogicalOp::And,
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }
    
    /// Parse equality expression
    fn parse_equality(&mut self) -> JsonPathResult<FilterExpression> {
        let mut left = self.parse_relational()?;
        
        while let Some(token) = self.tokens.front() {
            let op = match token {
                Token::Equal => ComparisonOp::Equal,
                Token::NotEqual => ComparisonOp::NotEqual,
                _ => break,
            };
            
            self.tokens.pop_front();
            let right = self.parse_relational()?;
            left = FilterExpression::Comparison {
                left: Box::new(left),
                operator: op,
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }
    
    /// Parse relational expression
    fn parse_relational(&mut self) -> JsonPathResult<FilterExpression> {
        let mut left = self.parse_primary()?;
        
        while let Some(token) = self.tokens.front() {
            let op = match token {
                Token::Less => ComparisonOp::Less,
                Token::LessEq => ComparisonOp::LessEq,
                Token::Greater => ComparisonOp::Greater,
                Token::GreaterEq => ComparisonOp::GreaterEq,
                _ => break,
            };
            
            self.tokens.pop_front();
            let right = self.parse_primary()?;
            left = FilterExpression::Comparison {
                left: Box::new(left),
                operator: op,
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }    /// Parse primary expression (literals, @, function calls, etc.)
    fn parse_primary(&mut self) -> JsonPathResult<FilterExpression> {
        match self.tokens.front() {
            Some(Token::At) => {
                self.tokens.pop_front();
                if matches!(self.tokens.front(), Some(Token::Dot)) {
                    self.tokens.pop_front();
                    self.parse_property_path()
                } else {
                    Ok(FilterExpression::Current)
                }
            }
            
            Some(Token::String(value)) => {
                let value = value.clone();
                self.tokens.pop_front();
                Ok(FilterExpression::Literal {
                    value: FilterValue::String(value),
                })
            }
            
            Some(Token::Integer(value)) => {
                let value = *value;
                self.tokens.pop_front();
                Ok(FilterExpression::Literal {
                    value: FilterValue::Integer(value),
                })
            }
            
            Some(Token::Number(value)) => {
                let value = *value;
                self.tokens.pop_front();
                Ok(FilterExpression::Literal {
                    value: FilterValue::Number(value),
                })
            }
            
            Some(Token::Boolean(value)) => {
                let value = *value;
                self.tokens.pop_front();
                Ok(FilterExpression::Literal {
                    value: FilterValue::Boolean(value),
                })
            }
            
            Some(Token::Null) => {
                self.tokens.pop_front();
                Ok(FilterExpression::Literal {
                    value: FilterValue::Null,
                })
            }
            
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                self.tokens.pop_front();
                
                if matches!(self.tokens.front(), Some(Token::LeftParen)) {
                    self.tokens.pop_front();
                    let args = self.parse_function_arguments()?;
                    self.expect_token(Token::RightParen)?;
                    Ok(FilterExpression::Function { name, args })
                } else {
                    Err(invalid_expression_error(
                        &self.input,
                        &format!("unexpected identifier: {}", name),
                        Some(self.position),
                    ))
                }
            }
            
            Some(Token::LeftParen) => {
                self.tokens.pop_front();
                let expr = self.parse_filter_expression()?;
                self.expect_token(Token::RightParen)?;
                Ok(expr)
            }
            
            _ => Err(invalid_expression_error(
                &self.input,
                "expected literal, @, function call, or parenthesized expression",
                Some(self.position),
            )),
        }
    }    /// Parse property path (@.property.nested)
    fn parse_property_path(&mut self) -> JsonPathResult<FilterExpression> {
        let mut path = Vec::new();
        
        if let Some(Token::Identifier(name)) = self.tokens.front() {
            path.push(name.clone());
            self.tokens.pop_front();
            
            while matches!(self.tokens.front(), Some(Token::Dot)) {
                self.tokens.pop_front();
                if let Some(Token::Identifier(name)) = self.tokens.front() {
                    path.push(name.clone());
                    self.tokens.pop_front();
                } else {
                    break;
                }
            }
        }
        
        if path.is_empty() {
            return Err(invalid_expression_error(
                &self.input,
                "expected property name after '@.'",
                Some(self.position),
            ));
        }
        
        Ok(FilterExpression::Property { path })
    }
    
    /// Parse function arguments
    fn parse_function_arguments(&mut self) -> JsonPathResult<Vec<FilterExpression>> {
        let mut args = Vec::new();
        
        // Check for empty argument list
        if matches!(self.tokens.front(), Some(Token::RightParen)) {
            return Ok(args);
        }
        
        // Parse first argument
        args.push(self.parse_filter_expression()?);
        
        // Parse remaining arguments
        while matches!(self.tokens.front(), Some(Token::Comma)) {
            self.tokens.pop_front();
            args.push(self.parse_filter_expression()?);
        }
        
        Ok(args)
    }
    
    /// Expect and consume a specific token
    fn expect_token(&mut self, expected: Token) -> JsonPathResult<()> {
        if let Some(token) = self.tokens.front() {
            if std::mem::discriminant(token) == std::mem::discriminant(&expected) {
                self.tokens.pop_front();
                Ok(())
            } else {
                Err(invalid_expression_error(
                    &self.input,
                    &format!("expected {:?}, found {:?}", expected, token),
                    Some(self.position),
                ))
            }
        } else {
            Err(invalid_expression_error(
                &self.input,
                &format!("expected {:?}, found end of input", expected),
                Some(self.position),
            ))
        }
    }
}