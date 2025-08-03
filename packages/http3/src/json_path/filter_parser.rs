//! Filter expression parsing for JSONPath predicates
//!
//! Handles parsing of complex filter expressions including comparisons,
//! logical operations, function calls, and property access patterns.

use std::collections::VecDeque;

use crate::json_path::{
    ast::{ComparisonOp, FilterExpression, FilterValue, LogicalOp},
    error::{JsonPathResult, invalid_expression_error},
    tokens::Token,
};

/// Parser for JSONPath filter expressions
pub struct FilterParser<'a> {
    tokens: &'a mut VecDeque<Token>,
    input: &'a str,
    position: usize,
}

impl<'a> FilterParser<'a> {
    /// Create new filter parser
    #[inline]
    pub fn new(tokens: &'a mut VecDeque<Token>, input: &'a str, position: usize) -> Self {
        Self {
            tokens,
            input,
            position,
        }
    }

    /// Parse complete filter expression
    #[inline]
    pub fn parse_filter_expression(&mut self) -> JsonPathResult<FilterExpression> {
        self.parse_logical_or()
    }

    /// Parse logical OR expressions (lowest precedence)
    fn parse_logical_or(&mut self) -> JsonPathResult<FilterExpression> {
        let mut left = self.parse_logical_and()?;

        while matches!(self.peek_token(), Some(Token::LogicalOr)) {
            self.consume_token();
            let right = self.parse_logical_and()?;
            left = FilterExpression::Logical {
                left: Box::new(left),
                operator: LogicalOp::Or,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse logical AND expressions
    fn parse_logical_and(&mut self) -> JsonPathResult<FilterExpression> {
        let mut left = self.parse_comparison()?;

        while matches!(self.peek_token(), Some(Token::LogicalAnd)) {
            self.consume_token();
            let right = self.parse_comparison()?;
            left = FilterExpression::Logical {
                left: Box::new(left),
                operator: LogicalOp::And,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse comparison expressions
    fn parse_comparison(&mut self) -> JsonPathResult<FilterExpression> {
        let left = self.parse_primary()?;

        if let Some(op) = self.parse_comparison_operator() {
            self.consume_token();
            let right = self.parse_primary()?;
            Ok(FilterExpression::Comparison {
                left: Box::new(left),
                operator: op,
                right: Box::new(right),
            })
        } else {
            Ok(left)
        }
    }

    /// Parse primary expressions (property access, literals, parentheses)
    fn parse_primary(&mut self) -> JsonPathResult<FilterExpression> {
        match self.peek_token() {
            Some(Token::At) => {
                self.consume_token();
                self.parse_property_access()
            }
            Some(Token::String(s)) => {
                let value = s.clone();
                self.consume_token();
                Ok(FilterExpression::Literal {
                    value: FilterValue::String(value),
                })
            }
            Some(Token::Number(n)) => {
                let value = *n;
                self.consume_token();
                if value.fract() == 0.0 {
                    Ok(FilterExpression::Literal {
                        value: FilterValue::Integer(value as i64),
                    })
                } else {
                    Ok(FilterExpression::Literal {
                        value: FilterValue::Number(value),
                    })
                }
            }
            Some(Token::Integer(int_val)) => {
                let value = *int_val;
                self.consume_token();
                Ok(FilterExpression::Literal {
                    value: FilterValue::Integer(value),
                })
            }
            Some(Token::True) => {
                self.consume_token();
                Ok(FilterExpression::Literal {
                    value: FilterValue::Boolean(true),
                })
            }
            Some(Token::False) => {
                self.consume_token();
                Ok(FilterExpression::Literal {
                    value: FilterValue::Boolean(false),
                })
            }
            Some(Token::Null) => {
                self.consume_token();
                Ok(FilterExpression::Literal {
                    value: FilterValue::Null,
                })
            }
            Some(Token::LeftParen) => {
                self.consume_token();
                let expr = self.parse_logical_or()?;
                self.expect_token(Token::RightParen)?;
                Ok(expr)
            }
            Some(Token::Identifier(name)) => {
                // Check if this is a function call
                let name = name.clone();
                self.consume_token();

                if matches!(self.peek_token(), Some(Token::LeftParen)) {
                    self.consume_token(); // consume '('
                    let args = self.parse_function_arguments()?;
                    self.expect_token(Token::RightParen)?;
                    Ok(FilterExpression::Function { name, args })
                } else {
                    Err(invalid_expression_error(
                        self.input,
                        &format!(
                            "unexpected identifier '{}' - did you mean a function call?",
                            name
                        ),
                        Some(self.position),
                    ))
                }
            }
            _ => Err(invalid_expression_error(
                self.input,
                "expected property access, literal, or parenthesized expression",
                Some(self.position),
            )),
        }
    }

    /// Parse function arguments (comma-separated filter expressions)
    fn parse_function_arguments(&mut self) -> JsonPathResult<Vec<FilterExpression>> {
        let mut args = Vec::new();

        // Handle empty argument list
        if matches!(self.peek_token(), Some(Token::RightParen)) {
            return Ok(args);
        }

        // Parse first argument
        args.push(self.parse_logical_or()?);

        // Parse remaining arguments
        while matches!(self.peek_token(), Some(Token::Comma)) {
            self.consume_token(); // consume comma
            args.push(self.parse_logical_or()?);
        }

        Ok(args)
    }

    /// Parse property access after @ token
    fn parse_property_access(&mut self) -> JsonPathResult<FilterExpression> {
        // Check for just @ (current node)
        if !matches!(self.peek_token(), Some(Token::Dot)) {
            return Ok(FilterExpression::Current);
        }

        let mut path = Vec::new();

        while matches!(self.peek_token(), Some(Token::Dot)) {
            self.consume_token(); // consume dot

            match self.peek_token() {
                Some(Token::Identifier(name)) => {
                    path.push(name.clone());
                    self.consume_token();
                }
                _ => {
                    return Err(invalid_expression_error(
                        self.input,
                        "expected property name after '.'",
                        Some(self.position),
                    ));
                }
            }
        }

        if path.is_empty() {
            Ok(FilterExpression::Current)
        } else {
            Ok(FilterExpression::Property { path })
        }
    }

    /// Parse comparison operator token
    fn parse_comparison_operator(&self) -> Option<ComparisonOp> {
        match self.peek_token() {
            Some(Token::Equal) => Some(ComparisonOp::Equal),
            Some(Token::NotEqual) => Some(ComparisonOp::NotEqual),
            Some(Token::Less) => Some(ComparisonOp::Less),
            Some(Token::LessEq) => Some(ComparisonOp::LessEq),
            Some(Token::Greater) => Some(ComparisonOp::Greater),
            Some(Token::GreaterEq) => Some(ComparisonOp::GreaterEq),
            _ => None,
        }
    }

    /// Peek at next token without consuming
    #[inline]
    fn peek_token(&self) -> Option<&Token> {
        self.tokens.front()
    }

    /// Consume and return next token
    #[inline]
    fn consume_token(&mut self) -> Option<Token> {
        self.tokens.pop_front()
    }

    /// Expect specific token and consume it
    fn expect_token(&mut self, expected: Token) -> JsonPathResult<()> {
        let token = self.consume_token();
        match token {
            Some(actual) if self.tokens_match(&actual, &expected) => Ok(()),
            Some(actual) => Err(invalid_expression_error(
                self.input,
                &format!("expected {:?}, found {:?}", expected, actual),
                Some(self.position),
            )),
            None => Err(invalid_expression_error(
                self.input,
                &format!("expected {:?}, found end of input", expected),
                Some(self.position),
            )),
        }
    }

    /// Check if two tokens match
    #[inline]
    fn tokens_match(&self, actual: &Token, expected: &Token) -> bool {
        use crate::json_path::tokens::TokenMatcher;
        TokenMatcher::tokens_match(actual, expected)
    }
}
