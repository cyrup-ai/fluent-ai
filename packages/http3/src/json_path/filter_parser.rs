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
                    
                    // RFC 9535: Validate function argument count
                    self.validate_function_arguments(&name, &args)?;
                    
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
            Some(Token::Root) => {
                // Parse JSONPath expression starting with $ or @
                let mut jsonpath_tokens = Vec::new();
                
                // Consume all tokens until we hit a delimiter or end
                while let Some(token) = self.peek_token() {
                    match token {
                        Token::Root | Token::At | Token::Dot | Token::DoubleDot | Token::LeftBracket | 
                        Token::RightBracket | Token::Star | Token::Identifier(_) | 
                        Token::Integer(_) | Token::String(_) | Token::Colon => {
                            if let Some(consumed_token) = self.consume_token() {
                                jsonpath_tokens.push(consumed_token);
                            }
                        }
                        _ => break, // Stop at other tokens (comma, right paren, operators, etc.)
                    }
                }
                
                // Parse the collected tokens into selectors
                use crate::json_path::ast::JsonSelector;
                
                if jsonpath_tokens.is_empty() {
                    return Err(invalid_expression_error(
                        self.input,
                        "empty JSONPath expression",
                        Some(self.position),
                    ));
                }
                
                // Convert tokens to selectors using a simple direct mapping
                let mut selectors = Vec::new();
                let mut i = 0;
                
                while i < jsonpath_tokens.len() {
                    match &jsonpath_tokens[i] {
                        Token::Root => {
                            selectors.push(JsonSelector::Root);
                            i += 1;
                        }
                        Token::At => {
                            // @ represents current node - in JSONPath context, this becomes root
                            selectors.push(JsonSelector::Root);
                            i += 1;
                        }
                        Token::DoubleDot => {
                            selectors.push(JsonSelector::RecursiveDescent);
                            i += 1;
                        }
                        Token::Star => {
                            selectors.push(JsonSelector::Wildcard);
                            i += 1;
                        }
                        Token::Identifier(name) => {
                            selectors.push(JsonSelector::Child {
                                name: name.clone(),
                                exact_match: true,
                            });
                            i += 1;
                        }
                        Token::Dot => {
                            // Skip dot tokens as they're structural
                            i += 1;
                        }
                        _ => {
                            // For now, skip other complex patterns
                            i += 1;
                        }
                    }
                }
                
                Ok(FilterExpression::JsonPath { selectors })
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
        if !matches!(self.peek_token(), Some(Token::Dot) | Some(Token::DoubleDot)) {
            return Ok(FilterExpression::Current);
        }

        // Handle property access patterns like @.author
        use crate::json_path::ast::JsonSelector;
        
        // After @, check for simple property access (dot followed by identifier)
        if matches!(self.peek_token(), Some(Token::Dot)) {
            self.consume_token(); // consume the dot
            
            // After dot, expect identifier
            if let Some(Token::Identifier(name)) = self.peek_token() {
                let name = name.clone();
                self.consume_token();
                
                // Check if there are more tokens - if not, this is a simple property access
                if self.peek_token().is_none() || matches!(self.peek_token(), Some(Token::EOF)) {
                    // Simple property access should create a Property expression, not JsonPath
                    return Ok(FilterExpression::Property { 
                        path: vec![name]
                    });
                }
            } else {
                return Err(invalid_expression_error(
                    self.input,
                    "expected property name after '.'",
                    Some(self.position),
                ));
            }
        }

        // Handle complex JSONPath patterns like @..book, @.*, etc.
        let mut selectors = Vec::new();
        
        // @ represents current node in filter context
        selectors.push(JsonSelector::Root);

        // Parse the remaining tokens as JSONPath selectors
        while let Some(token) = self.peek_token() {
            match token {
                Token::Dot => {
                    self.consume_token();
                    // After dot, expect identifier
                    if let Some(Token::Identifier(name)) = self.peek_token() {
                        let name = name.clone();
                        self.consume_token();
                        selectors.push(JsonSelector::Child {
                            name,
                            exact_match: true,
                        });
                    } else {
                        return Err(invalid_expression_error(
                            self.input,
                            "expected property name after '.'",
                            Some(self.position),
                        ));
                    }
                }
                Token::DoubleDot => {
                    self.consume_token();
                    selectors.push(JsonSelector::RecursiveDescent);
                }
                Token::Star => {
                    self.consume_token();
                    selectors.push(JsonSelector::Wildcard);
                }
                Token::Identifier(name) => {
                    // Bare identifier (should not happen after @ but handle gracefully)
                    let name = name.clone();
                    self.consume_token();
                    selectors.push(JsonSelector::Child {
                        name,
                        exact_match: true,
                    });
                }
                _ => break, // Stop at other tokens
            }
        }

        // Return appropriate expression type
        if selectors.len() == 1 {
            Ok(FilterExpression::Current) // Just @
        } else {
            Ok(FilterExpression::JsonPath { selectors })
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

    /// Validate function arguments according to RFC 9535
    fn validate_function_arguments(&self, function_name: &str, args: &[FilterExpression]) -> JsonPathResult<()> {
        // Check for known functions with case sensitivity
        let expected_count = match function_name {
            "count" => 1,
            "length" => 1,
            "value" => 1,
            "match" => 2,
            "search" => 2,
            _ => {
                // Check if this might be a case-sensitivity error
                let lowercase_name = function_name.to_lowercase();
                if matches!(lowercase_name.as_str(), "count" | "length" | "value" | "match" | "search") {
                    return Err(invalid_expression_error(
                        self.input,
                        &format!(
                            "unknown function '{}' - did you mean '{}'? (function names are case-sensitive)",
                            function_name,
                            lowercase_name
                        ),
                        Some(self.position),
                    ));
                }
                
                // Unknown function - let it pass for now (could be user-defined)
                return Ok(());
            }
        };

        if args.len() != expected_count {
            return Err(invalid_expression_error(
                self.input,
                &format!(
                    "function '{}' requires exactly {} argument{}, found {}",
                    function_name,
                    expected_count,
                    if expected_count == 1 { "" } else { "s" },
                    args.len()
                ),
                Some(self.position),
            ));
        }

        Ok(())
    }


}
