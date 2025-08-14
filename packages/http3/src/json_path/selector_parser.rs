//! JSONPath selector parsing implementation
//!
//! Handles parsing of individual JSONPath selectors including array indices,
//! slices, filters, and property access patterns.

use std::collections::VecDeque;

use crate::json_path::{
    ast::JsonSelector,
    error::{JsonPathResult, invalid_expression_error},
    filter_parser::FilterParser,
    tokens::Token,
};

/// Parser for individual JSONPath selectors
pub struct SelectorParser<'a> {
    tokens: &'a mut VecDeque<Token>,
    input: &'a str,
    position: usize,
}

impl<'a> SelectorParser<'a> {
    /// Create new selector parser
    #[inline]
    pub fn new(tokens: &'a mut VecDeque<Token>, input: &'a str, position: usize) -> Self {
        Self {
            tokens,
            input,
            position,
        }
    }

    /// Parse a single JSONPath selector
    pub fn parse_selector(&mut self) -> JsonPathResult<JsonSelector> {
        match self.peek_token() {
            Some(Token::Root) => {
                self.consume_token();
                Ok(JsonSelector::Root)
            }
            Some(Token::Dot) => {
                self.consume_token();
                self.parse_dot_selector()
            }
            Some(Token::DoubleDot) => {
                self.consume_token();
                Ok(JsonSelector::RecursiveDescent)
            }
            Some(Token::Star) => {
                self.consume_token();
                Ok(JsonSelector::Wildcard)
            }
            Some(Token::LeftBracket) => {
                self.consume_token();
                self.parse_bracket_selector()
            }
            Some(Token::Identifier(name)) => {
                // Handle standalone identifiers (e.g., 'author' in '$..author')
                let name = name.clone();
                self.consume_token();
                Ok(JsonSelector::Child {
                    name,
                    exact_match: true,
                })
            }
            Some(Token::At) => Err(invalid_expression_error(
                self.input,
                "current node identifier '@' is only valid within filter expressions [?...]",
                Some(self.position),
            )),
            _ => Err(invalid_expression_error(
                self.input,
                "expected selector (.property, [index], identifier, or [expression])",
                Some(self.position),
            )),
        }
    }

    /// Parse dot-notation selector (.property or ..)
    fn parse_dot_selector(&mut self) -> JsonPathResult<JsonSelector> {
        match self.peek_token() {
            Some(Token::Dot) => {
                self.consume_token();
                Ok(JsonSelector::RecursiveDescent)
            }
            Some(Token::Star) => {
                self.consume_token();
                Ok(JsonSelector::Wildcard)
            }
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                self.consume_token();
                Ok(JsonSelector::Child {
                    name,
                    exact_match: true,
                })
            }
            Some(Token::At) => Err(invalid_expression_error(
                self.input,
                "current node identifier '@' is only valid within filter expressions [?...]",
                Some(self.position),
            )),
            _ => Err(invalid_expression_error(
                self.input,
                "expected property name, '..' (recursive descent), or '*' (wildcard) after '.'",
                Some(self.position),
            )),
        }
    }

    /// Parse bracket-notation selector ([index], [start:end], [?expression])
    fn parse_bracket_selector(&mut self) -> JsonPathResult<JsonSelector> {
        match self.peek_token() {
            Some(Token::Star) => {
                self.consume_token();
                self.expect_token(Token::RightBracket)?;
                Ok(JsonSelector::Wildcard)
            }
            Some(Token::Question) => {
                self.consume_token();
                let mut filter_parser = FilterParser::new(self.tokens, self.input, self.position);
                let expression = filter_parser.parse_filter_expression()?;
                self.expect_token(Token::RightBracket)?;
                Ok(JsonSelector::Filter { expression })
            }
            Some(Token::String(s)) => {
                let name = s.clone();
                self.consume_token();

                // Check for comma-separated union selector
                if matches!(self.peek_token(), Some(Token::Comma)) {
                    let mut selectors = vec![JsonSelector::Child {
                        name,
                        exact_match: true,
                    }];

                    while matches!(self.peek_token(), Some(Token::Comma)) {
                        self.consume_token(); // consume comma

                        match self.peek_token() {
                            Some(Token::String(s)) => {
                                let name = s.clone();
                                self.consume_token();
                                selectors.push(JsonSelector::Child {
                                    name,
                                    exact_match: true,
                                });
                            }
                            Some(Token::Integer(n)) => {
                                let index = *n;
                                self.consume_token();
                                selectors.push(JsonSelector::Index {
                                    index,
                                    from_end: index < 0,
                                });
                            }
                            Some(Token::Star) => {
                                self.consume_token();
                                selectors.push(JsonSelector::Wildcard);
                            }
                            _ => {
                                return Err(invalid_expression_error(
                                    self.input,
                                    "expected string, integer, or '*' after comma in union selector",
                                    Some(self.position),
                                ));
                            }
                        }
                    }

                    self.expect_token(Token::RightBracket)?;
                    Ok(JsonSelector::Union { selectors })
                } else {
                    self.expect_token(Token::RightBracket)?;
                    Ok(JsonSelector::Child {
                        name,
                        exact_match: true,
                    })
                }
            }
            Some(Token::Integer(n)) => {
                let index = *n;
                self.consume_token();
                self.parse_index_or_slice(index)
            }
            Some(Token::Colon) => self.parse_slice_from_colon(),
            Some(Token::At) => Err(invalid_expression_error(
                self.input,
                "current node identifier '@' is only valid within filter expressions [?...]",
                Some(self.position),
            )),
            _ => Err(invalid_expression_error(
                self.input,
                "expected index, slice, filter, string, or wildcard in brackets",
                Some(self.position),
            )),
        }
    }

    /// Parse index or slice notation after initial integer
    fn parse_index_or_slice(&mut self, start: i64) -> JsonPathResult<JsonSelector> {
        match self.peek_token() {
            Some(Token::RightBracket) => {
                self.consume_token();
                Ok(JsonSelector::Index {
                    index: start,
                    from_end: start < 0,
                })
            }
            Some(Token::Colon) => self.parse_slice_from_start(start),
            Some(Token::Comma) => {
                // Parse union selector starting with integer
                let mut selectors = vec![JsonSelector::Index {
                    index: start,
                    from_end: start < 0,
                }];

                while matches!(self.peek_token(), Some(Token::Comma)) {
                    self.consume_token(); // consume comma

                    match self.peek_token() {
                        Some(Token::Integer(n)) => {
                            let index = *n;
                            self.consume_token();
                            selectors.push(JsonSelector::Index {
                                index,
                                from_end: index < 0,
                            });
                        }
                        Some(Token::String(s)) => {
                            let name = s.clone();
                            self.consume_token();
                            selectors.push(JsonSelector::Child {
                                name,
                                exact_match: true,
                            });
                        }
                        Some(Token::Star) => {
                            self.consume_token();
                            selectors.push(JsonSelector::Wildcard);
                        }
                        _ => {
                            return Err(invalid_expression_error(
                                self.input,
                                "expected integer, string, or '*' after comma in union selector",
                                Some(self.position),
                            ));
                        }
                    }
                }

                self.expect_token(Token::RightBracket)?;
                Ok(JsonSelector::Union { selectors })
            }
            _ => Err(invalid_expression_error(
                self.input,
                "expected ']', ':', or ',' after index",
                Some(self.position),
            )),
        }
    }

    /// Parse slice notation starting with integer (e.g., [1:5])
    fn parse_slice_from_start(&mut self, start: i64) -> JsonPathResult<JsonSelector> {
        self.consume_token(); // consume colon

        // Parse end index
        let end = if let Some(Token::Integer(n)) = self.peek_token() {
            let n = *n;
            self.consume_token();
            Some(n)
        } else if matches!(self.peek_token(), Some(Token::RightBracket)) {
            None // Open-ended slice like [1:]
        } else if matches!(self.peek_token(), Some(Token::Colon)) {
            None // Empty end in patterns like [1::2]
        } else {
            None
        };

        // Parse optional step
        let step = if matches!(self.peek_token(), Some(Token::Colon)) {
            self.consume_token(); // consume second colon
            // After second colon, step is REQUIRED per RFC 9535
            if let Some(Token::Integer(n)) = self.peek_token() {
                let n = *n;
                self.consume_token();
                // RFC 9535: step must not be zero
                if n == 0 {
                    return Err(invalid_expression_error(
                        self.input,
                        "step value cannot be zero in slice expression",
                        Some(self.position),
                    ));
                }
                Some(n)
            } else {
                return Err(invalid_expression_error(
                    self.input,
                    "step value required after second colon in slice",
                    Some(self.position),
                ));
            }
        } else {
            None
        };

        self.expect_token(Token::RightBracket)?;
        Ok(JsonSelector::Slice {
            start: Some(start),
            end,
            step,
        })
    }

    /// Parse slice notation starting with colon (e.g., [:5])
    fn parse_slice_from_colon(&mut self) -> JsonPathResult<JsonSelector> {
        self.consume_token(); // consume colon

        // Parse end index
        let end = if let Some(Token::Integer(n)) = self.peek_token() {
            let n = *n;
            self.consume_token();
            Some(n)
        } else if matches!(self.peek_token(), Some(Token::Colon)) {
            None // Empty end in patterns like [::2]
        } else {
            None
        };

        // Parse optional step
        let step = if matches!(self.peek_token(), Some(Token::Colon)) {
            self.consume_token(); // consume second colon
            // After second colon, step is REQUIRED per RFC 9535
            if let Some(Token::Integer(n)) = self.peek_token() {
                let n = *n;
                self.consume_token();
                // RFC 9535: step must not be zero
                if n == 0 {
                    return Err(invalid_expression_error(
                        self.input,
                        "step value cannot be zero in slice expression",
                        Some(self.position),
                    ));
                }
                Some(n)
            } else {
                return Err(invalid_expression_error(
                    self.input,
                    "step value required after second colon in slice",
                    Some(self.position),
                ));
            }
        } else {
            None
        };

        self.expect_token(Token::RightBracket)?;
        Ok(JsonSelector::Slice {
            start: None,
            end,
            step,
        })
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

    /// Check if two tokens match (handles different variants with same discriminant)
    #[inline]
    fn tokens_match(&self, actual: &Token, expected: &Token) -> bool {
        use crate::json_path::tokens::TokenMatcher;
        TokenMatcher::tokens_match(actual, expected)
    }
}
