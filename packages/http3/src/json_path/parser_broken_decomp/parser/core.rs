//! Core parser logic and main parsing orchestration
//!
//! Provides the main JsonPathParser struct and core parsing methods
//! for JSONPath expression parsing with recursive descent.

use super::super::types::{JsonSelector, FilterExpression};
use super::super::tokenizer::{Token, JsonPathTokenizer};
use crate::json_path::error::{invalid_expression_error, JsonPathResult};
use std::collections::VecDeque;

/// JSONPath expression parser
#[derive(Debug)]
pub struct JsonPathParser {
    pub(crate) input: String,
    pub(crate) position: usize,
    pub(crate) tokens: VecDeque<Token>,
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
    }

    /// Parse a single selector
    pub(crate) fn parse_selector(&mut self) -> JsonPathResult<JsonSelector> {
        match self.tokens.front() {
            Some(Token::Dot) => {
                self.tokens.pop_front();
                use super::selectors;
                selectors::parse_dot_selector(self)
            }
            Some(Token::DotDot) => {
                self.tokens.pop_front();
                Ok(JsonSelector::RecursiveDescent)
            }
            Some(Token::LeftBracket) => {
                self.tokens.pop_front();
                use super::selectors;
                selectors::parse_bracket_selector(self)
            }
            _ => Err(invalid_expression_error(
                &self.input,
                "expected '.', '..' or '[' after selector",
                Some(self.position),
            )),
        }
    }

    /// Expect and consume a specific token
    pub(crate) fn expect_token(&mut self, expected: Token) -> JsonPathResult<()> {
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