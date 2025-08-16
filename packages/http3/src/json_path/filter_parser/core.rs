//! Core filter parser structure and initialization
//!
//! Contains the main FilterParser struct and basic initialization logic
//! for parsing JSONPath filter expressions.

use std::collections::VecDeque;

use crate::json_path::{ast::FilterExpression, error::JsonPathResult, tokens::Token};

/// Parser for JSONPath filter expressions
pub struct FilterParser<'a> {
    pub(super) tokens: &'a mut VecDeque<Token>,
    pub(super) input: &'a str,
    pub(super) position: usize,
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
}
