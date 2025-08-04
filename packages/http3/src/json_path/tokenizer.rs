//! JSONPath expression tokenizer implementation
//!
//! Handles lexical analysis of JSONPath expressions, converting raw strings
//! into structured token sequences for parsing.

use std::collections::VecDeque;

use crate::json_path::{
    ast::JsonSelector,
    error::{JsonPathResult, invalid_expression_error},
    selector_parser::SelectorParser,
    tokens::Token,
};

/// Main expression parser that combines tokenization and parsing
pub struct ExpressionParser {
    input: String,
    tokens: VecDeque<Token>,
    position: usize,
}

impl ExpressionParser {
    /// Create new expression parser
    #[inline]
    pub fn new(input: &str) -> Self {
        Self {
            input: input.to_string(),
            tokens: VecDeque::new(),
            position: 0,
        }
    }

    /// Parse complete JSONPath expression into selector chain
    pub fn parse(&mut self) -> JsonPathResult<Vec<JsonSelector>> {
        self.tokenize()?;

        let mut selectors = Vec::new();

        while !matches!(self.peek_token(), Some(Token::EOF) | None) {
            let mut selector_parser =
                SelectorParser::new(&mut self.tokens, &self.input, self.position);
            selectors.push(selector_parser.parse_selector()?);
        }

        // RFC 9535: jsonpath-query = root-identifier segments
        // where segments = *(S segment) means zero or more segments
        // Therefore "$" (root-only) is valid and should return the root node
        if selectors.is_empty() {
            // For root-only queries like "$", we don't add any selectors
            // The JsonArrayStream will handle this by returning the root node itself
        }

        Ok(selectors)
    }

    /// Tokenize the input expression
    fn tokenize(&mut self) -> JsonPathResult<()> {
        let chars: Vec<char> = self.input.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            match chars[i] {
                ' ' | '\t' | '\n' | '\r' => {
                    // Skip whitespace
                }
                '$' => self.tokens.push_back(Token::Root),
                '.' => {
                    // Check for double dot (..)
                    if i + 1 < chars.len() && chars[i + 1] == '.' {
                        self.tokens.push_back(Token::DoubleDot);
                        i += 1; // Skip the second dot
                    } else {
                        self.tokens.push_back(Token::Dot);
                    }
                }
                '[' => self.tokens.push_back(Token::LeftBracket),
                ']' => self.tokens.push_back(Token::RightBracket),
                '(' => self.tokens.push_back(Token::LeftParen),
                ')' => self.tokens.push_back(Token::RightParen),
                ',' => self.tokens.push_back(Token::Comma),
                ':' => self.tokens.push_back(Token::Colon),
                '?' => self.tokens.push_back(Token::Question),
                '@' => self.tokens.push_back(Token::At),
                '*' => self.tokens.push_back(Token::Star),
                '\'' | '"' => {
                    let quote = chars[i];
                    i += 1; // Skip opening quote
                    let start = i;
                    while i < chars.len() && chars[i] != quote {
                        i += 1;
                    }
                    if i >= chars.len() {
                        return Err(invalid_expression_error(
                            &self.input,
                            "unterminated string literal",
                            Some(start),
                        ));
                    }
                    let string_value = chars[start..i].iter().collect();
                    self.tokens.push_back(Token::String(string_value));
                }
                c if c.is_ascii_digit() || c == '-' => {
                    let start = i;
                    if c == '-' {
                        i += 1;
                    }
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }

                    // Check for decimal point
                    let mut is_float = false;
                    if i < chars.len() && chars[i] == '.' {
                        is_float = true;
                        i += 1; // Skip decimal point
                        while i < chars.len() && chars[i].is_ascii_digit() {
                            i += 1;
                        }
                    }

                    let number_str: String = chars[start..i].iter().collect();
                    if is_float {
                        if let Ok(float_val) = number_str.parse::<f64>() {
                            self.tokens.push_back(Token::Number(float_val));
                        } else {
                            return Err(invalid_expression_error(
                                &self.input,
                                "invalid floating point number format",
                                Some(start),
                            ));
                        }
                    } else {
                        if let Ok(int_val) = number_str.parse::<i64>() {
                            self.tokens.push_back(Token::Integer(int_val));
                        } else {
                            return Err(invalid_expression_error(
                                &self.input,
                                "invalid integer format",
                                Some(start),
                            ));
                        }
                    }
                    i = i.saturating_sub(1); // Adjust for loop increment
                }
                '=' => {
                    if i + 1 < chars.len() && chars[i + 1] == '=' {
                        self.tokens.push_back(Token::Equal);
                        i += 1; // Skip next =
                    } else {
                        return Err(invalid_expression_error(
                            &self.input,
                            "single '=' not supported, use '==' for equality",
                            Some(i),
                        ));
                    }
                }
                '!' => {
                    if i + 1 < chars.len() && chars[i + 1] == '=' {
                        self.tokens.push_back(Token::NotEqual);
                        i += 1; // Skip next =
                    } else {
                        return Err(invalid_expression_error(
                            &self.input,
                            "single '!' not supported, use '!=' for inequality",
                            Some(i),
                        ));
                    }
                }
                '<' => {
                    if i + 1 < chars.len() && chars[i + 1] == '=' {
                        self.tokens.push_back(Token::LessEq);
                        i += 1; // Skip next =
                    } else {
                        self.tokens.push_back(Token::Less);
                    }
                }
                '>' => {
                    if i + 1 < chars.len() && chars[i + 1] == '=' {
                        self.tokens.push_back(Token::GreaterEq);
                        i += 1; // Skip next =
                    } else {
                        self.tokens.push_back(Token::Greater);
                    }
                }
                '&' => {
                    if i + 1 < chars.len() && chars[i + 1] == '&' {
                        self.tokens.push_back(Token::LogicalAnd);
                        i += 1; // Skip next &
                    } else {
                        return Err(invalid_expression_error(
                            &self.input,
                            "single '&' not supported, use '&&' for logical AND",
                            Some(i),
                        ));
                    }
                }
                '|' => {
                    if i + 1 < chars.len() && chars[i + 1] == '|' {
                        self.tokens.push_back(Token::LogicalOr);
                        i += 1; // Skip next |
                    } else {
                        return Err(invalid_expression_error(
                            &self.input,
                            "single '|' not supported, use '||' for logical OR",
                            Some(i),
                        ));
                    }
                }
                c if c.is_alphabetic() || c == '_' => {
                    let start = i;
                    while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                        i += 1;
                    }
                    let identifier: String = chars[start..i].iter().collect();

                    // Check for reserved keywords
                    let token = match identifier.as_str() {
                        "true" => Token::True,
                        "false" => Token::False,
                        "null" => Token::Null,
                        _ => Token::Identifier(identifier),
                    };

                    self.tokens.push_back(token);
                    i = i.saturating_sub(1); // Adjust for loop increment
                }
                _ => {
                    return Err(invalid_expression_error(
                        &self.input,
                        &format!("unexpected character '{}'", chars[i]),
                        Some(i),
                    ));
                }
            }
            i += 1;
        }

        self.tokens.push_back(Token::EOF);
        Ok(())
    }

    /// Peek at next token without consuming
    #[inline]
    fn peek_token(&self) -> Option<&Token> {
        self.tokens.front()
    }
}
