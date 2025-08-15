//! Tokenization logic for JSONPath expressions
//! 
//! Provides character-by-character parsing and token recognition for JSONPath syntax.

use crate::json_path::error::{invalid_expression_error, JsonPathResult};
use std::collections::VecDeque;

/// Token types for JSONPath expression parsing
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Structural tokens
    Dollar,          // $
    Dot,             // .
    DotDot,          // ..
    LeftBracket,     // [
    RightBracket,    // ]
    LeftParen,       // (
    RightParen,      // )
    Comma,           // ,
    Colon,           // :
    Question,        // ?
    At,              // @
    Star,            // *
    
    // Comparison operators
    Equal,           // ==
    NotEqual,        // !=
    Less,            // <
    LessEq,          // <=
    Greater,         // >
    GreaterEq,       // >=
    
    // Logical operators
    LogicalAnd,      // &&
    LogicalOr,       // ||
    
    // Literals
    String(String),
    Integer(i64),
    Number(f64),
    Boolean(bool),
    Null,
    
    // Identifiers
    Identifier(String),
}/// JSONPath tokenizer for lexical analysis
#[derive(Debug)]
pub struct JsonPathTokenizer {
    input: String,
    position: usize,
    pub(super) tokens: VecDeque<Token>,
}

impl JsonPathTokenizer {
    /// Create a new tokenizer for the given input
    pub fn new(input: String) -> Self {
        Self {
            input,
            position: 0,
            tokens: VecDeque::new(),
        }
    }
    
    /// Tokenize the entire input string
    pub fn tokenize(&mut self) -> JsonPathResult<()> {
        let mut chars = self.input.chars().peekable();
        let mut position = 0;
        
        while let Some(ch) = chars.next() {
            self.position = position;
            position += ch.len_utf8();
            
            match ch {
                ' ' | '\t' | '\n' | '\r' => {
                    // Skip whitespace
                    continue;
                }
                '$' => {
                    self.tokens.push_back(Token::Dollar);
                }
                '.' => {
                    if chars.peek() == Some(&'.') {
                        chars.next();
                        self.tokens.push_back(Token::DotDot);
                    } else {
                        self.tokens.push_back(Token::Dot);
                    }
                }
                '[' => {
                    self.tokens.push_back(Token::LeftBracket);
                }
                ']' => {
                    self.tokens.push_back(Token::RightBracket);
                }
                '(' => {
                    self.tokens.push_back(Token::LeftParen);
                }
                ')' => {
                    self.tokens.push_back(Token::RightParen);
                }
                ',' => {
                    self.tokens.push_back(Token::Comma);
                }
                ':' => {
                    self.tokens.push_back(Token::Colon);
                }
                '?' => {
                    self.tokens.push_back(Token::Question);
                }
                '@' => {
                    self.tokens.push_back(Token::At);
                }
                '*' => {
                    self.tokens.push_back(Token::Star);
                }                '=' => {
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        self.tokens.push_back(Token::Equal);
                    } else {
                        return Err(invalid_expression_error(
                            &self.input,
                            "unexpected '=' character",
                            Some(position),
                        ));
                    }
                }
                '!' => {
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        self.tokens.push_back(Token::NotEqual);
                    } else {
                        return Err(invalid_expression_error(
                            &self.input,
                            "unexpected '!' character",
                            Some(position),
                        ));
                    }
                }
                '<' => {
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        self.tokens.push_back(Token::LessEq);
                    } else {
                        self.tokens.push_back(Token::Less);
                    }
                }
                '>' => {
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        self.tokens.push_back(Token::GreaterEq);
                    } else {
                        self.tokens.push_back(Token::Greater);
                    }
                }
                '&' => {
                    if chars.peek() == Some(&'&') {
                        chars.next();
                        self.tokens.push_back(Token::LogicalAnd);
                    } else {
                        return Err(invalid_expression_error(
                            &self.input,
                            "unexpected '&' character",
                            Some(position),
                        ));
                    }
                }                '|' => {
                    if chars.peek() == Some(&'|') {
                        chars.next();
                        self.tokens.push_back(Token::LogicalOr);
                    } else {
                        return Err(invalid_expression_error(
                            &self.input,
                            "unexpected '|' character",
                            Some(position),
                        ));
                    }
                }
                '\'' | '"' => {
                    let quote_char = ch;
                    let mut string_value = String::new();
                    
                    while let Some(next_ch) = chars.next() {
                        if next_ch == quote_char {
                            break;
                        }
                        if next_ch == '\\' {
                            if let Some(escaped) = chars.next() {
                                match escaped {
                                    'n' => string_value.push('\n'),
                                    't' => string_value.push('\t'),
                                    'r' => string_value.push('\r'),
                                    '\\' => string_value.push('\\'),
                                    '\'' => string_value.push('\''),
                                    '"' => string_value.push('"'),
                                    _ => {
                                        string_value.push('\\');
                                        string_value.push(escaped);
                                    }
                                }
                            }
                        } else {
                            string_value.push(next_ch);
                        }
                    }
                    
                    self.tokens.push_back(Token::String(string_value));
                }                ch if ch.is_alphabetic() => {
                    let mut identifier = String::new();
                    identifier.push(ch);
                    
                    while let Some(&next_ch) = chars.peek() {
                        if next_ch.is_alphanumeric() || next_ch == '_' {
                            if let Some(ch) = chars.next() {
                                identifier.push(ch);
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    
                    // Check for boolean and null literals
                    match identifier.as_str() {
                        "true" => self.tokens.push_back(Token::Boolean(true)),
                        "false" => self.tokens.push_back(Token::Boolean(false)),
                        "null" => self.tokens.push_back(Token::Null),
                        _ => self.tokens.push_back(Token::Identifier(identifier)),
                    }
                }
                ch if ch.is_ascii_digit() || ch == '-' => {
                    let mut number_str = String::new();
                    number_str.push(ch);
                    
                    while let Some(&next_ch) = chars.peek() {
                        if next_ch.is_ascii_digit() || next_ch == '.' || next_ch == 'e' || next_ch == 'E' || next_ch == '+' || next_ch == '-' {
                            if let Some(ch) = chars.next() {
                                number_str.push(ch);
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    
                    if number_str.contains('.') || number_str.contains('e') || number_str.contains('E') {
                        if let Ok(num) = number_str.parse::<f64>() {
                            self.tokens.push_back(Token::Number(num));
                        } else {
                            return Err(invalid_expression_error(
                                &self.input,
                                &format!("invalid number: {}", number_str),
                                Some(position),
                            ));
                        }
                    } else {
                        if let Ok(num) = number_str.parse::<i64>() {
                            self.tokens.push_back(Token::Integer(num));
                        } else {
                            return Err(invalid_expression_error(
                                &self.input,
                                &format!("invalid integer: {}", number_str),
                                Some(position),
                            ));
                        }
                    }
                }
                _ => {
                    return Err(invalid_expression_error(
                        &self.input,
                        &format!("unexpected character: '{}'", ch),
                        Some(position),
                    ));
                }
            }
        }
        
        Ok(())
    }
}