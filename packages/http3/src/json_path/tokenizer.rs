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

        // RFC 9535 validation: Check for invalid token sequences
        // Valid: [$, Dot, Identifier] or [$, LeftBracket, ...] or [$, DoubleDot, ...]
        // Invalid: [$, Identifier] (direct identifier after root without dot or bracket)
        // Note: Multiple root identifiers are allowed in complex expressions with functions
        let tokens_vec: Vec<_> = self.tokens.iter().collect();

        if self.tokens.len() >= 3 {
            if matches!(tokens_vec[0], Token::Root) && matches!(tokens_vec[1], Token::Identifier(_))
            {
                // This is the invalid pattern: $identifier (without dot or bracket)
                return Err(invalid_expression_error(
                    &self.input,
                    "property access requires '.' (dot) or '[]' (bracket) notation after root '$'",
                    Some(1), // Position of the identifier token
                ));
            }
        }

        let mut selectors = Vec::new();

        while !matches!(self.peek_token(), Some(Token::EOF) | None) {
            let mut selector_parser =
                SelectorParser::new(&mut self.tokens, &self.input, self.position);
            selectors.push(selector_parser.parse_selector()?);
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
                        // Check for invalid triple dot (...)
                        if i + 2 < chars.len() && chars[i + 2] == '.' {
                            return Err(invalid_expression_error(
                                &self.input,
                                "triple dot '...' is invalid, use '..' for recursive descent",
                                Some(i),
                            ));
                        }
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
                    let mut string_value = String::new();

                    while i < chars.len() {
                        if chars[i] == quote {
                            // Found closing quote
                            break;
                        } else if chars[i] == '\\' && i + 1 < chars.len() {
                            // Handle escape sequence
                            i += 1; // Skip backslash
                            match chars[i] {
                                '"' => string_value.push('"'),
                                '\'' => string_value.push('\''),
                                '\\' => string_value.push('\\'),
                                '/' => string_value.push('/'),
                                'b' => string_value.push('\u{0008}'), // Backspace
                                'f' => string_value.push('\u{000C}'), // Form feed
                                'n' => string_value.push('\n'),       // Newline
                                'r' => string_value.push('\r'),       // Carriage return
                                't' => string_value.push('\t'),       // Tab
                                'u' => {
                                    // Unicode escape sequence \uXXXX
                                    if i + 4 >= chars.len() {
                                        return Err(invalid_expression_error(
                                            &self.input,
                                            "incomplete unicode escape sequence",
                                            Some(i),
                                        ));
                                    }
                                    let hex_digits: String = chars[i + 1..i + 5].iter().collect();
                                    if let Ok(code_point) = u32::from_str_radix(&hex_digits, 16) {
                                        // Handle Unicode surrogate pairs (UTF-16)
                                        if (0xD800..=0xDBFF).contains(&code_point) {
                                            // High surrogate - look for low surrogate
                                            if i + 10 < chars.len()
                                                && chars[i + 5] == '\\'
                                                && chars[i + 6] == 'u'
                                            {
                                                let low_hex: String =
                                                    chars[i + 7..i + 11].iter().collect();
                                                if let Ok(low_surrogate) =
                                                    u32::from_str_radix(&low_hex, 16)
                                                {
                                                    if (0xDC00..=0xDFFF).contains(&low_surrogate) {
                                                        // Valid surrogate pair - convert to Unicode scalar
                                                        let high = code_point - 0xD800;
                                                        let low = low_surrogate - 0xDC00;
                                                        let unicode_scalar =
                                                            0x10000 + (high << 10) + low;
                                                        if let Some(unicode_char) =
                                                            char::from_u32(unicode_scalar)
                                                        {
                                                            string_value.push(unicode_char);
                                                            i += 10; // Skip both \uXXXX sequences
                                                        } else {
                                                            return Err(invalid_expression_error(
                                                                &self.input,
                                                                "invalid surrogate pair result",
                                                                Some(i),
                                                            ));
                                                        }
                                                    } else {
                                                        return Err(invalid_expression_error(
                                                            &self.input,
                                                            "high surrogate not followed by valid low surrogate",
                                                            Some(i),
                                                        ));
                                                    }
                                                } else {
                                                    return Err(invalid_expression_error(
                                                        &self.input,
                                                        "invalid low surrogate hex digits",
                                                        Some(i),
                                                    ));
                                                }
                                            } else {
                                                return Err(invalid_expression_error(
                                                    &self.input,
                                                    "high surrogate not followed by low surrogate escape sequence",
                                                    Some(i),
                                                ));
                                            }
                                        } else if (0xDC00..=0xDFFF).contains(&code_point) {
                                            // Low surrogate without high surrogate is invalid
                                            return Err(invalid_expression_error(
                                                &self.input,
                                                "low surrogate without preceding high surrogate",
                                                Some(i),
                                            ));
                                        } else if let Some(unicode_char) =
                                            char::from_u32(code_point)
                                        {
                                            // Regular Unicode character (not surrogate)
                                            string_value.push(unicode_char);
                                            i += 4; // Skip the 4 hex digits
                                        } else {
                                            return Err(invalid_expression_error(
                                                &self.input,
                                                "invalid unicode code point",
                                                Some(i),
                                            ));
                                        }
                                    } else {
                                        return Err(invalid_expression_error(
                                            &self.input,
                                            "invalid unicode escape sequence",
                                            Some(i),
                                        ));
                                    }
                                }
                                _ => {
                                    return Err(invalid_expression_error(
                                        &self.input,
                                        "invalid escape sequence",
                                        Some(i),
                                    ));
                                }
                            }
                        } else {
                            // Regular character
                            string_value.push(chars[i]);
                        }
                        i += 1;
                    }

                    if i >= chars.len() {
                        return Err(invalid_expression_error(
                            &self.input,
                            "unterminated string literal",
                            Some(start),
                        ));
                    }

                    self.tokens.push_back(Token::String(string_value));
                }
                c if c.is_ascii_digit() || c == '-' => {
                    let start = i;
                    if c == '-' {
                        i += 1;
                    }

                    // RFC 9535: integers cannot have leading zeros (except for "0" itself)
                    let digit_start = i;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }

                    // Validate no leading zeros for multi-digit integers
                    if i > digit_start + 1 && chars[digit_start] == '0' {
                        return Err(invalid_expression_error(
                            &self.input,
                            "integers cannot have leading zeros",
                            Some(digit_start),
                        ));
                    }

                    // RFC 9535: negative zero "-0" is invalid per grammar: int = "0" / (["−"] (non-zero-digit *DIGIT))
                    if start < digit_start && chars[digit_start] == '0' && i == digit_start + 1 {
                        return Err(invalid_expression_error(
                            &self.input,
                            "negative zero is not allowed",
                            Some(start),
                        ));
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
