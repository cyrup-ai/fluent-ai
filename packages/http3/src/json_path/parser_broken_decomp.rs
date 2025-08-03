//! JSONPath Expression Compilation and Core Parsing Logic
//!
//! This module provides compile-time optimization of JSONPath expressions into
//! efficient runtime selectors. Focuses on tokenization, parsing, and selector
//! compilation with zero-allocation execution paths.

use crate::json_path::error::{invalid_expression_error, unsupported_feature_error, JsonPathError, JsonPathResult};
use crate::json_path::filter::FilterEvaluator;
use std::collections::VecDeque;

/// Compiled JSONPath expression optimized for streaming evaluation
#[derive(Debug, Clone)]
pub struct JsonPathExpression {
    /// Optimized selector chain for runtime execution
    selectors: Vec<JsonSelector>,
    /// Original expression string for debugging
    original: String,
    /// Whether this expression targets an array for streaming
    is_array_stream: bool,
}

/// Comprehensive complexity metrics for JSONPath expression analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComplexityMetrics {
    /// Count of recursive descent (..) selectors and their effective nesting depth
    pub recursive_descent_depth: u32,
    /// Total number of selectors in the expression chain
    pub total_selector_count: u32,
    /// Sum of all filter expression complexity scores
    pub filter_complexity_sum: u32,
    /// Largest slice range for slice operations (end - start)
    pub max_slice_range: u32,
    /// Number of selectors within union operations
    pub union_selector_count: u32,
}

impl ComplexityMetrics {
    /// Create new complexity metrics with zero values
    #[inline]
    pub const fn new() -> Self {
        Self {
            recursive_descent_depth: 0,
            total_selector_count: 0,
            filter_complexity_sum: 0,
            max_slice_range: 0,
            union_selector_count: 0,
        }
    }
    
    /// Add metrics from another ComplexityMetrics instance
    #[inline]
    pub fn add(&mut self, other: &Self) {
        self.recursive_descent_depth = self.recursive_descent_depth.saturating_add(other.recursive_descent_depth);
        self.total_selector_count = self.total_selector_count.saturating_add(other.total_selector_count);
        self.filter_complexity_sum = self.filter_complexity_sum.saturating_add(other.filter_complexity_sum);
        self.max_slice_range = self.max_slice_range.max(other.max_slice_range);
        self.union_selector_count = self.union_selector_count.saturating_add(other.union_selector_count);
    }
    
    /// Calculate overall complexity score
    #[inline]
    pub fn complexity_score(&self) -> u32 {
        // Weighted scoring algorithm that heavily penalizes exponential operations
        self.recursive_descent_depth.saturating_mul(5)
            .saturating_add(self.filter_complexity_sum.saturating_mul(3))
            .saturating_add(self.union_selector_count.saturating_mul(2))
            .saturating_add(self.max_slice_range)
            .saturating_add(self.total_selector_count)
    }
}

/// Individual JSONPath selector for runtime evaluation
#[derive(Debug, Clone, PartialEq)]
pub enum JsonSelector {
    /// Root selector ($)
    Root,
    /// Property access (.property or ['property'])
    Property {
        name: String,
    },
    /// Child property access (.property or ['property'])
    Child { 
        name: String,
        exact_match: bool,
    },
    /// Array index access ([0], [-1])
    Index {
        index: i32,
        from_end: bool,
    },
    /// Array slice ([start:end:step])
    Slice {
        start: Option<i32>,
        end: Option<i32>,
        step: Option<i32>,
    },
    /// Wildcard selector (*, [*])
    Wildcard,
    /// Recursive descent (..)
    RecursiveDescent,
    /// Filter expression ([?@.price > 10])
    Filter {
        expression: FilterExpression,
    },
    /// Union of multiple selectors ([0, 2, 4])
    Union {
        selectors: Vec<JsonSelector>,
    },
}

/// Filter expression components for JSONPath predicates
#[derive(Debug, Clone, PartialEq)]
pub enum FilterExpression {
    /// Current node reference (@)
    Current,
    /// Property access (@.property, @.nested.prop)
    Property {
        path: Vec<String>,
    },
    /// Literal value (string, number, boolean, null)
    Literal {
        value: FilterValue,
    },
    /// Comparison operation (@.price > 10)
    Comparison {
        left: Box<FilterExpression>,
        operator: ComparisonOp,
        right: Box<FilterExpression>,
    },
    /// Logical AND (&&)
    LogicalAnd {
        left: Box<FilterExpression>,
        right: Box<FilterExpression>,
    },
    /// Logical OR (||)
    LogicalOr {
        left: Box<FilterExpression>,
        right: Box<FilterExpression>,
    },
    /// Logical operations (&&, ||)
    Logical {
        left: Box<FilterExpression>,
        operator: LogicalOp,
        right: Box<FilterExpression>,
    },
    
    /// Function call (length(@.tags), match(@.name, 'pattern'))
    Function {
        name: String,
        args: Vec<FilterExpression>,
    },
}

/// Logical operators for filter expressions
#[derive(Debug, Clone, Copy)]
pub enum LogicalOp {
    And, // &&
    Or,  // ||
}

impl FilterExpression {
    /// Calculate complexity score for filter expressions
    #[inline]
    pub fn complexity_score(&self) -> u32 {
        match self {
            FilterExpression::Current => 1,
            FilterExpression::Property { path } => path.len() as u32,
            FilterExpression::Literal { .. } => 1,
            FilterExpression::Comparison { left, right, .. } => {
                2 + left.complexity_score() + right.complexity_score()
            }
            FilterExpression::Logical { left, right, .. } => {
                3 + left.complexity_score() + right.complexity_score()
            }
            FilterExpression::Function { args, .. } => {
                5 + args.iter().map(|arg| arg.complexity_score()).sum::<u32>()
            }
        }
    }
}

/// Comparison operators for filter expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    Equal,      // ==
    NotEqual,   // !=
    Less,       // <
    LessEq,     // <=
    Greater,    // >
    GreaterEq,  // >=
}

/// Filter value types for comparisons and function results
#[derive(Debug, Clone, PartialEq)]
pub enum FilterValue {
    String(String),
    Integer(i64),
    Number(f64),
    Boolean(bool),
    Null,
}

impl JsonPathExpression {
    /// Get the optimized selector chain
    #[inline]
    pub fn selectors(&self) -> &[JsonSelector] {
        &self.selectors
    }
    
    /// Get the original JSONPath expression string
    #[inline]
    pub fn original(&self) -> &str {
        &self.original
    }
    
    /// Check if this expression is optimized for array streaming
    #[inline]
    pub fn is_array_stream(&self) -> bool {
        self.is_array_stream
    }
    
    /// Calculate complexity score for performance estimation
    #[inline]
    pub fn complexity_score(&self) -> u32 {
        self.calculate_complexity_metrics().complexity_score()
    }
    
    /// Check if expression has recursive descent
    #[inline]
    pub fn has_recursive_descent(&self) -> bool {
        self.selectors.iter().any(|s| matches!(s, JsonSelector::RecursiveDescent))
    }
    
    /// Get root selector (first non-root selector)
    #[inline]
    pub fn root_selector(&self) -> Option<&JsonSelector> {
        self.selectors.iter().find(|s| !matches!(s, JsonSelector::Root))
    }
    
    /// Get recursive descent start position
    #[inline]
    pub fn recursive_descent_start(&self) -> Option<usize> {
        self.selectors.iter().position(|s| matches!(s, JsonSelector::RecursiveDescent))
    }
    
    /// Calculate detailed complexity metrics
    pub fn calculate_complexity_metrics(&self) -> ComplexityMetrics {
        let mut metrics = ComplexityMetrics::new();
        
        for selector in &self.selectors {
            match selector {
                JsonSelector::RecursiveDescent => {
                    metrics.recursive_descent_depth = metrics.recursive_descent_depth.saturating_add(1);
                    metrics.total_selector_count = metrics.total_selector_count.saturating_add(1);
                }
                JsonSelector::Filter { expression } => {
                    metrics.filter_complexity_sum = metrics.filter_complexity_sum.saturating_add(
                        Self::calculate_filter_complexity(expression)
                    );
                    metrics.total_selector_count = metrics.total_selector_count.saturating_add(1);
                }
                JsonSelector::Slice { start, end, .. } => {
                    let range = match (start, end) {
                        (Some(s), Some(e)) => (e - s).abs() as u32,
                        _ => 100, // Default estimate for unbounded slices
                    };
                    metrics.max_slice_range = metrics.max_slice_range.max(range);
                    metrics.total_selector_count = metrics.total_selector_count.saturating_add(1);
                }
                JsonSelector::Union { selectors } => {
                    metrics.union_selector_count = metrics.union_selector_count.saturating_add(selectors.len() as u32);
                    metrics.total_selector_count = metrics.total_selector_count.saturating_add(1);
                }
                _ => {
                    metrics.total_selector_count = metrics.total_selector_count.saturating_add(1);
                }
            }
        }
        
        metrics
    }
    
    /// Calculate complexity score for a filter expression
    fn calculate_filter_complexity(expr: &FilterExpression) -> u32 {
        match expr {
            FilterExpression::Current | FilterExpression::Literal { .. } => 1,
            FilterExpression::Property { path } => path.len() as u32,
            FilterExpression::Comparison { left, right, .. } => {
                3 + Self::calculate_filter_complexity(left) + Self::calculate_filter_complexity(right)
            }
            FilterExpression::LogicalAnd { left, right } | FilterExpression::LogicalOr { left, right } => {
                2 + Self::calculate_filter_complexity(left) + Self::calculate_filter_complexity(right)
            }
            FilterExpression::Function { args, .. } => {
                5 + args.iter().map(|arg| Self::calculate_filter_complexity(arg)).sum::<u32>()
            }
        }
    }
    
    /// Evaluate filter expression against JSON context
    #[inline]
    pub fn evaluate_filter(&self, context: &serde_json::Value, filter: &FilterExpression) -> JsonPathResult<bool> {
        FilterEvaluator::evaluate_predicate(context, filter)
    }
}

/// JSONPath expression parser and compiler
pub struct JsonPathParser {
    /// Input JSONPath expression
    input: String,
    /// Tokenized input for parsing
    tokens: VecDeque<Token>,
    /// Current parser position for error reporting
    position: usize,
}

/// Lexical tokens for JSONPath parsing
#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// $ root identifier
    Root,
    /// . property separator
    Dot,
    /// .. recursive descent
    DoubleDot,
    /// [ bracket open
    LeftBracket,
    /// ] bracket close
    RightBracket,
    /// ( parenthesis open
    LeftParen,
    /// ) parenthesis close
    RightParen,
    /// * wildcard
    Star,
    /// ? filter marker
    Question,
    /// @ current node
    At,
    /// , comma separator
    Comma,
    /// : slice separator
    Colon,
    /// == equality
    Equal,
    /// != inequality
    NotEqual,
    /// < less than
    Less,
    /// <= less than or equal
    LessEq,
    /// > greater than
    Greater,
    /// >= greater than or equal
    GreaterEq,
    /// && logical and
    LogicalAnd,
    /// || logical or
    LogicalOr,
    /// String literal
    String(String),
    /// Integer literal
    Integer(i64),
    /// Number literal
    Number(f64),
    /// true literal
    True,
    /// false literal
    False,
    /// null literal
    Null,
    /// Identifier (for function names and properties)
    Identifier(String),
    /// End of input
    EOF,
}

/// Lexical tokens for JSONPath parsing
#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// $ root identifier
    Root,
    /// . property separator
    Dot,
    /// .. recursive descent
    DoubleDot,
    /// [ bracket open
    LeftBracket,
    /// ] bracket close
    RightBracket,
    /// ( parenthesis open
    LeftParen,
    /// ) parenthesis close
    RightParen,
    /// * wildcard
    Star,
    /// ? filter marker
    Question,
    /// @ current node
    At,
    /// , comma separator
    Comma,
    /// : slice separator
    Colon,
    /// == equality
    Equal,
    /// != inequality
    NotEqual,
    /// < less than
    Less,
    /// <= less than or equal
    LessEq,
    /// > greater than
    Greater,
    /// >= greater than or equal
    GreaterEq,
    /// && logical and
    LogicalAnd,
    /// || logical or
    LogicalOr,
    /// String literal
    String(String),
    /// Integer literal
    Integer(i64),
    /// Number literal
    Number(f64),
    /// true literal
    True,
    /// false literal
    False,
    /// null literal
    Null,
    /// Identifier (for function names)
    Identifier(String),
    /// End of input
    EOF,
}

impl JsonPathParser {
    /// Compile JSONPath expression into optimized runtime selectors
    pub fn compile(input: &str) -> JsonPathResult<JsonPathExpression> {
        let mut parser = Self {
            input: input.to_string(),
            tokens: VecDeque::new(),
            position: 0,
        };
        
        parser.tokenize()?;
        let selectors = parser.parse_expression()?;
        
        let is_array_stream = Self::is_array_streaming_expression(&selectors);
        
        Ok(JsonPathExpression {
            selectors,
            original: input.to_string(),
            is_array_stream,
        })
    }
    
    /// Check if expression pattern indicates array streaming optimization
    fn is_array_streaming_expression(selectors: &[JsonSelector]) -> bool {
        selectors.len() >= 2 && 
        matches!(selectors.last(), Some(JsonSelector::Wildcard) | Some(JsonSelector::Index { .. }))
    }
    
    /// Tokenize JSONPath expression into tokens
    fn tokenize(&mut self) -> JsonPathResult<()> {
        // Simplified tokenizer implementation for RFC 9535 function support
        let mut chars = self.input.chars().peekable();
        let mut position = 0;
        
        while let Some(ch) = chars.next() {
            match ch {
                '$' => self.tokens.push_back(Token::Root),
                '.' => {
                    if chars.peek() == Some(&'.') {
                        chars.next();
                        self.tokens.push_back(Token::DoubleDot);
                    } else {
                        self.tokens.push_back(Token::Dot);
                    }
                }
                '[' => self.tokens.push_back(Token::LeftBracket),
                ']' => self.tokens.push_back(Token::RightBracket),
                '(' => self.tokens.push_back(Token::LeftParen),
                ')' => self.tokens.push_back(Token::RightParen),
                '?' => self.tokens.push_back(Token::Question),
                '@' => self.tokens.push_back(Token::At),
                '*' => self.tokens.push_back(Token::Star),
                ',' => self.tokens.push_back(Token::Comma),
                ':' => self.tokens.push_back(Token::Colon),
                '=' => {
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
                }
                '|' => {
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
                }
                ch if ch.is_alphabetic() => {
                    let mut identifier = String::new();
                    identifier.push(ch);
                    
                    while let Some(&next_ch) = chars.peek() {
                        if next_ch.is_alphanumeric() || next_ch == '_' {
                            identifier.push(chars.next().unwrap());
                        } else {
                            break;
                        }
                    }
                    
                    match identifier.as_str() {
                        "true" => self.tokens.push_back(Token::True),
                        "false" => self.tokens.push_back(Token::False),
                        "null" => self.tokens.push_back(Token::Null),
                        _ => self.tokens.push_back(Token::Identifier(identifier)),
                    }
                }
                ch if ch.is_ascii_digit() || ch == '-' => {
                    let mut number_str = String::new();
                    number_str.push(ch);
                    
                    while let Some(&next_ch) = chars.peek() {
                        if next_ch.is_ascii_digit() || next_ch == '.' {
                            number_str.push(chars.next().unwrap());
                        } else {
                            break;
                        }
                    }
                    
                    if number_str.contains('.') {
                        if let Ok(num) = number_str.parse::<f64>() {
                            self.tokens.push_back(Token::Number(num));
                        } else {
                            return Err(invalid_expression_error(
                                &self.input,
                                "invalid number format",
                                Some(position),
                            ));
                        }
                    } else if let Ok(num) = number_str.parse::<i64>() {
                        self.tokens.push_back(Token::Integer(num));
                    } else {
                        return Err(invalid_expression_error(
                            &self.input,
                            "invalid integer format",
                            Some(position),
                        ));
                    }
                }
                ch if ch.is_whitespace() => {
                    // Skip whitespace
                }
                _ => {
                    return Err(invalid_expression_error(
                        &self.input,
                        &format!("unexpected character '{}'", ch),
                        Some(position),
                    ));
                }
            }
            position += 1;
        }
        
        self.tokens.push_back(Token::EOF);
        Ok(())
    }
    
    /// Parse expression into selector chain
    fn parse_expression(&mut self) -> JsonPathResult<Vec<JsonSelector>> {
        let mut selectors = Vec::new();
        
        // JSONPath must start with $ (root)
        if !matches!(self.tokens.front(), Some(Token::Root)) {
            return Err(invalid_expression_error(
                &self.input,
                "JSONPath expression must start with '$'",
                Some(0),
            ));
        }
        
        self.tokens.pop_front(); // consume $
        selectors.push(JsonSelector::Root);
        
        // Parse remaining selectors
        while !matches!(self.tokens.front(), Some(Token::EOF) | None) {
            let selector = self.parse_selector()?;
            selectors.push(selector);
        }
        
        if selectors.len() == 1 {
            return Err(invalid_expression_error(
                &self.input,
                "JSONPath expression cannot be just '$'",
                Some(0),
            ));
        }
        
        Ok(selectors)
    }
    
    /// Parse individual selector
    fn parse_selector(&mut self) -> JsonPathResult<JsonSelector> {
        match self.tokens.front() {
            Some(Token::Dot) => {
                self.tokens.pop_front();
                if matches!(self.tokens.front(), Some(Token::Dot)) {
                    self.tokens.pop_front();
                    Ok(JsonSelector::RecursiveDescent)
                } else if matches!(self.tokens.front(), Some(Token::Star)) {
                    self.tokens.pop_front();
                    Ok(JsonSelector::Wildcard)
                } else if let Some(Token::Identifier(name)) = self.tokens.front() {
                    let name = name.clone();
                    self.tokens.pop_front();
                    Ok(JsonSelector::Property { name })
                } else {
                    Err(invalid_expression_error(
                        &self.input,
                        "expected property name after '.'",
                        Some(self.position),
                    ))
                }
            }
            
            Some(Token::DoubleDot) => {
                self.tokens.pop_front();
                Ok(JsonSelector::RecursiveDescent)
            }
            
            Some(Token::LeftBracket) => {
                self.tokens.pop_front();
                self.parse_bracket_expression()
            }
            
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                self.tokens.pop_front();
                Ok(JsonSelector::Property { name })
            }
            
            _ => Err(invalid_expression_error(
                &self.input,
                "unexpected token in JSONPath expression",
                Some(self.position),
            )),
        }
    }
    
    /// Parse bracket expression ([...])
    fn parse_bracket_expression(&mut self) -> JsonPathResult<JsonSelector> {
        match self.tokens.front() {
            Some(Token::Star) => {
                self.tokens.pop_front();
                self.expect_token(Token::RightBracket)?;
                Ok(JsonSelector::Wildcard)
            }
            
            Some(Token::Integer(index)) => {
                let index = *index as i32;
                self.tokens.pop_front();
                
                if matches!(self.tokens.front(), Some(Token::Colon)) {
                    // Slice notation
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
                        from_end: index < 0 
                    })
                }
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
    }
    
    /// Parse logical AND expression
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
    }
    
    /// Parse primary expression (literals, property access, function calls)
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
            
            Some(Token::True) => {
                self.tokens.pop_front();
                Ok(FilterExpression::Literal {
                    value: FilterValue::Boolean(true),
                })
            }
            
            Some(Token::False) => {
                self.tokens.pop_front();
                Ok(FilterExpression::Literal {
                    value: FilterValue::Boolean(false),
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
                
                // Check if this is a function call
                if matches!(self.tokens.front(), Some(Token::LeftParen)) {
                    self.tokens.pop_front();
                    let args = self.parse_function_arguments()?;
                    self.expect_token(Token::RightParen)?;
                    Ok(FilterExpression::Function { name, args })
                } else {
                    // This is a bare identifier, treat as property
                    Ok(FilterExpression::Property {
                        path: vec![name],
                    })
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
                "expected property access, literal, or parenthesized expression",
                Some(self.position),
            )),
        }
    }
    
    /// Parse property path (@.prop.nested)
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