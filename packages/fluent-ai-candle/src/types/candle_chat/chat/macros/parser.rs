//! Macro parsing and syntax analysis
//!
//! This module handles parsing macro definitions, validating syntax,
//! and converting text-based macro descriptions into executable structures.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use uuid::Uuid;

use super::types::*;
use crate::types::candle_chat::chat::commands::ImmutableChatCommand;

/// Macro parser for converting text definitions to executable macros
#[derive(Debug)]
pub struct MacroParser {
    /// Configuration for parsing behavior
    config: MacroParserConfig,
    /// Cache of parsed expressions for performance
    expression_cache: HashMap<String, ParsedExpression>,
}

/// Configuration for macro parser behavior
#[derive(Debug, Clone)]
pub struct MacroParserConfig {
    /// Enable strict syntax validation
    pub strict_validation: bool,
    /// Maximum expression length
    pub max_expression_length: usize,
    /// Enable expression caching
    pub enable_caching: bool,
    /// Allow custom functions in expressions
    pub allow_custom_functions: bool,
}

/// Parsed expression for condition evaluation
#[derive(Debug, Clone)]
pub enum ParsedExpression {
    /// Variable reference
    Variable(String),
    /// String literal
    Literal(String),
    /// Numeric literal
    Number(f64),
    /// Boolean literal
    Boolean(bool),
    /// Binary operation
    BinaryOp {
        operator: BinaryOperator,
        left: Box<ParsedExpression>,
        right: Box<ParsedExpression>,
    },
    /// Unary operation
    UnaryOp {
        operator: UnaryOperator,
        operand: Box<ParsedExpression>,
    },
    /// Function call
    FunctionCall {
        name: String,
        args: Vec<ParsedExpression>,
    },
}

/// Binary operators for expressions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryOperator {
    /// Equality comparison
    Equal,
    /// Inequality comparison
    NotEqual,
    /// Less than comparison
    LessThan,
    /// Greater than comparison
    GreaterThan,
    /// Less than or equal comparison
    LessThanOrEqual,
    /// Greater than or equal comparison
    GreaterThanOrEqual,
    /// Logical AND
    And,
    /// Logical OR
    Or,
    /// String contains
    Contains,
    /// String matches regex
    Matches,
    /// Addition
    Add,
    /// Subtraction
    Subtract,
    /// Multiplication
    Multiply,
    /// Division
    Divide,
}

/// Unary operators for expressions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnaryOperator {
    /// Logical NOT
    Not,
    /// Numeric negation
    Negate,
    /// Check if empty
    IsEmpty,
    /// Check if not empty
    IsNotEmpty,
}

/// Result of parsing a macro definition
#[derive(Debug, Clone)]
pub struct ParseResult {
    /// Successfully parsed macro
    pub macro_def: Option<StoredMacro>,
    /// Parsing errors encountered
    pub errors: Vec<ParseError>,
    /// Warnings during parsing
    pub warnings: Vec<ParseWarning>,
}

/// Parsing error information
#[derive(Debug, Clone)]
pub struct ParseError {
    /// Error message
    pub message: String,
    /// Line number where error occurred
    pub line: usize,
    /// Column number where error occurred
    pub column: usize,
    /// Error type classification
    pub error_type: ParseErrorType,
}

/// Types of parsing errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseErrorType {
    /// Syntax error in macro definition
    SyntaxError,
    /// Unknown command or action
    UnknownAction,
    /// Invalid variable name
    InvalidVariable,
    /// Invalid expression syntax
    InvalidExpression,
    /// Missing required parameter
    MissingParameter,
    /// Invalid parameter value
    InvalidParameter,
    /// Circular reference detected
    CircularReference,
}

/// Parsing warning information
#[derive(Debug, Clone)]
pub struct ParseWarning {
    /// Warning message
    pub message: String,
    /// Line number where warning occurred
    pub line: usize,
    /// Column number where warning occurred
    pub column: usize,
    /// Warning type classification
    pub warning_type: ParseWarningType,
}

/// Types of parsing warnings
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseWarningType {
    /// Unused variable
    UnusedVariable,
    /// Deprecated syntax
    DeprecatedSyntax,
    /// Performance warning
    PerformanceWarning,
    /// Style recommendation
    StyleRecommendation,
}

impl MacroParser {
    /// Create a new macro parser with default configuration
    pub fn new() -> Self {
        Self {
            config: MacroParserConfig::default(),
            expression_cache: HashMap::new(),
        }
    }

    /// Create a new macro parser with custom configuration
    pub fn with_config(config: MacroParserConfig) -> Self {
        Self {
            config,
            expression_cache: HashMap::new(),
        }
    }

    /// Parse a macro definition from text
    pub fn parse_macro(&mut self, definition: &str) -> ParseResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut actions = Vec::new();

        // Parse line by line
        for (line_num, line) in definition.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue; // Skip empty lines and comments
            }

            match self.parse_action_line(line, line_num + 1) {
                Ok(action) => actions.push(action),
                Err(error) => errors.push(error),
            }
        }

        let macro_def = if errors.is_empty() {
            Some(StoredMacro {
                metadata: MacroMetadata {
                    id: Uuid::new_v4(),
                    name: Arc::from("Parsed Macro"),
                    description: None,
                    tags: Vec::new(),
                    created_at: std::time::Instant::now(),
                    modified_at: std::time::Instant::now(),
                    author: None,
                    version: 1,
                },
                actions: actions.into(),
                variables: HashMap::new(),
            })
        } else {
            None
        };

        ParseResult {
            macro_def,
            errors,
            warnings,
        }
    }

    /// Parse a single action line
    fn parse_action_line(&mut self, line: &str, line_num: usize) -> Result<MacroAction, ParseError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Err(ParseError {
                message: "Empty action line".to_string(),
                line: line_num,
                column: 1,
                error_type: ParseErrorType::SyntaxError,
            });
        }

        let action_type = parts[0].to_lowercase();
        let timestamp = Duration::from_millis(0); // Default timestamp

        match action_type.as_str() {
            "send" => self.parse_send_action(&parts[1..], line_num, timestamp),
            "wait" => self.parse_wait_action(&parts[1..], line_num, timestamp),
            "set" => self.parse_set_action(&parts[1..], line_num, timestamp),
            "if" => self.parse_conditional_action(&parts[1..], line_num, timestamp),
            "loop" => self.parse_loop_action(&parts[1..], line_num, timestamp),
            "execute" => self.parse_execute_action(&parts[1..], line_num, timestamp),
            _ => Err(ParseError {
                message: format!("Unknown action type: {}", action_type),
                line: line_num,
                column: 1,
                error_type: ParseErrorType::UnknownAction,
            }),
        }
    }

    /// Parse a send message action
    fn parse_send_action(&self, parts: &[&str], line_num: usize, timestamp: Duration) -> Result<MacroAction, ParseError> {
        if parts.len() < 2 {
            return Err(ParseError {
                message: "Send action requires message type and content".to_string(),
                line: line_num,
                column: 1,
                error_type: ParseErrorType::MissingParameter,
            });
        }

        let message_type = Arc::from(parts[0]);
        let content = Arc::from(parts[1..].join(" "));

        Ok(MacroAction::SendMessage {
            content,
            message_type,
            timestamp,
        })
    }

    /// Parse a wait action
    fn parse_wait_action(&self, parts: &[&str], line_num: usize, timestamp: Duration) -> Result<MacroAction, ParseError> {
        if parts.is_empty() {
            return Err(ParseError {
                message: "Wait action requires duration".to_string(),
                line: line_num,
                column: 1,
                error_type: ParseErrorType::MissingParameter,
            });
        }

        let duration_str = parts[0];
        let duration = self.parse_duration(duration_str).map_err(|_| ParseError {
            message: format!("Invalid duration format: {}", duration_str),
            line: line_num,
            column: 1,
            error_type: ParseErrorType::InvalidParameter,
        })?;

        Ok(MacroAction::Wait {
            duration,
            timestamp,
        })
    }

    /// Parse a set variable action
    fn parse_set_action(&self, parts: &[&str], line_num: usize, timestamp: Duration) -> Result<MacroAction, ParseError> {
        if parts.len() < 2 {
            return Err(ParseError {
                message: "Set action requires variable name and value".to_string(),
                line: line_num,
                column: 1,
                error_type: ParseErrorType::MissingParameter,
            });
        }

        let name = Arc::from(parts[0]);
        let value = Arc::from(parts[1..].join(" "));

        Ok(MacroAction::SetVariable {
            name,
            value,
            timestamp,
        })
    }

    /// Parse a conditional action
    fn parse_conditional_action(&self, parts: &[&str], line_num: usize, timestamp: Duration) -> Result<MacroAction, ParseError> {
        if parts.is_empty() {
            return Err(ParseError {
                message: "Conditional action requires condition".to_string(),
                line: line_num,
                column: 1,
                error_type: ParseErrorType::MissingParameter,
            });
        }

        let condition = Arc::from(parts.join(" "));
        // Simplified: empty action arrays for now
        let then_actions = Arc::from([]);
        let else_actions = None;

        Ok(MacroAction::Conditional {
            condition,
            then_actions,
            else_actions,
            timestamp,
        })
    }

    /// Parse a loop action
    fn parse_loop_action(&self, parts: &[&str], line_num: usize, timestamp: Duration) -> Result<MacroAction, ParseError> {
        if parts.is_empty() {
            return Err(ParseError {
                message: "Loop action requires iteration count".to_string(),
                line: line_num,
                column: 1,
                error_type: ParseErrorType::MissingParameter,
            });
        }

        let iterations = parts[0].parse::<u32>().map_err(|_| ParseError {
            message: format!("Invalid iteration count: {}", parts[0]),
            line: line_num,
            column: 1,
            error_type: ParseErrorType::InvalidParameter,
        })?;

        let actions = Arc::from([]); // Simplified: empty actions for now

        Ok(MacroAction::Loop {
            iterations,
            actions,
            timestamp,
        })
    }

    /// Parse an execute command action
    fn parse_execute_action(&self, parts: &[&str], line_num: usize, timestamp: Duration) -> Result<MacroAction, ParseError> {
        if parts.is_empty() {
            return Err(ParseError {
                message: "Execute action requires command".to_string(),
                line: line_num,
                column: 1,
                error_type: ParseErrorType::MissingParameter,
            });
        }

        // Create a simple command (actual implementation would parse properly)
        let command = ImmutableChatCommand::new(parts.join(" "));

        Ok(MacroAction::ExecuteCommand {
            command,
            timestamp,
        })
    }

    /// Parse a duration string (e.g., "5s", "1m", "100ms")
    fn parse_duration(&self, duration_str: &str) -> Result<Duration, ParseError> {
        if duration_str.ends_with("ms") {
            let value = duration_str.trim_end_matches("ms").parse::<u64>()?;
            Ok(Duration::from_millis(value))
        } else if duration_str.ends_with('s') {
            let value = duration_str.trim_end_matches('s').parse::<u64>()?;
            Ok(Duration::from_secs(value))
        } else if duration_str.ends_with('m') {
            let value = duration_str.trim_end_matches('m').parse::<u64>()?;
            Ok(Duration::from_secs(value * 60))
        } else {
            // Default to milliseconds
            let value = duration_str.parse::<u64>()?;
            Ok(Duration::from_millis(value))
        }
    }

    /// Parse an expression for condition evaluation
    pub fn parse_expression(&mut self, expression: &str) -> Result<ParsedExpression, ParseError> {
        if self.config.enable_caching {
            if let Some(cached) = self.expression_cache.get(expression) {
                return Ok(cached.clone());
            }
        }

        let parsed = self.parse_expression_internal(expression)?;

        if self.config.enable_caching && expression.len() <= self.config.max_expression_length {
            self.expression_cache.insert(expression.to_string(), parsed.clone());
        }

        Ok(parsed)
    }

    /// Internal expression parsing implementation
    fn parse_expression_internal(&self, expression: &str) -> Result<ParsedExpression, ParseError> {
        let expression = expression.trim();

        // Simple parsing - real implementation would use proper tokenization
        if expression.starts_with('"') && expression.ends_with('"') {
            // String literal
            Ok(ParsedExpression::Literal(expression[1..expression.len()-1].to_string()))
        } else if expression == "true" || expression == "false" {
            // Boolean literal
            Ok(ParsedExpression::Boolean(expression == "true"))
        } else if let Ok(number) = expression.parse::<f64>() {
            // Numeric literal
            Ok(ParsedExpression::Number(number))
        } else if expression.starts_with('$') {
            // Variable reference
            Ok(ParsedExpression::Variable(expression[1..].to_string()))
        } else {
            // Default to variable for now
            Ok(ParsedExpression::Variable(expression.to_string()))
        }
    }

    /// Clear the expression cache
    pub fn clear_cache(&mut self) {
        self.expression_cache.clear();
    }
}

impl Default for MacroParserConfig {
    fn default() -> Self {
        Self {
            strict_validation: true,
            max_expression_length: 1024,
            enable_caching: true,
            allow_custom_functions: false,
        }
    }
}

impl Default for MacroParser {
    fn default() -> Self {
        Self::new()
    }
}

// Helper trait for ParseError to support ? operator with std::num::ParseIntError
impl From<std::num::ParseIntError> for ParseError {
    fn from(error: std::num::ParseIntError) -> Self {
        ParseError {
            message: error.to_string(),
            line: 0,
            column: 0,
            error_type: ParseErrorType::InvalidParameter,
        }
    }
}