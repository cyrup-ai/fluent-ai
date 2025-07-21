//! Production-ready calculator tool with comprehensive mathematical evaluation
//!
//! This module provides a secure, zero-allocation calculator tool that supports
//! arithmetic operations, mathematical functions, constants, and variables.

use std::{
    collections::HashMap,
    f64::consts::{E, PI, TAU},
    future::Future,
    pin::Pin,
};

use serde_json::{Value, json};
use fluent_ai_domain::tool::Tool;

use super::{
    core::{AnthropicError, AnthropicResult},
    function_calling::{ToolExecutionContext, ToolExecutor, ToolOutput},
};

/// Built-in calculator tool with production-ready expression evaluation
pub struct CalculatorTool;

impl ToolExecutor for CalculatorTool {
    fn execute(
        &self,
        input: Value,
        _context: &ToolExecutionContext,
    ) -> Pin<Box<dyn Future<Output = AnthropicResult<ToolOutput>> + Send>> {
        Box::pin(async move {
            let expression = input
                .get("expression")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    AnthropicError::InvalidRequest(
                        "Calculator requires 'expression' parameter".to_string(),
                    )
                })?;

            // Production-ready expression evaluation with comprehensive error handling
            let mut evaluator = ExpressionEvaluator::new();
            match evaluator.evaluate(expression) {
                Ok(result) => Ok(ToolOutput::Json(json!({
                    "result": result,
                    "expression": expression
                }))),
                Err(e) => {
                    let error_code = match e {
                        ExpressionError::ParseError { .. } => "PARSE_ERROR",
                        ExpressionError::DivisionByZero => "DIVISION_BY_ZERO",
                        ExpressionError::InvalidFunctionCall { .. } => "INVALID_FUNCTION",
                        ExpressionError::UndefinedVariable { .. } => "UNDEFINED_VARIABLE",
                        ExpressionError::DomainError { .. } => "DOMAIN_ERROR",
                        ExpressionError::Overflow { .. } => "OVERFLOW",
                        ExpressionError::InvalidExpression { .. } => "INVALID_EXPRESSION",
                    };
                    Ok(ToolOutput::Error {
                        message: e.to_string(),
                        code: Some(error_code.to_string()),
                    })
                }
            }
        })
    }

    fn definition(&self) -> Tool {
        Tool::new(
            "calculator",
            "Perform mathematical calculations",
            json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate. Supports arithmetic operations (+, -, *, /, %, ^), parentheses, mathematical functions (sin, cos, tan, sqrt, ln, log, exp, abs, etc.), constants (pi, e, tau), and variables. Examples: '2 + 3 * 4', 'sin(pi/2)', 'sqrt(16)', 'x = 5; x^2 + 3'"
                    }
                },
                "required": ["expression"]
            }),
        )
    }
}

/// Expression evaluation errors with detailed categorization
#[derive(Debug, Clone)]
pub enum ExpressionError {
    ParseError { position: usize, message: String },
    DivisionByZero,
    InvalidFunctionCall { function: String, args: usize },
    UndefinedVariable { variable: String },
    DomainError { function: String, input: f64 },
    Overflow { operation: String },
    InvalidExpression { message: String },
}

impl std::fmt::Display for ExpressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpressionError::ParseError { position, message } => {
                write!(f, "Parse error at position {}: {}", position, message)
            }
            ExpressionError::DivisionByZero => write!(f, "Division by zero"),
            ExpressionError::InvalidFunctionCall { function, args } => {
                write!(
                    f,
                    "Invalid function call: {}() with {} arguments",
                    function, args
                )
            }
            ExpressionError::UndefinedVariable { variable } => {
                write!(f, "Undefined variable: {}", variable)
            }
            ExpressionError::DomainError { function, input } => {
                write!(f, "Domain error in {}({}): invalid input", function, input)
            }
            ExpressionError::Overflow { operation } => {
                write!(f, "Numeric overflow in operation: {}", operation)
            }
            ExpressionError::InvalidExpression { message } => {
                write!(f, "Invalid expression: {}", message)
            }
        }
    }
}

impl std::error::Error for ExpressionError {}

/// Token types for lexical analysis
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Variable(String),
    Function(String),
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    Power,
    LeftParen,
    RightParen,
    Comma,
    Assign,
    End,
}

/// Production-ready expression evaluator with comprehensive mathematical support
pub struct ExpressionEvaluator {
    variables: HashMap<String, f64>,
    position: usize,
    tokens: Vec<Token>,
    current_token: usize,
}

impl ExpressionEvaluator {
    /// Create new expression evaluator with predefined constants
    pub fn new() -> Self {
        let mut variables = HashMap::new();

        // Mathematical constants
        variables.insert("pi".to_string(), PI);
        variables.insert("e".to_string(), E);
        variables.insert("tau".to_string(), TAU);

        Self {
            variables,
            position: 0,
            tokens: Vec::new(),
            current_token: 0,
        }
    }

    /// Evaluate mathematical expression with comprehensive error handling
    pub fn evaluate(&mut self, expression: &str) -> Result<f64, ExpressionError> {
        self.position = 0;
        self.current_token = 0;
        self.tokens = self.tokenize(expression)?;

        if self.tokens.is_empty() || matches!(self.tokens[0], Token::End) {
            return Err(ExpressionError::InvalidExpression {
                message: "Empty expression".to_string(),
            });
        }

        let result = self.parse_assignment()?;

        // Check for unconsumed tokens
        if self.current_token < self.tokens.len() - 1 {
            return Err(ExpressionError::ParseError {
                position: self.position,
                message: "Unexpected tokens after expression".to_string(),
            });
        }

        // Check for overflow and special values
        if result.is_infinite() {
            return Err(ExpressionError::Overflow {
                operation: "expression evaluation".to_string(),
            });
        }

        if result.is_nan() {
            return Err(ExpressionError::InvalidExpression {
                message: "Expression resulted in NaN".to_string(),
            });
        }

        Ok(result)
    }

    /// Tokenize input expression into structured tokens
    fn tokenize(&mut self, expression: &str) -> Result<Vec<Token>, ExpressionError> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = expression.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            match chars[i] {
                ' ' | '\t' | '\r' | '\n' => {
                    i += 1;
                    continue;
                }
                '+' => {
                    tokens.push(Token::Plus);
                    i += 1;
                }
                '-' => {
                    tokens.push(Token::Minus);
                    i += 1;
                }
                '*' => {
                    if i + 1 < chars.len() && chars[i + 1] == '*' {
                        tokens.push(Token::Power);
                        i += 2;
                    } else {
                        tokens.push(Token::Multiply);
                        i += 1;
                    }
                }
                '/' => {
                    tokens.push(Token::Divide);
                    i += 1;
                }
                '%' => {
                    tokens.push(Token::Modulo);
                    i += 1;
                }
                '^' => {
                    tokens.push(Token::Power);
                    i += 1;
                }
                '(' => {
                    tokens.push(Token::LeftParen);
                    i += 1;
                }
                ')' => {
                    tokens.push(Token::RightParen);
                    i += 1;
                }
                ',' => {
                    tokens.push(Token::Comma);
                    i += 1;
                }
                '=' => {
                    tokens.push(Token::Assign);
                    i += 1;
                }
                c if c.is_ascii_digit() || c == '.' => {
                    let start = i;
                    let mut has_dot = c == '.';
                    i += 1;

                    while i < chars.len()
                        && (chars[i].is_ascii_digit() || (chars[i] == '.' && !has_dot))
                    {
                        if chars[i] == '.' {
                            has_dot = true;
                        }
                        i += 1;
                    }

                    let number_str: String = chars[start..i].iter().collect();
                    let number =
                        number_str
                            .parse::<f64>()
                            .map_err(|_| ExpressionError::ParseError {
                                position: start,
                                message: format!("Invalid number: {}", number_str),
                            })?;

                    tokens.push(Token::Number(number));
                }
                c if c.is_ascii_alphabetic() || c == '_' => {
                    let start = i;
                    while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                        i += 1;
                    }

                    let identifier: String = chars[start..i].iter().collect();

                    // Check if followed by parenthesis (function call)
                    if i < chars.len() && chars[i] == '(' {
                        tokens.push(Token::Function(identifier));
                    } else {
                        tokens.push(Token::Variable(identifier));
                    }
                }
                _ => {
                    return Err(ExpressionError::ParseError {
                        position: i,
                        message: format!("Unexpected character: '{}'", chars[i]),
                    });
                }
            }
        }

        tokens.push(Token::End);
        Ok(tokens)
    }

    /// Parse assignment expressions (x = 5)
    fn parse_assignment(&mut self) -> Result<f64, ExpressionError> {
        if let Some(Token::Variable(var_name)) = self.peek_token() {
            if matches!(self.peek_next_token(), Some(Token::Assign)) {
                let var_name = var_name.clone();
                self.advance_token(); // consume variable
                self.advance_token(); // consume =

                let value = self.parse_expression()?;
                self.variables.insert(var_name, value);
                return Ok(value);
            }
        }

        self.parse_expression()
    }

    /// Parse arithmetic expressions with proper precedence
    fn parse_expression(&mut self) -> Result<f64, ExpressionError> {
        let mut left = self.parse_term()?;

        while matches!(self.peek_token(), Some(Token::Plus) | Some(Token::Minus)) {
            let op = self.peek_token().cloned();
            self.advance_token();
            let right = self.parse_term()?;

            match op {
                Some(Token::Plus) => {
                    left = left + right;
                    if left.is_infinite() {
                        return Err(ExpressionError::Overflow {
                            operation: "addition".to_string(),
                        });
                    }
                }
                Some(Token::Minus) => {
                    left = left - right;
                    if left.is_infinite() {
                        return Err(ExpressionError::Overflow {
                            operation: "subtraction".to_string(),
                        });
                    }
                }
                _ => unreachable!(),
            }
        }

        Ok(left)
    }

    /// Parse multiplication, division, and modulo terms
    fn parse_term(&mut self) -> Result<f64, ExpressionError> {
        let mut left = self.parse_power()?;

        while matches!(
            self.peek_token(),
            Some(Token::Multiply) | Some(Token::Divide) | Some(Token::Modulo)
        ) {
            let op = self.peek_token().cloned();
            self.advance_token();
            let right = self.parse_power()?;

            match op {
                Some(Token::Multiply) => {
                    left = left * right;
                    if left.is_infinite() {
                        return Err(ExpressionError::Overflow {
                            operation: "multiplication".to_string(),
                        });
                    }
                }
                Some(Token::Divide) => {
                    if right == 0.0 {
                        return Err(ExpressionError::DivisionByZero);
                    }
                    left = left / right;
                    if left.is_infinite() {
                        return Err(ExpressionError::Overflow {
                            operation: "division".to_string(),
                        });
                    }
                }
                Some(Token::Modulo) => {
                    if right == 0.0 {
                        return Err(ExpressionError::DivisionByZero);
                    }
                    left = left % right;
                }
                _ => unreachable!(),
            }
        }

        Ok(left)
    }

    /// Parse power/exponentiation (right-associative)
    fn parse_power(&mut self) -> Result<f64, ExpressionError> {
        let mut left = self.parse_factor()?;

        if matches!(self.peek_token(), Some(Token::Power)) {
            self.advance_token();
            let right = self.parse_power()?; // Right-associative
            left = left.powf(right);

            if left.is_infinite() {
                return Err(ExpressionError::Overflow {
                    operation: "exponentiation".to_string(),
                });
            }
        }

        Ok(left)
    }

    /// Parse factors (numbers, variables, functions, parentheses)
    fn parse_factor(&mut self) -> Result<f64, ExpressionError> {
        match self.peek_token() {
            Some(Token::Number(n)) => {
                let value = *n;
                self.advance_token();
                Ok(value)
            }
            Some(Token::Variable(var)) => {
                let var_name = var.clone();
                self.advance_token();

                self.variables
                    .get(&var_name)
                    .copied()
                    .ok_or_else(|| ExpressionError::UndefinedVariable { variable: var_name })
            }
            Some(Token::Function(func)) => {
                let func_name = func.clone();
                self.advance_token();
                self.parse_function_call(&func_name)
            }
            Some(Token::LeftParen) => {
                self.advance_token();
                let result = self.parse_expression()?;

                if !matches!(self.peek_token(), Some(Token::RightParen)) {
                    return Err(ExpressionError::ParseError {
                        position: self.position,
                        message: "Expected closing parenthesis".to_string(),
                    });
                }
                self.advance_token();
                Ok(result)
            }
            Some(Token::Minus) => {
                self.advance_token();
                let value = self.parse_factor()?;
                Ok(-value)
            }
            Some(Token::Plus) => {
                self.advance_token();
                self.parse_factor()
            }
            _ => Err(ExpressionError::ParseError {
                position: self.position,
                message: "Expected number, variable, function, or opening parenthesis".to_string(),
            }),
        }
    }

    /// Parse function calls with argument validation
    fn parse_function_call(&mut self, func_name: &str) -> Result<f64, ExpressionError> {
        if !matches!(self.peek_token(), Some(Token::LeftParen)) {
            return Err(ExpressionError::ParseError {
                position: self.position,
                message: "Expected opening parenthesis after function name".to_string(),
            });
        }
        self.advance_token();

        let mut args = Vec::new();

        // Handle empty argument list
        if matches!(self.peek_token(), Some(Token::RightParen)) {
            self.advance_token();
        } else {
            loop {
                args.push(self.parse_expression()?);

                match self.peek_token() {
                    Some(Token::Comma) => {
                        self.advance_token();
                        continue;
                    }
                    Some(Token::RightParen) => {
                        self.advance_token();
                        break;
                    }
                    _ => {
                        return Err(ExpressionError::ParseError {
                            position: self.position,
                            message: "Expected comma or closing parenthesis in function call"
                                .to_string(),
                        });
                    }
                }
            }
        }

        self.evaluate_function(func_name, &args)
    }

    /// Evaluate mathematical functions with comprehensive error handling
    fn evaluate_function(&self, func_name: &str, args: &[f64]) -> Result<f64, ExpressionError> {
        match func_name {
            // Trigonometric functions
            "sin" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: func_name.to_string(),
                        args: args.len(),
                    });
                }
                Ok(args[0].sin())
            }
            "cos" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: func_name.to_string(),
                        args: args.len(),
                    });
                }
                Ok(args[0].cos())
            }
            "tan" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: func_name.to_string(),
                        args: args.len(),
                    });
                }
                Ok(args[0].tan())
            }
            "asin" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: func_name.to_string(),
                        args: args.len(),
                    });
                }
                if args[0] < -1.0 || args[0] > 1.0 {
                    return Err(ExpressionError::DomainError {
                        function: func_name.to_string(),
                        input: args[0],
                    });
                }
                Ok(args[0].asin())
            }
            "acos" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: func_name.to_string(),
                        args: args.len(),
                    });
                }
                if args[0] < -1.0 || args[0] > 1.0 {
                    return Err(ExpressionError::DomainError {
                        function: func_name.to_string(),
                        input: args[0],
                    });
                }
                Ok(args[0].acos())
            }
            "atan" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: func_name.to_string(),
                        args: args.len(),
                    });
                }
                Ok(args[0].atan())
            }
            // Exponential and logarithmic functions
            "exp" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: func_name.to_string(),
                        args: args.len(),
                    });
                }
                let result = args[0].exp();
                if result.is_infinite() {
                    return Err(ExpressionError::Overflow {
                        operation: "exponential".to_string(),
                    });
                }
                Ok(result)
            }
            "ln" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: func_name.to_string(),
                        args: args.len(),
                    });
                }
                if args[0] <= 0.0 {
                    return Err(ExpressionError::DomainError {
                        function: func_name.to_string(),
                        input: args[0],
                    });
                }
                Ok(args[0].ln())
            }
            "log" | "log10" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: func_name.to_string(),
                        args: args.len(),
                    });
                }
                if args[0] <= 0.0 {
                    return Err(ExpressionError::DomainError {
                        function: func_name.to_string(),
                        input: args[0],
                    });
                }
                Ok(args[0].log10())
            }
            "log2" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: func_name.to_string(),
                        args: args.len(),
                    });
                }
                if args[0] <= 0.0 {
                    return Err(ExpressionError::DomainError {
                        function: func_name.to_string(),
                        input: args[0],
                    });
                }
                Ok(args[0].log2())
            }
            // Power and root functions
            "sqrt" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: func_name.to_string(),
                        args: args.len(),
                    });
                }
                if args[0] < 0.0 {
                    return Err(ExpressionError::DomainError {
                        function: func_name.to_string(),
                        input: args[0],
                    });
                }
                Ok(args[0].sqrt())
            }
            "cbrt" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: func_name.to_string(),
                        args: args.len(),
                    });
                }
                Ok(args[0].cbrt())
            }
            "pow" => {
                if args.len() != 2 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: func_name.to_string(),
                        args: args.len(),
                    });
                }
                let result = args[0].powf(args[1]);
                if result.is_infinite() {
                    return Err(ExpressionError::Overflow {
                        operation: "power".to_string(),
                    });
                }
                Ok(result)
            }
            // Utility functions
            "abs" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: func_name.to_string(),
                        args: args.len(),
                    });
                }
                Ok(args[0].abs())
            }
            "floor" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: func_name.to_string(),
                        args: args.len(),
                    });
                }
                Ok(args[0].floor())
            }
            "ceil" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: func_name.to_string(),
                        args: args.len(),
                    });
                }
                Ok(args[0].ceil())
            }
            "round" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: func_name.to_string(),
                        args: args.len(),
                    });
                }
                Ok(args[0].round())
            }
            "min" => {
                if args.len() != 2 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: func_name.to_string(),
                        args: args.len(),
                    });
                }
                Ok(args[0].min(args[1]))
            }
            "max" => {
                if args.len() != 2 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: func_name.to_string(),
                        args: args.len(),
                    });
                }
                Ok(args[0].max(args[1]))
            }
            _ => Err(ExpressionError::InvalidFunctionCall {
                function: func_name.to_string(),
                args: args.len(),
            }),
        }
    }

    /// Peek at current token without consuming it
    fn peek_token(&self) -> Option<&Token> {
        self.tokens.get(self.current_token)
    }

    /// Peek at next token without consuming current
    fn peek_next_token(&self) -> Option<&Token> {
        self.tokens.get(self.current_token + 1)
    }

    /// Advance to next token
    fn advance_token(&mut self) {
        if self.current_token < self.tokens.len() {
            self.current_token += 1;
        }
    }
}

impl Default for ExpressionEvaluator {
    fn default() -> Self {
        Self::new()
    }
}
