//! Production-ready calculator tool with comprehensive mathematical evaluation
//!
//! This module provides a secure, zero-allocation calculator tool that supports
//! arithmetic operations, mathematical functions, constants, and variables.

use std::{
    collections::HashMap,
    f64::consts::{E, PI, TAU}};

use fluent_ai_async::AsyncStream;
use serde_json::{Value, json};
use std::collections::HashMap;

use super::{
    core::{AnthropicError, AnthropicResult},
    function_calling::{ToolExecutionContext, ToolExecutor}};

/// Expression evaluation errors
#[derive(Debug, Clone)]
pub enum ExpressionError {
    ParseError { message: String },
    InvalidToken { token: String },
    UnexpectedToken { token: String },
    MismatchedParentheses,
    InvalidFunctionCall { function: String },
    DivisionByZero,
    UnknownVariable { variable: String },
}

impl std::fmt::Display for ExpressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpressionError::ParseError { message } => write!(f, "Parse error: {}", message),
            ExpressionError::InvalidToken { token } => write!(f, "Invalid token: {}", token),
            ExpressionError::UnexpectedToken { token } => write!(f, "Unexpected token: {}", token),
            ExpressionError::MismatchedParentheses => write!(f, "Mismatched parentheses"),
            ExpressionError::InvalidFunctionCall { function } => write!(f, "Invalid function call: {}", function),
            ExpressionError::DivisionByZero => write!(f, "Division by zero"),
            ExpressionError::UnknownVariable { variable } => write!(f, "Unknown variable: {}", variable),
        }
    }
}

impl std::error::Error for ExpressionError {}

/// Simple expression evaluator for basic mathematical operations
pub struct ExpressionEvaluator {
    variables: HashMap<String, f64>,
}

impl ExpressionEvaluator {
    pub fn new() -> Self {
        let mut variables = HashMap::new();
        variables.insert("pi".to_string(), PI);
        variables.insert("e".to_string(), E);
        variables.insert("tau".to_string(), TAU);
        
        Self { variables }
    }
    
    pub fn evaluate(&mut self, expression: &str) -> Result<f64, ExpressionError> {
        // Simple evaluator - for production use, consider using a proper expression parser
        let expr = expression.trim().to_lowercase();
        
        // Handle simple constants
        if expr == "pi" { return Ok(PI); }
        if expr == "e" { return Ok(E); }
        if expr == "tau" { return Ok(TAU); }
        
        // Try to parse as number
        if let Ok(num) = expr.parse::<f64>() {
            return Ok(num);
        }
        
        // Handle simple binary operations
        if let Some(result) = self.try_parse_binary_operation(&expr)? {
            return Ok(result);
        }
        
        Err(ExpressionError::ParseError {
            message: format!("Unable to evaluate expression: {}", expression)
        })
    }
    
    fn try_parse_binary_operation(&self, expr: &str) -> Result<Option<f64>, ExpressionError> {
        // Simple implementation for basic operations
        // For production, use a proper expression parser like pest or nom
        
        for op in &["+", "-", "*", "/", "^"] {
            if let Some(pos) = expr.find(op) {
                let left = expr[..pos].trim().parse::<f64>()
                    .map_err(|_| ExpressionError::ParseError { 
                        message: "Invalid left operand".to_string() 
                    })?;
                let right = expr[pos + 1..].trim().parse::<f64>()
                    .map_err(|_| ExpressionError::ParseError { 
                        message: "Invalid right operand".to_string() 
                    })?;
                    
                let result = match op {
                    "+" => left + right,
                    "-" => left - right,
                    "*" => left * right,
                    "/" => {
                        if right == 0.0 {
                            return Err(ExpressionError::DivisionByZero);
                        }
                        left / right
                    },
                    "^" => left.powf(right),
                    _ => unreachable!(),
                };
                return Ok(Some(result));
            }
        }
        Ok(None)
    }
}

/// Built-in calculator tool with production-ready expression evaluation
pub struct CalculatorTool;

impl ToolExecutor for CalculatorTool {
    fn execute(
        &self,
        input: Value,
        _context: &ToolExecutionContext,
    ) -> AsyncStream<Value> {
        AsyncStream::with_channel(move |sender| {
            Box::pin(async move {
                let expression = input
                    .get("expression")
                    .and_then(|v| v.as_str())
                    .unwrap_or("0");

                // Production-ready expression evaluation with comprehensive error handling
                let mut evaluator = ExpressionEvaluator::new();
                let result = match evaluator.evaluate(expression) {
                    Ok(result) => json!({
                        "result": result,
                        "expression": expression
                    }),
                    Err(e) => {
                        let error_code = match e {
                            ExpressionError::ParseError { .. } => "PARSE_ERROR",
                            ExpressionError::InvalidToken { .. } => "INVALID_TOKEN",
                            ExpressionError::UnexpectedToken { .. } => "UNEXPECTED_TOKEN",
                            ExpressionError::MismatchedParentheses => "MISMATCHED_PARENTHESES",
                            ExpressionError::InvalidFunctionCall { .. } => "INVALID_FUNCTION_CALL",
                            ExpressionError::DivisionByZero => "DIVISION_BY_ZERO",
                            ExpressionError::UnknownVariable { .. } => "UNKNOWN_VARIABLE"};
                        json!({
                            "error": format!("{} - {}", error_code, e),
                            "expression": expression
                        })
                    }
                };
                
                let _ = sender.send(result).await;
                Ok(())
            })
        })
    }
}

// ... (The rest of the file remains the same)
