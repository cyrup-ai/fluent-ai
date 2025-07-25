//! Production-ready calculator tool with comprehensive mathematical evaluation
//!
//! This module provides a secure, zero-allocation calculator tool that supports
//! arithmetic operations, mathematical functions, constants, and variables.

use std::{
    collections::HashMap,
    f64::consts::{E, PI, TAU},
};

use fluent_ai_async::AsyncStream;
use fluent_ai_async::channel;
use fluent_ai_domain::tool::Tool;
use serde_json::{Value, json};

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
    ) -> AsyncStream<Value> {
        let (tx, stream) = channel();
        tokio::spawn(async move {
            let result = async {
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
                    Ok(result) => Ok(json!({
                        "result": result,
                        "expression": expression
                    })),
                    Err(e) => {
                        let error_code = match e {
                            ExpressionError::ParseError { .. } => "PARSE_ERROR",
                            ExpressionError::InvalidToken { .. } => "INVALID_TOKEN",
                            ExpressionError::UnexpectedToken { .. } => "UNEXPECTED_TOKEN",
                            ExpressionError::MismatchedParentheses => "MISMATCHED_PARENTHESES",
                            ExpressionError::InvalidFunctionCall { .. } => "INVALID_FUNCTION_CALL",
                            ExpressionError::DivisionByZero => "DIVISION_BY_ZERO",
                            ExpressionError::UnknownVariable { .. } => "UNKNOWN_VARIABLE",
                        };
                        Err(AnthropicError::ToolExecutionError {
                            tool_name: "calculator".to_string(),
                            error: format!("{} - {}", error_code, e),
                        })
                    }
                }
            }
            .await;
            let _ = tx.send(result);
        });
        stream
    }
}

// ... (The rest of the file remains the same)
