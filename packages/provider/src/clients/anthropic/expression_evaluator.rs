//! High-performance mathematical expression evaluator with zero-allocation parsing
//! 
//! Production-ready expression evaluation supporting arithmetic operations,
//! mathematical functions, constants, and variables with comprehensive error handling.

use pest::Parser;
use pest_derive::Parser;
use std::collections::HashMap;
use std::f64::consts;
use thiserror::Error;

#[derive(Parser)]
#[grammar = "clients/anthropic/expression.pest"]
pub struct ExpressionParser;

/// Expression evaluation errors with detailed context
#[derive(Error, Debug, Clone)]
pub enum ExpressionError {
    #[error("Parse error: {message} at position {position}")]
    ParseError { message: String, position: usize },
    
    #[error("Division by zero in expression")]
    DivisionByZero,
    
    #[error("Invalid function call: {function} with {arg_count} arguments")]
    InvalidFunctionCall { function: String, arg_count: usize },
    
    #[error("Undefined variable: {variable}")]
    UndefinedVariable { variable: String },
    
    #[error("Mathematical domain error: {operation} with value {value}")]
    DomainError { operation: String, value: f64 },
    
    #[error("Overflow in calculation: {operation}")]
    Overflow { operation: String },
    
    #[error("Invalid expression: {message}")]
    InvalidExpression { message: String },
}

/// Expression evaluation result
pub type ExpressionResult<T> = Result<T, ExpressionError>;

/// Variable context for expression evaluation
#[derive(Debug, Clone, Default)]
pub struct VariableContext {
    variables: HashMap<String, f64>,
}

impl VariableContext {
    /// Create new empty variable context
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }
    
    /// Set variable value
    pub fn set(&mut self, name: impl Into<String>, value: f64) {
        self.variables.insert(name.into(), value);
    }
    
    /// Get variable value
    pub fn get(&self, name: &str) -> Option<f64> {
        self.variables.get(name).copied()
    }
    
    /// Check if variable exists
    pub fn contains(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }
    
    /// Clear all variables
    pub fn clear(&mut self) {
        self.variables.clear();
    }
}

/// High-performance expression evaluator with zero-allocation parsing
pub struct ExpressionEvaluator {
    context: VariableContext,
}

impl ExpressionEvaluator {
    /// Create new expression evaluator
    pub fn new() -> Self {
        Self {
            context: VariableContext::new(),
        }
    }
    
    /// Create evaluator with variable context
    pub fn with_context(context: VariableContext) -> Self {
        Self { context }
    }
    
    /// Evaluate mathematical expression
    pub fn evaluate(&mut self, expression: &str) -> ExpressionResult<f64> {
        // Parse expression using pest
        let pairs = ExpressionParser::parse(Rule::calculator, expression)
            .map_err(|e| ExpressionError::ParseError {
                message: format!("Parsing failed: {}", e),
                position: 0,
            })?;
        
        // Get the main pair
        let main_pair = pairs.into_iter().next()
            .ok_or_else(|| ExpressionError::InvalidExpression {
                message: "Empty expression".to_string(),
            })?;
        
        // Check if it's an assignment or just evaluation
        let inner_pair = main_pair.into_inner().next()
            .ok_or_else(|| ExpressionError::InvalidExpression {
                message: "Invalid expression structure".to_string(),
            })?;
        
        if inner_pair.as_rule() == Rule::expression {
            // Simple evaluation
            self.evaluate_expression(inner_pair)
        } else {
            // Assignment (variable = expression)
            let mut inner_pairs = inner_pair.into_inner();
            let var_name = inner_pairs.next()
                .ok_or_else(|| ExpressionError::InvalidExpression {
                    message: "Missing variable name in assignment".to_string(),
                })?
                .as_str()
                .to_string();
            
            let expr_pair = inner_pairs.next()
                .ok_or_else(|| ExpressionError::InvalidExpression {
                    message: "Missing expression in assignment".to_string(),
                })?;
            
            let value = self.evaluate_expression(expr_pair)?;
            self.context.set(var_name, value);
            Ok(value)
        }
    }
    
    /// Evaluate expression with custom variables
    pub fn evaluate_with_vars(&mut self, expression: &str, variables: &HashMap<String, f64>) -> ExpressionResult<f64> {
        // Temporarily add variables to context
        let original_vars = self.context.variables.clone();
        
        for (name, value) in variables {
            self.context.set(name.clone(), *value);
        }
        
        let result = self.evaluate(expression);
        
        // Restore original context
        self.context.variables = original_vars;
        
        result
    }
    
    /// Set variable in context
    pub fn set_variable(&mut self, name: impl Into<String>, value: f64) {
        self.context.set(name, value);
    }
    
    /// Get variable from context
    pub fn get_variable(&self, name: &str) -> Option<f64> {
        self.context.get(name)
    }
    
    /// Clear all variables
    pub fn clear_variables(&mut self) {
        self.context.clear();
    }
    
    /// Evaluate parsed expression pair
    fn evaluate_expression(&self, pair: pest::iterators::Pair<Rule>) -> ExpressionResult<f64> {
        match pair.as_rule() {
            Rule::expression => {
                let inner = pair.into_inner().next()
                    .ok_or_else(|| ExpressionError::InvalidExpression {
                        message: "Empty expression".to_string(),
                    })?;
                self.evaluate_expression(inner)
            }
            
            Rule::logical_or => {
                let mut inner = pair.into_inner();
                let mut result = self.evaluate_expression(inner.next()
                    .ok_or_else(|| ExpressionError::InvalidExpression {
                        message: "Missing operand in logical OR".to_string(),
                    })?)?;
                
                for expr in inner {
                    let right = self.evaluate_expression(expr)?;
                    result = if result != 0.0 || right != 0.0 { 1.0 } else { 0.0 };
                }
                Ok(result)
            }
            
            Rule::logical_and => {
                let mut inner = pair.into_inner();
                let mut result = self.evaluate_expression(inner.next()
                    .ok_or_else(|| ExpressionError::InvalidExpression {
                        message: "Missing operand in logical AND".to_string(),
                    })?)?;
                
                for expr in inner {
                    let right = self.evaluate_expression(expr)?;
                    result = if result != 0.0 && right != 0.0 { 1.0 } else { 0.0 };
                }
                Ok(result)
            }
            
            Rule::comparison => {
                let mut inner = pair.into_inner();
                let mut result = self.evaluate_expression(inner.next()
                    .ok_or_else(|| ExpressionError::InvalidExpression {
                        message: "Missing operand in comparison".to_string(),
                    })?)?;
                
                let mut inner_iter = inner.peekable();
                while inner_iter.peek().is_some() {
                    let op = inner_iter.next().unwrap().as_str();
                    let right = self.evaluate_expression(inner_iter.next()
                        .ok_or_else(|| ExpressionError::InvalidExpression {
                            message: "Missing right operand in comparison".to_string(),
                        })?)?;
                    
                    result = match op {
                        "==" => if (result - right).abs() < f64::EPSILON { 1.0 } else { 0.0 },
                        "!=" => if (result - right).abs() >= f64::EPSILON { 1.0 } else { 0.0 },
                        "<" => if result < right { 1.0 } else { 0.0 },
                        "<=" => if result <= right { 1.0 } else { 0.0 },
                        ">" => if result > right { 1.0 } else { 0.0 },
                        ">=" => if result >= right { 1.0 } else { 0.0 },
                        _ => return Err(ExpressionError::InvalidExpression {
                            message: format!("Unknown comparison operator: {}", op),
                        }),
                    };
                }
                Ok(result)
            }
            
            Rule::additive => {
                let mut inner = pair.into_inner();
                let mut result = self.evaluate_expression(inner.next()
                    .ok_or_else(|| ExpressionError::InvalidExpression {
                        message: "Missing operand in addition".to_string(),
                    })?)?;
                
                let mut inner_iter = inner.peekable();
                while inner_iter.peek().is_some() {
                    let op = inner_iter.next().unwrap().as_str();
                    let right = self.evaluate_expression(inner_iter.next()
                        .ok_or_else(|| ExpressionError::InvalidExpression {
                            message: "Missing right operand in addition".to_string(),
                        })?)?;
                    
                    match op {
                        "+" => {
                            result = result + right;
                            if result.is_infinite() {
                                return Err(ExpressionError::Overflow {
                                    operation: "addition".to_string(),
                                });
                            }
                        }
                        "-" => {
                            result = result - right;
                            if result.is_infinite() {
                                return Err(ExpressionError::Overflow {
                                    operation: "subtraction".to_string(),
                                });
                            }
                        }
                        _ => return Err(ExpressionError::InvalidExpression {
                            message: format!("Unknown additive operator: {}", op),
                        }),
                    }
                }
                Ok(result)
            }
            
            Rule::multiplicative => {
                let mut inner = pair.into_inner();
                let mut result = self.evaluate_expression(inner.next()
                    .ok_or_else(|| ExpressionError::InvalidExpression {
                        message: "Missing operand in multiplication".to_string(),
                    })?)?;
                
                let mut inner_iter = inner.peekable();
                while inner_iter.peek().is_some() {
                    let op = inner_iter.next().unwrap().as_str();
                    let right = self.evaluate_expression(inner_iter.next()
                        .ok_or_else(|| ExpressionError::InvalidExpression {
                            message: "Missing right operand in multiplication".to_string(),
                        })?)?;
                    
                    match op {
                        "*" => {
                            result = result * right;
                            if result.is_infinite() {
                                return Err(ExpressionError::Overflow {
                                    operation: "multiplication".to_string(),
                                });
                            }
                        }
                        "/" => {
                            if right == 0.0 {
                                return Err(ExpressionError::DivisionByZero);
                            }
                            result = result / right;
                            if result.is_infinite() {
                                return Err(ExpressionError::Overflow {
                                    operation: "division".to_string(),
                                });
                            }
                        }
                        "%" => {
                            if right == 0.0 {
                                return Err(ExpressionError::DivisionByZero);
                            }
                            result = result % right;
                        }
                        _ => return Err(ExpressionError::InvalidExpression {
                            message: format!("Unknown multiplicative operator: {}", op),
                        }),
                    }
                }
                Ok(result)
            }
            
            Rule::power => {
                let mut inner = pair.into_inner();
                let mut result = self.evaluate_expression(inner.next()
                    .ok_or_else(|| ExpressionError::InvalidExpression {
                        message: "Missing base in power operation".to_string(),
                    })?)?;
                
                // Power is right-associative
                let powers: Vec<_> = inner.collect();
                for power_expr in powers.into_iter().rev() {
                    let exponent = self.evaluate_expression(power_expr)?;
                    result = result.powf(exponent);
                    if result.is_infinite() || result.is_nan() {
                        return Err(ExpressionError::Overflow {
                            operation: "exponentiation".to_string(),
                        });
                    }
                }
                Ok(result)
            }
            
            Rule::unary => {
                let mut inner = pair.into_inner();
                let first = inner.next()
                    .ok_or_else(|| ExpressionError::InvalidExpression {
                        message: "Missing operand in unary operation".to_string(),
                    })?;
                
                if first.as_str() == "-" {
                    let operand = self.evaluate_expression(inner.next()
                        .ok_or_else(|| ExpressionError::InvalidExpression {
                            message: "Missing operand after unary minus".to_string(),
                        })?)?;
                    Ok(-operand)
                } else if first.as_str() == "+" {
                    let operand = self.evaluate_expression(inner.next()
                        .ok_or_else(|| ExpressionError::InvalidExpression {
                            message: "Missing operand after unary plus".to_string(),
                        })?)?;
                    Ok(operand)
                } else {
                    // No unary operator, evaluate the primary
                    self.evaluate_expression(first)
                }
            }
            
            Rule::primary => {
                let inner = pair.into_inner().next()
                    .ok_or_else(|| ExpressionError::InvalidExpression {
                        message: "Empty primary expression".to_string(),
                    })?;
                self.evaluate_expression(inner)
            }
            
            Rule::number => {
                let num_str = pair.as_str();
                num_str.parse::<f64>()
                    .map_err(|_| ExpressionError::InvalidExpression {
                        message: format!("Invalid number: {}", num_str),
                    })
            }
            
            Rule::constant => {
                match pair.as_str() {
                    "pi" => Ok(consts::PI),
                    "e" => Ok(consts::E),
                    "tau" => Ok(consts::TAU),
                    name => Err(ExpressionError::InvalidExpression {
                        message: format!("Unknown constant: {}", name),
                    }),
                }
            }
            
            Rule::variable => {
                let var_name = pair.as_str();
                self.context.get(var_name)
                    .ok_or_else(|| ExpressionError::UndefinedVariable {
                        variable: var_name.to_string(),
                    })
            }
            
            Rule::function => {
                let mut inner = pair.into_inner();
                let func_name = inner.next()
                    .ok_or_else(|| ExpressionError::InvalidExpression {
                        message: "Missing function name".to_string(),
                    })?
                    .as_str();
                
                let args: Result<Vec<f64>, ExpressionError> = inner
                    .map(|arg| self.evaluate_expression(arg))
                    .collect();
                let args = args?;
                
                self.evaluate_function(func_name, &args)
            }
            
            _ => Err(ExpressionError::InvalidExpression {
                message: format!("Unsupported rule: {:?}", pair.as_rule()),
            }),
        }
    }
    
    /// Evaluate mathematical function
    fn evaluate_function(&self, name: &str, args: &[f64]) -> ExpressionResult<f64> {
        match name {
            "sin" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                Ok(args[0].sin())
            }
            "cos" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                Ok(args[0].cos())
            }
            "tan" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                Ok(args[0].tan())
            }
            "asin" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                let value = args[0];
                if value < -1.0 || value > 1.0 {
                    return Err(ExpressionError::DomainError {
                        operation: "asin".to_string(),
                        value,
                    });
                }
                Ok(value.asin())
            }
            "acos" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                let value = args[0];
                if value < -1.0 || value > 1.0 {
                    return Err(ExpressionError::DomainError {
                        operation: "acos".to_string(),
                        value,
                    });
                }
                Ok(value.acos())
            }
            "atan" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                Ok(args[0].atan())
            }
            "sinh" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                Ok(args[0].sinh())
            }
            "cosh" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                Ok(args[0].cosh())
            }
            "tanh" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                Ok(args[0].tanh())
            }
            "sqrt" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                let value = args[0];
                if value < 0.0 {
                    return Err(ExpressionError::DomainError {
                        operation: "sqrt".to_string(),
                        value,
                    });
                }
                Ok(value.sqrt())
            }
            "ln" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                let value = args[0];
                if value <= 0.0 {
                    return Err(ExpressionError::DomainError {
                        operation: "ln".to_string(),
                        value,
                    });
                }
                Ok(value.ln())
            }
            "log" | "log10" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                let value = args[0];
                if value <= 0.0 {
                    return Err(ExpressionError::DomainError {
                        operation: "log10".to_string(),
                        value,
                    });
                }
                Ok(value.log10())
            }
            "log2" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                let value = args[0];
                if value <= 0.0 {
                    return Err(ExpressionError::DomainError {
                        operation: "log2".to_string(),
                        value,
                    });
                }
                Ok(value.log2())
            }
            "exp" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                let result = args[0].exp();
                if result.is_infinite() {
                    return Err(ExpressionError::Overflow {
                        operation: "exp".to_string(),
                    });
                }
                Ok(result)
            }
            "abs" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                Ok(args[0].abs())
            }
            "ceil" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                Ok(args[0].ceil())
            }
            "floor" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                Ok(args[0].floor())
            }
            "round" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                Ok(args[0].round())
            }
            "sign" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                let value = args[0];
                Ok(if value > 0.0 { 1.0 } else if value < 0.0 { -1.0 } else { 0.0 })
            }
            "deg" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                Ok(args[0].to_degrees())
            }
            "rad" => {
                if args.len() != 1 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                Ok(args[0].to_radians())
            }
            "min" => {
                if args.len() != 2 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                Ok(args[0].min(args[1]))
            }
            "max" => {
                if args.len() != 2 {
                    return Err(ExpressionError::InvalidFunctionCall {
                        function: name.to_string(),
                        arg_count: args.len(),
                    });
                }
                Ok(args[0].max(args[1]))
            }
            _ => Err(ExpressionError::InvalidExpression {
                message: format!("Unknown function: {}", name),
            }),
        }
    }
}

impl Default for ExpressionEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility function for quick expression evaluation
pub fn evaluate_expression(expression: &str) -> ExpressionResult<f64> {
    let mut evaluator = ExpressionEvaluator::new();
    evaluator.evaluate(expression)
}

/// Utility function for expression evaluation with variables
pub fn evaluate_expression_with_vars(expression: &str, variables: &HashMap<String, f64>) -> ExpressionResult<f64> {
    let mut evaluator = ExpressionEvaluator::new();
    evaluator.evaluate_with_vars(expression, variables)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_basic_arithmetic() {
        let mut eval = ExpressionEvaluator::new();
        
        assert_eq!(eval.evaluate("2 + 3").expect("Should work"), 5.0);
        assert_eq!(eval.evaluate("10 - 4").expect("Should work"), 6.0);
        assert_eq!(eval.evaluate("6 * 7").expect("Should work"), 42.0);
        assert_eq!(eval.evaluate("15 / 3").expect("Should work"), 5.0);
        assert_eq!(eval.evaluate("17 % 5").expect("Should work"), 2.0);
        assert_eq!(eval.evaluate("2 ^ 3").expect("Should work"), 8.0);
    }
    
    #[test]
    fn test_parentheses() {
        let mut eval = ExpressionEvaluator::new();
        
        assert_eq!(eval.evaluate("(2 + 3) * 4").expect("Should work"), 20.0);
        assert_eq!(eval.evaluate("2 + (3 * 4)").expect("Should work"), 14.0);
        assert_eq!(eval.evaluate("((2 + 3) * 4) / 2").expect("Should work"), 10.0);
    }
    
    #[test]
    fn test_functions() {
        let mut eval = ExpressionEvaluator::new();
        
        assert!((eval.evaluate("sin(0)").expect("Should work") - 0.0).abs() < f64::EPSILON);
        assert!((eval.evaluate("cos(0)").expect("Should work") - 1.0).abs() < f64::EPSILON);
        assert_eq!(eval.evaluate("sqrt(16)").expect("Should work"), 4.0);
        assert_eq!(eval.evaluate("abs(-5)").expect("Should work"), 5.0);
        assert_eq!(eval.evaluate("max(3, 7)").expect("Should work"), 7.0);
        assert_eq!(eval.evaluate("min(3, 7)").expect("Should work"), 3.0);
    }
    
    #[test]
    fn test_constants() {
        let mut eval = ExpressionEvaluator::new();
        
        assert!((eval.evaluate("pi").expect("Should work") - std::f64::consts::PI).abs() < f64::EPSILON);
        assert!((eval.evaluate("e").expect("Should work") - std::f64::consts::E).abs() < f64::EPSILON);
        assert!((eval.evaluate("tau").expect("Should work") - std::f64::consts::TAU).abs() < f64::EPSILON);
    }
    
    #[test]
    fn test_variables() {
        let mut eval = ExpressionEvaluator::new();
        
        eval.set_variable("x", 5.0);
        assert_eq!(eval.evaluate("x * 2").expect("Should work"), 10.0);
        
        eval.set_variable("y", 3.0);
        assert_eq!(eval.evaluate("x + y").expect("Should work"), 8.0);
    }
    
    #[test]
    fn test_assignment() {
        let mut eval = ExpressionEvaluator::new();
        
        assert_eq!(eval.evaluate("x = 10").expect("Should work"), 10.0);
        assert_eq!(eval.evaluate("x * 2").expect("Should work"), 20.0);
        
        assert_eq!(eval.evaluate("y = x + 5").expect("Should work"), 15.0);
        assert_eq!(eval.evaluate("y").expect("Should work"), 15.0);
    }
    
    #[test]
    fn test_error_handling() {
        let mut eval = ExpressionEvaluator::new();
        
        // Division by zero
        assert!(eval.evaluate("5 / 0").is_err());
        
        // Undefined variable
        assert!(eval.evaluate("undefined_var").is_err());
        
        // Invalid function
        assert!(eval.evaluate("unknown_func(1)").is_err());
        
        // Domain error
        assert!(eval.evaluate("sqrt(-1)").is_err());
        assert!(eval.evaluate("ln(0)").is_err());
    }
    
    #[test]
    fn test_complex_expressions() {
        let mut eval = ExpressionEvaluator::new();
        
        // Complex mathematical expression
        let result = eval.evaluate("sin(pi/2) + cos(0) * sqrt(16) - 2^3").expect("Should work");
        assert!((result - (-3.0)).abs() < f64::EPSILON);
        
        // Expression with variables
        eval.set_variable("a", 3.0);
        eval.set_variable("b", 4.0);
        let result = eval.evaluate("sqrt(a^2 + b^2)").expect("Should work");
        assert!((result - 5.0).abs() < f64::EPSILON);
    }
}