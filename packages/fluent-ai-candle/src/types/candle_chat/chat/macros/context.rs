//! Context handling and variable management for macros
//!
//! This module provides context management, variable substitution,
//! and condition evaluation for the macro execution environment.

use std::collections::HashMap;
use std::sync::Arc;

use regex::Regex;
use uuid::Uuid;

use super::types::*;
use super::parser::ParsedExpression;

/// Context manager for macro execution environments
#[derive(Debug)]
pub struct MacroContextManager {
    /// Global variable storage
    global_variables: HashMap<String, String>,
    /// Context-specific variable storage
    context_variables: HashMap<Uuid, HashMap<String, String>>,
    /// Variable change listeners
    listeners: Vec<VariableChangeListener>,
    /// Configuration for context behavior
    config: ContextConfig}

/// Configuration for context management
#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// Enable variable caching
    pub enable_caching: bool,
    /// Maximum cached variables
    pub max_cached_variables: usize,
    /// Variable name validation regex
    pub variable_name_pattern: Option<Regex>,
    /// Enable variable change notifications
    pub enable_notifications: bool}

/// Variable change listener for reactive updates
#[derive(Debug)]
pub struct VariableChangeListener {
    /// Listener ID
    pub id: Uuid,
    /// Variable name pattern to watch
    pub pattern: Regex,
    /// Callback function name
    pub callback: String}

/// Result of variable resolution
#[derive(Debug, Clone)]
pub enum VariableResolutionResult {
    /// Successfully resolved to value
    Resolved(String),
    /// Variable not found
    NotFound(String),
    /// Resolution error
    Error(String)}

/// Context evaluation utilities
pub struct ContextEvaluator {
    /// Regex cache for pattern matching
    regex_cache: HashMap<String, Regex>,
    /// Function registry for custom functions
    functions: HashMap<String, Box<dyn Fn(&[String]) -> String + Send + Sync>>}

impl std::fmt::Debug for ContextEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextEvaluator")
            .field("regex_cache", &format!("[{} cached regexes]", self.regex_cache.len()))
            .field("functions", &format!("[{} functions]", self.functions.len()))
            .finish()
    }
}

impl MacroContextManager {
    /// Create a new context manager
    pub fn new() -> Self {
        Self {
            global_variables: HashMap::new(),
            context_variables: HashMap::new(),
            listeners: Vec::new(),
            config: ContextConfig::default()}
    }

    /// Create a context manager with custom configuration
    pub fn with_config(config: ContextConfig) -> Self {
        Self {
            global_variables: HashMap::new(),
            context_variables: HashMap::new(),
            listeners: Vec::new(),
            config}
    }

    /// Create a new execution context
    pub fn create_context(&mut self, context_id: Uuid) -> MacroExecutionContext {
        let context_vars = HashMap::new();
        self.context_variables.insert(context_id, context_vars.clone());

        MacroExecutionContext {
            variables: context_vars.into_iter().map(|(k, v)| (Arc::from(k), Arc::from(v))).collect(),
            execution_id: context_id,
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            current_action: 0,
            loop_stack: Vec::new()}
    }

    /// Set a global variable
    pub fn set_global_variable(&mut self, name: String, value: String) {
        if let Some(ref pattern) = self.config.variable_name_pattern {
            if !pattern.is_match(&name) {
                return; // Invalid variable name
            }
        }

        let old_value = self.global_variables.insert(name.clone(), value.clone());

        if self.config.enable_notifications && old_value.as_ref() != Some(&value) {
            self.notify_variable_change(&name, old_value.as_deref(), Some(&value));
        }
    }

    /// Get a global variable
    pub fn get_global_variable(&self, name: &str) -> Option<&String> {
        self.global_variables.get(name)
    }

    /// Set a context-specific variable
    pub fn set_context_variable(&mut self, context_id: Uuid, name: String, value: String) {
        if let Some(context_vars) = self.context_variables.get_mut(&context_id) {
            let old_value = context_vars.insert(name.clone(), value.clone());

            if self.config.enable_notifications && old_value.as_ref() != Some(&value) {
                self.notify_variable_change(&name, old_value.as_deref(), Some(&value));
            }
        }
    }

    /// Get a context-specific variable
    pub fn get_context_variable(&self, context_id: Uuid, name: &str) -> Option<&String> {
        self.context_variables.get(&context_id)?.get(name)
    }

    /// Resolve a variable (context-specific first, then global)
    pub fn resolve_variable(&self, context_id: Uuid, name: &str) -> VariableResolutionResult {
        // Try context-specific first
        if let Some(value) = self.get_context_variable(context_id, name) {
            return VariableResolutionResult::Resolved(value.clone());
        }

        // Try global variables
        if let Some(value) = self.get_global_variable(name) {
            return VariableResolutionResult::Resolved(value.clone());
        }

        VariableResolutionResult::NotFound(name.to_string())
    }

    /// Substitute variables in text content
    pub fn substitute_variables(&self, context_id: Uuid, content: &str) -> String {
        let mut result = content.to_string();

        // Simple variable substitution - replace ${variable_name} patterns
        let re = Regex::new(r"\$\{([^}]+)\}").unwrap();
        for cap in re.captures_iter(content) {
            if let Some(var_name) = cap.get(1) {
                match self.resolve_variable(context_id, var_name.as_str()) {
                    VariableResolutionResult::Resolved(value) => {
                        result = result.replace(&cap[0], &value);
                    }
                    _ => {
                        // Leave unresolved variables as-is
                    }
                }
            }
        }

        result
    }

    /// Add a variable change listener
    pub fn add_listener(&mut self, pattern: String, callback: String) -> Uuid {
        let id = Uuid::new_v4();
        if let Ok(regex) = Regex::new(&pattern) {
            self.listeners.push(VariableChangeListener {
                id,
                pattern: regex,
                callback});
        }
        id
    }

    /// Remove a variable change listener
    pub fn remove_listener(&mut self, listener_id: Uuid) -> bool {
        let original_len = self.listeners.len();
        self.listeners.retain(|l| l.id != listener_id);
        self.listeners.len() != original_len
    }

    /// Notify listeners of variable changes
    fn notify_variable_change(&self, name: &str, old_value: Option<&str>, new_value: Option<&str>) {
        for listener in &self.listeners {
            if listener.pattern.is_match(name) {
                // In a real implementation, this would trigger the callback
                println!(
                    "Variable {} changed from {:?} to {:?} (listener: {})",
                    name, old_value, new_value, listener.callback
                );
            }
        }
    }

    /// Clear all variables for a context
    pub fn clear_context(&mut self, context_id: Uuid) {
        self.context_variables.remove(&context_id);
    }

    /// Clear all global variables
    pub fn clear_globals(&mut self) {
        self.global_variables.clear();
    }

    /// Get all variables for a context (includes globals)
    pub fn get_all_variables(&self, context_id: Uuid) -> HashMap<String, String> {
        let mut result = self.global_variables.clone();

        if let Some(context_vars) = self.context_variables.get(&context_id) {
            result.extend(context_vars.clone());
        }

        result
    }
}

impl ContextEvaluator {
    /// Create a new context evaluator
    pub fn new() -> Self {
        let mut evaluator = Self {
            regex_cache: HashMap::new(),
            functions: HashMap::new()};

        // Register built-in functions
        evaluator.register_builtin_functions();
        evaluator
    }

    /// Register built-in functions
    fn register_builtin_functions(&mut self) {
        // String length function
        self.functions.insert(
            "len".to_string(),
            Box::new(|args| {
                args.get(0).map(|s| s.len().to_string()).unwrap_or_default()
            }),
        );

        // String uppercase function
        self.functions.insert(
            "upper".to_string(),
            Box::new(|args| {
                args.get(0).map(|s| s.to_uppercase()).unwrap_or_default()
            }),
        );

        // String lowercase function
        self.functions.insert(
            "lower".to_string(),
            Box::new(|args| {
                args.get(0).map(|s| s.to_lowercase()).unwrap_or_default()
            }),
        );

        // String trim function
        self.functions.insert(
            "trim".to_string(),
            Box::new(|args| {
                args.get(0).map(|s| s.trim().to_string()).unwrap_or_default()
            }),
        );
    }

    /// Evaluate a parsed expression in a context
    pub fn evaluate_expression(
        &mut self,
        expression: &ParsedExpression,
        variables: &HashMap<String, String>,
    ) -> Result<String, MacroSystemError> {
        match expression {
            ParsedExpression::Variable(name) => {
                variables
                    .get(name)
                    .cloned()
                    .ok_or_else(|| MacroSystemError::VariableNotFound(name.clone()))
            }
            ParsedExpression::Literal(value) => Ok(value.clone()),
            ParsedExpression::Number(value) => Ok(value.to_string()),
            ParsedExpression::Boolean(value) => Ok(value.to_string()),
            ParsedExpression::BinaryOp {
                operator,
                left,
                right} => self.evaluate_binary_op(operator, left, right, variables),
            ParsedExpression::UnaryOp { operator, operand } => {
                self.evaluate_unary_op(operator, operand, variables)
            }
            ParsedExpression::FunctionCall { name, args } => {
                self.evaluate_function_call(name, args, variables)
            }
        }
    }

    /// Evaluate a binary operation
    fn evaluate_binary_op(
        &mut self,
        operator: &super::parser::BinaryOperator,
        left: &ParsedExpression,
        right: &ParsedExpression,
        variables: &HashMap<String, String>,
    ) -> Result<String, MacroSystemError> {
        use super::parser::BinaryOperator;

        let left_val = self.evaluate_expression(left, variables)?;
        let right_val = self.evaluate_expression(right, variables)?;

        match operator {
            BinaryOperator::Equal => Ok((left_val == right_val).to_string()),
            BinaryOperator::NotEqual => Ok((left_val != right_val).to_string()),
            BinaryOperator::Contains => Ok(left_val.contains(&right_val).to_string()),
            BinaryOperator::Add => {
                // Try numeric addition first
                if let (Ok(l), Ok(r)) = (left_val.parse::<f64>(), right_val.parse::<f64>()) {
                    Ok((l + r).to_string())
                } else {
                    // String concatenation
                    Ok(format!("{}{}", left_val, right_val))
                }
            }
            BinaryOperator::LessThan => {
                if let (Ok(l), Ok(r)) = (left_val.parse::<f64>(), right_val.parse::<f64>()) {
                    Ok((l < r).to_string())
                } else {
                    Ok((left_val < right_val).to_string())
                }
            }
            BinaryOperator::GreaterThan => {
                if let (Ok(l), Ok(r)) = (left_val.parse::<f64>(), right_val.parse::<f64>()) {
                    Ok((l > r).to_string())
                } else {
                    Ok((left_val > right_val).to_string())
                }
            }
            BinaryOperator::And => {
                let l_bool = self.is_truthy(&left_val);
                let r_bool = self.is_truthy(&right_val);
                Ok((l_bool && r_bool).to_string())
            }
            BinaryOperator::Or => {
                let l_bool = self.is_truthy(&left_val);
                let r_bool = self.is_truthy(&right_val);
                Ok((l_bool || r_bool).to_string())
            }
            BinaryOperator::Matches => {
                match self.get_or_compile_regex(&right_val) {
                    Ok(regex) => Ok(regex.is_match(&left_val).to_string()),
                    Err(_) => Err(MacroSystemError::ConditionError(
                        format!("Invalid regex pattern: {}", right_val)
                    ))}
            }
            _ => Err(MacroSystemError::ConditionError(
                format!("Unsupported binary operator: {:?}", operator)
            ))}
    }

    /// Evaluate a unary operation
    fn evaluate_unary_op(
        &mut self,
        operator: &super::parser::UnaryOperator,
        operand: &ParsedExpression,
        variables: &HashMap<String, String>,
    ) -> Result<String, MacroSystemError> {
        use super::parser::UnaryOperator;

        let operand_val = self.evaluate_expression(operand, variables)?;

        match operator {
            UnaryOperator::Not => Ok((!self.is_truthy(&operand_val)).to_string()),
            UnaryOperator::IsEmpty => Ok(operand_val.is_empty().to_string()),
            UnaryOperator::IsNotEmpty => Ok((!operand_val.is_empty()).to_string()),
            UnaryOperator::Negate => {
                if let Ok(num) = operand_val.parse::<f64>() {
                    Ok((-num).to_string())
                } else {
                    Err(MacroSystemError::ConditionError(
                        "Cannot negate non-numeric value".to_string()
                    ))
                }
            }
        }
    }

    /// Evaluate a function call
    fn evaluate_function_call(
        &mut self,
        name: &str,
        args: &[ParsedExpression],
        variables: &HashMap<String, String>,
    ) -> Result<String, MacroSystemError> {
        let mut arg_values = Vec::new();
        for arg in args {
            arg_values.push(self.evaluate_expression(arg, variables)?);
        }

        if let Some(function) = self.functions.get(name) {
            Ok(function(&arg_values))
        } else {
            Err(MacroSystemError::ConditionError(
                format!("Unknown function: {}", name)
            ))
        }
    }

    /// Check if a string value is "truthy"
    fn is_truthy(&self, value: &str) -> bool {
        match value.to_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => true,
            "false" | "0" | "no" | "off" | "" => false,
            _ => !value.is_empty()}
    }

    /// Get or compile a regex pattern
    fn get_or_compile_regex(&mut self, pattern: &str) -> Result<&Regex, regex::Error> {
        if !self.regex_cache.contains_key(pattern) {
            let regex = Regex::new(pattern)?;
            self.regex_cache.insert(pattern.to_string(), regex);
        }
        Ok(self.regex_cache.get(pattern).unwrap())
    }

    /// Register a custom function
    pub fn register_function<F>(&mut self, name: String, function: F)
    where
        F: Fn(&[String]) -> String + Send + Sync + 'static,
    {
        self.functions.insert(name, Box::new(function));
    }

    /// Clear the regex cache
    pub fn clear_cache(&mut self) {
        self.regex_cache.clear();
    }
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            max_cached_variables: 1000,
            variable_name_pattern: Some(Regex::new(r"^[a-zA-Z_][a-zA-Z0-9_]*$").unwrap()),
            enable_notifications: true}
    }
}

impl Default for MacroContextManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ContextEvaluator {
    fn default() -> Self {
        Self::new()
    }
}