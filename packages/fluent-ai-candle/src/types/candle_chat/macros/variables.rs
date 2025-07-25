//! Variable management and resolution for macros
//!
//! This module handles variable substitution, condition evaluation,
//! and variable management within macro execution contexts.

use std::collections::HashMap;

use super::types::MacroSystemError;

/// Resolve variables in a string
pub fn resolve_variables(content: &str, variables: &HashMap<String, String>) -> String {
    let mut result = content.to_string();
    for (key, value) in variables {
        let placeholder = format!("{{{}}}", key);
        result = result.replace(&placeholder, value);
    }
    result
}

/// Resolve variables in a string (static version for compatibility)
pub fn resolve_variables_static(content: &str, variables: &HashMap<String, String>) -> String {
    resolve_variables(content, variables)
}

/// Evaluate a condition string
pub fn evaluate_condition(condition: &str, variables: &HashMap<String, String>) -> bool {
    let resolved_condition = resolve_variables(condition, variables);
    
    // Basic boolean evaluation
    match resolved_condition.to_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => true,
        "false" | "0" | "no" | "off" => false,
        _ => {
            // Try to evaluate as comparison
            if resolved_condition.contains("==") {
                let parts: Vec<&str> = resolved_condition.split("==").collect();
                if parts.len() == 2 {
                    return parts[0].trim() == parts[1].trim();
                }
            }
            
            if resolved_condition.contains("!=") {
                let parts: Vec<&str> = resolved_condition.split("!=").collect();
                if parts.len() == 2 {
                    return parts[0].trim() != parts[1].trim();
                }
            }
            
            if resolved_condition.contains(">=") {
                let parts: Vec<&str> = resolved_condition.split(">=").collect();
                if parts.len() == 2 {
                    if let (Ok(left), Ok(right)) = (parts[0].trim().parse::<f64>(), parts[1].trim().parse::<f64>()) {
                        return left >= right;
                    }
                }
            }
            
            if resolved_condition.contains("<=") {
                let parts: Vec<&str> = resolved_condition.split("<=").collect();
                if parts.len() == 2 {
                    if let (Ok(left), Ok(right)) = (parts[0].trim().parse::<f64>(), parts[1].trim().parse::<f64>()) {
                        return left <= right;
                    }
                }
            }
            
            if resolved_condition.contains(">") {
                let parts: Vec<&str> = resolved_condition.split(">").collect();
                if parts.len() == 2 {
                    if let (Ok(left), Ok(right)) = (parts[0].trim().parse::<f64>(), parts[1].trim().parse::<f64>()) {
                        return left > right;
                    }
                }
            }
            
            if resolved_condition.contains("<") {
                let parts: Vec<&str> = resolved_condition.split("<").collect();
                if parts.len() == 2 {
                    if let (Ok(left), Ok(right)) = (parts[0].trim().parse::<f64>(), parts[1].trim().parse::<f64>()) {
                        return left < right;
                    }
                }
            }
            
            false
        }
    }
}

/// Evaluate a condition string (static version for compatibility)
pub fn evaluate_condition_static(condition: &str, variables: &HashMap<String, String>) -> bool {
    evaluate_condition(condition, variables)
}

/// Variable manager for macro contexts
pub struct VariableManager {
    variables: HashMap<String, String>}

impl VariableManager {
    /// Create new variable manager
    pub fn new() -> Self {
        Self {
            variables: HashMap::new()}
    }

    /// Create with initial variables
    pub fn with_variables(variables: HashMap<String, String>) -> Self {
        Self { variables }
    }

    /// Set a variable value
    pub fn set_variable(&mut self, name: String, value: String) {
        self.variables.insert(name, value);
    }

    /// Get a variable value
    pub fn get_variable(&self, name: &str) -> Option<&String> {
        self.variables.get(name)
    }

    /// Remove a variable
    pub fn remove_variable(&mut self, name: &str) -> Option<String> {
        self.variables.remove(name)
    }

    /// Clear all variables
    pub fn clear_variables(&mut self) {
        self.variables.clear();
    }

    /// Get all variables
    pub fn get_variables(&self) -> &HashMap<String, String> {
        &self.variables
    }

    /// Get mutable reference to variables
    pub fn get_variables_mut(&mut self) -> &mut HashMap<String, String> {
        &mut self.variables
    }

    /// Resolve variables in content
    pub fn resolve(&self, content: &str) -> String {
        resolve_variables(content, &self.variables)
    }

    /// Evaluate condition
    pub fn evaluate(&self, condition: &str) -> bool {
        evaluate_condition(condition, &self.variables)
    }

    /// Substitute variables in content with error handling
    pub fn substitute_variable(&self, content: &str, required_vars: &[&str]) -> Result<String, MacroSystemError> {
        let mut result = content.to_string();
        
        for var_name in required_vars {
            let placeholder = format!("{{{}}}", var_name);
            if let Some(value) = self.variables.get(*var_name) {
                result = result.replace(&placeholder, value);
            } else {
                return Err(MacroSystemError::VariableNotFound(var_name.to_string()));
            }
        }
        
        // Also substitute any other variables that might be present
        for (key, value) in &self.variables {
            let placeholder = format!("{{{}}}", key);
            result = result.replace(&placeholder, value);
        }
        
        Ok(result)
    }

    /// Check if variable exists
    pub fn has_variable(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }

    /// Get variable count
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    /// Get variable names
    pub fn variable_names(&self) -> Vec<String> {
        self.variables.keys().cloned().collect()
    }

    /// Merge variables from another manager
    pub fn merge_variables(&mut self, other: &VariableManager) {
        for (key, value) in &other.variables {
            self.variables.insert(key.clone(), value.clone());
        }
    }

    /// Create snapshot of current variables
    pub fn snapshot(&self) -> HashMap<String, String> {
        self.variables.clone()
    }

    /// Restore from snapshot
    pub fn restore_snapshot(&mut self, snapshot: HashMap<String, String>) {
        self.variables = snapshot;
    }
}

impl Default for VariableManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for VariableManager {
    fn clone(&self) -> Self {
        Self {
            variables: self.variables.clone()}
    }
}

/// Built-in variable functions
pub mod builtin {
    use std::collections::HashMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Get current timestamp as string
    pub fn current_timestamp() -> String {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .to_string()
    }

    /// Get current date as string (YYYY-MM-DD)
    pub fn current_date() -> String {
        use chrono::{Utc, Datelike};
        let now = Utc::now();
        format!("{:04}-{:02}-{:02}", now.year(), now.month(), now.day())
    }

    /// Get current time as string (HH:MM:SS)
    pub fn current_time() -> String {
        use chrono::{Utc, Timelike};
        let now = Utc::now();
        format!("{:02}:{:02}:{:02}", now.hour(), now.minute(), now.second())
    }

    /// Register built-in variables
    pub fn register_builtin_variables(variables: &mut HashMap<String, String>) {
        variables.insert("timestamp".to_string(), current_timestamp());
        variables.insert("date".to_string(), current_date());
        variables.insert("time".to_string(), current_time());
    }
}