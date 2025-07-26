//! Template parser implementation
//!
//! Provides high-performance parsing for template syntax with
//! zero-allocation patterns where possible.

use std::collections::HashMap;
use std::sync::Arc;

use super::core::{TemplateAst, TemplateError, TemplateResult, TemplateVariable, VariableType};

/// Template parser configuration
#[derive(Debug, Clone)]
pub struct ParserConfig {
    pub max_depth: usize,
    pub max_tokens: usize,
    pub allow_expressions: bool,
    pub allow_functions: bool,
    pub strict_variables: bool}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            max_depth: 32,
            max_tokens: 10000,
            allow_expressions: true,
            allow_functions: true,
            strict_variables: false}
    }
}

/// Template parser implementation
#[derive(Debug)]
pub struct TemplateParser {
    config: ParserConfig}

impl TemplateParser {
    pub fn new() -> Self {
        Self {
            config: ParserConfig::default()}
    }

    pub fn with_config(config: ParserConfig) -> Self {
        Self { config }
    }

    /// Parse template content into AST
    pub fn parse(&self, content: &str) -> TemplateResult<TemplateAst> {
        self.parse_with_depth(content, 0)
    }

    fn parse_with_depth(&self, content: &str, depth: usize) -> TemplateResult<TemplateAst> {
        if depth > self.config.max_depth {
            return Err(TemplateError::ParseError {
                message: Arc::from("Maximum parsing depth exceeded")});
        }

        let mut nodes = Vec::new();
        let mut current_text = String::new();
        let mut chars = content.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '{' && chars.peek() == Some(&'{') {
                chars.next(); // consume second '{'

                // Save accumulated text
                if !current_text.is_empty() {
                    nodes.push(TemplateAst::Text(Arc::from(current_text.clone())));
                    current_text.clear();
                }

                // Parse variable or expression
                let var_content = self.parse_until_closing(&mut chars)?;
                let ast_node = self.parse_variable_or_expression(&var_content, depth + 1)?;
                nodes.push(ast_node);
            } else {
                current_text.push(ch);
            }
        }

        // Add remaining text
        if !current_text.is_empty() {
            nodes.push(TemplateAst::Text(Arc::from(current_text)));
        }

        // Return single node or block
        match nodes.len() {
            0 => Ok(TemplateAst::Text(Arc::from(""))),
            1 => Ok(nodes.into_iter().next().unwrap()),
            _ => Ok(TemplateAst::Block(nodes.into()))}
    }

    fn parse_until_closing(
        &self,
        chars: &mut std::iter::Peekable<std::str::Chars>,
    ) -> TemplateResult<String> {
        let mut content = String::new();
        let mut brace_count = 0;

        while let Some(ch) = chars.next() {
            if ch == '}' && chars.peek() == Some(&'}') {
                if brace_count == 0 {
                    chars.next(); // consume second '}'
                    break;
                } else {
                    content.push(ch);
                    brace_count -= 1;
                }
            } else if ch == '{' && chars.peek() == Some(&'{') {
                content.push(ch);
                brace_count += 1;
            } else {
                content.push(ch);
            }
        }

        Ok(content)
    }

    fn parse_variable_or_expression(
        &self,
        content: &str,
        depth: usize,
    ) -> TemplateResult<TemplateAst> {
        let trimmed = content.trim();

        // Check for conditional statements
        if trimmed.starts_with("if ") {
            return self.parse_conditional(trimmed, depth);
        }

        // Check for loop statements
        if trimmed.starts_with("for ") {
            return self.parse_loop(trimmed, depth);
        }

        // Check for function calls
        if trimmed.contains('(') && trimmed.contains(')') {
            return self.parse_function_call(trimmed, depth);
        }

        // Check for expressions
        if self.config.allow_expressions
            && (trimmed.contains('+')
                || trimmed.contains('-')
                || trimmed.contains('*')
                || trimmed.contains('/'))
        {
            return self.parse_expression(trimmed, depth);
        }

        // Simple variable
        Ok(TemplateAst::Variable(Arc::from(trimmed)))
    }

    fn parse_conditional(&self, content: &str, _depth: usize) -> TemplateResult<TemplateAst> {
        // Simple conditional parsing - just return variable for now
        let condition_var = content.strip_prefix("if ").unwrap_or("").trim();
        Ok(TemplateAst::Conditional {
            condition: Arc::new(TemplateAst::Variable(Arc::from(condition_var))),
            if_true: Arc::new(TemplateAst::Text(Arc::from("true"))),
            if_false: Some(Arc::new(TemplateAst::Text(Arc::from("false"))))})
    }

    fn parse_loop(&self, content: &str, _depth: usize) -> TemplateResult<TemplateAst> {
        // Simple loop parsing - just return variable for now
        let loop_content = content.strip_prefix("for ").unwrap_or("").trim();
        let parts: Vec<&str> = loop_content.split(" in ").collect();

        if parts.len() != 2 {
            return Err(TemplateError::ParseError {
                message: Arc::from("Invalid loop syntax")});
        }

        Ok(TemplateAst::Loop {
            variable: Arc::from(parts[0].trim()),
            iterable: Arc::new(TemplateAst::Variable(Arc::from(parts[1].trim()))),
            body: Arc::new(TemplateAst::Variable(Arc::from(parts[0].trim())))})
    }

    fn parse_function_call(&self, content: &str, _depth: usize) -> TemplateResult<TemplateAst> {
        if !self.config.allow_functions {
            return Err(TemplateError::ParseError {
                message: Arc::from("Function calls not allowed")});
        }

        let paren_pos = content.find('(').unwrap_or(0);
        let func_name = content[..paren_pos].trim();

        Ok(TemplateAst::Function {
            name: Arc::from(func_name),
            args: Arc::new([]), // TODO: Parse arguments
        })
    }

    fn parse_expression(&self, content: &str, _depth: usize) -> TemplateResult<TemplateAst> {
        // Simple expression parsing - return as variable for now
        Ok(TemplateAst::Expression {
            operator: Arc::from("+"),
            operands: Arc::new([TemplateAst::Variable(Arc::from(content))])})
    }

    /// Extract variables from template content
    pub fn extract_variables(&self, content: &str) -> TemplateResult<Vec<TemplateVariable>> {
        let mut variables = HashMap::new();
        let mut chars = content.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '{' && chars.peek() == Some(&'{') {
                chars.next(); // consume second '{'

                let var_content = self.parse_until_closing(&mut chars)?;
                let var_name = var_content.trim();

                // Skip control structures
                if var_name.starts_with("if ")
                    || var_name.starts_with("for ")
                    || var_name.starts_with("end")
                {
                    continue;
                }

                // Extract simple variable name (before any operators or functions)
                let clean_name = var_name
                    .split_whitespace()
                    .next()
                    .unwrap_or(var_name)
                    .split('.')
                    .next()
                    .unwrap_or("")
                    .split('(')
                    .next()
                    .unwrap_or("")
                    .trim();

                if !clean_name.is_empty()
                    && clean_name.chars().all(|c| c.is_alphanumeric() || c == '_')
                {
                    variables.insert(
                        clean_name.to_string(),
                        TemplateVariable {
                            name: Arc::from(clean_name),
                            description: Arc::from(""),
                            var_type: VariableType::String,
                            default_value: None,
                            required: false,
                            validation_pattern: None,
                            valid_values: None,
                            min_value: None,
                            max_value: None},
                    );
                }
            }
        }

        Ok(variables.into_values().collect())
    }
}

impl Default for TemplateParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Parser statistics for performance monitoring
#[derive(Debug, Clone, PartialEq)]
pub struct ParserStats {
    pub parsed_count: usize,
    pub total_parse_time_us: usize,
    pub error_count: usize,
    pub peak_memory_bytes: usize}

/// Parse error type alias for convenience
pub type ParseError = TemplateError;

/// Parse result type alias for convenience
pub type ParseResult<T> = Result<T, ParseError>;

/// Quick parse function for convenience
pub fn parse_template(content: &str) -> TemplateResult<TemplateAst> {
    let parser = TemplateParser::new();
    parser.parse(content)
}

/// Extract variables from template content
pub fn extract_variables(content: &str) -> TemplateResult<Vec<TemplateVariable>> {
    let parser = TemplateParser::new();
    parser.extract_variables(content)
}

/// Validate template syntax
pub fn validate_template(content: &str) -> TemplateResult<()> {
    let parser = TemplateParser::new();
    parser.parse(content)?;
    Ok(())
}

