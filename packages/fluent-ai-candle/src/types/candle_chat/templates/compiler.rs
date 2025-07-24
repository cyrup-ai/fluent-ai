//! Template compiler for optimizing template execution
//!
//! Provides template compilation with optimization passes.

use crate::types::candle_chat::{templates::core::{ChatTemplate, CompiledTemplate, TemplateAst, TemplateResult}};

/// Template compiler configuration
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// Enable optimization passes
    pub optimize: bool,
    /// Maximum compilation time in milliseconds
    pub max_compile_time_ms: u64,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            optimize: true,
            max_compile_time_ms: 5000, // 5 seconds
        }
    }
}

/// Template compiler for optimizing templates
#[derive(Debug, Clone)]
pub struct TemplateCompiler {
    config: CompilerConfig,
}

impl TemplateCompiler {
    /// Create a new template compiler
    pub fn new() -> Self {
        Self {
            config: CompilerConfig::default(),
        }
    }

    /// Create compiler with configuration
    pub fn with_config(config: CompilerConfig) -> Self {
        Self { config }
    }

    /// Compile a template
    pub fn compile(&self, template: &ChatTemplate) -> TemplateResult<CompiledTemplate> {
        // For now, just create a simple AST from the content
        let ast = TemplateAst::Text(template.get_content().clone());

        // Use the template's existing variables
        let variables = template.variables.clone();

        Ok(CompiledTemplate::new(
            template.metadata.clone(),
            ast,
            variables,
        ))
    }

    /// Get compiler configuration
    pub fn config(&self) -> &CompilerConfig {
        &self.config
    }
}

impl Default for TemplateCompiler {
    fn default() -> Self {
        Self::new()
    }
}
