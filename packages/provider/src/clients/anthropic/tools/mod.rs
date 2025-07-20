//! Modular tool system for Anthropic Claude API with zero-allocation streaming
//!
//! This module provides a comprehensive, type-safe tool execution system with:
//! - Zero-allocation streaming architectures
//! - Lock-free concurrent execution
//! - Production-ready error handling
//! - Compile-time type safety
//! - HTTP/3 optimized networking

pub mod calculator;
pub mod core;
pub mod file_operations;
pub mod function_calling;

// Re-export core types for API compatibility
pub use core::{
    AnthropicError, AnthropicResult, ChainControl, Emitter, ErrorHandler, InvocationHandler,
    Message, ResultHandler, SchemaType, Tool, ToolError, ToolExecutionError, ToolRegistrationError,
};

// Re-export built-in tools
pub use calculator::{CalculatorTool, ExpressionError, ExpressionEvaluator};
pub use file_operations::FileOperationsTool;
// Convenience macro for tool creation
pub use function_calling::tool_builder;
// Re-export function calling system
pub use function_calling::{
    Conversation, DescribedTool, NamedTool, ToolBuilder, ToolExecutionContext, ToolExecutor,
    ToolOutput, ToolRegistry, ToolResult, ToolWithDeps, ToolWithInvocation, ToolWithRequestSchema,
    ToolWithSchemas, TypedTool, TypedToolStorage, TypedToolTrait,
};

/// Create a new tool registry with built-in tools pre-registered
///
/// This provides a convenient way to get started with common tools:
/// - Calculator: Mathematical expression evaluation
/// - File Operations: Anthropic Files API integration
pub fn with_builtins() -> ToolRegistry {
    let mut registry = ToolRegistry::new();

    // Register built-in tools with production-ready error handling
    match register_builtin_tools(&mut registry) {
        Ok(()) => registry,
        Err(e) => {
            tracing::error!("Failed to register built-in tools: {}", e);
            // Return empty registry rather than panicking
            ToolRegistry::new()
        }
    }
}

/// Register all built-in tools with comprehensive error handling
fn register_builtin_tools(registry: &mut ToolRegistry) -> AnthropicResult<()> {
    // Register calculator tool
    let calculator = Box::new(CalculatorTool);
    registry.register_tool("calculator".to_string(), calculator)?;

    // Register file operations tool
    let file_ops = Box::new(FileOperationsTool);
    registry.register_tool("file_operations".to_string(), file_ops)?;

    Ok(())
}

/// Production-ready tool registry builder with fluent API
pub struct ToolRegistryBuilder {
    registry: ToolRegistry,
}

impl ToolRegistryBuilder {
    /// Create new builder
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            registry: ToolRegistry::new(),
        }
    }

    /// Add built-in tools (calculator, file operations)
    #[inline(always)]
    pub fn with_builtins(mut self) -> AnthropicResult<Self> {
        register_builtin_tools(&mut self.registry)?;
        Ok(self)
    }

    /// Add calculator tool only
    #[inline(always)]
    pub fn with_calculator(mut self) -> AnthropicResult<Self> {
        let calculator = Box::new(CalculatorTool);
        self.registry
            .register_tool("calculator".to_string(), calculator)?;
        Ok(self)
    }

    /// Add file operations tool only (requires API key in execution context)
    #[inline(always)]
    pub fn with_file_operations(mut self) -> AnthropicResult<Self> {
        let file_ops = Box::new(FileOperationsTool);
        self.registry
            .register_tool("file_operations".to_string(), file_ops)?;
        Ok(self)
    }

    /// Add custom tool executor
    #[inline(always)]
    pub fn with_tool(
        mut self,
        name: String,
        executor: Box<dyn ToolExecutor + Send + Sync>,
    ) -> AnthropicResult<Self> {
        self.registry.register_tool(name, executor)?;
        Ok(self)
    }

    /// Add typed tool with zero-allocation storage
    #[inline(always)]
    pub fn with_typed_tool<D, Req, Res>(
        mut self,
        tool: TypedTool<D, Req, Res>,
    ) -> AnthropicResult<Self>
    where
        D: Send + Sync + 'static,
        Req: serde::de::DeserializeOwned + Send + 'static,
        Res: serde::Serialize + Send + 'static,
    {
        self.registry.register_typed_tool(tool)?;
        Ok(self)
    }

    /// Build the final registry
    #[inline(always)]
    pub fn build(self) -> ToolRegistry {
        self.registry
    }
}

impl Default for ToolRegistryBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to create tool registry builder
#[inline(always)]
pub fn builder() -> ToolRegistryBuilder {
    ToolRegistryBuilder::new()
}
