//! Modular tool system for Anthropic Claude API
//! 
//! This module has been decomposed into smaller, maintainable components
//! for better organization and production readiness.
//! 
//! ## Architecture
//! 
//! The tool system is organized into the following modules:
//! 
//! - `core`: Foundational traits, types, and error handling
//! - `function_calling`: Type-safe function calling and execution system  
//! - `calculator`: Production-ready mathematical expression evaluator
//! - `file_operations`: Anthropic Files API integration with HTTP/3
//! 
//! ## Usage
//! 
//! ```rust
//! use crate::clients::anthropic::tools::{with_builtins, ToolExecutionContext};
//! 
//! // Create registry with built-in tools
//! let registry = with_builtins();
//! 
//! // Execute calculator tool
//! let context = ToolExecutionContext::default();
//! let input = json!({"expression": "2 + 3 * 4"});
//! let result = registry.execute_tool("calculator", input, &context).await?;
//! ```
//! 
//! ## Builder Pattern
//! 
//! ```rust
//! use crate::clients::anthropic::tools::builder;
//! 
//! let registry = builder()
//!     .with_calculator()?
//!     .with_file_operations()?
//!     .build();
//! ```

// Import and re-export everything from the tools module
mod tools;

pub use tools::*;

// Legacy compatibility exports (if any external code imports these directly)
pub use tools::{
    // Core types
    Tool, ToolError, ToolExecutionError, ToolRegistrationError,
    Emitter, ChainControl, SchemaType, Message, AnthropicResult, AnthropicError,
    
    // Function calling system
    ToolBuilder, ToolExecutor, ToolRegistry, ToolExecutionContext, 
    ToolResult, ToolOutput, Conversation,
    
    // Built-in tools
    CalculatorTool, FileOperationsTool, ExpressionEvaluator, ExpressionError,
    
    // Convenience functions
    with_builtins, builder,
};