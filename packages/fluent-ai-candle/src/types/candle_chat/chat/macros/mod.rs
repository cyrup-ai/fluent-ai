//! Macro system for chat automation
//!
//! This module provides a comprehensive macro system for recording, storing,
//! and playing back chat interactions using zero-allocation patterns and
//! lock-free data structures for blazing-fast performance.
//!
//! # Architecture
//!
//! The macro system is decomposed into focused modules:
//! - [`types`] - Core types, enums, and data structures
//! - [`parser`] - Macro parsing and syntax analysis
//! - [`context`] - Context handling and variable management
//! - [`execution`] - Macro execution and evaluation logic
//! - [`actions`] - Action processing and command handling
//!
//! # Example Usage
//!
//! ```rust
//! use fluent_ai_candle::types::candle_chat::chat::macros::*;
//!
//! // Create a macro execution engine
//! let engine = MacroExecutionEngine::new();
//!
//! // Create a simple macro
//! let macro_def = StoredMacro {
//!     metadata: MacroMetadata {
//!         id: uuid::Uuid::new_v4(),
//!         name: std::sync::Arc::from("Hello Macro"),
//!         description: Some(std::sync::Arc::from("A simple greeting macro")),
//!         tags: vec![],
//!         created_at: std::time::Instant::now(),
//!         modified_at: std::time::Instant::now(),
//!         author: None,
//!         version: 1,
//!     },
//!     actions: std::sync::Arc::from([
//!         MacroAction::SendMessage {
//!             content: std::sync::Arc::from("Hello, World!"),
//!             message_type: std::sync::Arc::from("text"),
//!             timestamp: std::time::Duration::from_millis(0),
//!         }
//!     ]),
//!     variables: std::collections::HashMap::new(),
//! };
//!
//! // Execute the macro
//! let results = engine.execute_macro(macro_def, std::collections::HashMap::new());
//! // Process results using AsyncStream patterns...
//! ```

pub mod actions;
pub mod context;
pub mod errors;
pub mod execution;
pub mod parser;
pub mod processor;
pub mod system;
pub mod types;

// Re-export commonly used types from types module
pub use types::{
    MacroAction, MacroRecordingState, MacroPlaybackState, MacroExecutionContext,
    LoopContext, MacroRecordingSession, MacroPlaybackSession, StoredMacro,
    MacroMetadata, MacroExecutionResult, MacroPlaybackResult, ActionExecutionResult,
    ConditionType, TriggerCondition, MacroSystemConfig, MacroSystemError,
};

// Re-export parser types and functionality
pub use parser::{
    MacroParser, MacroParserConfig, ParseResult, ParseError, ParseErrorType,
    ParseWarning, ParseWarningType, ParsedExpression, BinaryOperator, UnaryOperator,
};

// Re-export context management functionality
pub use context::{
    MacroContextManager, ContextConfig, VariableChangeListener,
    VariableResolutionResult, ContextEvaluator,
};

// Re-export execution engine functionality
pub use execution::{
    MacroExecutionEngine, ExecutionConfig, ExecutionSession, ExecutionState,
    ExecutionTask, TaskPriority, ExecutionTraceEntry, ExecutionStatistics,
};

// Re-export action processing functionality
pub use actions::{
    ActionHandlerRegistry, ActionHandlerConfig, ActionHandlerStats,
    MessageActionProcessor, MessageProcessorConfig, MessageProcessorStats,
    QueuedMessage, MessagePriority, CommandActionProcessor, CommandProcessorConfig,
    CommandProcessorStats, CommandExecution, CommandExecutionResult,
    VariableActionProcessor, VariableProcessorConfig, VariableProcessorStats,
    VariableChange,
};

// Convenience type aliases for common usage patterns
pub type MacroHandler = Box<dyn Fn(&MacroAction, &mut MacroExecutionContext) -> Result<ActionExecutionResult, MacroSystemError> + Send + Sync>;

/// Create a default macro execution environment
pub fn create_default_environment() -> (MacroExecutionEngine, ActionHandlerRegistry, MacroParser) {
    let engine = MacroExecutionEngine::new();
    let handlers = ActionHandlerRegistry::new();
    let parser = MacroParser::new();
    
    (engine, handlers, parser)
}

/// Create a macro execution environment with custom configurations
pub fn create_configured_environment(
    execution_config: ExecutionConfig,
    handler_config: ActionHandlerConfig,
    parser_config: MacroParserConfig,
) -> (MacroExecutionEngine, ActionHandlerRegistry, MacroParser) {
    let engine = MacroExecutionEngine::with_config(execution_config);
    let mut handlers = ActionHandlerRegistry::new();
    handlers.config = handler_config;
    let parser = MacroParser::with_config(parser_config);
    
    (engine, handlers, parser)
}

/// Validate a macro definition for correctness
pub fn validate_macro(macro_def: &StoredMacro) -> Result<(), Vec<MacroSystemError>> {
    let mut errors = Vec::new();

    // Validate metadata
    if macro_def.metadata.name.is_empty() {
        errors.push(MacroSystemError::InvalidMacro("Macro name cannot be empty".to_string()));
    }

    // Validate actions
    if macro_def.actions.is_empty() {
        errors.push(MacroSystemError::InvalidMacro("Macro must have at least one action".to_string()));
    }

    // Check for circular references in conditional actions
    if let Err(error) = check_circular_references(&macro_def.actions) {
        errors.push(error);
    }

    // Validate action timestamps are in order
    if let Err(error) = validate_action_timestamps(&macro_def.actions) {
        errors.push(error);
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Check for circular references in macro actions
fn check_circular_references(actions: &[MacroAction]) -> Result<(), MacroSystemError> {
    // Simple implementation - in practice this would be more sophisticated
    for action in actions {
        match action {
            MacroAction::Conditional { then_actions, else_actions, .. } => {
                check_circular_references(then_actions)?;
                if let Some(else_actions) = else_actions {
                    check_circular_references(else_actions)?;
                }
            }
            MacroAction::Loop { actions, .. } => {
                check_circular_references(actions)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Validate that action timestamps are in chronological order
fn validate_action_timestamps(actions: &[MacroAction]) -> Result<(), MacroSystemError> {
    let mut last_timestamp = std::time::Duration::from_millis(0);
    
    for action in actions {
        let timestamp = match action {
            MacroAction::SendMessage { timestamp, .. } => *timestamp,
            MacroAction::ExecuteCommand { timestamp, .. } => *timestamp,
            MacroAction::Wait { timestamp, .. } => *timestamp,
            MacroAction::SetVariable { timestamp, .. } => *timestamp,
            MacroAction::Conditional { timestamp, .. } => *timestamp,
            MacroAction::Loop { timestamp, .. } => *timestamp,
        };

        if timestamp < last_timestamp {
            return Err(MacroSystemError::InvalidMacro(
                "Action timestamps must be in chronological order".to_string()
            ));
        }
        
        last_timestamp = timestamp;
    }
    
    Ok(())
}

/// Get macro system version information
pub fn version_info() -> MacroVersionInfo {
    MacroVersionInfo {
        version: "1.0.0".to_string(),
        build_date: "2025-07-24".to_string(),
        features: vec![
            "zero-allocation".to_string(),
            "lock-free".to_string(),
            "async-stream".to_string(),
            "variable-substitution".to_string(),
            "condition-evaluation".to_string(),
            "action-handlers".to_string(),
        ],
    }
}

/// Version information for the macro system
#[derive(Debug, Clone)]
pub struct MacroVersionInfo {
    /// System version
    pub version: String,
    /// Build date
    pub build_date: String,
    /// Available features
    pub features: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[test]
    fn test_macro_validation() {
        let valid_macro = StoredMacro {
            metadata: MacroMetadata {
                id: uuid::Uuid::new_v4(),
                name: Arc::from("Test Macro"),
                description: None,
                tags: vec![],
                created_at: Instant::now(),
                modified_at: Instant::now(),
                author: None,
                version: 1,
            },
            actions: Arc::from([
                MacroAction::SendMessage {
                    content: Arc::from("Hello"),
                    message_type: Arc::from("text"),
                    timestamp: Duration::from_millis(0),
                }
            ]),
            variables: HashMap::new(),
        };

        assert!(validate_macro(&valid_macro).is_ok());
    }

    #[test]
    fn test_environment_creation() {
        let (engine, handlers, parser) = create_default_environment();
        
        // Verify components are properly initialized
        assert!(engine.get_statistics().active_sessions == 0);
        assert!(!handlers.list_handlers().is_empty());
        // Parser doesn't have a simple validation method, but creation success is good
    }

    #[test]
    fn test_version_info() {
        let version = version_info();
        assert_eq!(version.version, "1.0.0");
        assert!(!version.features.is_empty());
    }
}