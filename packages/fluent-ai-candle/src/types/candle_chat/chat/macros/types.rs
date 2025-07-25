//! Macro system types and data structures
//!
//! This module contains all the core types, enums, and data structures for the
//! chat macro system including actions, states, contexts, and configurations.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::{Duration, Instant};

use atomic_counter::{AtomicCounter, ConsistentCounter};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::candle_chat::chat::commands::ImmutableChatCommand;

/// Macro action representing a single recorded operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MacroAction {
    /// Send a message with content
    SendMessage {
        content: Arc<str>,
        message_type: Arc<str>,
        timestamp: Duration,
    },
    /// Execute a command
    ExecuteCommand {
        command: ImmutableChatCommand,
        timestamp: Duration,
    },
    /// Wait for a specified duration
    Wait {
        duration: Duration,
        timestamp: Duration,
    },
    /// Set a variable value
    SetVariable {
        name: Arc<str>,
        value: Arc<str>,
        timestamp: Duration,
    },
    /// Conditional execution based on variable
    Conditional {
        condition: Arc<str>,
        then_actions: Arc<[MacroAction]>,
        else_actions: Option<Arc<[MacroAction]>>,
        timestamp: Duration,
    },
    /// Loop execution
    Loop {
        iterations: u32,
        actions: Arc<[MacroAction]>,
        timestamp: Duration,
    },
}

/// Macro recording state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroRecordingState {
    /// Not recording
    Idle,
    /// Currently recording
    Recording,
    /// Recording paused
    Paused,
    /// Recording completed
    Completed,
}

/// Macro playback state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroPlaybackState {
    /// Not playing
    Idle,
    /// Currently playing
    Playing,
    /// Playback paused
    Paused,
    /// Playback completed
    Completed,
    /// Playback failed
    Failed,
}

/// Macro execution context with variable substitution
#[derive(Debug, Clone)]
pub struct MacroExecutionContext {
    pub variables: HashMap<Arc<str>, Arc<str>>,
    pub execution_id: Uuid,
    pub start_time: Instant,
    pub current_action: usize,
    pub loop_stack: Vec<LoopContext>,
}

/// Loop execution context for tracking nested loops
#[derive(Debug, Clone)]
pub struct LoopContext {
    pub iteration: u32,
    pub max_iterations: u32,
    pub start_action: usize,
    pub end_action: usize,
}

/// Macro recording session for capturing actions
#[derive(Debug)]
pub struct MacroRecordingSession {
    pub id: Uuid,
    pub name: Arc<str>,
    pub description: Option<Arc<str>>,
    pub state: MacroRecordingState,
    pub start_time: Instant,
    pub actions: Vec<MacroAction>,
    pub variables: HashMap<Arc<str>, Arc<str>>,
    pub auto_save: bool,
}

/// Macro playback session for executing actions
#[derive(Debug)]
pub struct MacroPlaybackSession {
    pub id: Uuid,
    pub macro_id: Uuid,
    pub state: MacroPlaybackState,
    pub start_time: Instant,
    pub current_action: usize,
    pub context: MacroExecutionContext,
    pub error: Option<MacroSystemError>,
}

/// Stored macro definition with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMacro {
    pub metadata: MacroMetadata,
    pub actions: Arc<[MacroAction]>,
    pub variables: HashMap<Arc<str>, Arc<str>>,
}

/// Macro metadata for organization and discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroMetadata {
    pub id: Uuid,
    pub name: Arc<str>,
    pub description: Option<Arc<str>>,
    pub tags: Vec<Arc<str>>,
    pub created_at: Instant,
    pub modified_at: Instant,
    pub author: Option<Arc<str>>,
    pub version: u32,
}

/// Result of macro execution
#[derive(Debug, Clone)]
pub enum MacroExecutionResult {
    /// Execution completed successfully
    Success {
        execution_id: Uuid,
        duration: Duration,
        actions_executed: usize,
    },
    /// Execution failed with error
    Failed {
        execution_id: Uuid,
        error: MacroSystemError,
        actions_executed: usize,
    },
    /// Execution was cancelled
    Cancelled {
        execution_id: Uuid,
        actions_executed: usize,
    },
}

/// Result of macro playback operations
#[derive(Debug, Clone)]
pub enum MacroPlaybackResult {
    /// Session started successfully
    SessionStarted,
    /// Action executed successfully
    ActionExecuted,
    /// Playback completed
    Completed,
    /// Playback failed
    Failed,
    /// Session not found
    SessionNotFound,
    /// Session not active
    SessionNotActive,
}

/// Result of action execution
#[derive(Debug, Clone)]
pub enum ActionExecutionResult {
    /// Action executed successfully
    Success,
    /// Action requires waiting
    Wait(Duration),
    /// Skip to specific action index
    SkipToAction(usize),
    /// Action execution failed
    Error(MacroSystemError),
}

/// Conditional execution criteria
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConditionType {
    /// Variable equals value
    VariableEquals,
    /// Variable contains substring
    VariableContains,
    /// Variable matches regex
    VariableMatches,
    /// Variable is empty
    VariableEmpty,
    /// Variable is not empty
    VariableNotEmpty,
    /// Custom condition function
    Custom(Arc<str>),
}

/// Trigger conditions for automatic macro execution
#[derive(Debug, Clone)]
pub enum TriggerCondition {
    /// Execute on message containing text
    MessageContains(Arc<str>),
    /// Execute on command execution
    CommandExecuted(Arc<str>),
    /// Execute on time interval
    TimeInterval(Duration),
    /// Execute on variable change
    VariableChanged(Arc<str>),
    /// Execute on custom condition
    Custom(Arc<str>),
}

/// Macro system configuration
#[derive(Debug, Clone)]
pub struct MacroSystemConfig {
    /// Maximum recording duration
    pub max_recording_duration: Duration,
    /// Maximum actions per macro
    pub max_actions_per_macro: usize,
    /// Maximum concurrent playback sessions
    pub max_concurrent_sessions: usize,
    /// Enable automatic variable substitution
    pub enable_variable_substitution: bool,
    /// Maximum recursion depth
    pub max_recursion_depth: usize,
    /// Enable macro validation
    pub enable_validation: bool,
}

/// Macro system error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum MacroSystemError {
    #[error("Macro not found")]
    MacroNotFound,
    #[error("Session not found")]
    SessionNotFound,
    #[error("Invalid macro: {0}")]
    InvalidMacro(String),
    #[error("Execution error: {0}")]
    ExecutionError(String),
    #[error("Recording error: {0}")]
    RecordingError(String),
    #[error("Playback error: {0}")]
    PlaybackError(String),
    #[error("Variable not found: {0}")]
    VariableNotFound(String),
    #[error("Condition evaluation error: {0}")]
    ConditionError(String),
    #[error("Maximum recursion depth exceeded")]
    MaxRecursionDepthExceeded,
    #[error("Not implemented: {0}")]
    NotImplemented(String),
    #[error("Internal error: {0}")]
    InternalError(String),
    #[error("Subscriber not found: {0}")]
    SubscriberNotFound(String),
}

impl Default for MacroSystemConfig {
    fn default() -> Self {
        Self {
            max_recording_duration: Duration::from_secs(3600), // 1 hour
            max_actions_per_macro: 10000,
            max_concurrent_sessions: 100,
            enable_variable_substitution: true,
            max_recursion_depth: 10,
            enable_validation: true,
        }
    }
}

impl Default for MacroExecutionContext {
    fn default() -> Self {
        Self {
            variables: HashMap::new(),
            execution_id: Uuid::new_v4(),
            start_time: Instant::now(),
            current_action: 0,
            loop_stack: Vec::new(),
        }
    }
}