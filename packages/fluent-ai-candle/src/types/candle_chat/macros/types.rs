//! Macro types and data structures
//!
//! This module contains all the type definitions for the macro system,
//! including enums, structs, and configuration types.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use atomic_counter::{AtomicCounter, ConsistentCounter};
use crossbeam_queue::SegQueue;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::candle_chat::chat::commands::ImmutableChatCommand;

/// Macro action representing a single recorded operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MacroAction {
    /// Send a message with content
    SendMessage {
        content: String,
        message_type: String,
        timestamp: Duration},
    /// Execute a command
    ExecuteCommand {
        command: ImmutableChatCommand,
        timestamp: Duration},
    /// Wait for a specified duration
    Wait {
        duration: Duration,
        timestamp: Duration},
    /// Set a variable value
    SetVariable {
        name: String,
        value: String,
        timestamp: Duration},
    /// Conditional execution based on variable
    Conditional {
        condition: String,
        then_actions: Vec<MacroAction>,
        else_actions: Option<Vec<MacroAction>>,
        timestamp: Duration},
    /// Loop execution
    Loop {
        iterations: u32,
        actions: Vec<MacroAction>,
        timestamp: Duration}}

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
    Completed}

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
    Failed}

/// Macro execution context with variable substitution
#[derive(Debug, Clone)]
pub struct MacroExecutionContext {
    pub variables: HashMap<String, String>,
    pub execution_id: Uuid,
    pub start_time: Instant,
    pub current_action: usize,
    pub loop_stack: Vec<LoopContext>}

/// Loop execution context
#[derive(Debug, Clone)]
pub struct LoopContext {
    pub iteration: u32,
    pub max_iterations: u32,
    pub start_action: usize,
    pub end_action: usize}

/// Macro metadata and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroMetadata {
    pub id: Uuid,
    pub name: std::sync::Arc<str>,
    pub description: std::sync::Arc<str>,
    pub created_at: Duration,
    pub updated_at: Duration,
    pub version: u32,
    pub tags: Vec<std::sync::Arc<str>>,
    pub author: std::sync::Arc<str>,
    pub execution_count: u64,
    pub last_execution: Option<Duration>,
    pub average_duration: Duration,
    pub success_rate: f64,
    pub category: std::sync::Arc<str>,
    pub is_private: bool}

/// Complete macro definition with actions and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMacro {
    pub metadata: MacroMetadata,
    pub actions: Vec<MacroAction>,
    pub variables: HashMap<std::sync::Arc<str>, std::sync::Arc<str>>,
    pub triggers: Vec<String>,
    pub conditions: Vec<String>,
    pub dependencies: Vec<String>,
    pub execution_config: MacroExecutionConfig}

/// Macro execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroExecutionConfig {
    pub max_execution_time: Duration,
    pub retry_count: u32,
    pub retry_delay: Duration,
    pub abort_on_error: bool,
    pub parallel_execution: bool,
    pub priority: u8,
    pub resource_limits: ResourceLimits}

/// Resource limits for macro execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_memory_mb: u32,
    pub max_cpu_percent: u8,
    pub max_network_requests: u32,
    pub max_file_operations: u32}

/// Macro recording session
#[derive(Debug)]
pub struct MacroRecordingSession {
    pub id: Uuid,
    pub name: String,
    pub start_time: Instant,
    pub actions: SegQueue<MacroAction>,
    pub state: MacroRecordingState,
    pub variables: HashMap<String, String>,
    pub metadata: MacroMetadata}

impl Clone for MacroRecordingSession {
    fn clone(&self) -> Self {
        // Clone actions from SegQueue by draining and re-adding
        let actions = SegQueue::new();
        while let Some(action) = self.actions.pop() {
            actions.push(action.clone());
        }
        
        Self {
            id: self.id,
            name: self.name.clone(),
            start_time: self.start_time,
            actions,
            state: self.state.clone(),
            variables: self.variables.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

/// Macro playback session
#[derive(Debug)]
pub struct MacroPlaybackSession {
    pub id: Uuid,
    pub macro_id: Uuid,
    pub start_time: Instant,
    pub context: MacroExecutionContext,
    pub state: MacroPlaybackState,
    pub current_action: usize,
    pub total_actions: usize,
    pub error: Option<String>}

/// Macro execution statistics
#[derive(Debug, Default)]
pub struct ExecutionStats {
    pub total_executions: ConsistentCounter,
    pub successful_executions: ConsistentCounter,
    pub failed_executions: ConsistentCounter,
    pub total_duration: parking_lot::Mutex<Duration>,
    pub average_duration: parking_lot::Mutex<Duration>,
    pub last_execution: parking_lot::Mutex<Option<Instant>>}

impl Clone for ExecutionStats {
    fn clone(&self) -> Self {
        ExecutionStats {
            total_executions: ConsistentCounter::new(self.total_executions.get()),
            successful_executions: ConsistentCounter::new(self.successful_executions.get()),
            failed_executions: ConsistentCounter::new(self.failed_executions.get()),
            total_duration: parking_lot::Mutex::new(*self.total_duration.lock()),
            average_duration: parking_lot::Mutex::new(*self.average_duration.lock()),
            last_execution: parking_lot::Mutex::new(*self.last_execution.lock())}
    }
}

impl Default for MacroExecutionConfig {
    fn default() -> Self {
        Self {
            max_execution_time: Duration::from_secs(300), // 5 minutes
            retry_count: 3,
            retry_delay: Duration::from_millis(1000),
            abort_on_error: false,
            parallel_execution: false,
            priority: 5,
            resource_limits: ResourceLimits::default()}
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 100,
            max_cpu_percent: 25,
            max_network_requests: 50,
            max_file_operations: 20}
    }
}

/// Macro system error types
#[derive(Debug, thiserror::Error)]
pub enum MacroSystemError {
    #[error("Macro not found: {0}")]
    MacroNotFound(Uuid),
    #[error("Session not found: {0}")]
    SessionNotFound(Uuid),
    #[error("Recording already in progress: {0}")]
    RecordingInProgress(Uuid),
    #[error("Playback already in progress: {0}")]
    PlaybackInProgress(Uuid),
    #[error("Invalid macro action: {0}")]
    InvalidAction(String),
    #[error("Execution timeout")]
    ExecutionTimeout,
    #[error("Variable not found: {0}")]
    VariableNotFound(String),
    #[error("Condition evaluation error: {0}")]
    ConditionError(String),
    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Lock poisoned: {0}")]
    LockPoisoned(String),
    #[error("Time error: {0}")]
    TimeError(String),
    #[error("Lock error")]
    LockError,
    #[error("Internal error: {0}")]
    InternalError(String),
    #[error("Invalid macro: {0}")]
    InvalidMacro(String),
    #[error("Maximum recursion depth exceeded")]
    MaxRecursionDepthExceeded,
    #[error("Storage error: {0}")]
    StorageError(String),
    #[error("System error: {0}")]
    SystemError(String)}

impl<T> From<std::sync::PoisonError<T>> for MacroSystemError {
    fn from(err: std::sync::PoisonError<T>) -> Self {
        MacroSystemError::LockPoisoned(err.to_string())
    }
}

/// Result type for macro operations
pub type MacroResult<T> = Result<T, MacroSystemError>;

/// Action execution result
#[derive(Debug)]
pub enum ActionExecutionResult {
    /// Action executed successfully
    Success,
    /// Action failed with error
    Error(String),
    /// Wait for duration before continuing
    Wait(Duration),
    /// Skip to specific action index
    SkipToAction(usize),
    /// Complete execution
    Complete}

/// Macro playback result
#[derive(Debug)]
pub enum MacroPlaybackResult {
    /// Action executed successfully
    ActionExecuted,
    /// All actions completed
    Completed,
    /// Playback failed
    Failed,
    /// Playback paused
    Paused}