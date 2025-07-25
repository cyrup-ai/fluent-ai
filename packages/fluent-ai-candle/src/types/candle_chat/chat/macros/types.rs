//! Macro system types and data structures
//!
//! This module contains all the core types, enums, and data structures for the
//! chat macro system including actions, states, contexts, and configurations.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

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
        name: Arc<str>,
        value: Arc<str>,
        timestamp: Duration},
    /// Conditional execution based on variable
    Conditional {
        condition: Arc<str>,
        then_actions: Arc<[MacroAction]>,
        else_actions: Option<Arc<[MacroAction]>>,
        timestamp: Duration},
    /// Loop execution
    Loop {
        iterations: u32,
        actions: Arc<[MacroAction]>,
        timestamp: Duration}}

impl MacroAction {
    /// Convert to legacy MacroAction type for compatibility
    pub fn to_legacy(&self) -> crate::types::candle_chat::macros::types::MacroAction {
        use crate::types::candle_chat::macros::types::MacroAction as LegacyAction;
        match self {
            MacroAction::SendMessage { content, message_type, timestamp } => {
                LegacyAction::SendMessage {
                    content: content.to_string(),
                    message_type: message_type.to_string(),
                    timestamp: *timestamp,
                }
            }
            MacroAction::ExecuteCommand { command, timestamp } => {
                LegacyAction::ExecuteCommand {
                    command: command.clone(),
                    timestamp: *timestamp,
                }
            }
            MacroAction::Wait { duration, timestamp } => {
                LegacyAction::Wait {
                    duration: *duration,
                    timestamp: *timestamp,
                }
            }
            MacroAction::SetVariable { name, value, timestamp } => {
                LegacyAction::SetVariable {
                    name: name.to_string(),
                    value: value.to_string(),
                    timestamp: *timestamp,
                }
            }
            MacroAction::Conditional { condition, then_actions, else_actions, timestamp } => {
                LegacyAction::Conditional {
                    condition: condition.to_string(),
                    then_actions: then_actions.iter().map(|a| a.to_legacy()).collect(),
                    else_actions: else_actions.as_ref().map(|actions| 
                        actions.iter().map(|a| a.to_legacy()).collect()
                    ),
                    timestamp: *timestamp,
                }
            }
            MacroAction::Loop { iterations, actions, timestamp } => {
                LegacyAction::Loop {
                    iterations: *iterations,
                    actions: actions.iter().map(|a| a.to_legacy()).collect(),
                    timestamp: *timestamp,
                }
            }
        }
    }
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
    pub variables: HashMap<Arc<str>, Arc<str>>,
    pub execution_id: Uuid,
    pub start_time: u64,
    pub current_action: usize,
    pub loop_stack: Vec<LoopContext>}

/// Loop execution context for tracking nested loops
#[derive(Debug, Clone)]
pub struct LoopContext {
    pub iteration: u32,
    pub max_iterations: u32,
    pub start_action: usize,
    pub end_action: usize}

/// Macro recording session for capturing actions
#[derive(Debug)]
pub struct MacroRecordingSession {
    pub id: Uuid,
    pub name: Arc<str>,
    pub description: Option<Arc<str>>,
    pub state: MacroRecordingState,
    pub start_time: u64,
    pub actions: Vec<MacroAction>,
    pub variables: HashMap<Arc<str>, Arc<str>>,
    pub auto_save: bool,
    pub metadata: MacroMetadata}

impl MacroRecordingSession {
    /// Create new recording session with metadata
    pub fn new(id: Uuid, name: Arc<str>, description: Option<Arc<str>>) -> Self {
        let metadata = MacroMetadata {
            id,
            name: name.clone(),
            description: description.clone(),
            tags: vec![Arc::from("user-defined")],
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            modified_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            author: Some(Arc::from("system")),
            version: 1};
        
        Self {
            id,
            name,
            description,
            state: MacroRecordingState::Idle,
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            actions: Vec::new(),
            variables: HashMap::new(),
            auto_save: false,
            metadata}
    }
    
    /// Start recording session
    pub fn start_recording(&mut self) {
        self.state = MacroRecordingState::Recording;
        self.start_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }
    
    /// Stop recording session
    pub fn stop_recording(&mut self) {
        self.state = MacroRecordingState::Completed;
    }
    
    /// Add action to recording
    pub fn add_action(&mut self, action: MacroAction) {
        if self.state == MacroRecordingState::Recording {
            self.actions.push(action);
        }
    }
    
    /// Get recorded action count
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }
    
    /// Check if recording is active
    pub fn is_recording(&self) -> bool {
        self.state == MacroRecordingState::Recording
    }
}

/// Macro playback session for executing actions
#[derive(Debug)]
pub struct MacroPlaybackSession {
    pub id: Uuid,
    pub macro_id: Uuid,
    pub state: MacroPlaybackState,
    pub start_time: u64,
    pub current_action: usize,
    pub total_actions: usize,
    pub context: MacroExecutionContext,
    pub error: Option<MacroSystemError>}

impl MacroPlaybackSession {
    /// Create new playback session
    pub fn new(id: Uuid, macro_id: Uuid, total_actions: usize) -> Self {
        Self {
            id,
            macro_id,
            state: MacroPlaybackState::Idle,
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            current_action: 0,
            total_actions,
            context: MacroExecutionContext::default(),
            error: None}
    }
    
    /// Start playback session
    pub fn start_playback(&mut self) {
        self.state = MacroPlaybackState::Playing;
        self.start_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }
    
    /// Complete playback session
    pub fn complete_playback(&mut self) {
        self.state = MacroPlaybackState::Completed;
    }
    
    /// Fail playback session with error
    pub fn fail_playback(&mut self, error: MacroSystemError) {
        self.state = MacroPlaybackState::Failed;
        self.error = Some(error);
    }
    
    /// Advance to next action
    pub fn advance_action(&mut self) {
        if self.current_action < self.total_actions {
            self.current_action += 1;
        }
    }
    
    /// Check if playback is complete
    pub fn is_complete(&self) -> bool {
        self.current_action >= self.total_actions
    }
    
    /// Get progress percentage
    pub fn progress(&self) -> f32 {
        if self.total_actions == 0 {
            0.0
        } else {
            (self.current_action as f32 / self.total_actions as f32) * 100.0
        }
    }
}

/// Stored macro definition with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMacro {
    pub metadata: MacroMetadata,
    pub actions: Arc<[MacroAction]>,
    pub variables: HashMap<Arc<str>, Arc<str>>}

/// Macro metadata for organization and discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroMetadata {
    pub id: Uuid,
    pub name: Arc<str>,
    pub description: Option<Arc<str>>,
    pub tags: Vec<Arc<str>>,
    pub created_at: u64,
    pub modified_at: u64,
    pub author: Option<Arc<str>>,
    pub version: u32}

impl MacroMetadata {
    /// Convert to legacy MacroMetadata type for compatibility
    pub fn to_legacy(&self) -> crate::types::candle_chat::macros::types::MacroMetadata {
        use std::time::Duration;
        crate::types::candle_chat::macros::types::MacroMetadata {
            id: self.id,
            name: self.name.to_string().into(),
            description: self.description.as_ref().map(|s| s.to_string()).unwrap_or_default().into(),
            created_at: Duration::from_secs(self.created_at),
            updated_at: Duration::from_secs(self.modified_at),
            version: self.version,
            tags: self.tags.iter().cloned().collect(),
            author: self.author.as_ref().map(|s| s.to_string()).unwrap_or_default().into(),
            execution_count: 0,
            last_execution: None,
            average_duration: Duration::from_millis(0),
            success_rate: 1.0,
            category: "general".into(),
            is_private: false,
        }
    }
}

/// Result of macro execution
#[derive(Debug, Clone)]
pub enum MacroExecutionResult {
    /// Execution completed successfully
    Success {
        execution_id: Uuid,
        duration: Duration,
        actions_executed: usize},
    /// Execution failed with error
    Failed {
        execution_id: Uuid,
        error: MacroSystemError,
        actions_executed: usize},
    /// Execution was cancelled
    Cancelled {
        execution_id: Uuid,
        actions_executed: usize}}

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
    SessionNotActive}

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
    Error(MacroSystemError)}

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
    Custom(Arc<str>)}

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
    Custom(Arc<str>)}

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
    pub enable_validation: bool}

/// Macro system error types
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
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
    SubscriberNotFound(String)}

impl Default for MacroSystemConfig {
    fn default() -> Self {
        Self {
            max_recording_duration: Duration::from_secs(3600), // 1 hour
            max_actions_per_macro: 10000,
            max_concurrent_sessions: 100,
            enable_variable_substitution: true,
            max_recursion_depth: 10,
            enable_validation: true}
    }
}

impl Default for MacroExecutionContext {
    fn default() -> Self {
        Self {
            variables: HashMap::new(),
            execution_id: Uuid::new_v4(),
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            current_action: 0,
            loop_stack: Vec::new()}
    }
}