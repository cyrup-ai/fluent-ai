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
        /// Message content to send
        content: Arc<str>,
        /// Type classification for the message
        message_type: Arc<str>,
        /// Timestamp when action was recorded
        timestamp: Duration
    },
    /// Execute a command
    ExecuteCommand {
        /// Command to execute
        command: ImmutableChatCommand,
        /// Timestamp when action was recorded
        timestamp: Duration
    },
    /// Wait for a specified duration
    Wait {
        /// Duration to wait
        duration: Duration,
        /// Timestamp when action was recorded
        timestamp: Duration
    },
    /// Set a variable value
    SetVariable {
        /// Variable name
        name: Arc<str>,
        /// Variable value
        value: Arc<str>,
        /// Timestamp when action was recorded
        timestamp: Duration
    },
    /// Conditional execution based on variable
    Conditional {
        /// Condition expression to evaluate
        condition: Arc<str>,
        /// Actions to execute if condition is true
        then_actions: Arc<[MacroAction]>,
        /// Optional actions to execute if condition is false
        else_actions: Option<Arc<[MacroAction]>>,
        /// Timestamp when action was recorded
        timestamp: Duration
    },
    /// Loop execution
    Loop {
        /// Number of iterations to perform
        iterations: u32,
        /// Actions to execute in each iteration
        actions: Arc<[MacroAction]>,
        /// Timestamp when action was recorded
        timestamp: Duration
    }
}

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
    Completed
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
    Failed
}

/// Macro execution context with variable substitution
#[derive(Debug, Clone)]
pub struct MacroExecutionContext {
    /// Variables available during execution
    pub variables: HashMap<Arc<str>, Arc<str>>,
    /// Unique identifier for this execution
    pub execution_id: Uuid,
    /// Unix timestamp when execution started
    pub start_time: u64,
    /// Index of currently executing action
    pub current_action: usize,
    /// Stack of nested loop contexts
    pub loop_stack: Vec<LoopContext>
}

/// Loop execution context for tracking nested loops
#[derive(Debug, Clone)]
pub struct LoopContext {
    /// Current iteration number (0-based)
    pub iteration: u32,
    /// Maximum number of iterations
    pub max_iterations: u32,
    /// Index of first action in loop body
    pub start_action: usize,
    /// Index of last action in loop body
    pub end_action: usize
}

/// Macro recording session for capturing actions
#[derive(Debug)]
pub struct MacroRecordingSession {
    /// Unique identifier for this recording session
    pub id: Uuid,
    /// Name for the recording session
    pub name: Arc<str>,
    /// Optional description of what is being recorded
    pub description: Option<Arc<str>>,
    /// Current recording state
    pub state: MacroRecordingState,
    /// Unix timestamp when recording started
    pub start_time: u64,
    /// List of recorded actions
    pub actions: Vec<MacroAction>,
    /// Variables captured during recording
    pub variables: HashMap<Arc<str>, Arc<str>>,
    /// Whether to automatically save changes
    pub auto_save: bool,
    /// Metadata for the macro being recorded
    pub metadata: MacroMetadata
}

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
    /// Unique identifier for this playback session
    pub id: Uuid,
    /// ID of the macro being played back
    pub macro_id: Uuid,
    /// Current state of the playback session
    pub state: MacroPlaybackState,
    /// Unix timestamp when the session started
    pub start_time: u64,
    /// Index of the currently executing action
    pub current_action: usize,
    /// Total number of actions in the macro
    pub total_actions: usize,
    /// Execution context for the macro
    pub context: MacroExecutionContext,
    /// Error that occurred during playback, if any
    pub error: Option<MacroSystemError>
}

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
    /// Macro metadata containing ID, name, and other properties
    pub metadata: MacroMetadata,
    /// Sequence of actions to execute in this macro
    pub actions: Arc<[MacroAction]>,
    /// Variable definitions for macro parameters
    pub variables: HashMap<Arc<str>, Arc<str>>
}

/// Macro metadata for organization and discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroMetadata {
    /// Unique identifier for this macro
    pub id: Uuid,
    /// Human-readable name of the macro
    pub name: Arc<str>,
    /// Optional description of what the macro does
    pub description: Option<Arc<str>>,
    /// Tags for categorizing and searching macros
    pub tags: Vec<Arc<str>>,
    /// Unix timestamp when the macro was created
    pub created_at: u64,
    /// Unix timestamp when the macro was last modified
    pub modified_at: u64,
    /// Optional author name who created the macro
    pub author: Option<Arc<str>>,
    /// Version number of the macro
    pub version: u32
}

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
        /// Unique identifier for this execution attempt
        execution_id: Uuid,
        /// Total time taken to execute the macro
        duration: Duration,
        /// Number of actions that were successfully executed
        actions_executed: usize
    },
    /// Execution failed with error
    Failed {
        /// Unique identifier for this execution attempt
        execution_id: Uuid,
        /// Error that caused the execution to fail
        error: MacroSystemError,
        /// Number of actions that were executed before failure
        actions_executed: usize
    },
    /// Execution was cancelled
    Cancelled {
        /// Unique identifier for this execution attempt
        execution_id: Uuid,
        /// Number of actions that were executed before cancellation
        actions_executed: usize
    }
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
    SessionNotActive}

/// Result of action execution
#[derive(Debug, Clone)]
pub enum ActionExecutionResult {
    /// Action executed successfully
    Success,
    /// Action requires waiting
    Wait(
        /// Duration to wait before continuing
        Duration
    ),
    /// Skip to specific action index
    SkipToAction(
        /// Index of action to jump to
        usize
    ),
    /// Action execution failed
    Error(
        /// Error that caused the failure
        MacroSystemError
    )
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
    Custom(
        /// Custom condition function name
        Arc<str>
    )
}

/// Trigger conditions for automatic macro execution
#[derive(Debug, Clone)]
pub enum TriggerCondition {
    /// Execute on message containing text
    MessageContains(
        /// Text pattern to match in messages
        Arc<str>
    ),
    /// Execute on command execution
    CommandExecuted(
        /// Command name that triggers execution
        Arc<str>
    ),
    /// Execute on time interval
    TimeInterval(
        /// Interval duration between executions
        Duration
    ),
    /// Execute on variable change
    VariableChanged(
        /// Variable name to monitor for changes
        Arc<str>
    ),
    /// Execute on custom condition
    Custom(
        /// Custom trigger condition function
        Arc<str>
    )
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
    pub enable_validation: bool
}

/// Macro system error types
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MacroSystemError {
    /// Requested macro was not found
    #[error("Macro not found")]
    MacroNotFound,
    /// Requested session was not found
    #[error("Session not found")]
    SessionNotFound,
    /// Macro definition is invalid
    #[error("Invalid macro: {0}")]
    InvalidMacro(
        /// Description of why macro is invalid
        String
    ),
    /// Error during macro execution
    #[error("Execution error: {0}")]
    ExecutionError(
        /// Description of the execution error
        String
    ),
    /// Error during macro recording
    #[error("Recording error: {0}")]
    RecordingError(
        /// Description of the recording error
        String
    ),
    /// Error during macro playback
    #[error("Playback error: {0}")]
    PlaybackError(
        /// Description of the playback error
        String
    ),
    /// Required variable was not found
    #[error("Variable not found: {0}")]
    VariableNotFound(
        /// Name of the missing variable
        String
    ),
    /// Error evaluating conditional expression
    #[error("Condition evaluation error: {0}")]
    ConditionError(
        /// Description of the condition error
        String
    ),
    /// Macro recursion exceeded maximum allowed depth
    #[error("Maximum recursion depth exceeded")]
    MaxRecursionDepthExceeded,
    /// Feature not yet implemented
    #[error("Not implemented: {0}")]
    NotImplemented(
        /// Description of unimplemented feature
        String
    ),
    /// Internal system error
    #[error("Internal error: {0}")]
    InternalError(
        /// Description of the internal error
        String
    ),
    /// Event subscriber was not found
    #[error("Subscriber not found: {0}")]
    SubscriberNotFound(
        /// ID of the missing subscriber
        String
    )
}

impl Default for MacroSystemConfig {
    /// Create default macro system configuration with safe defaults
    fn default() -> Self {
        Self {
            max_recording_duration: Duration::from_secs(3600), // 1 hour
            max_actions_per_macro: 10000,
            max_concurrent_sessions: 100,
            enable_variable_substitution: true,
            max_recursion_depth: 10,
            enable_validation: true
        }
    }
}

impl Default for MacroExecutionContext {
    /// Create default macro execution context with new UUID and current time
    fn default() -> Self {
        Self {
            variables: HashMap::new(),
            execution_id: Uuid::new_v4(),
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            current_action: 0,
            loop_stack: Vec::new()
        }
    }
}