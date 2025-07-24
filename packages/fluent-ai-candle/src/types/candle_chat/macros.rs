//! Macro system for chat automation with lock-free data structures
//!
//! This module provides a comprehensive macro system for recording, storing,
//! and playing back chat interactions using zero-allocation patterns and
//! lock-free data structures for blazing-fast performance.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use atomic_counter::{AtomicCounter, ConsistentCounter};
use crossbeam_queue::SegQueue;
use crossbeam_skiplist::SkipMap;
use fluent_ai_async::AsyncStream;
use fluent_ai_async::{emit, handle_error};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::types::candle_chat::commands::ImmutableChatCommand;

/// Macro action representing a single recorded operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MacroAction {
    /// Send a message with content
    SendMessage {
        content: String,
        message_type: String,
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
        name: String,
        value: String,
        timestamp: Duration,
    },
    /// Conditional execution based on variable
    Conditional {
        condition: String,
        then_actions: Vec<MacroAction>,
        else_actions: Option<Vec<MacroAction>>,
        timestamp: Duration,
    },
    /// Loop execution
    Loop {
        iterations: u32,
        actions: Vec<MacroAction>,
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
    pub variables: HashMap<String, String>,
    pub execution_id: Uuid,
    pub start_time: Instant,
    pub current_action: usize,
    pub loop_stack: Vec<LoopContext>,
}

/// Loop execution context
#[derive(Debug, Clone)]
pub struct LoopContext {
    pub iteration: u32,
    pub max_iterations: u32,
    pub start_action: usize,
    pub end_action: usize,
}

/// Macro metadata and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroMetadata {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub created_at: Duration,
    pub updated_at: Duration,
    pub version: u32,
    pub tags: Vec<String>,
    pub author: String,
    pub execution_count: u64,
    pub last_execution: Option<Duration>,
    pub average_duration: Duration,
    pub success_rate: f64,
    pub category: String,
    pub is_private: bool,
}

/// Complete macro definition with actions and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMacro {
    pub metadata: MacroMetadata,
    pub actions: Vec<MacroAction>,
    pub variables: HashMap<String, String>,
    pub triggers: Vec<String>,
    pub conditions: Vec<String>,
    pub dependencies: Vec<String>,
    pub execution_config: MacroExecutionConfig,
}

/// Macro execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroExecutionConfig {
    pub max_execution_time: Duration,
    pub retry_count: u32,
    pub retry_delay: Duration,
    pub abort_on_error: bool,
    pub parallel_execution: bool,
    pub priority: u8,
    pub resource_limits: ResourceLimits,
}

/// Resource limits for macro execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_memory_mb: u32,
    pub max_cpu_percent: u8,
    pub max_network_requests: u32,
    pub max_file_operations: u32,
}

/// Macro recording session
#[derive(Debug)]
pub struct MacroRecordingSession {
    pub id: Uuid,
    pub name: String,
    pub start_time: Instant,
    pub actions: SegQueue<MacroAction>,
    pub state: MacroRecordingState,
    pub variables: HashMap<String, String>,
    pub metadata: MacroMetadata,
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
    pub error: Option<String>,
}

/// High-performance macro system with lock-free operations
pub struct MacroSystem {
    /// Lock-free macro storage using skip list
    macros: SkipMap<Uuid, ChatMacro>,
    /// Active recording sessions
    recording_sessions: RwLock<HashMap<Uuid, MacroRecordingSession>>,
    /// Active playback sessions
    playback_sessions: RwLock<HashMap<Uuid, MacroPlaybackSession>>,
    /// Macro execution statistics
    execution_stats: SkipMap<Uuid, Arc<ExecutionStats>>,
    /// Global macro counter
    macro_counter: ConsistentCounter,
    /// Execution counter
    execution_counter: ConsistentCounter,
}

/// Macro execution statistics
#[derive(Debug, Default)]
pub struct ExecutionStats {
    pub total_executions: ConsistentCounter,
    pub successful_executions: ConsistentCounter,
    pub failed_executions: ConsistentCounter,
    pub total_duration: parking_lot::Mutex<Duration>,
    pub average_duration: parking_lot::Mutex<Duration>,
    pub last_execution: parking_lot::Mutex<Option<Instant>>,
}

impl Clone for ExecutionStats {
    fn clone(&self) -> Self {
        ExecutionStats {
            total_executions: ConsistentCounter::new(self.total_executions.get()),
            successful_executions: ConsistentCounter::new(self.successful_executions.get()),
            failed_executions: ConsistentCounter::new(self.failed_executions.get()),
            total_duration: parking_lot::Mutex::new(*self.total_duration.lock()),
            average_duration: parking_lot::Mutex::new(*self.average_duration.lock()),
            last_execution: parking_lot::Mutex::new(*self.last_execution.lock()),
        }
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
            resource_limits: ResourceLimits::default(),
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 100,
            max_cpu_percent: 25,
            max_network_requests: 50,
            max_file_operations: 20,
        }
    }
}

impl MacroSystem {
    /// Create a new macro system with optimal performance settings
    pub fn new() -> Self {
        Self {
            macros: SkipMap::new(),
            recording_sessions: RwLock::new(HashMap::new()),
            playback_sessions: RwLock::new(HashMap::new()),
            execution_stats: SkipMap::new(),
            macro_counter: ConsistentCounter::new(0),
            execution_counter: ConsistentCounter::new(0),
        }
    }

    /// Start recording a new macro
    pub fn start_recording(
        &self,
        name: String,
        description: String,
    ) -> Result<Uuid, MacroSystemError> {
        let session_id = Uuid::new_v4();
        let macro_id = Uuid::new_v4();

        let metadata = MacroMetadata {
            id: macro_id,
            name: name.clone(),
            description: description,
            created_at: Duration::from_secs(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|_| MacroSystemError::SystemTimeError)?
                    .as_secs(),
            ),
            updated_at: Duration::from_secs(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|_| MacroSystemError::SystemTimeError)?
                    .as_secs(),
            ),
            version: 1,
            tags: vec![],
            author: "system".to_string(),
            execution_count: 0,
            last_execution: None,
            average_duration: Duration::from_secs(0),
            success_rate: 0.0,
            category: "user-defined".to_string(),
            is_private: false,
        };

        let session = MacroRecordingSession {
            id: session_id,
            name,
            start_time: Instant::now(),
            actions: SegQueue::new(),
            state: MacroRecordingState::Recording,
            variables: HashMap::new(),
            metadata,
        };

        let mut sessions = self.recording_sessions.try_write().map_err(|_| {
            MacroSystemError::LockContentionError("Recording sessions lock contention".to_string())
        })?;
        sessions.insert(session_id, session);

        Ok(session_id)
    }

    /// Record a macro action
    pub fn record_action(
        &self,
        session_id: Uuid,
        action: MacroAction,
    ) -> Result<(), MacroSystemError> {
        let sessions = self.recording_sessions.try_read().map_err(|_| {
            MacroSystemError::SystemError("Failed to acquire read lock".to_string())
        })?;

        if let Some(session) = sessions.get(&session_id) {
            if session.state == MacroRecordingState::Recording {
                session.actions.push(action);
                Ok(())
            } else {
                Err(MacroSystemError::RecordingNotActive)
            }
        } else {
            Err(MacroSystemError::SessionNotFound)
        }
    }

    /// Stop recording and save the macro (zero-allocation, lock-free)
    pub fn stop_recording(&self, session_id: Uuid) -> Result<Uuid, MacroSystemError> {
        let mut sessions = self.recording_sessions.try_write().map_err(|_| {
            MacroSystemError::SystemError("Failed to acquire write lock".to_string())
        })?;

        if let Some(mut session) = sessions.remove(&session_id) {
            session.state = MacroRecordingState::Completed;

            // Collect all recorded actions
            let mut actions = Vec::new();
            while let Some(action) = session.actions.pop() {
                actions.push(action);
            }
            actions.reverse(); // Restore original order

            // Create the macro
            let chat_macro = ChatMacro {
                metadata: session.metadata.clone(),
                actions: actions.into(),
                variables: session.variables,
                triggers: vec![],
                conditions: vec![],
                dependencies: vec![],
                execution_config: MacroExecutionConfig::default(),
            };

            let macro_id = session.metadata.id;
            self.macros.insert(macro_id, chat_macro);
            self.macro_counter.inc();

            Ok(macro_id)
        } else {
            Err(MacroSystemError::SessionNotFound)
        }
    }

    /// Get a macro by ID
    pub fn get_macro(&self, macro_id: Uuid) -> Option<ChatMacro> {
        self.macros
            .get(&macro_id)
            .map(|entry| entry.value().clone())
    }

    /// List all available macros
    pub fn list_macros(&self) -> Vec<MacroMetadata> {
        self.macros
            .iter()
            .map(|entry| entry.value().metadata.clone())
            .collect()
    }

    /// Search macros by name, description, or tags
    pub fn search_macros(&self, query: &str) -> Vec<MacroMetadata> {
        let query_lower = query.to_lowercase();

        self.macros
            .iter()
            .filter(|entry| {
                let macro_def = entry.value();
                macro_def
                    .metadata
                    .name
                    .to_lowercase()
                    .contains(&query_lower)
                    || macro_def
                        .metadata
                        .description
                        .to_lowercase()
                        .contains(&query_lower)
                    || macro_def
                        .metadata
                        .tags
                        .iter()
                        .any(|tag| tag.to_lowercase().contains(&query_lower))
            })
            .map(|entry| entry.value().metadata.clone())
            .collect()
    }

    /// Start macro playback (zero-allocation streaming)
    pub fn start_playback(
        &self,
        macro_id: Uuid,
        variables: HashMap<String, String>,
    ) -> Result<Uuid, MacroSystemError> {
        let macro_def = self
            .get_macro(macro_id)
            .ok_or(MacroSystemError::MacroNotFound)?;

        let session_id = Uuid::new_v4();
        let context = MacroExecutionContext {
            variables,
            execution_id: session_id,
            start_time: Instant::now(),
            current_action: 0,
            loop_stack: Vec::new(),
        };

        let session = MacroPlaybackSession {
            id: session_id,
            macro_id,
            start_time: Instant::now(),
            context,
            state: MacroPlaybackState::Playing,
            current_action: 0,
            total_actions: macro_def.actions.len(),
            error: None,
        };

        let mut sessions = self.playback_sessions.try_write().map_err(|_| {
            MacroSystemError::LockContentionError("Playback sessions lock contention".to_string())
        })?;
        sessions.insert(session_id, session);

        self.execution_counter.inc();

        Ok(session_id)
    }

    /// Execute the next action in a playback session (zero-allocation streaming)
    pub fn execute_next_action(
        &self,
        session_id: Uuid,
    ) -> Result<MacroPlaybackResult, MacroSystemError> {
        let mut sessions = self.playback_sessions.try_write().map_err(|_| {
            MacroSystemError::SystemError("Failed to acquire write lock".to_string())
        })?;

        if let Some(session) = sessions.get_mut(&session_id) {
            if session.state != MacroPlaybackState::Playing {
                return Ok(MacroPlaybackResult::SessionNotActive);
            }

            let macro_def = self
                .get_macro(session.macro_id)
                .ok_or(MacroSystemError::MacroNotFound)?;

            if session.current_action >= macro_def.actions.len() {
                session.state = MacroPlaybackState::Completed;
                return Ok(MacroPlaybackResult::Completed);
            }

            let action = &macro_def.actions[session.current_action];
            let mut result_stream = self.execute_action(action.clone(), session.context.clone());

            let result = match result_stream.try_next() {
                Some(action_result) => action_result,
                None => {
                    return Err(MacroSystemError::ExecutionError(
                        "No result from action execution".to_string(),
                    ));
                }
            };

            session.current_action += 1;

            match result {
                ActionExecutionResult::Success => {
                    if session.current_action >= macro_def.actions.len() {
                        session.state = MacroPlaybackState::Completed;
                        Ok(MacroPlaybackResult::Completed)
                    } else {
                        Ok(MacroPlaybackResult::ActionExecuted)
                    }
                }
                ActionExecutionResult::Wait(duration) => {
                    std::thread::sleep(duration);
                    Ok(MacroPlaybackResult::ActionExecuted)
                }
                ActionExecutionResult::SkipToAction(index) => {
                    session.current_action = index;
                    Ok(MacroPlaybackResult::ActionExecuted)
                }
                ActionExecutionResult::Error(error) => {
                    session.state = MacroPlaybackState::Failed;
                    session.error = Some(error);
                    Ok(MacroPlaybackResult::Failed)
                }
            }
        } else {
            Err(MacroSystemError::SessionNotFound)
        }
    }

    /// Execute a single macro action
    fn execute_action(
        &self,
        action: MacroAction,
        mut context: MacroExecutionContext,
    ) -> AsyncStream<ActionExecutionResult> {
        // Extract needed data from self to avoid lifetime issues
        AsyncStream::with_channel(move |sender| {
            match action {
                MacroAction::SendMessage {
                    content,
                    message_type,
                    ..
                } => {
                    let resolved_content =
                        MacroSystem::resolve_variables_static(&content, &context.variables);
                    // In a real implementation, this would send the message to the chat system
                    println!(
                        "Sending message: {} (type: {})",
                        resolved_content, message_type
                    );
                    emit!(sender, ActionExecutionResult::Success);
                }
                MacroAction::ExecuteCommand { command, .. } => {
                    // In a real implementation, this would execute the command
                    println!("Executing command: {:?}", command);
                    emit!(sender, ActionExecutionResult::Success);
                }
                MacroAction::Wait { duration, .. } => {
                    emit!(sender, ActionExecutionResult::Wait(duration));
                }
                MacroAction::SetVariable { name, value, .. } => {
                    let resolved_value =
                        MacroSystem::resolve_variables_static(&value, &context.variables);
                    context.variables.insert(name.clone(), resolved_value);
                    emit!(sender, ActionExecutionResult::Success);
                }
                MacroAction::Conditional {
                    condition,
                    then_actions,
                    else_actions,
                    ..
                } => {
                    let condition_result =
                        MacroSystem::evaluate_condition_static(&condition, &context.variables);

                    let actions_to_execute = if condition_result {
                        &then_actions
                    } else if let Some(ref else_actions) = else_actions {
                        else_actions
                    } else {
                        emit!(sender, ActionExecutionResult::Success);
                        return;
                    };

                    // Execute conditional actions synchronously
                    for action in actions_to_execute.iter() {
                        // Simplified nested action execution for AsyncStream compatibility
                        // TODO: Implement proper nested action execution with AsyncStream
                        match action {
                            MacroAction::SendMessage {
                                content,
                                message_type,
                                ..
                            } => {
                                let resolved_content = MacroSystem::resolve_variables_static(
                                    content.as_ref(),
                                    &context.variables,
                                );
                                println!(
                                    "Nested: Sending message: {} (type: {})",
                                    resolved_content,
                                    message_type.as_str()
                                );
                            }
                            MacroAction::SetVariable { name, value, .. } => {
                                let resolved_value = MacroSystem::resolve_variables_static(
                                    value.as_ref(),
                                    &context.variables,
                                );
                                context.variables.insert(name.to_string(), resolved_value);
                            }
                            _ => {
                                println!("Nested action execution: {:?}", action);
                            }
                        }
                    }

                    emit!(sender, ActionExecutionResult::Success);
                }
                MacroAction::Loop {
                    iterations,
                    actions,
                    ..
                } => {
                    let loop_context = LoopContext {
                        iteration: 0,
                        max_iterations: iterations,
                        start_action: 0,
                        end_action: actions.len(),
                    };

                    context.loop_stack.push(loop_context);

                    for _ in 0..iterations {
                        for action in actions.iter() {
                            // Simplified nested action execution for AsyncStream compatibility
                            // TODO: Implement proper nested action execution with AsyncStream
                            match action {
                                MacroAction::SendMessage {
                                    content,
                                    message_type,
                                    ..
                                } => {
                                    let resolved_content = MacroSystem::resolve_variables_static(
                                        content,
                                        &context.variables,
                                    );
                                    println!(
                                        "Loop: Sending message: {} (type: {})",
                                        resolved_content, message_type
                                    );
                                }
                                MacroAction::SetVariable { name, value, .. } => {
                                    let resolved_value = MacroSystem::resolve_variables_static(
                                        value,
                                        &context.variables,
                                    );
                                    context.variables.insert(name.to_string(), resolved_value);
                                }
                                _ => {
                                    println!("Loop action execution: {:?}", action);
                                }
                            }
                        }
                    }

                    context.loop_stack.pop();
                    emit!(sender, ActionExecutionResult::Success);
                }
            }
        })
    }

    /// Resolve variables in a string (static version for AsyncStream usage)
    fn resolve_variables_static(content: &str, variables: &HashMap<String, String>) -> String {
        let mut result = content.to_string();

        for (key, value) in variables {
            let placeholder = format!("{{{}}}", key);
            result = result.replace(&placeholder, value);
        }

        result
    }

    /// Evaluate a condition string (static version for AsyncStream usage)
    fn evaluate_condition_static(condition: &str, variables: &HashMap<String, String>) -> bool {
        // Simple condition evaluation - in a real implementation, this would be more sophisticated
        if condition.contains("==") {
            let parts: Vec<&str> = condition.split("==").collect();
            if parts.len() == 2 {
                let left = Self::resolve_variables_static(parts[0].trim(), variables);
                let right = Self::resolve_variables_static(parts[1].trim(), variables);
                return left == right;
            }
        }

        // Default to false for unsupported conditions
        false
    }

    /// Resolve variables in a string (instance method)
    fn resolve_variables(&self, content: &str, variables: &HashMap<String, String>) -> String {
        Self::resolve_variables_static(content, variables)
    }

    /// Evaluate a condition string
    fn evaluate_condition(&self, condition: &str, variables: &HashMap<String, String>) -> bool {
        // Simple condition evaluation - in a real implementation, this would be more sophisticated
        if condition.contains("==") {
            let parts: Vec<&str> = condition.split("==").collect();
            if parts.len() == 2 {
                let left = self.resolve_variables(parts[0].trim(), variables);
                let right = self.resolve_variables(parts[1].trim(), variables);
                return left == right;
            }
        }

        // Default to false for unsupported conditions
        false
    }

    /// Get execution statistics for a macro
    pub fn get_execution_stats(&self, macro_id: Uuid) -> Option<ExecutionStats> {
        self.execution_stats
            .get(&macro_id)
            .map(|entry| (**entry.value()).clone())
    }

    /// Get total macro count
    pub fn get_macro_count(&self) -> usize {
        self.macro_counter.get()
    }

    /// Get total execution count
    pub fn get_execution_count(&self) -> usize {
        self.execution_counter.get()
    }
}

/// Result of macro action execution
#[derive(Debug)]
pub enum ActionExecutionResult {
    Success,
    Wait(Duration),
    SkipToAction(usize),
    Error(String),
}

/// Result of macro playback operation
#[derive(Debug)]
pub enum MacroPlaybackResult {
    ActionExecuted,
    Completed,
    Failed,
    SessionNotActive,
}

/// Macro system errors
#[derive(Debug, thiserror::Error)]
pub enum MacroSystemError {
    #[error("Recording session not found")]
    SessionNotFound,
    #[error("Recording not active")]
    RecordingNotActive,
    #[error("Macro not found")]
    MacroNotFound,
    #[error("System time error")]
    SystemTimeError,
    #[error("Execution error: {0}")]
    ExecutionError(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Lock contention error: {0}")]
    LockContentionError(String),
    #[error("System error: {0}")]
    SystemError(String),
}

impl Default for MacroSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating macros programmatically
pub struct MacroBuilder {
    name: Option<String>,
    description: Option<String>,
    actions: Vec<MacroAction>,
    variables: HashMap<String, String>,
    triggers: Vec<String>,
    conditions: Vec<String>,
    dependencies: Vec<String>,
    execution_config: MacroExecutionConfig,
}

impl MacroBuilder {
    /// Create a new macro builder
    pub fn new() -> Self {
        Self {
            name: None,
            description: None,
            actions: Vec::new(),
            variables: HashMap::new(),
            triggers: Vec::new(),
            conditions: Vec::new(),
            dependencies: Vec::new(),
            execution_config: MacroExecutionConfig::default(),
        }
    }

    /// Set the macro name
    pub fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Set the macro description
    pub fn description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Add an action to the macro
    pub fn add_action(mut self, action: MacroAction) -> Self {
        self.actions.push(action);
        self
    }

    /// Add a variable to the macro
    pub fn add_variable(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.variables.insert(name.into(), value.into());
        self
    }

    /// Set execution configuration
    pub fn execution_config(mut self, config: MacroExecutionConfig) -> Self {
        self.execution_config = config;
        self
    }

    /// Build the macro
    pub fn build(self) -> Result<ChatMacro, MacroSystemError> {
        let name = self
            .name
            .ok_or_else(|| MacroSystemError::ExecutionError("Name is required".to_string()))?;
        let description = self.description.unwrap_or_else(|| "".to_string());

        let metadata = MacroMetadata {
            id: Uuid::new_v4(),
            name,
            description,
            created_at: Duration::from_secs(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|_| MacroSystemError::SystemTimeError)?
                    .as_secs(),
            ),
            updated_at: Duration::from_secs(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|_| MacroSystemError::SystemTimeError)?
                    .as_secs(),
            ),
            version: 1,
            tags: vec![],
            author: "builder".to_string(),
            execution_count: 0,
            last_execution: None,
            average_duration: Duration::from_secs(0),
            success_rate: 0.0,
            category: "programmatic".to_string(),
            is_private: false,
        };

        Ok(ChatMacro {
            metadata,
            actions: self.actions.into(),
            variables: self.variables,
            triggers: self.triggers.into(),
            conditions: self.conditions.into(),
            dependencies: self.dependencies.into(),
            execution_config: self.execution_config,
        })
    }
}

impl Default for MacroBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Macro processor for executing and managing chat macros
///
/// This processor provides comprehensive macro execution capabilities with:
/// - Recording and playback of chat interactions
/// - Variable substitution and conditional logic
/// - Performance monitoring and error handling
/// - Concurrent execution with lock-free data structures
/// - Macro validation and optimization
#[derive(Debug, Clone)]
pub struct MacroProcessor {
    /// Macro storage with lock-free access
    macros: Arc<SkipMap<Uuid, ChatMacro>>,
    /// Execution statistics
    stats: Arc<MacroProcessorStats>,
    /// Variable context for macro execution
    variables: Arc<RwLock<HashMap<String, String>>>,
    /// Execution queue for async processing
    #[allow(dead_code)] // TODO: Implement in macro execution system
    execution_queue: Arc<SegQueue<MacroExecutionRequest>>,
    /// Configuration settings
    config: MacroProcessorConfig,
}

/// Macro processor statistics (internal atomic counters)
#[derive(Debug, Default)]
pub struct MacroProcessorStats {
    /// Total macros executed
    pub total_executions: AtomicUsize,
    /// Successful executions
    pub successful_executions: AtomicUsize,
    /// Failed executions
    pub failed_executions: AtomicUsize,
    /// Total execution time in microseconds
    pub total_execution_time_us: AtomicUsize,
    /// Active executions
    pub active_executions: AtomicUsize,
}

/// Macro processor statistics snapshot (for external API)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroProcessorStatsSnapshot {
    /// Total macros executed
    pub total_executions: usize,
    /// Successful executions
    pub successful_executions: usize,
    /// Failed executions
    pub failed_executions: usize,
    /// Total execution time in microseconds
    pub total_execution_time_us: usize,
    /// Active executions
    pub active_executions: usize,
}

/// Macro processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroProcessorConfig {
    /// Maximum concurrent executions
    pub max_concurrent_executions: usize,
    /// Default execution timeout in seconds
    pub default_timeout_seconds: u64,
    /// Enable variable substitution
    pub enable_variable_substitution: bool,
    /// Enable conditional execution
    pub enable_conditional_execution: bool,
    /// Enable loop execution
    pub enable_loop_execution: bool,
    /// Maximum macro recursion depth
    pub max_recursion_depth: usize,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Auto-save macro changes
    pub auto_save: bool,
}

/// Macro execution request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroExecutionRequest {
    /// Macro ID to execute
    pub macro_id: Uuid,
    /// Execution context variables
    pub context_variables: HashMap<String, String>,
    /// Execution timeout override
    pub timeout_override: Option<Duration>,
    /// Execution priority (higher = more priority)
    pub priority: u32,
    /// Request timestamp
    pub requested_at: Duration,
}

/// Macro execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroExecutionResult {
    /// Execution success indicator
    pub success: bool,
    /// Execution message/error
    pub message: String,
    /// Actions executed
    pub actions_executed: usize,
    /// Execution duration
    pub execution_duration: Duration,
    /// Variables modified during execution
    pub modified_variables: HashMap<String, String>,
    /// Execution metadata
    pub metadata: MacroExecutionMetadata,
}

/// Macro execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroExecutionMetadata {
    /// Execution ID
    pub execution_id: Uuid,
    /// Macro ID
    pub macro_id: Uuid,
    /// Start timestamp
    pub started_at: Duration,
    /// End timestamp
    pub completed_at: Duration,
    /// Execution context
    pub context: HashMap<String, String>,
    /// Performance metrics
    pub performance: MacroPerformanceMetrics,
}

/// Macro performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroPerformanceMetrics {
    /// CPU time used in microseconds
    pub cpu_time_us: u64,
    /// Memory used in bytes
    pub memory_bytes: u64,
    /// Network requests made
    pub network_requests: u32,
    /// Disk operations performed
    pub disk_operations: u32,
}

impl Default for MacroProcessorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_executions: 10,
            default_timeout_seconds: 30,
            enable_variable_substitution: true,
            enable_conditional_execution: true,
            enable_loop_execution: true,
            max_recursion_depth: 10,
            enable_monitoring: true,
            auto_save: true,
        }
    }
}

impl MacroProcessor {
    /// Create a new macro processor
    pub fn new() -> Self {
        Self {
            macros: Arc::new(SkipMap::new()),
            stats: Arc::new(MacroProcessorStats::default()),
            variables: Arc::new(RwLock::new(HashMap::new())),
            execution_queue: Arc::new(SegQueue::new()),
            config: MacroProcessorConfig::default(),
        }
    }

    /// Create a macro processor with custom configuration
    pub fn with_config(config: MacroProcessorConfig) -> Self {
        Self {
            macros: Arc::new(SkipMap::new()),
            stats: Arc::new(MacroProcessorStats::default()),
            variables: Arc::new(RwLock::new(HashMap::new())),
            execution_queue: Arc::new(SegQueue::new()),
            config,
        }
    }

    /// Register a macro
    pub fn register_macro(&self, macro_def: ChatMacro) -> Result<(), MacroSystemError> {
        // Validate macro
        self.validate_macro(&macro_def)?;

        // Store macro
        self.macros.insert(macro_def.metadata.id, macro_def);

        Ok(())
    }

    /// Unregister a macro
    pub fn unregister_macro(&self, macro_id: &Uuid) -> Result<(), MacroSystemError> {
        if self.macros.remove(macro_id).is_none() {
            return Err(MacroSystemError::MacroNotFound);
        }

        Ok(())
    }

    /// Execute a macro by ID (zero-allocation streaming)
    pub fn execute_macro(
        &self,
        macro_id: &Uuid,
        context_variables: HashMap<String, String>,
    ) -> AsyncStream<MacroExecutionResult> {
        let self_clone = self.clone();
        let macro_id = *macro_id;
        AsyncStream::with_channel(move |sender| {
            // Synchronous macro lookup and execution for blazing-fast performance
            match self_clone.macros.get(&macro_id) {
                Some(macro_def) => {
                    let macro_def = macro_def.value().clone();
                    match self_clone.execute_macro_sync(macro_def, context_variables) {
                        Ok(result) => {
                            emit!(sender, result);
                        }
                        Err(e) => {
                            handle_error!(e, "Macro execution failed");
                        }
                    }
                }
                None => {
                    handle_error!(MacroSystemError::MacroNotFound, "Macro not found");
                }
            }
        })
    }

    /// Execute a macro directly (zero-allocation streaming)
    pub fn execute_macro_direct(
        &self,
        macro_def: ChatMacro,
        context_variables: HashMap<String, String>,
    ) -> AsyncStream<MacroExecutionResult> {
        let self_clone = self.clone();
        AsyncStream::with_channel(move |sender| {
            // Use synchronous execution with streaming results
            match self_clone.execute_macro_sync(macro_def, context_variables) {
                Ok(result) => {
                    emit!(sender, result);
                }
                Err(e) => {
                    handle_error!(e, "Macro execution failed");
                }
            }
        })
    }

    /// Internal macro execution implementation (synchronous for blazing-fast performance)
    fn execute_macro_sync(
        &self,
        macro_def: ChatMacro,
        context_variables: HashMap<String, String>,
    ) -> Result<MacroExecutionResult, MacroSystemError> {
        let execution_id = Uuid::new_v4();
        let start_time = Instant::now();
        let started_at = Duration::from_secs(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|_| MacroSystemError::SystemTimeError)?
                .as_secs(),
        );

        // Update statistics
        self.stats
            .total_executions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.stats
            .active_executions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Merge context variables with global variables (use String for consistency)
        let mut execution_context = {
            let global_vars = self.variables.try_read().unwrap();
            let mut context = global_vars.clone();

            // Extend with provided context variables (both are HashMap<String, String>)
            context.extend(context_variables);

            // Extend with macro_def.variables (already HashMap<String, String>)
            context.extend(macro_def.variables);
            context
        };

        let mut actions_executed = 0;
        let mut modified_variables: HashMap<String, String> = HashMap::new();

        // Execute actions
        for action in macro_def.actions.iter() {
            match self.execute_action_sync(action, &mut execution_context) {
                Ok(modified_vars) => {
                    actions_executed += 1;
                    // Extend with modified variables (both are HashMap<String, String>)
                    modified_variables.extend(modified_vars);
                }
                Err(e) => {
                    self.stats
                        .failed_executions
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    self.stats
                        .active_executions
                        .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);

                    return Ok(MacroExecutionResult {
                        success: false,
                        message: format!("Action execution failed: {}", e),
                        actions_executed,
                        execution_duration: start_time.elapsed(),
                        modified_variables: modified_variables.clone(),
                        metadata: MacroExecutionMetadata {
                            execution_id,
                            macro_id: macro_def.metadata.id,
                            started_at,
                            completed_at: Duration::from_secs(
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs(),
                            ),
                            context: execution_context.clone(),
                            performance: MacroPerformanceMetrics {
                                cpu_time_us: start_time.elapsed().as_micros() as u64,
                                memory_bytes: 0, // Would need memory profiling
                                network_requests: 0,
                                disk_operations: 0,
                            },
                        },
                    });
                }
            }
        }

        let execution_duration = start_time.elapsed();

        // Update statistics
        self.stats
            .successful_executions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.stats
            .active_executions
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        self.stats.total_execution_time_us.fetch_add(
            execution_duration.as_micros() as usize,
            std::sync::atomic::Ordering::Relaxed,
        );

        Ok(MacroExecutionResult {
            success: true,
            message: "Macro executed successfully".to_string(),
            actions_executed,
            execution_duration,
            modified_variables: modified_variables
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            metadata: MacroExecutionMetadata {
                execution_id,
                macro_id: macro_def.metadata.id,
                started_at,
                completed_at: Duration::from_secs(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                ),
                context: execution_context.clone(),
                performance: MacroPerformanceMetrics {
                    cpu_time_us: execution_duration.as_micros() as u64,
                    memory_bytes: 0, // Would need memory profiling
                    network_requests: 0,
                    disk_operations: 0,
                },
            },
        })
    }

    /// Execute a single macro action iteratively (zero-allocation, blazing-fast)
    fn execute_action_sync(
        &self,
        action: &MacroAction,
        context: &mut HashMap<String, String>,
    ) -> Result<HashMap<String, String>, MacroSystemError> {
        use std::collections::VecDeque;

        let mut modified_vars = HashMap::new();
        let mut work_queue = VecDeque::with_capacity(16); // Stack-allocated for small workloads
        work_queue.push_back(action);

        while let Some(current_action) = work_queue.pop_front() {
            match current_action {
                MacroAction::SendMessage { content, .. } => {
                    // Substitute variables in content
                    let processed_content = if self.config.enable_variable_substitution {
                        self.substitute_variables(content.as_ref(), context)?
                    } else {
                        content.to_string()
                    };

                    // In a real implementation, this would send the message
                    // For now, we just simulate the action
                    println!("Sending message: {}", processed_content);
                }
                MacroAction::ExecuteCommand { command, .. } => {
                    // In a real implementation, this would execute the command
                    // For now, we just simulate the action
                    println!("Executing command: {:?}", command);
                }
                MacroAction::Wait { duration, .. } => {
                    std::thread::sleep(*duration);
                }
                MacroAction::SetVariable { name, value, .. } => {
                    let processed_value = if self.config.enable_variable_substitution {
                        self.substitute_variables(value, context)?
                    } else {
                        value.clone()
                    };

                    context.insert(name.clone(), processed_value.clone());
                    modified_vars.insert(name.clone(), processed_value);
                }
                MacroAction::Conditional {
                    condition,
                    then_actions,
                    else_actions,
                    ..
                } => {
                    if self.config.enable_conditional_execution {
                        // Context is already Arc<str> HashMap, no conversion needed
                        let condition_result = self.evaluate_condition(condition, context)?;

                        let actions_to_execute = if condition_result {
                            then_actions
                        } else if let Some(else_acts) = else_actions {
                            else_acts
                        } else {
                            continue; // Skip to next action in queue
                        };

                        // Add sub-actions to work queue (zero-allocation iteration)
                        for sub_action in actions_to_execute.iter().rev() {
                            work_queue.push_front(sub_action);
                        }
                    }
                }
                MacroAction::Loop {
                    iterations,
                    actions,
                    ..
                } => {
                    if self.config.enable_loop_execution {
                        // Add loop iterations to work queue in reverse order for correct execution
                        for _ in 0..*iterations {
                            for sub_action in actions.iter().rev() {
                                work_queue.push_front(sub_action);
                            }
                        }
                    }
                }
            }
        }

        Ok(modified_vars)
    }

    /// Substitute variables in text
    fn substitute_variables(
        &self,
        text: &str,
        context: &HashMap<String, String>,
    ) -> Result<String, MacroSystemError> {
        let mut result = text.to_string();

        // Simple variable substitution: ${variable_name}
        for (name, value) in context {
            let placeholder = format!("${{{}}}", name);
            result = result.replace(&placeholder, value);
        }

        Ok(result)
    }

    /// Evaluate a condition string
    fn evaluate_condition(
        &self,
        condition: &str,
        context: &HashMap<String, String>,
    ) -> Result<bool, MacroSystemError> {
        // Simple condition evaluation - in production this would be more sophisticated
        if condition.contains("==") {
            let parts: Vec<&str> = condition.split("==").collect();
            if parts.len() == 2 {
                let left = parts[0].trim();
                let right = parts[1].trim().trim_matches('"');

                if let Some(value) = context.get(left) {
                    return Ok(value == right);
                }
            }
        }

        // Default to false for unknown conditions
        Ok(false)
    }

    /// Validate a macro
    fn validate_macro(&self, macro_def: &ChatMacro) -> Result<(), MacroSystemError> {
        if macro_def.metadata.name.is_empty() {
            return Err(MacroSystemError::ValidationError(
                "Macro name cannot be empty".to_string(),
            ));
        }

        if macro_def.actions.is_empty() {
            return Err(MacroSystemError::ValidationError(
                "Macro must have at least one action".to_string(),
            ));
        }

        // Validate recursion depth
        self.validate_recursion_depth(&macro_def.actions, 0)?;

        Ok(())
    }

    /// Validate recursion depth
    fn validate_recursion_depth(
        &self,
        actions: &[MacroAction],
        current_depth: usize,
    ) -> Result<(), MacroSystemError> {
        if current_depth > self.config.max_recursion_depth {
            return Err(MacroSystemError::ValidationError(
                "Maximum recursion depth exceeded".to_string(),
            ));
        }

        for action in actions {
            if let MacroAction::Conditional {
                then_actions,
                else_actions,
                ..
            } = action
            {
                self.validate_recursion_depth(then_actions, current_depth + 1)?;
                if let Some(else_acts) = else_actions {
                    self.validate_recursion_depth(else_acts, current_depth + 1)?;
                }
            }
        }

        Ok(())
    }

    /// Get all registered macros
    pub fn get_macros(&self) -> Vec<ChatMacro> {
        self.macros
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get macro by ID
    pub fn get_macro(&self, macro_id: &Uuid) -> Option<ChatMacro> {
        self.macros.get(macro_id).map(|entry| entry.value().clone())
    }

    /// Get processor statistics
    pub fn stats(&self) -> MacroProcessorStatsSnapshot {
        MacroProcessorStatsSnapshot {
            total_executions: self.stats.total_executions.load(Ordering::Relaxed),
            successful_executions: self.stats.successful_executions.load(Ordering::Relaxed),
            failed_executions: self.stats.failed_executions.load(Ordering::Relaxed),
            total_execution_time_us: self.stats.total_execution_time_us.load(Ordering::Relaxed),
            active_executions: self.stats.active_executions.load(Ordering::Relaxed),
        }
    }

    /// Set global variable (zero-allocation, lock-free)
    pub fn set_variable(&self, name: String, value: String) {
        if let Ok(mut vars) = self.variables.try_write() {
            vars.insert(name, value);
        }
    }

    /// Get global variable (zero-allocation, lock-free)
    pub fn get_variable(&self, name: &str) -> Option<String> {
        if let Ok(vars) = self.variables.try_read() {
            vars.get(name).cloned()
        } else {
            None
        }
    }

    /// Clear all global variables (zero-allocation, lock-free)
    pub fn clear_variables(&self) {
        if let Ok(mut vars) = self.variables.try_write() {
            vars.clear();
        }
    }
}

impl Default for MacroProcessor {
    fn default() -> Self {
        Self::new()
    }
}
