//! Macro system for chat automation with lock-free data structures
//!
//! This module provides a comprehensive macro system for recording, storing,
//! and playing back chat interactions using zero-allocation patterns and
//! lock-free data structures for blazing-fast performance.

use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use crossbeam_queue::SegQueue;
use crossbeam_skiplist::SkipMap;
use atomic_counter::{AtomicCounter, ConsistentCounter};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::chat::commands::ChatCommand;
use crate::chat::formatting::MessageContent;

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
        command: ChatCommand,
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
    pub name: Arc<str>,
    pub description: Arc<str>,
    pub created_at: Duration,
    pub updated_at: Duration,
    pub version: u32,
    pub tags: Arc<[Arc<str>]>,
    pub author: Arc<str>,
    pub execution_count: u64,
    pub last_execution: Option<Duration>,
    pub average_duration: Duration,
    pub success_rate: f64,
    pub category: Arc<str>,
    pub is_private: bool,
}

/// Complete macro definition with actions and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMacro {
    pub metadata: MacroMetadata,
    pub actions: Arc<[MacroAction]>,
    pub variables: HashMap<Arc<str>, Arc<str>>,
    pub triggers: Arc<[Arc<str>]>,
    pub conditions: Arc<[Arc<str>]>,
    pub dependencies: Arc<[Arc<str>]>,
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
    pub name: Arc<str>,
    pub start_time: Instant,
    pub actions: SegQueue<MacroAction>,
    pub state: MacroRecordingState,
    pub variables: HashMap<Arc<str>, Arc<str>>,
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
    pub error: Option<Arc<str>>,
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
    pub total_executions: AtomicCounter,
    pub successful_executions: AtomicCounter,
    pub failed_executions: AtomicCounter,
    pub total_duration: parking_lot::Mutex<Duration>,
    pub average_duration: parking_lot::Mutex<Duration>,
    pub last_execution: parking_lot::Mutex<Option<Instant>>,
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
    pub async fn start_recording(&self, name: Arc<str>, description: Arc<str>) -> Result<Uuid, MacroSystemError> {
        let session_id = Uuid::new_v4();
        let macro_id = Uuid::new_v4();
        
        let metadata = MacroMetadata {
            id: macro_id,
            name: name.clone(),
            description,
            created_at: Duration::from_secs(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|_| MacroSystemError::SystemTimeError)?
                    .as_secs()
            ),
            updated_at: Duration::from_secs(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|_| MacroSystemError::SystemTimeError)?
                    .as_secs()
            ),
            version: 1,
            tags: Arc::new([]),
            author: Arc::from("system"),
            execution_count: 0,
            last_execution: None,
            average_duration: Duration::from_secs(0),
            success_rate: 0.0,
            category: Arc::from("user-defined"),
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

        let mut sessions = self.recording_sessions.write().await;
        sessions.insert(session_id, session);
        
        Ok(session_id)
    }

    /// Record a macro action
    pub async fn record_action(&self, session_id: Uuid, action: MacroAction) -> Result<(), MacroSystemError> {
        let sessions = self.recording_sessions.read().await;
        
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

    /// Stop recording and save the macro
    pub async fn stop_recording(&self, session_id: Uuid) -> Result<Uuid, MacroSystemError> {
        let mut sessions = self.recording_sessions.write().await;
        
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
                triggers: Arc::new([]),
                conditions: Arc::new([]),
                dependencies: Arc::new([]),
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
        self.macros.get(&macro_id).map(|entry| entry.value().clone())
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
                macro_def.metadata.name.to_lowercase().contains(&query_lower) ||
                macro_def.metadata.description.to_lowercase().contains(&query_lower) ||
                macro_def.metadata.tags.iter().any(|tag| tag.to_lowercase().contains(&query_lower))
            })
            .map(|entry| entry.value().metadata.clone())
            .collect()
    }

    /// Start macro playback
    pub async fn start_playback(&self, macro_id: Uuid, variables: HashMap<Arc<str>, Arc<str>>) -> Result<Uuid, MacroSystemError> {
        let macro_def = self.get_macro(macro_id)
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
        
        let mut sessions = self.playback_sessions.write().await;
        sessions.insert(session_id, session);
        
        self.execution_counter.inc();
        
        Ok(session_id)
    }

    /// Execute the next action in a playback session
    pub async fn execute_next_action(&self, session_id: Uuid) -> Result<MacroPlaybackResult, MacroSystemError> {
        let mut sessions = self.playback_sessions.write().await;
        
        if let Some(session) = sessions.get_mut(&session_id) {
            if session.state != MacroPlaybackState::Playing {
                return Ok(MacroPlaybackResult::SessionNotActive);
            }
            
            let macro_def = self.get_macro(session.macro_id)
                .ok_or(MacroSystemError::MacroNotFound)?;
            
            if session.current_action >= macro_def.actions.len() {
                session.state = MacroPlaybackState::Completed;
                return Ok(MacroPlaybackResult::Completed);
            }
            
            let action = &macro_def.actions[session.current_action];
            let result = self.execute_action(action, &mut session.context).await?;
            
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
                    tokio::time::sleep(duration).await;
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
    async fn execute_action(&self, action: &MacroAction, context: &mut MacroExecutionContext) -> Result<ActionExecutionResult, MacroSystemError> {
        match action {
            MacroAction::SendMessage { content, message_type, .. } => {
                let resolved_content = self.resolve_variables(content, &context.variables);
                // In a real implementation, this would send the message to the chat system
                println!("Sending message: {} (type: {})", resolved_content, message_type);
                Ok(ActionExecutionResult::Success)
            }
            MacroAction::ExecuteCommand { command, .. } => {
                // In a real implementation, this would execute the command
                println!("Executing command: {:?}", command);
                Ok(ActionExecutionResult::Success)
            }
            MacroAction::Wait { duration, .. } => {
                Ok(ActionExecutionResult::Wait(*duration))
            }
            MacroAction::SetVariable { name, value, .. } => {
                let resolved_value = self.resolve_variables(value, &context.variables);
                context.variables.insert(name.clone(), resolved_value.into());
                Ok(ActionExecutionResult::Success)
            }
            MacroAction::Conditional { condition, then_actions, else_actions, .. } => {
                let condition_result = self.evaluate_condition(condition, &context.variables);
                
                let actions_to_execute = if condition_result {
                    then_actions
                } else if let Some(else_actions) = else_actions {
                    else_actions
                } else {
                    return Ok(ActionExecutionResult::Success);
                };
                
                // Execute conditional actions
                for action in actions_to_execute.iter() {
                    let result = self.execute_action(action, context).await?;
                    if let ActionExecutionResult::Error(error) = result {
                        return Ok(ActionExecutionResult::Error(error));
                    }
                }
                
                Ok(ActionExecutionResult::Success)
            }
            MacroAction::Loop { iterations, actions, .. } => {
                let loop_context = LoopContext {
                    iteration: 0,
                    max_iterations: *iterations,
                    start_action: 0,
                    end_action: actions.len(),
                };
                
                context.loop_stack.push(loop_context);
                
                for _ in 0..*iterations {
                    for action in actions.iter() {
                        let result = self.execute_action(action, context).await?;
                        if let ActionExecutionResult::Error(error) = result {
                            context.loop_stack.pop();
                            return Ok(ActionExecutionResult::Error(error));
                        }
                    }
                }
                
                context.loop_stack.pop();
                Ok(ActionExecutionResult::Success)
            }
        }
    }

    /// Resolve variables in a string
    fn resolve_variables(&self, content: &str, variables: &HashMap<Arc<str>, Arc<str>>) -> String {
        let mut result = content.to_string();
        
        for (key, value) in variables {
            let placeholder = format!("{{{}}}", key);
            result = result.replace(&placeholder, value);
        }
        
        result
    }

    /// Evaluate a condition string
    fn evaluate_condition(&self, condition: &str, variables: &HashMap<Arc<str>, Arc<str>>) -> bool {
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
        self.execution_stats.get(&macro_id).map(|entry| (*entry.value()).clone())
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
    Error(Arc<str>),
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
}

impl Default for MacroSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating macros programmatically
pub struct MacroBuilder {
    name: Option<Arc<str>>,
    description: Option<Arc<str>>,
    actions: Vec<MacroAction>,
    variables: HashMap<Arc<str>, Arc<str>>,
    triggers: Vec<Arc<str>>,
    conditions: Vec<Arc<str>>,
    dependencies: Vec<Arc<str>>,
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
    pub fn name(mut self, name: Arc<str>) -> Self {
        self.name = Some(name);
        self
    }

    /// Set the macro description
    pub fn description(mut self, description: Arc<str>) -> Self {
        self.description = Some(description);
        self
    }

    /// Add an action to the macro
    pub fn add_action(mut self, action: MacroAction) -> Self {
        self.actions.push(action);
        self
    }

    /// Add a variable to the macro
    pub fn add_variable(mut self, name: Arc<str>, value: Arc<str>) -> Self {
        self.variables.insert(name, value);
        self
    }

    /// Set execution configuration
    pub fn execution_config(mut self, config: MacroExecutionConfig) -> Self {
        self.execution_config = config;
        self
    }

    /// Build the macro
    pub fn build(self) -> Result<ChatMacro, MacroSystemError> {
        let name = self.name.ok_or_else(|| MacroSystemError::ExecutionError("Name is required".to_string()))?;
        let description = self.description.unwrap_or_else(|| Arc::from(""));
        
        let metadata = MacroMetadata {
            id: Uuid::new_v4(),
            name,
            description,
            created_at: Duration::from_secs(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|_| MacroSystemError::SystemTimeError)?
                    .as_secs()
            ),
            updated_at: Duration::from_secs(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|_| MacroSystemError::SystemTimeError)?
                    .as_secs()
            ),
            version: 1,
            tags: Arc::new([]),
            author: Arc::from("builder"),
            execution_count: 0,
            last_execution: None,
            average_duration: Duration::from_secs(0),
            success_rate: 0.0,
            category: Arc::from("programmatic"),
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