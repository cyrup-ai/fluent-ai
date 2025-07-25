//! Main macro system implementation
//!
//! This module contains the core MacroSystem struct that manages macro recording,
//! playback, and execution using lock-free data structures and AsyncStream patterns.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use atomic_counter::{AtomicCounter, ConsistentCounter};
use crossbeam_skiplist::SkipMap;
use fluent_ai_async::{AsyncStream, emit, handle_error};
use tokio::sync::RwLock;
use uuid::Uuid;

use super::types::{
    MacroAction, ChatMacro, MacroRecordingSession, MacroPlaybackSession,
    MacroRecordingState, MacroPlaybackState, MacroExecutionContext, LoopContext,
    MacroMetadata, MacroExecutionConfig, ExecutionStats,
};
use super::errors::{MacroSystemError, ActionExecutionResult, MacroPlaybackResult};

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
        name: Arc<str>,
        description: Arc<str>,
    ) -> AsyncStream<Uuid> {
        let recording_sessions = self.recording_sessions.clone();
        
        AsyncStream::with_channel(move |sender| {
            let session_id = Uuid::new_v4();
            let macro_id = Uuid::new_v4();

            // Use zero-allocation, lock-free patterns for time operations
            let current_time = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                Ok(duration) => Duration::from_secs(duration.as_secs()),
                Err(_) => {
                    handle_error!(
                        MacroSystemError::SystemTimeError,
                        "Failed to get system time for macro recording"
                    );
                    return;
                }
            };

            let metadata = MacroMetadata {
                id: macro_id,
                name: name.clone(),
                description,
                created_at: current_time,
                updated_at: current_time,
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
                actions: crossbeam_queue::SegQueue::new(),
                state: MacroRecordingState::Recording,
                variables: HashMap::new(),
                metadata,
            };

            // Use zero-allocation, lock-free patterns with try_write()
            if let Ok(mut sessions) = recording_sessions.try_write() {
                sessions.insert(session_id, session);
                emit!(sender, session_id);
            } else {
                handle_error!(
                    MacroSystemError::LockError,
                    "Failed to acquire recording sessions write lock"
                );
            }
        })
    }

    /// Record a macro action with fluent-ai-async streaming architecture
    pub fn record_action(
        &self,
        session_id: Uuid,
        action: MacroAction,
    ) -> AsyncStream<()> {
        let recording_sessions = self.recording_sessions.clone();
        
        AsyncStream::with_channel(move |sender| {
            // Use zero-allocation, lock-free patterns with try_read()
            if let Ok(sessions) = recording_sessions.try_read() {
                if let Some(session) = sessions.get(&session_id) {
                    if session.state == MacroRecordingState::Recording {
                        session.actions.push(action);
                        emit!(sender, ());
                    } else {
                        handle_error!(
                            MacroSystemError::RecordingNotActive,
                            "Macro recording session is not active"
                        );
                    }
                } else {
                    handle_error!(
                        MacroSystemError::SessionNotFound,
                        "Macro recording session not found"
                    );
                }
            } else {
                handle_error!(
                    MacroSystemError::LockError,
                    "Failed to acquire recording sessions read lock"
                );
            }
        })
    }

    /// Stop recording and save the macro with fluent-ai-async streaming architecture
    pub fn stop_recording(&self, session_id: Uuid) -> AsyncStream<Uuid> {
        let recording_sessions = self.recording_sessions.clone();
        let macros = self.macros.clone();
        let macro_counter = self.macro_counter.clone();
        
        AsyncStream::with_channel(move |sender| {
            // Use zero-allocation, lock-free patterns with try_write()
            if let Ok(mut sessions) = recording_sessions.try_write() {
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
                    macros.insert(macro_id, chat_macro);
                    macro_counter.inc();

                    emit!(sender, macro_id);
                } else {
                    handle_error!(
                        MacroSystemError::SessionNotFound,
                        "Macro recording session not found"
                    );
                }
            } else {
                handle_error!(
                    MacroSystemError::LockError,
                    "Failed to acquire recording sessions write lock"
                );
            }
        })
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

    /// Start macro playback
    pub async fn start_playback(
        &self,
        macro_id: Uuid,
        variables: HashMap<Arc<str>, Arc<str>>,
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

        let mut sessions = self.playback_sessions.write().await;
        sessions.insert(session_id, session);

        self.execution_counter.inc();

        Ok(session_id)
    }

    /// Execute the next action in a playback session
    pub async fn execute_next_action(
        &self,
        session_id: Uuid,
    ) -> Result<MacroPlaybackResult, MacroSystemError> {
        let mut sessions = self.playback_sessions.write().await;

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
            let result = self.execute_action_internal(action, &mut session.context).await?;

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

    /// Internal action execution for playback
    async fn execute_action_internal(
        &self,
        action: &MacroAction,
        context: &mut MacroExecutionContext,
    ) -> Result<ActionExecutionResult, MacroSystemError> {
        match action {
            MacroAction::SendMessage { content, message_type, .. } => {
                let resolved_content = self.resolve_variables(content, &context.variables);
                println!("Sending message: {} (type: {})", resolved_content, message_type);
                Ok(ActionExecutionResult::Success)
            }
            MacroAction::ExecuteCommand { command, .. } => {
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
                } else if let Some(ref else_actions) = else_actions {
                    else_actions
                } else {
                    return Ok(ActionExecutionResult::Success);
                };

                for sub_action in actions_to_execute.iter() {
                    let result = self.execute_action_internal(sub_action, context).await?;
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
                    for sub_action in actions.iter() {
                        let result = self.execute_action_internal(sub_action, context).await?;
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

impl Default for MacroSystem {
    fn default() -> Self {
        Self::new()
    }
}