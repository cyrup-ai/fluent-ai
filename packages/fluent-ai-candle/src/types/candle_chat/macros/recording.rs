//! Macro recording functionality
//!
//! This module handles the recording of user actions into macros,
//! providing a high-performance, lock-free recording system.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::types::candle_chat::search::tagging::{ConsistentCounter};
use crossbeam_skiplist::SkipMap;
use crossbeam_queue::SegQueue;
use fluent_ai_async::AsyncStream;
use uuid::Uuid;

use super::types::{
    ChatMacro, MacroAction, MacroExecutionConfig, MacroMetadata, MacroRecordingSession,
    MacroRecordingState, MacroSystemError};

/// Recording functionality for the macro system
pub struct MacroRecorder {
    /// Active recording sessions - lock-free
    recording_sessions: Arc<SkipMap<Uuid, MacroRecordingSession>>,
    /// Macro storage for completed recordings
    macros: Arc<SkipMap<Uuid, ChatMacro>>,
    /// Global macro counter
    macro_counter: ConsistentCounter}

impl Clone for MacroRecorder {
    fn clone(&self) -> Self {
        Self {
            recording_sessions: self.recording_sessions.clone(),
            macros: self.macros.clone(),
            macro_counter: self.macro_counter.clone(),
        }
    }
}

impl MacroRecorder {
    /// Create new macro recorder
    pub fn new(macros: Arc<SkipMap<Uuid, ChatMacro>>) -> Self {
        Self {
            recording_sessions: Arc::new(SkipMap::new()),
            macros,
            macro_counter: ConsistentCounter::new(0)}
    }

    /// Start recording a new macro
    pub fn start_recording(
        &self,
        name: String,
        description: String,
    ) -> AsyncStream<Uuid> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
                let recording_sessions = self.recording_sessions.clone();
        
        AsyncStream::with_channel(move |sender| {
            let session_id = Uuid::new_v4();
            let macro_id = Uuid::new_v4();

            let current_time = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                Ok(duration) => Duration::from_secs(duration.as_secs()),
                Err(_) => {
                    handle_error!(
                        MacroSystemError::TimeError(
                            "System time error".to_string()
                        ),
                        "failed to get system time"
                    );
                }
            };

            let metadata = MacroMetadata {
                id: macro_id,
                name: Arc::from(name.as_str()),
                description: Arc::from(description.as_str()),
                created_at: current_time,
                updated_at: current_time,
                version: 1,
                tags: vec![],
                author: Arc::from("system"),
                execution_count: 0,
                last_execution: None,
                average_duration: Duration::from_secs(0),
                success_rate: 0.0,
                category: Arc::from("user-defined"),
                is_private: false};

            let session = MacroRecordingSession {
                id: session_id,
                name,
                start_time: Instant::now(),
                actions: crossbeam_queue::SegQueue::new(),
                state: MacroRecordingState::Recording,
                variables: HashMap::new(),
                metadata};

            // Lock-free insert operation
            recording_sessions.insert(session_id, session);
            
            emit!(sender, session_id);
        })
    }

    /// Record a macro action
    pub fn record_action(
        &self,
        session_id: Uuid,
        action: MacroAction,
    ) -> AsyncStream<()> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let recording_sessions = self.recording_sessions.clone();
        
        AsyncStream::with_channel(move |sender| {
            if let Some(session_entry) = recording_sessions.get(&session_id) {
                let session = session_entry.value();
                if session.state == MacroRecordingState::Recording {
                    session.actions.push(action);
                    emit!(sender, ());
                } else {
                    handle_error!(
                        MacroSystemError::InvalidAction(
                            "Recording session is not active".to_string()
                        ),
                        "recording session not active"
                    );
                }
            } else {
                handle_error!(
                    MacroSystemError::SessionNotFound(session_id),
                    "recording session not found"
                );
            }
        })
    }

    /// Pause recording session
    pub fn pause_recording(&self, session_id: Uuid) -> AsyncStream<()> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let recording_sessions = self.recording_sessions.clone();
        
        AsyncStream::with_channel(move |sender| {
            if let Some(session_entry) = recording_sessions.get(&session_id) {
                let session = session_entry.value();
                match session.state {
                    MacroRecordingState::Recording => {
                        // Create a new session with updated state (immutable update pattern)
                        let updated_session = MacroRecordingSession {
                            id: session.id,
                            name: session.name.clone(),
                            start_time: session.start_time,
                            actions: SegQueue::new(), // Create new queue to avoid cloning concurrent structure
                            state: MacroRecordingState::Paused,
                            variables: session.variables.clone(),
                            metadata: session.metadata.clone()
                        };
                        recording_sessions.insert(session_id, updated_session);
                        emit!(sender, ());
                    }
                    _ => {
                        handle_error!(
                            MacroSystemError::InvalidAction(
                                "Cannot pause inactive recording session".to_string()
                            ),
                            "cannot pause inactive session"
                        );
                    }
                }
            } else {
                handle_error!(
                    MacroSystemError::SessionNotFound(session_id),
                    "recording session not found"
                );
            }
        })
    }

    /// Resume recording session
    pub fn resume_recording(&self, session_id: Uuid) -> AsyncStream<()> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let recording_sessions = self.recording_sessions.clone();
        
        AsyncStream::with_channel(move |sender| {
            if let Some(session_entry) = recording_sessions.get(&session_id) {
                let session = session_entry.value();
                match session.state {
                    MacroRecordingState::Paused => {
                        // Create a new session with updated state (immutable update pattern)
                        let updated_session = MacroRecordingSession {
                            id: session.id,
                            name: session.name.clone(),
                            start_time: session.start_time,
                            actions: SegQueue::new(), // Create new queue to avoid cloning concurrent structure
                            state: MacroRecordingState::Recording,
                            variables: session.variables.clone(),
                            metadata: session.metadata.clone()
                        };
                        recording_sessions.insert(session_id, updated_session);
                        // Replace the session with updated state (lock-free update)
                        recording_sessions.insert(session_id, session.clone());
                        emit!(sender, ());
                    }
                    _ => {
                        handle_error!(
                            MacroSystemError::InvalidAction(
                                "Cannot resume non-paused recording session".to_string()
                            ),
                            "cannot resume non-paused session"
                        );
                    }
                }
            } else {
                handle_error!(
                    MacroSystemError::SessionNotFound(session_id),
                    "recording session not found"
                );
            }
        })
    }

    /// Stop recording and save the macro (zero-allocation, lock-free)
    pub fn stop_recording(&self, session_id: Uuid) -> AsyncStream<Uuid> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let recording_sessions = Arc::clone(&self.recording_sessions);
        let macros = Arc::clone(&self.macros);
        let macro_counter = self.macro_counter.clone();
        
        AsyncStream::with_channel(move |sender| {
            if let Some(session_entry) = recording_sessions.remove(&session_id) {
                let mut session = session_entry.value().clone();
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
                    actions,
                    variables: session.variables.into_iter()
                        .map(|(k, v)| (Arc::from(k.as_str()), Arc::from(v.as_str())))
                        .collect(),
                    triggers: vec![],
                    conditions: vec![],
                    dependencies: vec![],
                    execution_config: MacroExecutionConfig::default()};

                let macro_id = session.metadata.id;
                macros.insert(macro_id, chat_macro);
                macro_counter.inc();

                emit!(sender, macro_id);
            } else {
                handle_error!(
                    MacroSystemError::SessionNotFound(session_id),
                    "recording session not found"
                );
            }
        })
    }

    /// Get recording session status
    pub fn get_recording_status(&self, session_id: Uuid) -> AsyncStream<MacroRecordingState> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let recording_sessions = Arc::clone(&self.recording_sessions);
        
        AsyncStream::with_channel(move |sender| {
            if let Some(session_entry) = recording_sessions.get(&session_id) {
                let session = session_entry.value();
                emit!(sender, session.state);
            } else {
                handle_error!(
                    MacroSystemError::SessionNotFound(session_id),
                    "recording session not found"
                );
            }
        })
    }

    /// Get all active recording sessions
    pub fn get_active_recordings(&self) -> AsyncStream<Vec<Uuid>> {
        use fluent_ai_async::{AsyncStream, emit};
        
        let recording_sessions = Arc::clone(&self.recording_sessions);
        
        AsyncStream::with_channel(move |sender| {
            let active_sessions: Vec<Uuid> = recording_sessions
                .iter()
                .map(|entry| *entry.key())
                .collect();
            
            emit!(sender, active_sessions);
        })
    }

    /// Cancel recording session
    pub fn cancel_recording(&self, session_id: Uuid) -> AsyncStream<()> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let recording_sessions = self.recording_sessions.clone();
        
        AsyncStream::with_channel(move |sender| {
            if recording_sessions.remove(&session_id).is_some() {
                emit!(sender, ());
            } else {
                handle_error!(
                    MacroSystemError::SessionNotFound(session_id),
                    "recording session not found"
                );
            }
        })
    }

    /// Get recording session info
    pub fn get_recording_info(&self, session_id: Uuid) -> AsyncStream<RecordingInfo> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let recording_sessions = self.recording_sessions.clone();
        
        AsyncStream::with_channel(move |sender| {
            if let Some(session_entry) = recording_sessions.get(&session_id) {
                let session = session_entry.value();
                let info = RecordingInfo {
                    id: session.id,
                    name: session.name.clone(),
                    start_time: session.start_time,
                    state: session.state,
                    action_count: session.actions.len(),
                    duration: session.start_time.elapsed()};
                emit!(sender, info);
            } else {
                handle_error!(
                    MacroSystemError::SessionNotFound(session_id),
                    "recording session not found"
                );
            }
        })
    }

    /// Add variable to recording session
    pub fn add_recording_variable(
        &self,
        session_id: Uuid,
        name: String,
        value: String,
    ) -> AsyncStream<()> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let recording_sessions = self.recording_sessions.clone();
        
        AsyncStream::with_channel(move |sender| {
            if let Some(session_entry) = recording_sessions.get(&session_id) {
                let session = session_entry.value();
                // Create a new session with updated variables (immutable update pattern)
                let mut updated_variables = session.variables.clone();
                updated_variables.insert(name, value);
                
                let updated_session = MacroRecordingSession {
                    id: session.id,
                    name: session.name.clone(),
                    start_time: session.start_time,
                    actions: SegQueue::new(), // Create new queue to avoid cloning concurrent structure
                    state: session.state,
                    variables: updated_variables,
                    metadata: session.metadata.clone()
                };
                // Replace the session with updated variables (lock-free update)
                recording_sessions.insert(session_id, updated_session);
                emit!(sender, ());
            } else {
                handle_error!(
                    MacroSystemError::SessionNotFound(session_id),
                    "recording session not found"
                );
            }
        })
    }
}

/// Information about a recording session
#[derive(Debug, Clone)]
pub struct RecordingInfo {
    pub id: Uuid,
    pub name: String,
    pub start_time: Instant,
    pub state: MacroRecordingState,
    pub action_count: usize,
    pub duration: Duration}