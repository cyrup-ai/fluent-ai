//! Macro recording functionality
//!
//! This module handles the recording of user actions into macros,
//! providing a high-performance, lock-free recording system.

use std::time::{Duration, Instant};

use atomic_counter::ConsistentCounter;
use crossbeam_skiplist::SkipMap;
use uuid::Uuid;

use super::types::{
    ChatMacro, MacroAction, MacroExecutionConfig, MacroMetadata, MacroRecordingSession,
    MacroRecordingState, MacroSystemError,
};

/// Recording functionality for the macro system
pub struct MacroRecorder {
    /// Active recording sessions - lock-free
    recording_sessions: SkipMap<Uuid, MacroRecordingSession>,
    /// Macro storage for completed recordings
    macros: SkipMap<Uuid, ChatMacro>,
    /// Global macro counter
    macro_counter: ConsistentCounter,
}

impl MacroRecorder {
    /// Create new macro recorder
    pub fn new(macros: SkipMap<Uuid, ChatMacro>) -> Self {
        Self {
            recording_sessions: SkipMap::new(),
            macros,
            macro_counter: ConsistentCounter::new(0),
        }
    }

    /// Start recording a new macro
    pub fn start_recording(
        &self,
        name: String,
        description: String,
    ) -> AsyncStream<Uuid> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        use std::collections::HashMap;
        
        let recording_sessions = self.recording_sessions.clone();
        
        AsyncStream::with_channel(move |sender| {
            let session_id = Uuid::new_v4();
            let macro_id = Uuid::new_v4();

            let current_time = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                Ok(duration) => Duration::from_secs(duration.as_secs()),
                Err(_) => {
                    handle_error!(
                        MacroSystemError::SerializationError(
                            serde_json::Error::custom("System time error")
                        ),
                        "failed to get system time"
                    );
                }
            };

            let metadata = MacroMetadata {
                id: macro_id,
                name: name.clone(),
                description,
                created_at: current_time,
                updated_at: current_time,
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
                actions: crossbeam_queue::SegQueue::new(),
                state: MacroRecordingState::Recording,
                variables: HashMap::new(),
                metadata,
            };

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
                let mut session = session_entry.value().clone();
                match session.state {
                    MacroRecordingState::Recording => {
                        session.state = MacroRecordingState::Paused;
                        // Replace the session with updated state (lock-free update)
                        recording_sessions.insert(session_id, session);
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
                let mut session = session_entry.value().clone();
                match session.state {
                    MacroRecordingState::Paused => {
                        session.state = MacroRecordingState::Recording;
                        // Replace the session with updated state (lock-free update)
                        recording_sessions.insert(session_id, session);
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
        
        let recording_sessions = self.recording_sessions.clone();
        let macros = self.macros.clone();
        let macro_counter = self.macro_counter.clone();
        
        AsyncStream::with_channel(move |sender| {
            if let Some(session_entry) = recording_sessions.remove(&session_id) {
                let mut session = session_entry.1; // Extract value from removed entry
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
                    variables: session.variables,
                    triggers: vec![],
                    conditions: vec![],
                    dependencies: vec![],
                    execution_config: MacroExecutionConfig::default(),
                };

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
    pub async fn get_recording_status(&self, session_id: Uuid) -> Result<MacroRecordingState, MacroSystemError> {
        let sessions = self.recording_sessions.read().expect("lock poisoned");
        
        if let Some(session) = sessions.get(&session_id) {
            Ok(session.state)
        } else {
            Err(MacroSystemError::SessionNotFound(session_id))
        }
    }

    /// Get all active recording sessions
    pub async fn get_active_recordings(&self) -> Vec<Uuid> {
        let sessions = self.recording_sessions.read().expect("lock poisoned");
        sessions.keys().copied().collect()
    }

    /// Cancel recording session
    pub async fn cancel_recording(&self, session_id: Uuid) -> Result<(), MacroSystemError> {
        let mut sessions = self.recording_sessions.write().await;
        
        if sessions.remove(&session_id).is_some() {
            Ok(())
        } else {
            Err(MacroSystemError::SessionNotFound(session_id))
        }
    }

    /// Get recording session info
    pub async fn get_recording_info(&self, session_id: Uuid) -> Result<RecordingInfo, MacroSystemError> {
        let sessions = self.recording_sessions.read().expect("lock poisoned");
        
        if let Some(session) = sessions.get(&session_id) {
            Ok(RecordingInfo {
                id: session.id,
                name: session.name.clone(),
                start_time: session.start_time,
                state: session.state,
                action_count: session.actions.len(),
                duration: session.start_time.elapsed(),
            })
        } else {
            Err(MacroSystemError::SessionNotFound(session_id))
        }
    }

    /// Add variable to recording session
    pub async fn add_recording_variable(
        &self,
        session_id: Uuid,
        name: String,
        value: String,
    ) -> Result<(), MacroSystemError> {
        let mut sessions = self.recording_sessions.write().await;
        
        if let Some(session) = sessions.get_mut(&session_id) {
            session.variables.insert(name, value);
            Ok(())
        } else {
            Err(MacroSystemError::SessionNotFound(session_id))
        }
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
    pub duration: Duration,
}