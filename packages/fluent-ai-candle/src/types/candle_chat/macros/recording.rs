//! Macro recording functionality
//!
//! This module handles the recording of user actions into macros,
//! providing a high-performance, lock-free recording system.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use atomic_counter::ConsistentCounter;
use crossbeam_skiplist::SkipMap;
use tokio::sync::RwLock;
use uuid::Uuid;

use super::types::{
    ChatMacro, MacroAction, MacroExecutionConfig, MacroMetadata, MacroRecordingSession,
    MacroRecordingState, MacroSystemError,
};

/// Recording functionality for the macro system
pub struct MacroRecorder {
    /// Active recording sessions
    recording_sessions: RwLock<HashMap<Uuid, MacroRecordingSession>>,
    /// Macro storage for completed recordings
    macros: SkipMap<Uuid, ChatMacro>,
    /// Global macro counter
    macro_counter: ConsistentCounter,
}

impl MacroRecorder {
    /// Create new macro recorder
    pub fn new(macros: SkipMap<Uuid, ChatMacro>) -> Self {
        Self {
            recording_sessions: RwLock::new(HashMap::new()),
            macros,
            macro_counter: ConsistentCounter::new(0),
        }
    }

    /// Start recording a new macro
    pub async fn start_recording(
        &self,
        name: String,
        description: String,
    ) -> Result<Uuid, MacroSystemError> {
        let session_id = Uuid::new_v4();
        let macro_id = Uuid::new_v4();

        let metadata = MacroMetadata {
            id: macro_id,
            name: name.clone(),
            description,
            created_at: Duration::from_secs(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|_| MacroSystemError::SerializationError(
                        serde_json::Error::custom("System time error")
                    ))?
                    .as_secs(),
            ),
            updated_at: Duration::from_secs(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|_| MacroSystemError::SerializationError(
                        serde_json::Error::custom("System time error")
                    ))?
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
            actions: crossbeam_queue::SegQueue::new(),
            state: MacroRecordingState::Recording,
            variables: HashMap::new(),
            metadata,
        };

        let mut sessions = self.recording_sessions.write().await;
        sessions.insert(session_id, session);

        Ok(session_id)
    }

    /// Record a macro action
    pub async fn record_action(
        &self,
        session_id: Uuid,
        action: MacroAction,
    ) -> Result<(), MacroSystemError> {
        let sessions = self.recording_sessions.read().await;

        if let Some(session) = sessions.get(&session_id) {
            if session.state == MacroRecordingState::Recording {
                session.actions.push(action);
                Ok(())
            } else {
                Err(MacroSystemError::InvalidAction(
                    "Recording session is not active".to_string()
                ))
            }
        } else {
            Err(MacroSystemError::SessionNotFound(session_id))
        }
    }

    /// Pause recording session
    pub async fn pause_recording(&self, session_id: Uuid) -> Result<(), MacroSystemError> {
        let mut sessions = self.recording_sessions.write().await;

        if let Some(session) = sessions.get_mut(&session_id) {
            match session.state {
                MacroRecordingState::Recording => {
                    session.state = MacroRecordingState::Paused;
                    Ok(())
                }
                _ => Err(MacroSystemError::InvalidAction(
                    "Cannot pause inactive recording session".to_string()
                ))
            }
        } else {
            Err(MacroSystemError::SessionNotFound(session_id))
        }
    }

    /// Resume recording session
    pub async fn resume_recording(&self, session_id: Uuid) -> Result<(), MacroSystemError> {
        let mut sessions = self.recording_sessions.write().await;

        if let Some(session) = sessions.get_mut(&session_id) {
            match session.state {
                MacroRecordingState::Paused => {
                    session.state = MacroRecordingState::Recording;
                    Ok(())
                }
                _ => Err(MacroSystemError::InvalidAction(
                    "Cannot resume non-paused recording session".to_string()
                ))
            }
        } else {
            Err(MacroSystemError::SessionNotFound(session_id))
        }
    }

    /// Stop recording and save the macro (zero-allocation, lock-free)
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
                actions,
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
            Err(MacroSystemError::SessionNotFound(session_id))
        }
    }

    /// Get recording session status
    pub async fn get_recording_status(&self, session_id: Uuid) -> Result<MacroRecordingState, MacroSystemError> {
        let sessions = self.recording_sessions.read().await;
        
        if let Some(session) = sessions.get(&session_id) {
            Ok(session.state)
        } else {
            Err(MacroSystemError::SessionNotFound(session_id))
        }
    }

    /// Get all active recording sessions
    pub async fn get_active_recordings(&self) -> Vec<Uuid> {
        let sessions = self.recording_sessions.read().await;
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
        let sessions = self.recording_sessions.read().await;
        
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