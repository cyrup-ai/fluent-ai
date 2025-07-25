//! Macro system for chat automation
//!
//! This module provides a comprehensive macro system for recording, storing,
//! and playing back chat interactions using zero-allocation patterns and
//! lock-free data structures for blazing-fast performance.

pub mod types;
pub mod recording;
pub mod playback;
pub mod variables;
pub mod storage;

// Re-export commonly used types
pub use types::{
    MacroAction, MacroRecordingState, MacroPlaybackState, MacroExecutionContext,
    LoopContext, MacroMetadata, ChatMacro, MacroExecutionConfig, ResourceLimits,
    MacroRecordingSession, MacroPlaybackSession, ExecutionStats, MacroSystemError,
    MacroResult, ActionExecutionResult, MacroPlaybackResult,
};

pub use recording::{MacroRecorder, RecordingInfo};
pub use playback::{MacroPlayer, PlaybackInfo};
pub use variables::{
    VariableManager, resolve_variables, resolve_variables_static,
    evaluate_condition, evaluate_condition_static,
};
pub use storage::{MacroStorage, StorageStats, MacroExport};

use std::collections::HashMap;
use std::sync::Arc;

use atomic_counter::ConsistentCounter;
use crossbeam_skiplist::SkipMap;
use std::sync::RwLock;
use uuid::Uuid;

/// High-performance macro system with lock-free operations
pub struct MacroSystem {
    /// Macro storage
    storage: MacroStorage,
    /// Macro recorder
    recorder: MacroRecorder,
    /// Macro player
    player: MacroPlayer,
    /// Variable manager
    variables: RwLock<VariableManager>,
    /// Execution counter
    execution_counter: ConsistentCounter,
}

impl MacroSystem {
    /// Create new macro system
    pub fn new() -> Self {
        let storage = MacroStorage::new();
        let macros = storage.macros().clone();
        
        let recorder = MacroRecorder::new(macros.clone());
        let player = MacroPlayer::new(macros);
        
        Self {
            storage,
            recorder,
            player,
            variables: RwLock::new(VariableManager::new()),
            execution_counter: ConsistentCounter::new(0),
        }
    }

    /// Get storage reference
    pub fn storage(&self) -> &MacroStorage {
        &self.storage
    }

    /// Get recorder reference
    pub fn recorder(&self) -> &MacroRecorder {
        &self.recorder
    }

    /// Get player reference
    pub fn player(&self) -> &MacroPlayer {
        &self.player
    }

    /// Start recording a new macro
    pub fn start_recording(&self, name: String, description: String) -> fluent_ai_async::AsyncStream<Uuid> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let recorder = self.recorder.clone();
        AsyncStream::with_channel(move |sender| {
            match recorder.start_recording_sync(name, description) {
                Ok(uuid) => emit!(sender, uuid),
                Err(e) => handle_error!(e, "Failed to start macro recording"),
            }
        })
    }

    /// Record a macro action
    pub fn record_action(&self, session_id: Uuid, action: MacroAction) -> fluent_ai_async::AsyncStream<()> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let recorder = self.recorder.clone();
        AsyncStream::with_channel(move |sender| {
            match recorder.record_action_sync(session_id, action) {
                Ok(()) => emit!(sender, ()),
                Err(e) => handle_error!(e, "Failed to record macro action"),
            }
        })
    }

    /// Stop recording and save the macro
    pub fn stop_recording(&self, session_id: Uuid) -> fluent_ai_async::AsyncStream<Uuid> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let recorder = self.recorder.clone();
        AsyncStream::with_channel(move |sender| {
            match recorder.stop_recording_sync(session_id) {
                Ok(uuid) => emit!(sender, uuid),
                Err(e) => handle_error!(e, "Failed to stop macro recording"),
            }
        })
    }

    /// Start macro playback
    pub fn start_playback(&self, macro_id: Uuid, variables: HashMap<String, String>) -> fluent_ai_async::AsyncStream<Uuid> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        self.execution_counter.inc();
        let player = self.player.clone();
        AsyncStream::with_channel(move |sender| {
            match player.start_playback_sync(macro_id, variables) {
                Ok(uuid) => emit!(sender, uuid),
                Err(e) => handle_error!(e, "Failed to start macro playback"),
            }
        })
    }

    /// Execute the next action in a playback session
    pub fn execute_next_action(&self, session_id: Uuid) -> fluent_ai_async::AsyncStream<MacroPlaybackResult> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let player = self.player.clone();
        AsyncStream::with_channel(move |sender| {
            match player.execute_next_action_sync(session_id) {
                Ok(result) => emit!(sender, result),
                Err(e) => handle_error!(e, "Failed to execute next macro action"),
            }
        })
    }

    /// Get a macro by ID
    pub fn get_macro(&self, macro_id: Uuid) -> Option<ChatMacro> {
        self.storage.get_macro(macro_id)
    }

    /// Store a macro
    pub fn store_macro(&self, macro_def: ChatMacro) -> MacroResult<Uuid> {
        self.storage.store_macro(macro_def)
    }

    /// Delete a macro
    pub fn delete_macro(&self, macro_id: Uuid) -> MacroResult<bool> {
        self.storage.delete_macro(macro_id)
    }

    /// List all macros
    pub fn list_macros(&self) -> Vec<ChatMacro> {
        self.storage.list_macros()
    }

    /// Search macros by pattern
    pub fn search_macros(&self, pattern: &str) -> Vec<ChatMacro> {
        self.storage.search_macros(pattern)
    }

    /// Set global variable
    pub fn set_variable(&self, name: String, value: String) -> fluent_ai_async::AsyncStream<()> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let variables = self.variables.clone();
        AsyncStream::with_channel(move |sender| {
            match variables.write() {
                Ok(mut vars) => {
                    vars.set_variable(name, value);
                    emit!(sender, ());
                }
                Err(e) => handle_error!(
                    MacroSystemError::StorageError(format!("Failed to acquire variable lock: {}", e)),
                    "Failed to set variable"
                ),
            }
        })
    }

    /// Get global variable
    pub fn get_variable(&self, name: &str) -> fluent_ai_async::AsyncStream<Option<String>> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let name = name.to_string();
        let variables = self.variables.clone();
        AsyncStream::with_channel(move |sender| {
            match variables.read() {
                Ok(vars) => {
                    let result = vars.get_variable(&name).cloned();
                    emit!(sender, result);
                }
                Err(e) => handle_error!(
                    MacroSystemError::StorageError(format!("Failed to acquire variable lock: {}", e)),
                    "Failed to get variable"
                ),
            }
        })
    }

    /// Clear all global variables
    pub fn clear_variables(&self) -> fluent_ai_async::AsyncStream<()> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let variables = self.variables.clone();
        AsyncStream::with_channel(move |sender| {
            match variables.write() {
                Ok(mut vars) => {
                    vars.clear_variables();
                    emit!(sender, ());
                }
                Err(e) => handle_error!(
                    MacroSystemError::StorageError(format!("Failed to acquire variable lock: {}", e)),
                    "Failed to clear variables"
                ),
            }
        })
    }

    /// Get execution statistics
    pub fn get_execution_stats(&self, macro_id: Uuid) -> Option<Arc<ExecutionStats>> {
        self.storage.get_execution_stats(macro_id)
    }

    /// Get system statistics
    pub fn get_system_stats(&self) -> SystemStats {
        SystemStats {
            total_executions: self.execution_counter.get(),
            storage_stats: self.storage.get_storage_stats(),
        }
    }
}

impl Default for MacroSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// System-wide statistics
#[derive(Debug, Clone)]
pub struct SystemStats {
    pub total_executions: usize,
    pub storage_stats: StorageStats,
}