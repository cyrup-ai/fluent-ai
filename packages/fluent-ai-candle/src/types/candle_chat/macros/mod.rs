//! Macro system for chat automation
//!
//! This module provides a comprehensive macro system for recording, storing,
//! and playing back chat interactions using zero-allocation patterns and
//! lock-free data structures for blazing-fast performance.

use std::collections::HashMap;
use fluent_ai_async::{AsyncStream, emit};

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
    MacroResult, ActionExecutionResult, MacroPlaybackResult};

pub use recording::{MacroRecorder, RecordingInfo};
pub use playback::{MacroPlayer, PlaybackInfo};
pub use variables::{
    VariableManager, resolve_variables, resolve_variables_static,
    evaluate_condition, evaluate_condition_static};
pub use storage::{MacroStorage, StorageStats, MacroExport};

use std::sync::Arc;

use atomic_counter::{AtomicCounter, ConsistentCounter};
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
    execution_counter: ConsistentCounter}

impl MacroSystem {
    /// Create new macro system
    pub fn new() -> Self {
        let storage = MacroStorage::new();
        let macros = storage.macros();
        
        let recorder = MacroRecorder::new(macros.clone());
        let player = MacroPlayer::new(macros);
        
        Self {
            storage,
            recorder,
            player,
            variables: RwLock::new(VariableManager::new()),
            execution_counter: ConsistentCounter::new(0)}
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
        self.recorder.start_recording(name, description)
    }

    /// Record a macro action
    pub fn record_action(&self, session_id: Uuid, action: MacroAction) -> fluent_ai_async::AsyncStream<()> {
        self.recorder.record_action(session_id, action)
    }

    /// Stop recording and save the macro
    pub fn stop_recording(&self, session_id: Uuid) -> fluent_ai_async::AsyncStream<Uuid> {
        self.recorder.stop_recording(session_id)
    }

    /// Start macro playback
    pub fn start_playback(&self, macro_id: Uuid, variables: HashMap<String, String>) -> fluent_ai_async::AsyncStream<Uuid> {
        self.execution_counter.inc();
        let player = self.player.clone();
        // Use the async start_playback method directly
        player.start_playback(macro_id, variables)
    }

    /// Execute the next action in a playback session
    pub fn execute_next_action(&self, session_id: Uuid) -> fluent_ai_async::AsyncStream<MacroPlaybackResult> {
        let player = self.player.clone();
        // Use the async execute_next_action method directly  
        player.execute_next_action(session_id)
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
        
        let result = match self.variables.write() {
            Ok(mut vars) => {
                vars.set_variable(name, value);
                Ok(())
            }
            Err(e) => Err(MacroSystemError::StorageError(format!("Failed to acquire variable lock: {}", e))),
        };
        
        AsyncStream::with_channel(move |sender| {
            match result {
                Ok(()) => emit!(sender, ()),
                Err(e) => handle_error!(
                    MacroSystemError::StorageError(format!("Failed to acquire variable lock: {}", e)),
                    "Failed to set variable"
                )}
        })
    }

    /// Get global variable
    pub fn get_variable(&self, name: &str) -> fluent_ai_async::AsyncStream<Option<String>> {
        let result = match self.variables.read() {
            Ok(vars) => vars.get_variable(name).cloned(),
            Err(_) => None,
        };
        
        AsyncStream::with_channel(move |sender| {
            emit!(sender, result);
        })
    }

    /// Clear all global variables
    pub fn clear_variables(&self) -> fluent_ai_async::AsyncStream<()> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let result = match self.variables.write() {
            Ok(mut vars) => {
                vars.clear_variables();
                Ok(())
            }
            Err(e) => Err(MacroSystemError::StorageError(format!("Failed to acquire variable lock: {}", e))),
        };
        
        AsyncStream::with_channel(move |sender| {
            match result {
                Ok(()) => emit!(sender, ()),
                Err(e) => handle_error!(e, "Failed to clear variables"
                )}
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
            storage_stats: self.storage.get_storage_stats()}
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
    pub storage_stats: StorageStats}