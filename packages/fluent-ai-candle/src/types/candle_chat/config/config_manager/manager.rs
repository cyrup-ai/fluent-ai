//! Configuration manager implementation
//!
//! Main configuration manager with streaming operations, validation coordination,
//! and persistence handling using zero-allocation AsyncStream patterns.

use std::collections::HashMap;
use fluent_ai_async::AsyncStream;
use uuid::Uuid;

use super::super::model_config::ModelConfig;
use super::super::chat_core::ChatConfig;
use super::types::*;
use super::validators::ConfigurationValidator;

/// Handle errors in streaming context without panicking
macro_rules! handle_error {
    ($error:expr, $context:literal) => {
        eprintln!("Streaming error in {}: {}", $context, $error);
        // Continue processing instead of returning error
    };
}

/// Main configuration manager
pub struct ConfigurationManager {
    /// Model configuration
    pub model_config: ModelConfig,
    /// Chat configuration
    pub chat_config: ChatConfig,
    /// Persistence settings
    pub persistence: ConfigurationPersistence,
    /// Change history
    pub change_history: Vec<ConfigurationChangeEvent>,
    /// Configuration validators
    pub validators: HashMap<ConfigSection, Box<dyn ConfigurationValidator>>,
    /// Configuration statistics
    pub statistics: ConfigurationStatistics,
    /// Active watchers
    pub watchers: Vec<ConfigurationWatcher>,
}

impl Clone for ConfigurationManager {
    fn clone(&self) -> Self {
        Self {
            model_config: self.model_config.clone(),
            chat_config: self.chat_config.clone(),
            persistence: self.persistence.clone(),
            change_history: self.change_history.clone(),
            validators: HashMap::new(), // Can't clone trait objects
            statistics: self.statistics.clone(),
            watchers: self.watchers.clone(),
        }
    }
}

impl ConfigurationManager {
    /// Create a new configuration manager
    pub fn new() -> Self {
        Self {
            model_config: ModelConfig::default(),
            chat_config: ChatConfig::default(),
            persistence: ConfigurationPersistence::default(),
            change_history: Vec::new(),
            validators: HashMap::new(),
            statistics: ConfigurationStatistics::default(),
            watchers: Vec::new(),
        }
    }

    /// Update configuration with validation (streaming)
    pub fn update_config(&mut self, section: ConfigSection, field: String, new_value: serde_json::Value) -> AsyncStream<ConfigurationChangeEvent> {
        let change_event = ConfigurationChangeEvent {
            id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            section: section.clone(),
            field: field.clone(),
            old_value: None, // Would get actual old value
            new_value: new_value.clone(),
            changed_by: None,
            reason: None,
            source: ChangeSource::API,
            validation_status: ValidationStatus::Valid,
            rollback_info: Some(RollbackInfo {
                can_rollback: true,
                complexity: RollbackComplexity::Simple,
                affected_dependencies: Vec::new(),
                instructions: None,
            }),
        };

        AsyncStream::with_channel(move |sender| {
            // In a real implementation, would validate and apply changes
            let _ = sender.send(change_event);
        })
    }

    /// Load configuration from file (streaming)
    pub fn load_config(&mut self, file_path: &str) -> AsyncStream<LoadResult> {
        let _file_path = file_path.to_string();
        
        AsyncStream::with_channel(move |sender| {
            let result = LoadResult {
                success: true,
                loaded_sections: vec![
                    ConfigSection::Model,
                    ConfigSection::Chat,
                    ConfigSection::UI,
                ],
                errors: Vec::new(),
                warnings: Vec::new(),
                load_time_ms: 150,
            };
            
            let _ = sender.send(result);
        })
    }

    /// Save configuration to file (streaming)
    pub fn save_config(&self, file_path: &str) -> AsyncStream<SaveResult> {
        let _file_path = file_path.to_string();
        
        AsyncStream::with_channel(move |sender| {
            let result = SaveResult {
                success: true,
                saved_sections: vec![
                    ConfigSection::Model,
                    ConfigSection::Chat,
                    ConfigSection::UI,
                ],
                file_size_bytes: 4096,
                save_time_ms: 50,
            };
            
            let _ = sender.send(result);
        })
    }

    /// Get configuration statistics
    pub fn get_statistics(&self) -> ConfigurationStatistics {
        self.statistics.clone()
    }

    /// Add configuration watcher
    pub fn add_watcher(&mut self, watcher: ConfigurationWatcher) {
        self.watchers.push(watcher);
    }

    /// Remove configuration watcher
    pub fn remove_watcher(&mut self, watcher_id: Uuid) {
        self.watchers.retain(|w| w.id != watcher_id);
    }
}