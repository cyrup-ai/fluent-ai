//! Configuration manager with atomic updates and lock-free operations
//!
//! This module provides the main configuration management system with
//! atomic updates, validation, and change notifications.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use arc_swap::ArcSwap;
use crossbeam_queue::SegQueue;
use fluent_ai_async::AsyncStream;
use tokio::sync::{RwLock, broadcast};
use uuid::Uuid;

use super::core::ChatConfig;
use super::events::{ConfigurationChangeEvent, ConfigurationChangeType, ConfigurationPersistence};
use super::validation::{ConfigurationValidator, PersonalityValidator, BehaviorValidator, UIValidator, IntegrationValidator};
use super::model::ModelValidator;

/// Configuration manager with atomic updates and lock-free operations
#[derive(Debug)]
pub struct ConfigurationManager {
    /// Current configuration with atomic swapping
    current_config: ArcSwap<ChatConfig>,
    /// Configuration change events queue
    change_events: SegQueue<ConfigurationChangeEvent>,
    /// Change notification broadcast channel
    change_notifier: broadcast::Sender<ConfigurationChangeEvent>,
    /// Configuration persistence settings
    persistence: Arc<RwLock<ConfigurationPersistence>>,
    /// Configuration version counter
    version_counter: AtomicUsize,
    /// Statistics
    stats: ConfigurationStatistics,
}

impl Clone for ConfigurationManager {
    fn clone(&self) -> Self {
        Self {
            current_config: ArcSwap::new(self.current_config.load().clone()),
            change_events: SegQueue::new(),
            change_notifier: {
                let (tx, _) = broadcast::channel(1000);
                tx
            },
            persistence: Arc::new(RwLock::new(
                self.persistence.try_read()
                    .map(|p| p.clone())
                    .unwrap_or_default()
            )),
            version_counter: AtomicUsize::new(self.version_counter.load(Ordering::Relaxed)),
            stats: ConfigurationStatistics::default(),
        }
    }
}

/// Configuration statistics
#[derive(Debug, Default)]
pub struct ConfigurationStatistics {
    pub total_changes: AtomicUsize,
    pub validation_errors: AtomicUsize,
    pub last_change_time: ArcSwap<Option<Instant>>,
}

impl ConfigurationManager {
    /// Create a new configuration manager
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(1000);
        
        Self {
            current_config: ArcSwap::new(Arc::new(ChatConfig::default())),
            change_events: SegQueue::new(),
            change_notifier: tx,
            persistence: Arc::new(RwLock::new(ConfigurationPersistence::default())),
            version_counter: AtomicUsize::new(1),
            stats: ConfigurationStatistics::default(),
        }
    }

    /// Create with initial configuration
    pub fn with_config(config: ChatConfig) -> Self {
        let (tx, _) = broadcast::channel(1000);
        
        Self {
            current_config: ArcSwap::new(Arc::new(config)),
            change_events: SegQueue::new(),
            change_notifier: tx,
            persistence: Arc::new(RwLock::new(ConfigurationPersistence::default())),
            version_counter: AtomicUsize::new(1),
            stats: ConfigurationStatistics::default(),
        }
    }

    /// Get current configuration (zero-allocation clone of Arc)
    pub fn get_config(&self) -> Arc<ChatConfig> {
        self.current_config.load()
    }

    /// Update configuration atomically with validation
    pub fn update_config(&self, new_config: ChatConfig) -> AsyncStream<Result<(), Vec<String>>> {
        let current_config = self.current_config.load();
        let change_notifier = self.change_notifier.clone();
        let change_events = &self.change_events;
        let version_counter = &self.version_counter;
        let stats = &self.stats;
        
        AsyncStream::with_channel(move |sender| {
            // Validate configuration
            if let Err(validation_errors) = Self::validate_full_config(&new_config) {
                stats.validation_errors.fetch_add(1, Ordering::Relaxed);
                fluent_ai_async::emit!(sender, Err(validation_errors));
                return;
            }

            // Create change event
            let change_event = ConfigurationChangeEvent::new(
                new_config.id,
                ConfigurationChangeType::Updated,
                "configuration",
                Some(serde_json::to_value(&**current_config).unwrap_or_default()),
                serde_json::to_value(&new_config).unwrap_or_default(),
                None,
                "manager",
            );

            // Update configuration atomically
            self.current_config.store(Arc::new(new_config));
            version_counter.fetch_add(1, Ordering::Relaxed);
            stats.total_changes.fetch_add(1, Ordering::Relaxed);
            stats.last_change_time.store(Arc::new(Some(Instant::now())));

            // Queue change event
            change_events.push(change_event.clone());

            // Notify subscribers (ignore if no subscribers)
            let _ = change_notifier.send(change_event);

            fluent_ai_async::emit!(sender, Ok(()));
        })
    }

    /// Subscribe to configuration changes
    pub fn subscribe_to_changes(&self) -> broadcast::Receiver<ConfigurationChangeEvent> {
        self.change_notifier.subscribe()
    }

    /// Get configuration version
    pub fn get_version(&self) -> usize {
        self.version_counter.load(Ordering::Relaxed)
    }

    /// Get recent change events
    pub fn get_recent_changes(&self, limit: usize) -> Vec<ConfigurationChangeEvent> {
        let mut events = Vec::with_capacity(limit.min(1000));
        let mut count = 0;
        
        while let Some(event) = self.change_events.pop() {
            events.push(event);
            count += 1;
            if count >= limit {
                break;
            }
        }
        
        events.reverse(); // Most recent first
        events
    }

    /// Get configuration statistics
    pub fn get_statistics(&self) -> ConfigurationStatistics {
        ConfigurationStatistics {
            total_changes: AtomicUsize::new(self.stats.total_changes.load(Ordering::Relaxed)),
            validation_errors: AtomicUsize::new(self.stats.validation_errors.load(Ordering::Relaxed)),
            last_change_time: ArcSwap::new(self.stats.last_change_time.load().as_ref().clone()),
        }
    }

    /// Reset configuration to defaults
    pub fn reset_to_defaults(&self) -> AsyncStream<Result<(), Vec<String>>> {
        let default_config = ChatConfig::default();
        self.update_config(default_config)
    }

    /// Validate full configuration using all validators
    fn validate_full_config(config: &ChatConfig) -> Result<(), Vec<String>> {
        let mut all_errors = Vec::new();

        // Validate model configuration
        let model_validator = ModelValidator;
        if let Err(errors) = model_validator.validate(&config.model) {
            all_errors.extend(errors.into_iter().map(|e| e.description()));
        }

        // Validate personality configuration
        let personality_validator = PersonalityValidator;
        if let Err(errors) = personality_validator.validate(&config.personality) {
            all_errors.extend(errors.into_iter().map(|e| e.description()));
        }

        // Validate behavior configuration
        let behavior_validator = BehaviorValidator;
        if let Err(errors) = behavior_validator.validate(&config.behavior) {
            all_errors.extend(errors.into_iter().map(|e| e.description()));
        }

        // Validate UI configuration
        let ui_validator = UIValidator;
        if let Err(errors) = ui_validator.validate(&config.ui) {
            all_errors.extend(errors.into_iter().map(|e| e.description()));
        }

        // Validate integration configuration
        let integration_validator = IntegrationValidator;
        if let Err(errors) = integration_validator.validate(&config.integrations) {
            all_errors.extend(errors.into_iter().map(|e| e.description()));
        }

        if all_errors.is_empty() {
            Ok(())
        } else {
            Err(all_errors)
        }
    }
}

impl Default for ConfigurationManager {
    fn default() -> Self {
        Self::new()
    }
}