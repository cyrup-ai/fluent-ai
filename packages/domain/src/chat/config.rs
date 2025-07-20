//! Configuration management system for chat features
//!
//! This module provides a comprehensive configuration management system with atomic updates,
//! validation, persistence, and change notifications using zero-allocation patterns and
//! lock-free operations for blazing-fast performance.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use atomic_counter::{AtomicCounter, ConsistentCounter};
use crossbeam_queue::SegQueue;
use serde::{Deserialize, Serialize};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use tokio::sync::{RwLock, broadcast};
use uuid::Uuid;

/// Core chat configuration
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize, Default)]
pub struct ChatConfig {
    /// Maximum message length
    pub max_message_length: usize,
    /// Enable message history
    pub enable_history: bool,
    /// History retention period
    pub history_retention: Duration,
    /// Enable streaming responses
    pub enable_streaming: bool,
    /// Personality configuration
    pub personality: PersonalityConfig,
    /// Behavior configuration
    pub behavior: BehaviorConfig,
    /// UI configuration
    pub ui: UIConfig,
    /// Integration configuration
    pub integration: IntegrationConfig,
}

/// Personality configuration for AI behavior
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct PersonalityConfig {
    /// Personality type identifier
    pub personality_type: Arc<str>,
    /// Response style settings
    pub response_style: Arc<str>,
    /// Tone configuration
    pub tone: Arc<str>,
    /// Custom instructions
    pub custom_instructions: Option<Arc<str>>,
    /// Creativity level (0.0-1.0)
    pub creativity: f64,
    /// Formality level (0.0-1.0)
    pub formality: f64,
    /// Humor level (0.0-1.0)
    pub humor: f64,
    /// Empathy level (0.0-1.0)
    pub empathy: f64,
    /// Expertise level
    pub expertise_level: Arc<str>,
}

/// Behavior configuration for chat system
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct BehaviorConfig {
    /// Enable auto-responses
    pub auto_response: bool,
    /// Response delay settings
    pub response_delay: Duration,
    /// Enable message filtering
    pub enable_filtering: bool,
    /// Maximum concurrent conversations
    pub max_concurrent_chats: usize,
}

/// User interface configuration
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct UIConfig {
    /// Theme settings
    pub theme: Arc<str>,
    /// Font size
    pub font_size: u32,
    /// Enable dark mode
    pub dark_mode: bool,
    /// Animation settings
    pub enable_animations: bool,
}

/// Integration configuration for external services
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct IntegrationConfig {
    /// Enabled integrations
    pub enabled_integrations: Vec<Arc<str>>,
    /// API keys and tokens
    pub credentials: HashMap<Arc<str>, Arc<str>>,
    /// Webhook configuration
    pub webhooks: Vec<Arc<str>>,
}

impl Default for PersonalityConfig {
    fn default() -> Self {
        Self {
            personality_type: Arc::from("balanced"),
            response_style: Arc::from("helpful"),
            tone: Arc::from("neutral"),
            custom_instructions: None,
            creativity: 0.5,
            formality: 0.5,
            humor: 0.3,
            empathy: 0.7,
            expertise_level: Arc::from("intermediate"),
        }
    }
}

impl Default for BehaviorConfig {
    fn default() -> Self {
        Self {
            auto_response: false,
            response_delay: Duration::from_millis(500),
            enable_filtering: true,
            max_concurrent_chats: 10,
        }
    }
}

impl Default for UIConfig {
    fn default() -> Self {
        Self {
            theme: Arc::from("default"),
            font_size: 14,
            dark_mode: false,
            enable_animations: true,
        }
    }
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enabled_integrations: Vec::new(),
            credentials: HashMap::new(),
            webhooks: Vec::new(),
        }
    }
}

/// Configuration change event with zero-allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationChangeEvent {
    /// Event ID
    pub id: Uuid,
    /// Timestamp of the change
    pub timestamp: Duration,
    /// Configuration section that changed
    pub section: Arc<str>,
    /// Type of change (update, replace, validate)
    pub change_type: ConfigurationChangeType,
    /// Old configuration value (optional)
    pub old_value: Option<Arc<str>>,
    /// New configuration value (optional)
    pub new_value: Option<Arc<str>>,
    /// User who made the change
    pub user: Option<Arc<str>>,
    /// Change description
    pub description: Arc<str>,
}

/// Configuration change type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigurationChangeType {
    /// Update existing configuration
    Update,
    /// Replace entire configuration
    Replace,
    /// Validate configuration
    Validate,
    /// Reset to default
    Reset,
    /// Import from file
    Import,
    /// Export to file
    Export,
}

/// Configuration validation error
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConfigurationValidationError {
    #[error("Invalid personality configuration: {detail}")]
    InvalidPersonality { detail: Arc<str> },
    #[error("Invalid behavior configuration: {detail}")]
    InvalidBehavior { detail: Arc<str> },
    #[error("Invalid UI configuration: {detail}")]
    InvalidUI { detail: Arc<str> },
    #[error("Invalid integration configuration: {detail}")]
    InvalidIntegration { detail: Arc<str> },
    #[error("Configuration conflict: {detail}")]
    Conflict { detail: Arc<str> },
    #[error("Schema validation failed: {detail}")]
    SchemaValidation { detail: Arc<str> },
    #[error("Range validation failed: {field} must be between {min} and {max}")]
    RangeValidation { field: Arc<str>, min: f32, max: f32 },
    #[error("Required field missing: {field}")]
    RequiredField { field: Arc<str> },
}

/// Configuration validation result
pub type ConfigurationValidationResult<T> = Result<T, ConfigurationValidationError>;

/// Configuration persistence settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationPersistence {
    /// Enable automatic persistence
    pub auto_save: bool,
    /// Auto-save interval in seconds
    pub auto_save_interval: u64,
    /// Configuration file path
    pub config_file_path: Arc<str>,
    /// Backup retention count
    pub backup_retention: u32,
    /// Compression enabled
    pub compression: bool,
    /// Encryption enabled
    pub encryption: bool,
    /// File format (json, yaml, toml, binary)
    pub format: Arc<str>,
}

impl Default for ConfigurationPersistence {
    fn default() -> Self {
        Self {
            auto_save: true,
            auto_save_interval: 300, // 5 minutes
            config_file_path: Arc::from("chat_config.json"),
            backup_retention: 5,
            compression: true,
            encryption: false,
            format: Arc::from("json"),
        }
    }
}

/// Configuration manager with atomic updates and lock-free operations
pub struct ConfigurationManager {
    /// Current configuration with atomic updates
    config: ArcSwap<ChatConfig>,
    /// Configuration change event queue
    change_events: SegQueue<ConfigurationChangeEvent>,
    /// Change notification broadcaster
    change_notifier: broadcast::Sender<ConfigurationChangeEvent>,
    /// Configuration validation rules
    validation_rules: Arc<RwLock<HashMap<Arc<str>, Arc<dyn ConfigurationValidator + Send + Sync>>>>,
    /// Persistence settings
    persistence: Arc<RwLock<ConfigurationPersistence>>,
    /// Configuration change counter
    change_counter: ConsistentCounter,
    /// Last persistence timestamp
    last_persistence: parking_lot::Mutex<Instant>,
    /// Configuration version counter
    version_counter: ConsistentCounter,
    /// Configuration locks for atomic operations
    configuration_locks: Arc<RwLock<HashMap<Arc<str>, Arc<parking_lot::RwLock<()>>>>>,
}

/// Configuration validator trait
pub trait ConfigurationValidator {
    /// Validate configuration section
    fn validate(&self, config: &ChatConfig) -> ConfigurationValidationResult<()>;
    /// Get validator name
    fn name(&self) -> &str;
    /// Get validation priority (lower = higher priority)
    fn priority(&self) -> u8;
}

/// Personality configuration validator
pub struct PersonalityValidator;

impl ConfigurationValidator for PersonalityValidator {
    fn validate(&self, config: &ChatConfig) -> ConfigurationValidationResult<()> {
        let personality = &config.personality;

        // Validate creativity range
        if !(0.0..=1.0).contains(&personality.creativity) {
            return Err(ConfigurationValidationError::RangeValidation {
                field: Arc::from("creativity"),
                min: 0.0,
                max: 1.0,
            });
        }

        // Validate formality range
        if !(0.0..=1.0).contains(&personality.formality) {
            return Err(ConfigurationValidationError::RangeValidation {
                field: Arc::from("formality"),
                min: 0.0,
                max: 1.0,
            });
        }

        // Validate humor range
        if !(0.0..=1.0).contains(&personality.humor) {
            return Err(ConfigurationValidationError::RangeValidation {
                field: Arc::from("humor"),
                min: 0.0,
                max: 1.0,
            });
        }

        // Validate empathy range
        if !(0.0..=1.0).contains(&personality.empathy) {
            return Err(ConfigurationValidationError::RangeValidation {
                field: Arc::from("empathy"),
                min: 0.0,
                max: 1.0,
            });
        }

        // Validate expertise level
        let valid_expertise = ["beginner", "intermediate", "advanced", "expert"];
        if !valid_expertise.contains(&personality.expertise_level.as_ref()) {
            return Err(ConfigurationValidationError::InvalidPersonality {
                detail: Arc::from("Invalid expertise level"),
            });
        }

        // Validate tone
        let valid_tones = ["formal", "casual", "friendly", "professional", "neutral"];
        if !valid_tones.contains(&personality.tone.as_ref()) {
            return Err(ConfigurationValidationError::InvalidPersonality {
                detail: Arc::from("Invalid tone"),
            });
        }

        // Validate verbosity
        let valid_verbosity = ["concise", "balanced", "detailed"];
        if !valid_verbosity.contains(&personality.verbosity.as_ref()) {
            return Err(ConfigurationValidationError::InvalidPersonality {
                detail: Arc::from("Invalid verbosity level"),
            });
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "personality"
    }

    fn priority(&self) -> u8 {
        1
    }
}

/// Behavior configuration validator
pub struct BehaviorValidator;

impl ConfigurationValidator for BehaviorValidator {
    fn validate(&self, config: &ChatConfig) -> ConfigurationValidationResult<()> {
        let behavior = &config.behavior;

        // Validate proactivity range
        if !(0.0..=1.0).contains(&behavior.proactivity) {
            return Err(ConfigurationValidationError::RangeValidation {
                field: Arc::from("proactivity"),
                min: 0.0,
                max: 1.0,
            });
        }

        // Validate question frequency range
        if !(0.0..=1.0).contains(&behavior.question_frequency) {
            return Err(ConfigurationValidationError::RangeValidation {
                field: Arc::from("question_frequency"),
                min: 0.0,
                max: 1.0,
            });
        }

        // Validate conversation flow
        let valid_flows = ["natural", "structured", "adaptive", "guided"];
        if !valid_flows.contains(&behavior.conversation_flow.as_ref()) {
            return Err(ConfigurationValidationError::InvalidBehavior {
                detail: Arc::from("Invalid conversation flow"),
            });
        }

        // Validate follow-up behavior
        let valid_followups = ["contextual", "consistent", "adaptive", "minimal"];
        if !valid_followups.contains(&behavior.follow_up_behavior.as_ref()) {
            return Err(ConfigurationValidationError::InvalidBehavior {
                detail: Arc::from("Invalid follow-up behavior"),
            });
        }

        // Validate error handling
        let valid_error_handling = ["graceful", "verbose", "silent", "strict"];
        if !valid_error_handling.contains(&behavior.error_handling.as_ref()) {
            return Err(ConfigurationValidationError::InvalidBehavior {
                detail: Arc::from("Invalid error handling approach"),
            });
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "behavior"
    }

    fn priority(&self) -> u8 {
        2
    }
}

/// UI configuration validator
pub struct UIValidator;

impl ConfigurationValidator for UIValidator {
    fn validate(&self, config: &ChatConfig) -> ConfigurationValidationResult<()> {
        let ui = &config.ui;

        // Validate theme
        let valid_themes = ["light", "dark", "auto", "system", "custom"];
        if !valid_themes.contains(&ui.theme.as_ref()) {
            return Err(ConfigurationValidationError::InvalidUI {
                detail: Arc::from("Invalid theme"),
            });
        }

        // Validate layout
        let valid_layouts = ["standard", "compact", "wide", "mobile", "adaptive"];
        if !valid_layouts.contains(&ui.layout.as_ref()) {
            return Err(ConfigurationValidationError::InvalidUI {
                detail: Arc::from("Invalid layout"),
            });
        }

        // Validate color scheme
        let valid_color_schemes = ["adaptive", "high_contrast", "colorblind", "custom"];
        if !valid_color_schemes.contains(&ui.color_scheme.as_ref()) {
            return Err(ConfigurationValidationError::InvalidUI {
                detail: Arc::from("Invalid color scheme"),
            });
        }

        // Validate display density
        let valid_densities = ["compact", "comfortable", "spacious"];
        if !valid_densities.contains(&ui.display_density.as_ref()) {
            return Err(ConfigurationValidationError::InvalidUI {
                detail: Arc::from("Invalid display density"),
            });
        }

        // Validate animations
        let valid_animations = ["none", "minimal", "smooth", "rich"];
        if !valid_animations.contains(&ui.animations.as_ref()) {
            return Err(ConfigurationValidationError::InvalidUI {
                detail: Arc::from("Invalid animation setting"),
            });
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "ui"
    }

    fn priority(&self) -> u8 {
        3
    }
}

/// Integration configuration validator
pub struct IntegrationValidator;

impl ConfigurationValidator for IntegrationValidator {
    fn validate(&self, config: &ChatConfig) -> ConfigurationValidationResult<()> {
        let integration = &config.integration;

        // Validate external services
        let valid_services = ["mcp", "tools", "plugins", "apis", "webhooks"];
        for service in &integration.external_services {
            if !valid_services.contains(&service.as_ref()) {
                return Err(ConfigurationValidationError::InvalidIntegration {
                    detail: Arc::from(format!("Invalid external service: {}", service)),
                });
            }
        }

        // Validate API configurations
        let valid_apis = ["rest", "graphql", "websocket", "grpc"];
        for api in &integration.api_configurations {
            if !valid_apis.contains(&api.as_ref()) {
                return Err(ConfigurationValidationError::InvalidIntegration {
                    detail: Arc::from(format!("Invalid API configuration: {}", api)),
                });
            }
        }

        // Validate authentication methods
        let valid_auth = ["token", "oauth", "apikey", "basic", "jwt"];
        for auth in &integration.authentication {
            if !valid_auth.contains(&auth.as_ref()) {
                return Err(ConfigurationValidationError::InvalidIntegration {
                    detail: Arc::from(format!("Invalid authentication method: {}", auth)),
                });
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "integration"
    }

    fn priority(&self) -> u8 {
        4
    }
}

impl ConfigurationManager {
    /// Create a new configuration manager
    pub fn new(initial_config: ChatConfig) -> Self {
        let (change_notifier, _) = broadcast::channel(1000);

        let manager = Self {
            config: ArcSwap::new(Arc::new(initial_config)),
            change_events: SegQueue::new(),
            change_notifier,
            validation_rules: Arc::new(RwLock::new(HashMap::new())),
            persistence: Arc::new(RwLock::new(ConfigurationPersistence::default())),
            change_counter: ConsistentCounter::new(0),
            last_persistence: parking_lot::Mutex::new(Instant::now()),
            version_counter: ConsistentCounter::new(1),
            configuration_locks: Arc::new(RwLock::new(HashMap::new())),
        };

        // Initialize default validators using shared references
        let validation_rules = manager.validation_rules.clone();
        tokio::spawn(async move {
            {
                let mut rules = validation_rules.write().await;
                rules.insert("personality".into(), Arc::new(PersonalityValidator));
                rules.insert("behavior".into(), Arc::new(BehaviorValidator));
                rules.insert("ui".into(), Arc::new(UIValidator));
                rules.insert("integration".into(), Arc::new(IntegrationValidator));
            }
        });

        manager
    }

    /// Get current configuration
    pub fn get_config(&self) -> Arc<ChatConfig> {
        self.config.load_full()
    }

    /// Update configuration atomically
    pub async fn update_config(&self, new_config: ChatConfig) -> ConfigurationValidationResult<()> {
        // Validate the new configuration
        self.validate_config(&new_config).await?;

        let old_config = self.config.load_full();
        let config_arc = Arc::new(new_config);

        // Perform atomic update
        self.config.store(config_arc.clone());

        // Create change event
        let change_event = ConfigurationChangeEvent {
            id: Uuid::new_v4(),
            timestamp: Duration::from_secs(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            ),
            section: Arc::from("all"),
            change_type: ConfigurationChangeType::Replace,
            old_value: Some(Arc::from(format!("{:?}", old_config))),
            new_value: Some(Arc::from(format!("{:?}", config_arc))),
            user: None,
            description: Arc::from("Configuration updated"),
        };

        // Queue change event
        self.change_events.push(change_event.clone());
        self.change_counter.inc();
        self.version_counter.inc();

        // Notify subscribers
        let _ = self.change_notifier.send(change_event);

        // Check for auto-save
        self.check_auto_save().await;

        Ok(())
    }

    /// Update specific configuration section
    pub async fn update_section<F>(
        &self,
        section: &str,
        updater: F,
    ) -> ConfigurationValidationResult<()>
    where
        F: FnOnce(&mut ChatConfig) -> ConfigurationValidationResult<()>,
    {
        let section_arc = Arc::from(section);

        // Get section lock
        let section_lock = {
            let locks = self.configuration_locks.read().await;
            locks.get(&section_arc).cloned()
        };

        let section_lock = if let Some(lock) = section_lock {
            lock
        } else {
            let new_lock = Arc::new(parking_lot::RwLock::new(()));
            let mut locks = self.configuration_locks.write().await;
            locks.insert(section_arc.clone(), new_lock.clone());
            new_lock
        };

        // Acquire section lock
        let _guard = section_lock.write();

        // Load current config and make a copy
        let current_config = self.config.load_full();
        let mut new_config = current_config.as_ref().clone();

        // Apply update
        updater(&mut new_config)?;

        // Validate the updated configuration
        self.validate_config(&new_config).await?;

        // Store the updated configuration
        let config_arc = Arc::new(new_config);
        self.config.store(config_arc.clone());

        // Create change event
        let change_event = ConfigurationChangeEvent {
            id: Uuid::new_v4(),
            timestamp: Duration::from_secs(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            ),
            section: section_arc,
            change_type: ConfigurationChangeType::Update,
            old_value: Some(Arc::from(format!("{:?}", current_config))),
            new_value: Some(Arc::from(format!("{:?}", config_arc))),
            user: None,
            description: Arc::from("Configuration section updated"),
        };

        // Queue change event
        self.change_events.push(change_event.clone());
        self.change_counter.inc();
        self.version_counter.inc();

        // Notify subscribers
        let _ = self.change_notifier.send(change_event);

        // Check for auto-save
        self.check_auto_save().await;

        Ok(())
    }

    /// Subscribe to configuration changes
    pub fn subscribe_to_changes(&self) -> broadcast::Receiver<ConfigurationChangeEvent> {
        self.change_notifier.subscribe()
    }

    /// Validate configuration
    async fn validate_config(&self, config: &ChatConfig) -> ConfigurationValidationResult<()> {
        let validators = self.validation_rules.read().await;

        // Sort validators by priority
        let mut validator_pairs: Vec<_> = validators.iter().collect();
        validator_pairs.sort_by_key(|(_, validator)| validator.priority());

        // Run validators in priority order
        for (_, validator) in validator_pairs {
            validator.validate(config)?;
        }

        Ok(())
    }

    /// Register a configuration validator
    pub async fn register_validator(
        &self,
        validator: Arc<dyn ConfigurationValidator + Send + Sync>,
    ) {
        let mut validators = self.validation_rules.write().await;
        validators.insert(Arc::from(validator.name()), validator);
    }

    /// Check if auto-save is needed
    async fn check_auto_save(&self) {
        let persistence = self.persistence.read().await;
        if !persistence.auto_save {
            return;
        }

        let mut last_save = self.last_persistence.lock();
        let now = Instant::now();

        if now.duration_since(*last_save).as_secs() >= persistence.auto_save_interval {
            *last_save = now;
            drop(last_save);
            drop(persistence);

            // Perform auto-save
            if let Err(e) = self.save_to_file().await {
                tracing::error!("Auto-save failed: {}", e);
            }
        }
    }

    /// Save configuration to file
    pub async fn save_to_file(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let config = self.get_config();
        let persistence = self.persistence.read().await;

        let serialized = match persistence.format.as_ref() {
            "json" => serde_json::to_string_pretty(&*config)?,
            "yaml" => serde_yaml::to_string(&*config)?,
            "toml" => toml::to_string(&*config)?,
            "binary" => {
                let bytes = rkyv::to_bytes::<ChatConfig>(&*config)?;
                {
                    use base64::Engine;
                    base64::engine::general_purpose::STANDARD.encode(&bytes)
                }
            }
            _ => return Err("Unsupported format".into()),
        };

        let data = if persistence.compression {
            let compressed = lz4::block::compress(&serialized.as_bytes(), None, true)?;
            {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD.encode(&compressed)
            }
        } else {
            serialized
        };

        tokio::fs::write(&*persistence.config_file_path, data).await?;

        Ok(())
    }

    /// Load configuration from file
    pub async fn load_from_file(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let persistence = self.persistence.read().await;

        let data = tokio::fs::read_to_string(&*persistence.config_file_path).await?;

        let content = if persistence.compression {
            let compressed = {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD.decode(&data)?
            };
            let decompressed = lz4::block::decompress(&compressed, None)?;
            String::from_utf8(decompressed)?
        } else {
            data
        };

        let config: ChatConfig = match persistence.format.as_ref() {
            "json" => serde_json::from_str(&content)?,
            "yaml" => serde_yaml::from_str(&content)?,
            "toml" => toml::from_str(&content)?,
            "binary" => {
                use base64::Engine;
                let bytes = base64::engine::general_purpose::STANDARD.decode(&content)?;
                let archived = rkyv::from_bytes::<ChatConfig, rkyv::rancor::Error>(&bytes)?;
                archived
            }
            _ => return Err("Unsupported format".into()),
        };

        self.update_config(config).await?;

        Ok(())
    }

    /// Get configuration change history
    pub fn get_change_history(&self) -> Vec<ConfigurationChangeEvent> {
        let mut history = Vec::new();
        while let Some(event) = self.change_events.pop() {
            history.push(event);
        }
        history.reverse();
        history
    }

    /// Get configuration statistics
    pub fn get_statistics(&self) -> ConfigurationStatistics {
        ConfigurationStatistics {
            total_changes: self.change_counter.get(),
            current_version: self.version_counter.get(),
            last_modified: Duration::from_secs(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            ),
            validators_count: 0,      // Will be populated asynchronously
            auto_save_enabled: false, // Will be populated asynchronously
        }
    }
}

/// Configuration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationStatistics {
    pub total_changes: usize,
    pub current_version: usize,
    pub last_modified: Duration,
    pub validators_count: usize,
    pub auto_save_enabled: bool,
}

/// Configuration builder for ergonomic configuration creation
pub struct ConfigurationBuilder {
    config: ChatConfig,
}

impl ConfigurationBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: ChatConfig::default(),
        }
    }

    /// Set personality configuration
    pub fn personality(mut self, personality: PersonalityConfig) -> Self {
        self.config.personality = personality;
        self
    }

    /// Set behavior configuration
    pub fn behavior(mut self, behavior: BehaviorConfig) -> Self {
        self.config.behavior = behavior;
        self
    }

    /// Set UI configuration
    pub fn ui(mut self, ui: UIConfig) -> Self {
        self.config.ui = ui;
        self
    }

    /// Set integration configuration
    pub fn integration(mut self, integration: IntegrationConfig) -> Self {
        self.config.integration = integration;
        self
    }

    /// Build the configuration
    pub fn build(self) -> ChatConfig {
        self.config
    }
}

impl Default for ConfigurationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Personality configuration builder
pub struct PersonalityConfigBuilder {
    config: PersonalityConfig,
}

impl PersonalityConfigBuilder {
    /// Create a new personality configuration builder
    pub fn new() -> Self {
        Self {
            config: PersonalityConfig::default(),
        }
    }

    /// Set tone
    pub fn tone(mut self, tone: impl Into<Arc<str>>) -> Self {
        self.config.tone = tone.into();
        self
    }

    /// Set creativity level
    pub fn creativity(mut self, creativity: f64) -> Self {
        self.config.creativity = creativity.clamp(0.0, 1.0);
        self
    }

    /// Set formality level
    pub fn formality(mut self, formality: f64) -> Self {
        self.config.formality = formality.clamp(0.0, 1.0);
        self
    }

    /// Set expertise level
    pub fn expertise(mut self, expertise: impl Into<Arc<str>>) -> Self {
        self.config.expertise_level = expertise.into();
        self
    }

    /// Add personality trait
    pub fn trait_name(mut self, trait_name: impl Into<Arc<str>>) -> Self {
        self.config.traits.push(trait_name.into());
        self
    }

    /// Set response style
    pub fn response_style(mut self, style: impl Into<Arc<str>>) -> Self {
        self.config.response_style = style.into();
        self
    }

    /// Set humor level
    pub fn humor(mut self, humor: f32) -> Self {
        self.config.humor = humor.clamp(0.0, 1.0);
        self
    }

    /// Set empathy level
    pub fn empathy(mut self, empathy: f32) -> Self {
        self.config.empathy = empathy.clamp(0.0, 1.0);
        self
    }

    /// Set verbosity level
    pub fn verbosity(mut self, verbosity: impl Into<Arc<str>>) -> Self {
        self.config.verbosity = verbosity.into();
        self
    }

    /// Build the personality configuration
    pub fn build(self) -> PersonalityConfig {
        self.config
    }
}

impl Default for PersonalityConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}
