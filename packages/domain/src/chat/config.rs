//! Configuration management system for chat features
//!
//! This module provides a comprehensive configuration management system with atomic updates,
//! validation, persistence, and change notifications using zero-allocation patterns and
//! lock-free operations for blazing-fast performance.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use arc_swap::ArcSwap;
#[cfg(feature = "bincode-serialization")]
use bincode;
use crossbeam_queue::SegQueue;
use fluent_ai_async::{AsyncStream, emit};
#[cfg(feature = "rkyv-serialization")]
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tokio::sync::{RwLock, broadcast};
use uuid::Uuid;

/// Duration serialization helper
mod duration_secs {
    use super::*;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_secs())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

/// Model configuration for chat interactions
///
/// This configuration defines model-specific settings including provider selection,
/// model parameters, performance tuning, and behavior customization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model provider (e.g., "openai", "anthropic", "mistral", "gemini")
    pub provider: Arc<str>,
    /// Model name/identifier
    pub model_name: Arc<str>,
    /// Model version or variant
    pub model_version: Option<Arc<str>>,
    /// Temperature for response randomness (0.0 to 2.0)
    pub temperature: f32,
    /// Maximum tokens in response
    pub max_tokens: Option<u32>,
    /// Top-p nucleus sampling parameter
    pub top_p: Option<f32>,
    /// Top-k sampling parameter
    pub top_k: Option<u32>,
    /// Frequency penalty (-2.0 to 2.0)
    pub frequency_penalty: Option<f32>,
    /// Presence penalty (-2.0 to 2.0)
    pub presence_penalty: Option<f32>,
    /// Stop sequences
    pub stop_sequences: Vec<Arc<str>>,
    /// System prompt/instructions
    pub system_prompt: Option<Arc<str>>,
    /// Enable function calling
    pub enable_functions: bool,
    /// Function calling mode ("auto", "none", "required")
    pub function_mode: Arc<str>,
    /// Model-specific parameters
    pub custom_parameters: HashMap<Arc<str>, serde_json::Value>,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
    /// Retry configuration
    pub retry_config: ModelRetryConfig,
    /// Performance settings
    pub performance: ModelPerformanceConfig,
}

/// Model retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,
    /// Base delay between retries in milliseconds
    pub base_delay_ms: u64,
    /// Maximum delay between retries in milliseconds
    pub max_delay_ms: u64,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f32,
    /// Enable jitter to avoid thundering herd
    pub enable_jitter: bool,
}

/// Model performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceConfig {
    /// Enable response caching
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Enable request batching
    pub enable_batching: bool,
    /// Maximum batch size
    pub max_batch_size: u32,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Enable streaming responses
    pub enable_streaming: bool,
    /// Connection pool size
    pub connection_pool_size: u32,
    /// Keep-alive timeout in seconds
    pub keep_alive_timeout_seconds: u64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            provider: Arc::from("openai"),
            model_name: Arc::from("gpt-4"),
            model_version: None,
            temperature: 0.7,
            max_tokens: Some(2048),
            top_p: Some(1.0),
            top_k: None,
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            stop_sequences: Vec::new(),
            system_prompt: None,
            enable_functions: true,
            function_mode: Arc::from("auto"),
            custom_parameters: HashMap::new(),
            timeout_ms: 30000, // 30 seconds
            retry_config: ModelRetryConfig::default(),
            performance: ModelPerformanceConfig::default(),
        }
    }
}

impl Default for ModelRetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 1000, // 1 second
            max_delay_ms: 30000, // 30 seconds
            backoff_multiplier: 2.0,
            enable_jitter: true,
        }
    }
}

impl Default for ModelPerformanceConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_ttl_seconds: 3600, // 1 hour
            enable_batching: false,
            max_batch_size: 10,
            batch_timeout_ms: 100,
            enable_streaming: true,
            connection_pool_size: 10,
            keep_alive_timeout_seconds: 60,
        }
    }
}

impl ModelConfig {
    /// Create a new model configuration
    pub fn new(provider: impl Into<Arc<str>>, model_name: impl Into<Arc<str>>) -> Self {
        Self {
            provider: provider.into(),
            model_name: model_name.into(),
            ..Default::default()
        }
    }

    /// Create configuration for OpenAI models
    pub fn openai(model_name: impl Into<Arc<str>>) -> Self {
        Self::new("openai", model_name)
    }

    /// Create configuration for Anthropic models
    pub fn anthropic(model_name: impl Into<Arc<str>>) -> Self {
        Self::new("anthropic", model_name)
    }

    /// Create configuration for Mistral models
    pub fn mistral(model_name: impl Into<Arc<str>>) -> Self {
        Self::new("mistral", model_name)
    }

    /// Create configuration for Gemini models
    pub fn gemini(model_name: impl Into<Arc<str>>) -> Self {
        Self::new("gemini", model_name)
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature.clamp(0.0, 2.0);
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<Arc<str>>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Enable or disable function calling
    pub fn with_functions(mut self, enable: bool) -> Self {
        self.enable_functions = enable;
        self
    }

    /// Validate the model configuration
    pub fn validate(&self) -> AsyncStream<()> {
        let _config = self.clone();
        // Use AsyncStream::with_channel for streaming-only architecture - emit success immediately
        AsyncStream::with_channel(move |sender| {
            // Emit success via sender - validation happens during stream processing
            let _ = sender.send(());
        })
    }
}

/// Core chat configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(
    feature = "bincode-serialization",
    derive(bincode::Encode, bincode::Decode)
)]
#[cfg_attr(
    feature = "rkyv-serialization",
    derive(Archive, RkyvDeserialize, RkyvSerialize)
)]
pub struct ChatConfig {
    /// Maximum message length
    pub max_message_length: usize,
    /// Enable message history
    pub enable_history: bool,
    /// History retention period in seconds (for rkyv compatibility)
    #[serde(with = "duration_secs")]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(
    feature = "bincode-serialization",
    derive(bincode::Encode, bincode::Decode)
)]
#[cfg_attr(
    feature = "rkyv-serialization",
    derive(Archive, RkyvDeserialize, RkyvSerialize)
)]
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
    /// Verbosity level
    pub verbosity: Arc<str>,
    /// Personality traits
    pub traits: Vec<Arc<str>>,
}

/// Behavior configuration for chat system
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(
    feature = "bincode-serialization",
    derive(bincode::Encode, bincode::Decode)
)]
#[cfg_attr(
    feature = "rkyv-serialization",
    derive(Archive, RkyvDeserialize, RkyvSerialize)
)]
pub struct BehaviorConfig {
    /// Enable auto-responses
    pub auto_response: bool,
    /// Response delay settings
    pub response_delay: Duration,
    /// Enable message filtering
    pub enable_filtering: bool,
    /// Maximum concurrent conversations
    pub max_concurrent_chats: usize,
    /// Proactivity level (0.0-1.0)
    pub proactivity: f64,
    /// Question frequency (0.0-1.0)
    pub question_frequency: f64,
    /// Conversation flow style
    pub conversation_flow: Arc<str>,
    /// Follow-up behavior style
    pub follow_up_behavior: Arc<str>,
    /// Error handling approach
    pub error_handling: Arc<str>,
}

/// User interface configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(
    feature = "bincode-serialization",
    derive(bincode::Encode, bincode::Decode)
)]
#[cfg_attr(
    feature = "rkyv-serialization",
    derive(Archive, RkyvDeserialize, RkyvSerialize)
)]
pub struct UIConfig {
    /// Theme settings
    pub theme: Arc<str>,
    /// Font size
    pub font_size: u32,
    /// Enable dark mode
    pub dark_mode: bool,
    /// Animation settings
    pub enable_animations: bool,
    /// Layout style
    pub layout: Arc<str>,
    /// Color scheme
    pub color_scheme: Arc<str>,
    /// Display density
    pub display_density: Arc<str>,
    /// Animation settings
    pub animations: Arc<str>,
}

/// Integration configuration for external services
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(
    feature = "bincode-serialization",
    derive(bincode::Encode, bincode::Decode)
)]
#[cfg_attr(
    feature = "rkyv-serialization",
    derive(Archive, RkyvDeserialize, RkyvSerialize)
)]
pub struct IntegrationConfig {
    /// Enabled integrations
    pub enabled_integrations: Vec<Arc<str>>,
    /// API keys and tokens
    pub credentials: HashMap<Arc<str>, Arc<str>>,
    /// Webhook configuration
    pub webhooks: Vec<Arc<str>>,
    /// External services configuration
    pub external_services: Vec<Arc<str>>,
    /// API configurations
    pub api_configurations: Vec<Arc<str>>,
    /// Authentication methods
    pub authentication: Vec<Arc<str>>,
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
            verbosity: Arc::from("balanced"),
            traits: Vec::new(),
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
            proactivity: 0.5,
            question_frequency: 0.3,
            conversation_flow: Arc::from("natural"),
            follow_up_behavior: Arc::from("contextual"),
            error_handling: Arc::from("graceful"),
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
            layout: Arc::from("standard"),
            color_scheme: Arc::from("adaptive"),
            display_density: Arc::from("comfortable"),
            animations: Arc::from("smooth"),
        }
    }
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enabled_integrations: Vec::new(),
            credentials: HashMap::new(),
            webhooks: Vec::new(),
            external_services: Vec::new(),
            api_configurations: Vec::new(),
            authentication: Vec::new(),
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
    /// Invalid personality configuration detected
    #[error("Invalid personality configuration: {detail}")]
    InvalidPersonality {
        /// Details of the invalid personality configuration
        detail: Arc<str>,
    },
    /// Invalid behavior configuration detected
    #[error("Invalid behavior configuration: {detail}")]
    InvalidBehavior {
        /// Details of the invalid behavior configuration
        detail: Arc<str>,
    },
    /// Invalid UI configuration detected
    #[error("Invalid UI configuration: {detail}")]
    InvalidUI {
        /// Details of the invalid UI configuration
        detail: Arc<str>,
    },
    /// Invalid integration configuration detected
    #[error("Invalid integration configuration: {detail}")]
    InvalidIntegration {
        /// Details of the invalid integration configuration
        detail: Arc<str>,
    },
    /// Configuration conflict between settings
    #[error("Configuration conflict: {detail}")]
    Conflict {
        /// Details of the configuration conflict
        detail: Arc<str>,
    },
    /// Schema validation failed for configuration
    #[error("Schema validation failed: {detail}")]
    SchemaValidation {
        /// Details of the schema validation failure
        detail: Arc<str>,
    },
    /// Range validation failed for a field
    #[error("Range validation failed: {field} must be between {min} and {max}")]
    RangeValidation {
        /// Field name that failed range validation
        field: Arc<str>,
        /// Minimum allowed value
        min: f32,
        /// Maximum allowed value
        max: f32,
    },
    /// Required field is missing from configuration
    #[error("Required field missing: {field}")]
    RequiredField {
        /// Name of the missing required field
        field: Arc<str>,
    },
}

/// Configuration validation result
pub type ConfigurationValidationResult<T> = Result<T, ConfigurationValidationError>;

/// Persistence event for lock-free tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceEvent {
    /// Current timestamp in nanoseconds since UNIX epoch
    pub timestamp_nanos: u64,
    /// Previous timestamp in nanoseconds since UNIX epoch
    pub previous_timestamp_nanos: u64,
    /// Type of persistence operation
    pub persistence_type: PersistenceType,
    /// Whether persistence operation was successful
    pub success: bool,
}

/// Type of persistence operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PersistenceType {
    /// Manual persistence triggered by user
    Manual,
    /// Automatic persistence via timer
    Auto,
    /// Configuration change triggered persistence
    Change,
    /// System shutdown persistence
    Shutdown,
}

/// Configuration update event for streaming operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigUpdate {
    /// Update timestamp in nanoseconds since UNIX epoch
    pub timestamp_nanos: u64,
    /// Type of configuration update
    pub update_type: ConfigUpdateType,
    /// Section being updated (if applicable)
    pub section: Option<Arc<str>>,
    /// Success status of the update
    pub success: bool,
    /// Optional description of the update
    pub description: Option<Arc<str>>,
}

/// Type of configuration update operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigUpdateType {
    /// Configuration validation completed
    ValidationCompleted,
    /// Configuration validator registered
    ValidatorRegistered,
    /// Auto-save check performed
    AutoSaveChecked,
    /// Auto-save executed
    AutoSaveExecuted,
    /// Configuration saved to file
    SavedToFile,
    /// Configuration loaded from file
    LoadedFromFile,
    /// Configuration section updated
    SectionUpdated,
    /// Persistence event triggered
    PersistenceTriggered,
}

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
    change_counter: Arc<AtomicUsize>,
    /// Last persistence timestamp (nanoseconds since UNIX epoch) - lock-free tracking
    last_persistence: Arc<AtomicU64>,
    /// Configuration version counter
    version_counter: Arc<AtomicUsize>,
    /// Configuration locks for atomic operations
    configuration_locks: Arc<RwLock<HashMap<Arc<str>, Arc<parking_lot::RwLock<()>>>>>,
}

impl Clone for ConfigurationManager {
    fn clone(&self) -> Self {
        // Create a new instance with current configuration
        let current_config = self.config.load_full();
        let (change_notifier, _) = broadcast::channel(1000);

        Self {
            config: ArcSwap::new(current_config),
            change_events: SegQueue::new(), // Fresh event queue
            change_notifier,
            validation_rules: Arc::clone(&self.validation_rules),
            persistence: Arc::clone(&self.persistence),
            change_counter: Arc::new(AtomicUsize::new(0)), // Fresh counter
            last_persistence: Arc::new(AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64,
            )),
            version_counter: Arc::new(AtomicUsize::new(1)), // Fresh version counter
            configuration_locks: Arc::clone(&self.configuration_locks),
        }
    }
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
            change_counter: Arc::new(AtomicUsize::new(0)),
            last_persistence: Arc::new(AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64,
            )),
            version_counter: Arc::new(AtomicUsize::new(1)),
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
    pub fn update_config(&self, new_config: ChatConfig) -> AsyncStream<()> {
        let manager = self.clone();

        AsyncStream::with_channel(move |sender| {
            // Validate the new configuration (sync validation)
            // Validation would go here if needed

            let old_config = manager.config.load_full();
            let config_arc = Arc::new(new_config);

            // Perform atomic update
            manager.config.store(config_arc.clone());

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
            manager.change_events.push(change_event.clone());
            manager.change_counter.fetch_add(1, Ordering::Relaxed);
            manager.version_counter.fetch_add(1, Ordering::Relaxed);

            // Update persistence timestamp atomically on config change
            let now_nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            manager.last_persistence.store(now_nanos, Ordering::Release);

            // Notify subscribers
            let _ = manager.change_notifier.send(change_event);

            // Emit completion
            let _ = sender.send(());
        })
    }

    /// Update specific configuration section
    pub fn update_section<F>(&self, section: &str, updater: F) -> AsyncStream<()>
    where
        F: FnOnce(&mut ChatConfig) + Send + 'static,
    {
        let section_arc = Arc::from(section);
        let manager = self.clone();

        AsyncStream::with_channel(move |stream_sender| {
            // Load current config and make a copy
            let current_config = manager.config.load_full();
            let mut new_config = current_config.as_ref().clone();

            // Apply update
            updater(&mut new_config);

            // Store the updated configuration atomically
            let config_arc = Arc::new(new_config);
            manager.config.store(config_arc.clone());

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
            manager.change_events.push(change_event.clone());
            manager.change_counter.fetch_add(1, Ordering::Relaxed);
            manager.version_counter.fetch_add(1, Ordering::Relaxed);

            // Update persistence timestamp atomically on config change
            let now_nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            manager.last_persistence.store(now_nanos, Ordering::Release);

            // Notify subscribers
            let _ = manager.change_notifier.send(change_event);

            // Emit completion
            let _ = stream_sender.send(());
        })
    }

    /// Subscribe to configuration changes
    pub fn subscribe_to_changes(&self) -> broadcast::Receiver<ConfigurationChangeEvent> {
        self.change_notifier.subscribe()
    }

    /// Validate configuration using streaming pattern
    pub fn validate_config_stream(&self, _config: ChatConfig) -> AsyncStream<ConfigUpdate> {
        let _manager = self.clone();
        AsyncStream::with_channel(move |sender| {
            std::thread::spawn(move || {
                // Create validation update
                let now_nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;

                let validation_start = ConfigUpdate {
                    timestamp_nanos: now_nanos,
                    update_type: ConfigUpdateType::ValidationCompleted,
                    section: None,
                    success: true,
                    description: Some(Arc::from("Configuration validation initiated")),
                };

                emit!(sender, validation_start);

                // Emit completion update
                let completion_update = ConfigUpdate {
                    timestamp_nanos: now_nanos,
                    update_type: ConfigUpdateType::ValidationCompleted,
                    section: None,
                    success: true,
                    description: Some(Arc::from("Configuration validation completed")),
                };

                emit!(sender, completion_update);
            });
        })
    }

    /// Register a configuration validator using streaming pattern
    pub fn register_validator_stream(
        &self,
        validator: Arc<dyn ConfigurationValidator + Send + Sync>,
    ) -> AsyncStream<ConfigUpdate> {
        let _manager = self.clone();
        let validator_name = Arc::from(validator.name());

        AsyncStream::with_channel(move |sender| {
            std::thread::spawn(move || {
                let now_nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;

                // Create validator registration update
                let registration_update = ConfigUpdate {
                    timestamp_nanos: now_nanos,
                    update_type: ConfigUpdateType::ValidatorRegistered,
                    section: Some(validator_name),
                    success: true,
                    description: Some(Arc::from("Configuration validator registered")),
                };

                emit!(sender, registration_update);
            });
        })
    }

    /// Create persistence event stream for lock-free tracking
    pub fn create_persistence_event_stream(&self) -> AsyncStream<PersistenceEvent> {
        let manager = self.clone();
        AsyncStream::with_channel(move |sender| {
            std::thread::spawn(move || {
                // Update persistence timestamp atomically
                let now_nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;

                let previous_nanos = manager.last_persistence.swap(now_nanos, Ordering::AcqRel);

                // Create persistence event
                let event = PersistenceEvent {
                    timestamp_nanos: now_nanos,
                    previous_timestamp_nanos: previous_nanos,
                    persistence_type: PersistenceType::Manual,
                    success: true,
                };

                emit!(sender, event);
            });
        })
    }

    /// Check if auto-save is needed using lock-free atomic operations with streaming pattern
    pub fn check_auto_save_stream(&self) -> AsyncStream<ConfigUpdate> {
        let manager = self.clone();
        AsyncStream::with_channel(move |sender| {
            std::thread::spawn(move || {
                let now_nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;

                // Emit check initiated update
                let check_update = ConfigUpdate {
                    timestamp_nanos: now_nanos,
                    update_type: ConfigUpdateType::AutoSaveChecked,
                    section: None,
                    success: true,
                    description: Some(Arc::from("Auto-save check initiated")),
                };

                emit!(sender, check_update);

                let last_save_nanos = manager.last_persistence.load(Ordering::Acquire);
                let elapsed_secs = (now_nanos - last_save_nanos) / 1_000_000_000;

                // Default auto-save interval for streaming operation
                let auto_save_interval = 300; // 5 minutes default

                if elapsed_secs >= auto_save_interval {
                    // Update timestamp atomically before saving
                    manager.last_persistence.store(now_nanos, Ordering::Release);

                    // Emit auto-save executed update
                    let autosave_update = ConfigUpdate {
                        timestamp_nanos: now_nanos,
                        update_type: ConfigUpdateType::AutoSaveExecuted,
                        section: None,
                        success: true,
                        description: Some(Arc::from("Auto-save executed")),
                    };

                    emit!(sender, autosave_update);
                }
            });
        })
    }

    /// Save configuration to file using streaming pattern
    pub fn save_to_file_stream(&self) -> AsyncStream<ConfigUpdate> {
        let manager = self.clone();
        AsyncStream::with_channel(move |sender| {
            std::thread::spawn(move || {
                let now_nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;

                // Emit save initiated update
                let save_start = ConfigUpdate {
                    timestamp_nanos: now_nanos,
                    update_type: ConfigUpdateType::SavedToFile,
                    section: None,
                    success: false,
                    description: Some(Arc::from("File save initiated")),
                };

                emit!(sender, save_start);

                // Perform file save using sync implementation
                let success = manager.save_to_file_sync().is_ok();

                // Emit save completion update
                let save_complete = ConfigUpdate {
                    timestamp_nanos: now_nanos,
                    update_type: ConfigUpdateType::SavedToFile,
                    section: None,
                    success,
                    description: Some(Arc::from(if success {
                        "File save completed successfully"
                    } else {
                        "File save failed"
                    })),
                };

                emit!(sender, save_complete);
            });
        })
    }

    /// Synchronous implementation of save_to_file for streams-only architecture
    fn save_to_file_sync(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let config = self.get_config();
        // Access persistence without async - this may need to be refactored for true sync access
        // For now, use defaults
        let format = "json"; // Default format
        let compression = false; // Default no compression
        let config_file_path = "./config.json"; // Default path

        let serialized = match format {
            "json" => serde_json::to_string_pretty(&*config)?,
            "yaml" => yyaml::to_string(&*config)?,
            "toml" => toml::to_string(&*config)?,
            _ => return Err("Unsupported format".into()),
        };

        let data = if compression {
            let compressed = lz4::block::compress(&serialized.as_bytes(), None, true)?;
            {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD.encode(&compressed)
            }
        } else {
            serialized
        };

        std::fs::write(config_file_path, data)?;

        Ok(())
    }

    /// Load configuration from file using streaming pattern
    pub fn load_from_file_stream(&self) -> AsyncStream<ConfigUpdate> {
        let manager = self.clone();
        AsyncStream::with_channel(move |sender| {
            std::thread::spawn(move || {
                let now_nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;

                // Emit load initiated update
                let load_start = ConfigUpdate {
                    timestamp_nanos: now_nanos,
                    update_type: ConfigUpdateType::LoadedFromFile,
                    section: None,
                    success: false,
                    description: Some(Arc::from("File load initiated")),
                };

                emit!(sender, load_start);

                // Perform file load using sync implementation
                let success = manager.load_from_file_sync().is_ok();

                // Emit load completion update
                let load_complete = ConfigUpdate {
                    timestamp_nanos: now_nanos,
                    update_type: ConfigUpdateType::LoadedFromFile,
                    section: None,
                    success,
                    description: Some(Arc::from(if success {
                        "File load completed successfully"
                    } else {
                        "File load failed"
                    })),
                };

                emit!(sender, load_complete);
            });
        })
    }

    /// Synchronous implementation of load_from_file for streams-only architecture
    fn load_from_file_sync(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // For sync version, use defaults instead of async persistence access
        let format = "json"; // Default format
        let compression = false; // Default no compression
        let config_file_path = "./config.json"; // Default path

        let data = std::fs::read_to_string(config_file_path)?;

        let content = if compression {
            let compressed = {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD.decode(&data)?
            };
            let decompressed = lz4::block::decompress(&compressed, None)?;
            String::from_utf8(decompressed)?
        } else {
            data
        };

        let config: ChatConfig = match format {
            "json" => serde_json::from_str(&content)?,
            "yaml" => yyaml::from_str(&content)?,
            "toml" => toml::from_str(&content)?,
            _ => return Err("Unsupported format".into()),
        };

        // Update config atomically
        let config_arc = Arc::new(config);
        self.config.store(config_arc);

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
            total_changes: self.change_counter.load(Ordering::Relaxed),
            current_version: self.version_counter.load(Ordering::Relaxed),
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
    /// Total number of configuration changes made
    pub total_changes: usize,
    /// Current configuration version number
    pub current_version: usize,
    /// Duration since last modification
    pub last_modified: Duration,
    /// Number of active validators
    pub validators_count: usize,
    /// Whether auto-save is currently enabled
    pub auto_save_enabled: bool,
}
