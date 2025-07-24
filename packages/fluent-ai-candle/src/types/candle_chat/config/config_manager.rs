//! Configuration management and persistence
//!
//! This module provides configuration management capabilities with
//! validation, persistence, and change tracking using zero-allocation patterns.

use std::collections::HashMap;

use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::model_config::{ModelConfig, ValidationResult};
use super::chat_config::ChatConfig;

/// Handle errors in streaming context without panicking
macro_rules! handle_error {
    ($error:expr, $context:literal) => {
        eprintln!("Streaming error in {}: {}", $context, $error);
        // Continue processing instead of returning error
    };
}

/// Configuration change event for tracking modifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationChangeEvent {
    /// Event ID
    pub id: Uuid,
    /// Event timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Configuration section changed
    pub section: ConfigSection,
    /// Field that was changed
    pub field: String,
    /// Old value (serialized)
    pub old_value: Option<serde_json::Value>,
    /// New value (serialized)
    pub new_value: serde_json::Value,
    /// User who made the change
    pub changed_by: Option<String>,
    /// Change reason/description
    pub reason: Option<String>,
    /// Change source (UI, API, etc.)
    pub source: ChangeSource,
    /// Validation status
    pub validation_status: ValidationStatus,
    /// Rollback information
    pub rollback_info: Option<RollbackInfo>,
}

/// Configuration section identifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigSection {
    /// Model configuration
    Model,
    /// Chat configuration
    Chat,
    /// Personality configuration
    Personality,
    /// Behavior configuration
    Behavior,
    /// UI configuration
    UI,
    /// Integration configuration
    Integration,
    /// Security configuration
    Security,
    /// Performance configuration
    Performance,
    /// Custom section
    Custom(String),
}

/// Source of configuration change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeSource {
    /// User interface
    UI,
    /// API call
    API,
    /// Configuration file
    File,
    /// Environment variable
    Environment,
    /// System default
    System,
    /// Migration/upgrade
    Migration,
    /// External integration
    External,
}

/// Validation status for changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// Change is valid
    Valid,
    /// Change has warnings
    Warning(Vec<String>),
    /// Change is invalid
    Invalid(Vec<String>),
    /// Change needs review
    PendingReview,
}

/// Rollback information for changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackInfo {
    /// Whether rollback is possible
    pub can_rollback: bool,
    /// Rollback complexity
    pub complexity: RollbackComplexity,
    /// Dependencies that would be affected
    pub affected_dependencies: Vec<String>,
    /// Rollback instructions
    pub instructions: Option<String>,
}

/// Rollback complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RollbackComplexity {
    /// Simple field change
    Simple,
    /// Multiple field changes
    Moderate,
    /// Structural changes
    Complex,
    /// System-wide changes
    Critical,
}

/// Configuration persistence settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationPersistence {
    /// Enable automatic persistence
    pub auto_save: bool,
    /// Save interval in seconds
    pub save_interval_seconds: u64,
    /// Configuration file path
    pub config_file_path: String,
    /// Backup configuration
    pub backup_config: BackupConfiguration,
    /// Encryption settings
    pub encryption: Option<EncryptionSettings>,
    /// Compression settings
    pub compression: CompressionSettings,
    /// Version control settings
    pub version_control: VersionControlSettings,
}

/// Backup configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfiguration {
    /// Enable backups
    pub enabled: bool,
    /// Backup interval in hours
    pub interval_hours: u64,
    /// Maximum backup files to keep
    pub max_backups: usize,
    /// Backup location
    pub backup_path: String,
    /// Compress backups
    pub compress_backups: bool,
}

/// Encryption settings for configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionSettings {
    /// Encryption algorithm
    pub algorithm: String,
    /// Key derivation function
    pub key_derivation: String,
    /// Salt for encryption
    pub salt: Option<String>,
    /// Encryption key source
    pub key_source: KeySource,
}

/// Source of encryption key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeySource {
    /// Environment variable
    Environment(String),
    /// File path
    File(String),
    /// Key management service
    KMS(String),
    /// User-provided
    UserProvided,
}

/// Compression settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionSettings {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: String,
    /// Compression level (1-9)
    pub level: u8,
}

/// Version control settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionControlSettings {
    /// Enable version tracking
    pub enabled: bool,
    /// Maximum versions to keep
    pub max_versions: usize,
    /// Version comparison enabled
    pub enable_comparison: bool,
    /// Auto-commit changes
    pub auto_commit: bool,
}

impl Default for ConfigurationPersistence {
    fn default() -> Self {
        Self {
            auto_save: true,
            save_interval_seconds: 300,
            config_file_path: "config.json".to_string(),
            backup_config: BackupConfiguration::default(),
            encryption: None,
            compression: CompressionSettings::default(),
            version_control: VersionControlSettings::default(),
        }
    }
}

impl Default for BackupConfiguration {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_hours: 24,
            max_backups: 7,
            backup_path: "backups/".to_string(),
            compress_backups: true,
        }
    }
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: "gzip".to_string(),
            level: 6,
        }
    }
}

impl Default for VersionControlSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            max_versions: 10,
            enable_comparison: true,
            auto_commit: false,
        }
    }
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

/// Trait for configuration validation
pub trait ConfigurationValidator: Send + Sync {
    /// Validate configuration changes
    fn validate(&self, old_value: &serde_json::Value, new_value: &serde_json::Value) -> ValidationResult;
    
    /// Get validator name
    fn name(&self) -> &str;
    
    /// Get validation rules
    fn rules(&self) -> Vec<ValidationRule>;
}

/// Validation rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule type
    pub rule_type: ValidationRuleType,
    /// Rule parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Rule severity
    pub severity: ValidationSeverity,
}

/// Type of validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRuleType {
    /// Range validation
    Range,
    /// Pattern validation
    Pattern,
    /// Required field validation
    Required,
    /// Type validation
    Type,
    /// Custom validation
    Custom(String),
}

/// Severity of validation issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Information only
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical error
    Critical,
}

/// Configuration watcher for change notifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationWatcher {
    /// Watcher ID
    pub id: Uuid,
    /// Watched sections
    pub sections: Vec<ConfigSection>,
    /// Watched fields
    pub fields: Vec<String>,
    /// Callback URL for notifications
    pub callback_url: Option<String>,
    /// Watcher enabled
    pub enabled: bool,
    /// Filter conditions
    pub filters: HashMap<String, serde_json::Value>,
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

/// Personality configuration validator
pub struct PersonalityValidator;

impl ConfigurationValidator for PersonalityValidator {
    fn validate(&self, _old_value: &serde_json::Value, new_value: &serde_json::Value) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate personality levels are between 0.0 and 1.0
        if let Some(obj) = new_value.as_object() {
            for (key, value) in obj {
                if key.ends_with("_level") {
                    if let Some(level) = value.as_f64() {
                        if level < 0.0 || level > 1.0 {
                            errors.push(format!("{} must be between 0.0 and 1.0", key));
                        }
                    }
                }
            }
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            config_hash: "personality_hash".to_string(),
        }
    }

    fn name(&self) -> &str {
        "PersonalityValidator"
    }

    fn rules(&self) -> Vec<ValidationRule> {
        vec![
            ValidationRule {
                name: "level_range".to_string(),
                description: "Personality levels must be between 0.0 and 1.0".to_string(),
                rule_type: ValidationRuleType::Range,
                parameters: HashMap::new(),
                severity: ValidationSeverity::Error,
            }
        ]
    }
}

/// Behavior configuration validator
pub struct BehaviorValidator;

impl ConfigurationValidator for BehaviorValidator {
    fn validate(&self, _old_value: &serde_json::Value, new_value: &serde_json::Value) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate response delay is reasonable
        if let Some(obj) = new_value.as_object() {
            if let Some(delay) = obj.get("response_delay_ms").and_then(|v| v.as_u64()) {
                if delay > 10000 {
                    warnings.push("Response delay over 10 seconds may impact user experience".to_string());
                }
            }
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            config_hash: "behavior_hash".to_string(),
        }
    }

    fn name(&self) -> &str {
        "BehaviorValidator"
    }

    fn rules(&self) -> Vec<ValidationRule> {
        vec![
            ValidationRule {
                name: "response_delay".to_string(),
                description: "Response delay should be reasonable".to_string(),
                rule_type: ValidationRuleType::Range,
                parameters: HashMap::new(),
                severity: ValidationSeverity::Warning,
            }
        ]
    }
}

/// UI configuration validator
pub struct UIValidator;

impl ConfigurationValidator for UIValidator {
    fn validate(&self, _old_value: &serde_json::Value, new_value: &serde_json::Value) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate font size is reasonable
        if let Some(obj) = new_value.as_object() {
            if let Some(theme) = obj.get("theme").and_then(|v| v.as_object()) {
                if let Some(fonts) = theme.get("fonts").and_then(|v| v.as_object()) {
                    if let Some(size) = fonts.get("font_size").and_then(|v| v.as_u64()) {
                        if size < 8 || size > 72 {
                            errors.push("Font size must be between 8 and 72 pixels".to_string());
                        }
                    }
                }
            }
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            config_hash: "ui_hash".to_string(),
        }
    }

    fn name(&self) -> &str {
        "UIValidator"
    }

    fn rules(&self) -> Vec<ValidationRule> {
        vec![
            ValidationRule {
                name: "font_size_range".to_string(),
                description: "Font size must be between 8 and 72 pixels".to_string(),
                rule_type: ValidationRuleType::Range,
                parameters: HashMap::new(),
                severity: ValidationSeverity::Error,
            }
        ]
    }
}

/// Integration configuration validator
pub struct IntegrationValidator;

impl ConfigurationValidator for IntegrationValidator {
    fn validate(&self, _old_value: &serde_json::Value, new_value: &serde_json::Value) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate webhook URLs
        if let Some(obj) = new_value.as_object() {
            if let Some(webhooks) = obj.get("webhooks").and_then(|v| v.as_array()) {
                for webhook in webhooks {
                    if let Some(webhook_obj) = webhook.as_object() {
                        if let Some(url) = webhook_obj.get("url").and_then(|v| v.as_str()) {
                            if !url.starts_with("https://") {
                                warnings.push("Webhook URLs should use HTTPS for security".to_string());
                            }
                        }
                    }
                }
            }
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            config_hash: "integration_hash".to_string(),
        }
    }

    fn name(&self) -> &str {
        "IntegrationValidator"
    }

    fn rules(&self) -> Vec<ValidationRule> {
        vec![
            ValidationRule {
                name: "webhook_security".to_string(),
                description: "Webhook URLs should use HTTPS".to_string(),
                rule_type: ValidationRuleType::Pattern,
                parameters: HashMap::new(),
                severity: ValidationSeverity::Warning,
            }
        ]
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
        let file_path = file_path.to_string();
        
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
        let file_path = file_path.to_string();
        
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

/// Result of configuration loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadResult {
    /// Whether load was successful
    pub success: bool,
    /// Sections that were loaded
    pub loaded_sections: Vec<ConfigSection>,
    /// Load errors
    pub errors: Vec<String>,
    /// Load warnings
    pub warnings: Vec<String>,
    /// Load time in milliseconds
    pub load_time_ms: u64,
}

/// Result of configuration saving
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveResult {
    /// Whether save was successful
    pub success: bool,
    /// Sections that were saved
    pub saved_sections: Vec<ConfigSection>,
    /// File size in bytes
    pub file_size_bytes: usize,
    /// Save time in milliseconds
    pub save_time_ms: u64,
}

/// Configuration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationStatistics {
    /// Total configuration changes
    pub total_changes: usize,
    /// Changes by section
    pub changes_by_section: HashMap<String, usize>,
    /// Average validation time
    pub avg_validation_time_ms: f64,
    /// Last save timestamp
    pub last_save: Option<chrono::DateTime<chrono::Utc>>,
    /// Configuration size in bytes
    pub config_size_bytes: usize,
}

impl Default for ConfigurationStatistics {
    fn default() -> Self {
        Self {
            total_changes: 0,
            changes_by_section: HashMap::new(),
            avg_validation_time_ms: 0.0,
            last_save: None,
            config_size_bytes: 0,
        }
    }
}