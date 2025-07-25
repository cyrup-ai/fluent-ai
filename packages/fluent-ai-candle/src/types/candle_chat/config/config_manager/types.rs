//! Configuration types and events
//!
//! Core types for configuration management including events, persistence settings,
//! and result structures with zero-allocation patterns.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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

// Default implementations
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