//! Configuration change events and related types
//!
//! This module defines events for configuration changes, validation errors,
//! and persistence settings.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Configuration change event with zero-allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationChangeEvent {
    /// Event ID
    pub id: Uuid,
    /// Configuration ID that changed
    pub config_id: Uuid,
    /// Change type
    pub change_type: ConfigurationChangeType,
    /// Field path that changed (e.g., "model.temperature")
    pub field_path: Arc<str>,
    /// Old value (JSON representation)
    pub old_value: Option<serde_json::Value>,
    /// New value (JSON representation)
    pub new_value: serde_json::Value,
    /// Timestamp of change
    pub timestamp: Instant,
    /// User who made the change
    pub user_id: Option<Arc<str>>,
    /// Source of the change ("ui", "api", "system")
    pub source: Arc<str>,
}

/// Configuration change type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfigurationChangeType {
    /// Configuration created
    Created,
    /// Configuration updated
    Updated,
    /// Configuration deleted
    Deleted,
    /// Configuration imported
    Imported,
    /// Configuration exported
    Exported,
    /// Configuration reset to defaults
    Reset,
}

/// Configuration validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigurationValidationError {
    /// Invalid value for field
    InvalidValue {
        field: Arc<str>,
        value: serde_json::Value,
        expected: Arc<str>,
    },
    /// Missing required field
    MissingField(Arc<str>),
    /// Field out of range
    OutOfRange {
        field: Arc<str>,
        value: f64,
        min: f64,
        max: f64,
    },
    /// Invalid combination of fields
    InvalidCombination {
        fields: Vec<Arc<str>>,
        reason: Arc<str>,
    },
    /// Custom validation error
    Custom(Arc<str>),
}

/// Configuration validation result
pub type ConfigurationValidationResult = Result<(), Vec<ConfigurationValidationError>>;

/// Configuration persistence settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationPersistence {
    /// Enable automatic saving
    pub auto_save: bool,
    /// Auto-save interval in seconds
    pub auto_save_interval_secs: u64,
    /// Save location ("file", "database", "memory")
    pub save_location: Arc<str>,
    /// File path for file-based persistence
    pub file_path: Option<Arc<str>>,
    /// Backup settings
    pub backup: BackupConfig,
    /// Encryption settings
    pub encryption: EncryptionConfig,
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Enable automatic backups
    pub enabled: bool,
    /// Backup interval in hours
    pub interval_hours: u32,
    /// Maximum number of backups to keep
    pub max_backups: u32,
    /// Backup location
    pub backup_path: Option<Arc<str>>,
    /// Compress backups
    pub compress: bool,
}

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Enable encryption
    pub enabled: bool,
    /// Encryption algorithm
    pub algorithm: Arc<str>,
    /// Key derivation method
    pub key_derivation: Arc<str>,
}

impl Default for ConfigurationPersistence {
    fn default() -> Self {
        Self {
            auto_save: true,
            auto_save_interval_secs: 30,
            save_location: Arc::from("file"),
            file_path: None,
            backup: BackupConfig::default(),
            encryption: EncryptionConfig::default(),
        }
    }
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_hours: 24,
            max_backups: 7,
            backup_path: None,
            compress: true,
        }
    }
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: Arc::from("AES-256-GCM"),
            key_derivation: Arc::from("PBKDF2"),
        }
    }
}

impl ConfigurationChangeEvent {
    /// Create a new configuration change event
    pub fn new(
        config_id: Uuid,
        change_type: ConfigurationChangeType,
        field_path: impl Into<Arc<str>>,
        old_value: Option<serde_json::Value>,
        new_value: serde_json::Value,
        user_id: Option<Arc<str>>,
        source: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            config_id,
            change_type,
            field_path: field_path.into(),
            old_value,
            new_value,
            timestamp: Instant::now(),
            user_id,
            source: source.into(),
        }
    }

    /// Get the age of this event
    pub fn age(&self) -> Duration {
        Instant::now().duration_since(self.timestamp)
    }

    /// Check if this event is recent (within specified duration)
    pub fn is_recent(&self, within: Duration) -> bool {
        self.age() <= within
    }
}

impl ConfigurationValidationError {
    /// Get a human-readable description of the error
    pub fn description(&self) -> String {
        match self {
            Self::InvalidValue { field, expected, .. } => {
                format!("Invalid value for field '{}': expected {}", field, expected)
            }
            Self::MissingField(field) => {
                format!("Missing required field: '{}'", field)
            }
            Self::OutOfRange { field, value, min, max } => {
                format!(
                    "Field '{}' value {} is out of range [{}, {}]",
                    field, value, min, max
                )
            }
            Self::InvalidCombination { fields, reason } => {
                format!(
                    "Invalid combination of fields [{}]: {}",
                    fields.iter().map(|f| f.as_ref()).collect::<Vec<_>>().join(", "),
                    reason
                )
            }
            Self::Custom(message) => message.to_string(),
        }
    }

    /// Get the field(s) involved in this error
    pub fn fields(&self) -> Vec<&str> {
        match self {
            Self::InvalidValue { field, .. } => vec![field.as_ref()],
            Self::MissingField(field) => vec![field.as_ref()],
            Self::OutOfRange { field, .. } => vec![field.as_ref()],
            Self::InvalidCombination { fields, .. } => {
                fields.iter().map(|f| f.as_ref()).collect()
            }
            Self::Custom(_) => vec![],
        }
    }

    /// Get the severity level of this error
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::MissingField(_) => ErrorSeverity::Error,
            Self::OutOfRange { .. } => ErrorSeverity::Error,
            Self::InvalidValue { .. } => ErrorSeverity::Warning,
            Self::InvalidCombination { .. } => ErrorSeverity::Error,
            Self::Custom(_) => ErrorSeverity::Warning,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Informational
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical error
    Critical,
}

impl std::fmt::Display for ConfigurationChangeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Created => write!(f, "created"),
            Self::Updated => write!(f, "updated"),
            Self::Deleted => write!(f, "deleted"),
            Self::Imported => write!(f, "imported"),
            Self::Exported => write!(f, "exported"),
            Self::Reset => write!(f, "reset"),
        }
    }
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "info"),
            Self::Warning => write!(f, "warning"),
            Self::Error => write!(f, "error"),
            Self::Critical => write!(f, "critical"),
        }
    }
}