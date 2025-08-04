// Removed unused imports

use serde::{Deserialize, Serialize};

use super::database::DatabaseConfig;
use super::vector::VectorStoreConfig;
use crate::domain::chat::config::CandleModelConfig as ModelConfig;
// Removed unused import: use crate::completion::CompletionModel;
use super::super::cognitive::types::{CognitiveMemoryConfig, CognitiveProcessorConfig};

/// Comprehensive memory system configuration
///
/// This configuration brings together all memory subsystem configurations:
/// - Database configuration for persistent storage
/// - Vector store configuration for embeddings and similarity search
/// - LLM configuration for language model integration
/// - Cognitive configuration for advanced cognitive processing
/// - Performance and optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Database configuration for persistent memory storage
    pub database: DatabaseConfig,

    /// Vector store configuration for embeddings and similarity search
    pub vector_store: VectorStoreConfig,

    /// Provider configuration for completion services
    pub provider_model: ModelConfig,

    /// Cognitive memory configuration for advanced processing
    pub cognitive: CognitiveMemoryConfig,

    /// Cognitive processor configuration
    pub cognitive_processor: CognitiveProcessorConfig,

    /// Memory system performance configuration
    pub performance: MemoryPerformanceConfig,

    /// Memory retention and cleanup configuration
    pub retention: MemoryRetentionConfig,

    /// Security and access control configuration
    pub security: MemorySecurityConfig,

    /// Monitoring and observability configuration
    pub monitoring: MemoryMonitoringConfig}

/// Memory system performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPerformanceConfig {
    /// Maximum concurrent memory operations
    pub max_concurrent_operations: usize,

    /// Memory operation timeout in milliseconds
    pub operation_timeout_ms: u64,

    /// Cache size for frequently accessed memories
    pub cache_size: usize,

    /// Enable memory compression
    pub enable_compression: bool,

    /// Memory batch processing size
    pub batch_size: usize,

    /// Enable memory prefetching
    pub enable_prefetching: bool,

    /// Memory access optimization level (0-3)
    pub optimization_level: u8}

/// Memory retention and cleanup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRetentionConfig {
    /// Default memory retention period in seconds
    pub default_retention_seconds: u64,

    /// Maximum memory age before automatic cleanup in seconds
    pub max_age_seconds: u64,

    /// Memory cleanup interval in seconds
    pub cleanup_interval_seconds: u64,

    /// Enable automatic memory archiving
    pub enable_archiving: bool,

    /// Archive threshold (memories older than this are archived)
    pub archive_threshold_seconds: u64,

    /// Maximum number of memories to keep in active storage
    pub max_active_memories: usize,

    /// Memory importance threshold for retention decisions
    pub importance_threshold: f32}

/// Memory security and access control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySecurityConfig {
    /// Enable memory encryption at rest
    pub enable_encryption: bool,

    /// Encryption key derivation method
    pub key_derivation: KeyDerivationMethod,

    /// Enable access logging
    pub enable_access_logging: bool,

    /// Maximum failed access attempts before lockout
    pub max_failed_attempts: u32,

    /// Access lockout duration in seconds
    pub lockout_duration_seconds: u64,

    /// Enable memory integrity checks
    pub enable_integrity_checks: bool,

    /// Memory access permissions
    pub access_permissions: MemoryAccessPermissions}

/// Memory monitoring and observability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMonitoringConfig {
    /// Enable performance metrics collection
    pub enable_metrics: bool,

    /// Metrics collection interval in seconds
    pub metrics_interval_seconds: u64,

    /// Enable memory usage tracking
    pub enable_usage_tracking: bool,

    /// Enable error tracking and reporting
    pub enable_error_tracking: bool,

    /// Maximum number of metrics to retain
    pub max_metrics_history: usize,

    /// Enable health checks
    pub enable_health_checks: bool,

    /// Health check interval in seconds
    pub health_check_interval_seconds: u64}

/// Key derivation methods for memory encryption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyDerivationMethod {
    /// PBKDF2 with SHA-256
    PBKDF2,
    /// Argon2id (recommended)
    Argon2id,
    /// scrypt
    Scrypt}

/// Memory access permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAccessPermissions {
    /// Allow read access
    pub read: bool,
    /// Allow write access
    pub write: bool,
    /// Allow delete access
    pub delete: bool,
    /// Allow admin access
    pub admin: bool,
    /// Allowed user roles
    pub allowed_roles: Vec<String>}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            database: DatabaseConfig::default(),
            vector_store: VectorStoreConfig::default(),
            provider_model: ModelConfig::default(),
            cognitive: CognitiveMemoryConfig::default(),
            cognitive_processor: CognitiveProcessorConfig::default(),
            performance: MemoryPerformanceConfig::default(),
            retention: MemoryRetentionConfig::default(),
            security: MemorySecurityConfig::default(),
            monitoring: MemoryMonitoringConfig::default()}
    }
}

impl Default for MemoryPerformanceConfig {
    fn default() -> Self {
        Self {
            max_concurrent_operations: num_cpus::get() * 4,
            operation_timeout_ms: 30000, // 30 seconds
            cache_size: 10000,
            enable_compression: true,
            batch_size: 100,
            enable_prefetching: true,
            optimization_level: 2}
    }
}

impl Default for MemoryRetentionConfig {
    fn default() -> Self {
        Self {
            default_retention_seconds: 86400 * 30, // 30 days
            max_age_seconds: 86400 * 365,          // 1 year
            cleanup_interval_seconds: 3600,        // 1 hour
            enable_archiving: true,
            archive_threshold_seconds: 86400 * 90, // 90 days
            max_active_memories: 100000,
            importance_threshold: 0.5}
    }
}

impl Default for MemorySecurityConfig {
    fn default() -> Self {
        Self {
            enable_encryption: true,
            key_derivation: KeyDerivationMethod::Argon2id,
            enable_access_logging: true,
            max_failed_attempts: 5,
            lockout_duration_seconds: 300, // 5 minutes
            enable_integrity_checks: true,
            access_permissions: MemoryAccessPermissions::default()}
    }
}

impl Default for MemoryMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_interval_seconds: 60, // 1 minute
            enable_usage_tracking: true,
            enable_error_tracking: true,
            max_metrics_history: 10000,
            enable_health_checks: true,
            health_check_interval_seconds: 300, // 5 minutes
        }
    }
}

impl Default for MemoryAccessPermissions {
    fn default() -> Self {
        Self {
            read: true,
            write: true,
            delete: false,
            admin: false,
            allowed_roles: vec!["user".to_string()]}
    }
}

impl MemoryConfig {
    /// Create a new memory configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a memory configuration optimized for development
    pub fn development() -> Self {
        let mut config = Self::default();
        config.performance.max_concurrent_operations = 2;
        config.performance.cache_size = 1000;
        config.retention.max_active_memories = 10000;
        config.security.enable_encryption = false;
        config.monitoring.metrics_interval_seconds = 300; // 5 minutes
        config
    }

    /// Create a memory configuration optimized for production
    pub fn production() -> Self {
        let mut config = Self::default();
        config.performance.max_concurrent_operations = num_cpus::get() * 8;
        config.performance.cache_size = 50000;
        config.retention.max_active_memories = 1000000;
        config.security.enable_encryption = true;
        config.security.enable_integrity_checks = true;
        config.monitoring.metrics_interval_seconds = 30; // 30 seconds
        config
    }

    /// Validate the memory configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.performance.max_concurrent_operations == 0 {
            return Err("max_concurrent_operations must be greater than 0".to_string());
        }

        if self.performance.cache_size == 0 {
            return Err("cache_size must be greater than 0".to_string());
        }

        if self.retention.max_active_memories == 0 {
            return Err("max_active_memories must be greater than 0".to_string());
        }

        if self.retention.importance_threshold < 0.0 || self.retention.importance_threshold > 1.0 {
            return Err("importance_threshold must be between 0.0 and 1.0".to_string());
        }

        Ok(())
    }
}
