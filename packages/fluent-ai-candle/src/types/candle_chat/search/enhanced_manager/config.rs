//! Configuration types for enhanced history manager
//!
//! This module defines all configuration structures and their defaults
//! for the enhanced history management system.

use serde::{Deserialize, Serialize};

/// Configuration for enhanced history manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagerConfig {
    /// Enable caching
    pub enable_caching: bool,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Enable automatic indexing
    pub auto_indexing: bool,
    /// Indexing batch size
    pub indexing_batch_size: usize,
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
    /// Cleanup interval in seconds
    pub cleanup_interval_seconds: u64,
    /// Memory limit in bytes
    pub memory_limit_bytes: usize,
    /// Enable compression
    pub enable_compression: bool,
    /// Backup configuration
    pub backup_config: BackupConfig}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_size_limit: 1000,
            cache_ttl_seconds: 3600,
            auto_indexing: true,
            indexing_batch_size: 100,
            enable_performance_monitoring: true,
            cleanup_interval_seconds: 300,
            memory_limit_bytes: 1024 * 1024 * 100, // 100MB
            enable_compression: false,
            backup_config: BackupConfig::default()}
    }
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Enable automatic backups
    pub enabled: bool,
    /// Backup interval in hours
    pub interval_hours: u64,
    /// Maximum backup files to keep
    pub max_backups: usize,
    /// Backup compression
    pub compress_backups: bool,
    /// Backup location
    pub backup_path: Option<String>}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_hours: 24,
            max_backups: 7,
            compress_backups: true,
            backup_path: None}
    }
}

/// Cleanup result information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupResult {
    /// Number of messages cleaned
    pub messages_cleaned: usize,
    /// Number of cache entries removed
    pub cache_entries_removed: usize,
    /// Memory freed in bytes
    pub memory_freed: usize,
    /// Time taken for cleanup in milliseconds
    pub cleanup_time_ms: u64}

/// Optimization result information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Index entries optimized
    pub index_entries_optimized: usize,
    /// Memory usage before optimization
    pub memory_before: usize,
    /// Memory usage after optimization
    pub memory_after: usize,
    /// Time taken for optimization in milliseconds
    pub optimization_time_ms: u64}

/// Builder pattern for history manager configuration
#[derive(Debug, Default)]
pub struct HistoryManagerBuilder {
    config: ManagerConfig}

impl HistoryManagerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: ManagerConfig::default()}
    }

    /// Enable or disable caching
    pub fn with_caching(mut self, enabled: bool) -> Self {
        self.config.enable_caching = enabled;
        self
    }

    /// Set cache size limit
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.config.cache_size_limit = size;
        self
    }

    /// Set cache TTL
    pub fn with_cache_ttl(mut self, ttl_seconds: u64) -> Self {
        self.config.cache_ttl_seconds = ttl_seconds;
        self
    }

    /// Enable or disable auto indexing
    pub fn with_auto_indexing(mut self, enabled: bool) -> Self {
        self.config.auto_indexing = enabled;
        self
    }

    /// Set indexing batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.config.indexing_batch_size = batch_size;
        self
    }

    /// Enable or disable performance monitoring
    pub fn with_performance_monitoring(mut self, enabled: bool) -> Self {
        self.config.enable_performance_monitoring = enabled;
        self
    }

    /// Set cleanup interval
    pub fn with_cleanup_interval(mut self, interval_seconds: u64) -> Self {
        self.config.cleanup_interval_seconds = interval_seconds;
        self
    }

    /// Set memory limit
    pub fn with_memory_limit(mut self, limit_bytes: usize) -> Self {
        self.config.memory_limit_bytes = limit_bytes;
        self
    }

    /// Configure backup settings
    pub fn with_backup_config(mut self, backup_config: BackupConfig) -> Self {
        self.config.backup_config = backup_config;
        self
    }

    /// Build the configuration
    pub fn build(self) -> ManagerConfig {
        self.config
    }
}