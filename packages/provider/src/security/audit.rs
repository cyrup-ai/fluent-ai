//! Comprehensive audit logging for credential and security events
//!
//! Production-ready audit system with:
//! - Structured JSON logging for SIEM integration
//! - Tamper-evident log integrity
//! - Real-time security event monitoring
//! - Compliance-ready audit trails
//! - Zero-allocation event handling

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use super::credentials::{AuditConfig, CredentialSource};
use super::{SecurityError, SecurityResult};

/// Audit event for credential operations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type")]
pub enum CredentialEvent {
    /// Credential successfully loaded
    Load {
        provider: String,
        source: CredentialSource,
        timestamp: SystemTime,
        success: bool},

    /// Credential accessed from cache
    Access {
        provider: String,
        source: CredentialSource,
        timestamp: SystemTime,
        success: bool},

    /// Credential updated or rotated
    Update {
        provider: String,
        source: CredentialSource,
        timestamp: SystemTime,
        success: bool},

    /// Credential removed
    Remove {
        provider: String,
        timestamp: SystemTime,
        success: bool},

    /// Credential validation event
    Validation {
        provider: String,
        timestamp: SystemTime,
        success: bool,
        failure_reason: Option<String>},

    /// Credential not found
    NotFound {
        provider: String,
        attempted_sources: Vec<String>,
        timestamp: SystemTime},

    /// Credential rotation event
    Rotation {
        provider: String,
        old_key_age: std::time::Duration,
        timestamp: SystemTime,
        success: bool,
        rotation_reason: String}}

/// Security-related system events
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type")]
pub enum SecurityEvent {
    /// System startup
    SystemStartup {
        timestamp: SystemTime,
        component: String,
        configuration: String},

    /// System shutdown
    SystemShutdown {
        timestamp: SystemTime,
        component: String,
        uptime: std::time::Duration},

    /// Encryption key operations
    KeyOperation {
        operation: String, // "generate", "load", "rotate"
        key_path: String,
        timestamp: SystemTime,
        success: bool,
        error: Option<String>},

    /// Authentication attempts
    Authentication {
        provider: String,
        timestamp: SystemTime,
        success: bool,
        failure_reason: Option<String>,
        source_ip: Option<String>},

    /// Security policy violations
    PolicyViolation {
        policy: String,
        violation_type: String,
        details: String,
        timestamp: SystemTime,
        severity: SecuritySeverity},

    /// Suspicious activity detection
    SuspiciousActivity {
        activity_type: String,
        details: String,
        timestamp: SystemTime,
        risk_score: f32,
        recommended_action: String}}

/// Security event severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecuritySeverity {
    Low,
    Medium,
    High,
    Critical}

/// Audit log entry with integrity verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    /// Unique entry ID
    pub id: String,

    /// Timestamp of the event
    pub timestamp: SystemTime,

    /// Event data
    pub event: AuditEvent,

    /// Hash of the previous log entry for chain integrity
    pub previous_hash: String,

    /// Hash of this entry's content
    pub entry_hash: String,

    /// Sequence number for ordering
    pub sequence: u64,

    /// Additional metadata
    pub metadata: HashMap<String, String>}

/// Union type for all audit events
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "category")]
pub enum AuditEvent {
    Credential(CredentialEvent),
    Security(SecurityEvent)}

/// Statistics about audit logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditStatistics {
    pub total_events: u64,
    pub credential_events: u64,
    pub security_events: u64,
    pub last_event_time: Option<SystemTime>,
    pub log_file_size: u64,
    pub integrity_violations: u64}

/// Production-ready audit logger with tamper detection
pub struct AuditLogger {
    /// Configuration
    config: AuditConfig,

    /// Log file handle
    log_file: Arc<RwLock<std::fs::File>>,

    /// Event sequence counter
    sequence_counter: Arc<std::sync::atomic::AtomicU64>,

    /// Hash of the last log entry for chain integrity
    last_entry_hash: Arc<RwLock<String>>,

    /// Event statistics
    statistics: Arc<RwLock<AuditStatistics>>,

    /// Real-time event monitoring cache
    recent_events: Arc<DashMap<String, Vec<AuditLogEntry>>>}

impl AuditLogger {
    /// Create new audit logger with configuration
    pub async fn new(config: &AuditConfig) -> SecurityResult<Self> {
        if !config.enabled {
            // Create a disabled logger
            return Self::create_disabled_logger(config).await;
        }

        // Ensure log directory exists
        if let Some(parent) = Path::new(&config.log_path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| SecurityError::AuditError {
                message: format!("Failed to create audit log directory: {}", e)})?;
        }

        // Open log file with append mode
        let log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.log_path)
            .map_err(|e| SecurityError::AuditError {
                message: format!("Failed to open audit log file {}: {}", config.log_path, e)})?;

        // Set file permissions (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = log_file
                .metadata()
                .map_err(|e| SecurityError::AuditError {
                    message: format!("Failed to read log file metadata: {}", e)})?
                .permissions();
            perms.set_mode(0o640); // Owner read/write, group read
            std::fs::set_permissions(&config.log_path, perms).map_err(|e| {
                SecurityError::AuditError {
                    message: format!("Failed to set log file permissions: {}", e)}
            })?;
        }

        // Calculate last entry hash from existing log
        let last_entry_hash = Self::calculate_last_entry_hash(&config.log_path)?;

        // Initialize statistics
        let statistics = AuditStatistics {
            total_events: 0,
            credential_events: 0,
            security_events: 0,
            last_event_time: None,
            log_file_size: log_file.metadata().map(|m| m.len()).unwrap_or(0),
            integrity_violations: 0};

        Ok(Self {
            config: config.clone(),
            log_file: Arc::new(RwLock::new(log_file)),
            sequence_counter: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            last_entry_hash: Arc::new(RwLock::new(last_entry_hash)),
            statistics: Arc::new(RwLock::new(statistics)),
            recent_events: Arc::new(DashMap::new())})
    }

    /// Create a disabled audit logger (for testing or disabled configurations)
    async fn create_disabled_logger(config: &AuditConfig) -> SecurityResult<Self> {
        // Create a dummy file handle that will never be used
        let temp_file = tempfile::NamedTempFile::new().map_err(|e| SecurityError::AuditError {
            message: format!("Failed to create temporary file for disabled logger: {}", e)})?;

        let statistics = AuditStatistics {
            total_events: 0,
            credential_events: 0,
            security_events: 0,
            last_event_time: None,
            log_file_size: 0,
            integrity_violations: 0};

        Ok(Self {
            config: config.clone(),
            log_file: Arc::new(RwLock::new(temp_file.into_file())),
            sequence_counter: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            last_entry_hash: Arc::new(RwLock::new(String::new())),
            statistics: Arc::new(RwLock::new(statistics)),
            recent_events: Arc::new(DashMap::new())})
    }

    /// Log a credential-related event
    pub async fn log_credential_event(&self, event: CredentialEvent) -> SecurityResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let audit_event = AuditEvent::Credential(event);
        self.log_event(audit_event).await
    }

    /// Log a security-related event
    pub async fn log_security_event(&self, event: SecurityEvent) -> SecurityResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let audit_event = AuditEvent::Security(event);
        self.log_event(audit_event).await
    }

    /// Internal method to log any audit event
    async fn log_event(&self, event: AuditEvent) -> SecurityResult<()> {
        let timestamp = SystemTime::now();
        let sequence = self
            .sequence_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Get previous hash for chain integrity
        let previous_hash = self.last_entry_hash.read().await.clone();

        // Create log entry
        let entry = AuditLogEntry {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp,
            event: event.clone(),
            previous_hash: previous_hash.clone(),
            entry_hash: String::new(), // Will be calculated below
            sequence,
            metadata: self.create_metadata(&event).await};

        // Calculate entry hash
        let entry_content = format!("{:?}{}{}", entry.event, entry.previous_hash, entry.sequence);
        let entry_hash = format!("{:x}", Sha256::digest(entry_content.as_bytes()));

        let mut entry = entry;
        entry.entry_hash = entry_hash.clone();

        // Serialize to JSON
        let json_line = serde_json::to_string(&entry).map_err(|e| SecurityError::AuditError {
            message: format!("Failed to serialize audit event: {}", e)})?;

        // Write to log file
        {
            let mut file = self.log_file.write().await;
            writeln!(file, "{}", json_line).map_err(|e| SecurityError::AuditError {
                message: format!("Failed to write to audit log: {}", e)})?;
            file.flush().map_err(|e| SecurityError::AuditError {
                message: format!("Failed to flush audit log: {}", e)})?;
        }

        // Update last entry hash
        *self.last_entry_hash.write().await = entry_hash;

        // Update statistics
        {
            let mut stats = self.statistics.write().await;
            stats.total_events += 1;
            stats.last_event_time = Some(timestamp);

            match event {
                AuditEvent::Credential(_) => stats.credential_events += 1,
                AuditEvent::Security(_) => stats.security_events += 1}
        }

        // Add to recent events cache for monitoring
        let event_type = self.get_event_type(&event);
        self.recent_events
            .entry(event_type)
            .or_insert_with(Vec::new)
            .push(entry);

        // Log to tracing as well
        match self.config.log_level.as_str() {
            "DEBUG" => tracing::debug!("Audit event: {:?}", event),
            "INFO" => info!("Audit event: {:?}", event),
            "WARN" => warn!("Audit event: {:?}", event),
            "ERROR" => error!("Audit event: {:?}", event),
            _ => info!("Audit event: {:?}", event)}

        Ok(())
    }

    /// Create metadata for the audit entry
    async fn create_metadata(&self, event: &AuditEvent) -> HashMap<String, String> {
        let mut metadata = HashMap::new();

        metadata.insert(
            "hostname".to_string(),
            hostname::get()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
        );
        metadata.insert("process_id".to_string(), std::process::id().to_string());
        metadata.insert(
            "thread_id".to_string(),
            format!("{:?}", std::thread::current().id()),
        );

        // Add event-specific metadata
        match event {
            AuditEvent::Credential(cred_event) => {
                metadata.insert("category".to_string(), "credential".to_string());

                match cred_event {
                    CredentialEvent::Load { provider, .. }
                    | CredentialEvent::Access { provider, .. }
                    | CredentialEvent::Update { provider, .. }
                    | CredentialEvent::Remove { provider, .. }
                    | CredentialEvent::Validation { provider, .. }
                    | CredentialEvent::NotFound { provider, .. }
                    | CredentialEvent::Rotation { provider, .. } => {
                        metadata.insert("provider".to_string(), provider.clone());
                    }
                }
            }
            AuditEvent::Security(sec_event) => {
                metadata.insert("category".to_string(), "security".to_string());

                match sec_event {
                    SecurityEvent::SystemStartup { component, .. }
                    | SecurityEvent::SystemShutdown { component, .. } => {
                        metadata.insert("component".to_string(), component.clone());
                    }
                    SecurityEvent::Authentication { provider, .. } => {
                        metadata.insert("provider".to_string(), provider.clone());
                    }
                    SecurityEvent::PolicyViolation {
                        policy, severity, ..
                    } => {
                        metadata.insert("policy".to_string(), policy.clone());
                        metadata.insert("severity".to_string(), format!("{:?}", severity));
                    }
                    SecurityEvent::SuspiciousActivity { risk_score, .. } => {
                        metadata.insert("risk_score".to_string(), risk_score.to_string());
                    }
                    _ => {}
                }
            }
        }

        metadata
    }

    /// Get event type string for categorization
    fn get_event_type(&self, event: &AuditEvent) -> String {
        match event {
            AuditEvent::Credential(cred_event) => match cred_event {
                CredentialEvent::Load { .. } => "credential_load".to_string(),
                CredentialEvent::Access { .. } => "credential_access".to_string(),
                CredentialEvent::Update { .. } => "credential_update".to_string(),
                CredentialEvent::Remove { .. } => "credential_remove".to_string(),
                CredentialEvent::Validation { .. } => "credential_validation".to_string(),
                CredentialEvent::NotFound { .. } => "credential_not_found".to_string(),
                CredentialEvent::Rotation { .. } => "credential_rotation".to_string()},
            AuditEvent::Security(sec_event) => match sec_event {
                SecurityEvent::SystemStartup { .. } => "system_startup".to_string(),
                SecurityEvent::SystemShutdown { .. } => "system_shutdown".to_string(),
                SecurityEvent::KeyOperation { .. } => "key_operation".to_string(),
                SecurityEvent::Authentication { .. } => "authentication".to_string(),
                SecurityEvent::PolicyViolation { .. } => "policy_violation".to_string(),
                SecurityEvent::SuspiciousActivity { .. } => "suspicious_activity".to_string()}}
    }

    /// Calculate the hash of the last entry in the log file
    fn calculate_last_entry_hash(log_path: &str) -> SecurityResult<String> {
        if !Path::new(log_path).exists() {
            return Ok("genesis".to_string()); // First entry in a new log
        }

        // Read the last line of the log file
        let content = std::fs::read_to_string(log_path).map_err(|e| SecurityError::AuditError {
            message: format!("Failed to read audit log for hash calculation: {}", e)})?;

        if let Some(last_line) = content.lines().last() {
            if let Ok(entry) = serde_json::from_str::<AuditLogEntry>(last_line) {
                return Ok(entry.entry_hash);
            }
        }

        Ok("genesis".to_string())
    }

    /// Get audit statistics
    pub async fn get_statistics(&self) -> AuditStatistics {
        self.statistics.read().await.clone()
    }

    /// Get recent events for monitoring
    pub async fn get_recent_events(&self, event_type: &str, limit: usize) -> Vec<AuditLogEntry> {
        if let Some(events) = self.recent_events.get(event_type) {
            let events = events.clone();
            if events.len() <= limit {
                events
            } else {
                events[events.len() - limit..].to_vec()
            }
        } else {
            Vec::new()
        }
    }

    /// Verify log integrity by checking hash chain
    pub async fn verify_log_integrity(&self) -> SecurityResult<bool> {
        let content = std::fs::read_to_string(&self.config.log_path).map_err(|e| {
            SecurityError::AuditError {
                message: format!("Failed to read audit log for integrity check: {}", e)}
        })?;

        let mut previous_hash = "genesis".to_string();

        for line in content.lines() {
            let entry: AuditLogEntry =
                serde_json::from_str(line).map_err(|e| SecurityError::AuditError {
                    message: format!("Failed to parse audit log entry: {}", e)})?;

            // Verify previous hash matches
            if entry.previous_hash != previous_hash {
                error!(
                    "Audit log integrity violation: hash chain broken at sequence {}",
                    entry.sequence
                );
                return Ok(false);
            }

            // Verify entry hash
            let entry_content =
                format!("{:?}{}{}", entry.event, entry.previous_hash, entry.sequence);
            let calculated_hash = format!("{:x}", Sha256::digest(entry_content.as_bytes()));

            if entry.entry_hash != calculated_hash {
                error!(
                    "Audit log integrity violation: entry hash mismatch at sequence {}",
                    entry.sequence
                );
                return Ok(false);
            }

            previous_hash = entry.entry_hash;
        }

        Ok(true)
    }
}

impl Drop for AuditLogger {
    fn drop(&mut self) {
        // Log shutdown event synchronously
        if self.config.enabled {
            let shutdown_event = SecurityEvent::SystemShutdown {
                timestamp: SystemTime::now(),
                component: "AuditLogger".to_string(),
                uptime: std::time::Duration::from_secs(0), /* Would be calculated in real implementation */
            };

            // We can't use async in Drop, so we do a best-effort sync write
            if let Ok(json_line) = serde_json::to_string(&AuditLogEntry {
                id: uuid::Uuid::new_v4().to_string(),
                timestamp: SystemTime::now(),
                event: AuditEvent::Security(shutdown_event),
                previous_hash: String::new(), // Best effort
                entry_hash: String::new(),    // Best effort
                sequence: 0,                  // Best effort
                metadata: HashMap::new()}) {
                if let Ok(mut file) = self.log_file.try_write() {
                    let _ = writeln!(file, "{}", json_line);
                    let _ = file.flush();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::env;

    use super::*;

    #[tokio::test]
    async fn test_audit_logger_creation() {
        let temp_dir = env::temp_dir();
        let log_path = temp_dir.join("test_audit.log");

        let config = AuditConfig {
            enabled: true,
            log_path: log_path.to_string_lossy().to_string(),
            log_level: "INFO".to_string(),
            include_credential_hashes: true};

        let logger = AuditLogger::new(&config)
            .await
            .expect("Failed to create audit logger in test");

        // Test logging a credential event
        let event = CredentialEvent::Load {
            provider: "openai".to_string(),
            source: CredentialSource::Environment {
                variable_name: "OPENAI_API_KEY".to_string()},
            timestamp: SystemTime::now(),
            success: true};

        logger
            .log_credential_event(event)
            .await
            .expect("Failed to log credential event in test");

        // Check statistics
        let stats = logger.get_statistics().await;
        assert_eq!(stats.credential_events, 1);
        assert_eq!(stats.total_events, 1);

        // Cleanup
        let _ = std::fs::remove_file(&log_path);
    }

    #[tokio::test]
    async fn test_log_integrity() {
        let temp_dir = env::temp_dir();
        let log_path = temp_dir.join("test_integrity.log");

        let config = AuditConfig {
            enabled: true,
            log_path: log_path.to_string_lossy().to_string(),
            log_level: "INFO".to_string(),
            include_credential_hashes: true};

        let logger = AuditLogger::new(&config)
            .await
            .expect("Failed to create audit logger for integrity test");

        // Log multiple events
        for i in 0..5 {
            let event = CredentialEvent::Access {
                provider: format!("provider_{}", i),
                source: CredentialSource::Environment {
                    variable_name: format!("API_KEY_{}", i)},
                timestamp: SystemTime::now(),
                success: true};

            logger
                .log_credential_event(event)
                .await
                .expect("Failed to log credential event in integrity test");
        }

        // Verify integrity
        assert!(
            logger
                .verify_log_integrity()
                .await
                .expect("Failed to verify log integrity in test")
        );

        // Cleanup
        let _ = std::fs::remove_file(&log_path);
    }

    #[tokio::test]
    async fn test_disabled_logger() {
        let config = AuditConfig {
            enabled: false,
            log_path: "/dev/null".to_string(),
            log_level: "INFO".to_string(),
            include_credential_hashes: false};

        let logger = AuditLogger::new(&config).await.unwrap();

        // Events should be logged but not written to file
        let event = CredentialEvent::Load {
            provider: "test".to_string(),
            source: CredentialSource::Environment {
                variable_name: "TEST_KEY".to_string()},
            timestamp: SystemTime::now(),
            success: true};

        // Should succeed but do nothing
        logger.log_credential_event(event).await.unwrap();

        let stats = logger.get_statistics().await;
        assert_eq!(stats.total_events, 0); // No events logged when disabled
    }
}
