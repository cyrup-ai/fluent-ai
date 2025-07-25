//! Secure credential management for AI provider API keys
//!
//! This module provides production-ready credential management with:
//! - Environment variable configuration with validation
//! - Encrypted credential storage using ChaCha20Poly1305
//! - Key rotation with audit logging
//! - Zero-allocation string handling for sensitive data
//! - Automatic key expiration and refresh
//! - Comprehensive audit trail for compliance

pub mod audit;
pub mod credentials;
pub mod encryption;
pub mod rotation;

pub use audit::{AuditLogger, CredentialEvent, SecurityEvent};
pub use credentials::{CredentialConfig, CredentialManager, SecureCredential};
pub use encryption::EncryptionEngine;
pub use rotation::{KeyRotationScheduler, RotationPolicy};
use thiserror::Error;

/// Security-related errors
#[derive(Error, Debug)]
pub enum SecurityError {
    #[error("Credential not found: {credential_name}")]
    CredentialNotFound { credential_name: String },

    #[error("Encryption error: {message}")]
    EncryptionError { message: String },

    #[error("Key rotation failed: {reason}")]
    RotationFailed { reason: String },

    #[error("Audit logging error: {message}")]
    AuditError { message: String },

    #[error("Configuration error: {field} - {message}")]
    ConfigurationError { field: String, message: String },

    #[error("Validation error: {message}")]
    ValidationError { message: String },

    #[error("Access denied: {reason}")]
    AccessDenied { reason: String }}

/// Result type for security operations
pub type SecurityResult<T> = Result<T, SecurityError>;
