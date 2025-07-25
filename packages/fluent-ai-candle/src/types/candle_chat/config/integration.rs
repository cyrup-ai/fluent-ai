//! Integration configuration for external services
//!
//! This module defines configuration structures for integrating with
//! external services, webhooks, plugins, and API management.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Integration configuration for external services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Enabled integrations
    pub enabled_integrations: Vec<String>,
    /// Integration settings
    pub integration_settings: HashMap<String, serde_json::Value>,
    /// API keys for integrations
    pub api_keys: HashMap<String, String>,
    /// Webhook configurations
    pub webhooks: Vec<WebhookConfig>,
    /// Plugin configurations
    pub plugins: Vec<PluginConfig>,
}

/// Webhook configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    /// Webhook name
    pub name: String,
    /// Webhook URL
    pub url: String,
    /// Events to trigger on
    pub events: Vec<String>,
    /// Authentication method
    pub auth_method: WebhookAuth,
    /// Enabled flag
    pub enabled: bool,
}

/// Webhook authentication method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebhookAuth {
    /// No authentication
    None,
    /// Bearer token
    Bearer(String),
    /// API key in header
    ApiKey(String, String),
    /// Basic authentication
    Basic(String, String),
}

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin settings
    pub settings: HashMap<String, serde_json::Value>,
    /// Plugin enabled flag
    pub enabled: bool,
    /// Plugin priority
    pub priority: i32,
}

/// History configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryConfig {
    /// Enable history storage
    pub enabled: bool,
    /// Maximum history entries
    pub max_entries: usize,
    /// History retention days
    pub retention_days: u64,
    /// Enable search in history
    pub enable_search: bool,
    /// Enable history export
    pub enable_export: bool,
    /// Compression for old entries
    pub compress_old_entries: bool,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable encryption
    pub enable_encryption: bool,
    /// Encryption algorithm
    pub encryption_algorithm: String,
    /// Enable audit logging
    pub enable_audit_logging: bool,
    /// Session security settings
    pub session_security: SessionSecurityConfig,
    /// Content security policy
    pub content_security_policy: Option<String>,
}

/// Session security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSecurityConfig {
    /// Require authentication
    pub require_auth: bool,
    /// Session token expiry minutes
    pub token_expiry_minutes: u64,
    /// Enable CSRF protection
    pub enable_csrf_protection: bool,
    /// Secure cookie settings
    pub secure_cookies: bool,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable response caching
    pub enable_caching: bool,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// Enable lazy loading
    pub enable_lazy_loading: bool,
    /// Message batch size
    pub message_batch_size: usize,
    /// Debounce delay for typing
    pub debounce_delay_ms: u64,
}

// Default implementations
impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enabled_integrations: Vec::new(),
            integration_settings: HashMap::new(),
            api_keys: HashMap::new(),
            webhooks: Vec::new(),
            plugins: Vec::new(),
        }
    }
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: 10000,
            retention_days: 365,
            enable_search: true,
            enable_export: true,
            compress_old_entries: true,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_encryption: true,
            encryption_algorithm: "AES-256-GCM".to_string(),
            enable_audit_logging: true,
            session_security: SessionSecurityConfig::default(),
            content_security_policy: None,
        }
    }
}

impl Default for SessionSecurityConfig {
    fn default() -> Self {
        Self {
            require_auth: false,
            token_expiry_minutes: 60,
            enable_csrf_protection: true,
            secure_cookies: true,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_size_limit: 1000,
            enable_lazy_loading: true,
            message_batch_size: 50,
            debounce_delay_ms: 300,
        }
    }
}