//! Integration types, traits, and configuration
//!
//! Provides zero-allocation integration type definitions with comprehensive
//! configuration options and ergonomic trait abstractions.

use std::sync::Arc;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Integration type options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegrationType {
    /// Webhook integration
    Webhook,
    /// REST API integration
    RestApi,
    /// Plugin integration
    Plugin,
    /// External service integration
    ExternalService,
}

/// Integration configuration with zero-allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Integration type
    pub integration_type: IntegrationType,
    /// Integration name
    pub name: Arc<str>,
    /// Endpoint URL or identifier
    pub endpoint: Arc<str>,
    /// Authentication token (if required)
    pub auth_token: Option<Arc<str>>,
    /// Custom headers
    pub headers: Vec<(Arc<str>, Arc<str>)>,
    /// Timeout in seconds
    pub timeout_seconds: u32,
    /// Retry attempts
    pub retry_attempts: u32,
    /// Enable integration
    pub enabled: bool,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            integration_type: IntegrationType::RestApi,
            name: Arc::from("default_integration"),
            endpoint: Arc::from(""),
            auth_token: None,
            headers: Vec::new(),
            timeout_seconds: 30,
            retry_attempts: 3,
            enabled: true,
        }
    }
}

/// Integration error types
#[derive(Error, Debug, Clone)]
pub enum IntegrationError {
    #[error("Connection error: {detail}")]
    ConnectionError { detail: Arc<str> },

    #[error("Authentication error: {detail}")]
    AuthenticationError { detail: Arc<str> },

    #[error("Timeout error: {detail}")]
    TimeoutError { detail: Arc<str> },

    #[error("Configuration error: {detail}")]
    ConfigurationError { detail: Arc<str> },

    #[error("Plugin error: {detail}")]
    PluginError { detail: Arc<str> },
}

/// Result type for integration operations
pub type IntegrationResult<T> = Result<T, IntegrationError>;

/// Integration statistics for monitoring and optimization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IntegrationStats {
    /// Total requests made
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: u64,
    /// Total bytes sent
    pub total_bytes_sent: u64,
    /// Total bytes received
    pub total_bytes_received: u64,
    /// Last successful request timestamp
    pub last_success_timestamp: Option<std::time::SystemTime>,
    /// Last error timestamp
    pub last_error_timestamp: Option<std::time::SystemTime>,
}

/// Plugin trait for external plugins
pub trait Plugin: Send + Sync + std::fmt::Debug {
    /// Plugin name
    fn name(&self) -> &str;

    /// Plugin version
    fn version(&self) -> &str;

    /// Initialize the plugin
    fn initialize(&mut self, config: &PluginConfig) -> IntegrationResult<()>;

    /// Execute plugin action
    fn execute(
        &self,
        action: &str,
        data: &serde_json::Value,
    ) -> IntegrationResult<serde_json::Value>;

    /// Cleanup plugin resources
    fn cleanup(&mut self) -> IntegrationResult<()>;
}

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Plugin identifier
    pub plugin_id: Arc<str>,
    /// Plugin name
    pub name: Arc<str>,
    /// Plugin version
    pub version: Arc<str>,
    /// Plugin settings
    pub settings: std::collections::HashMap<Arc<str>, serde_json::Value>,
    /// Enable plugin
    pub enabled: bool,
}

/// Integration request data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationRequest {
    /// Request method (GET, POST, etc.)
    pub method: Arc<str>,
    /// Request path or action
    pub path: Arc<str>,
    /// Request headers
    pub headers: std::collections::HashMap<Arc<str>, Arc<str>>,
    /// Request body
    pub body: Option<serde_json::Value>,
    /// Request timeout override
    pub timeout_ms: Option<u64>,
}

/// Integration response data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationResponse {
    /// Response status code
    pub status_code: u16,
    /// Response headers
    pub headers: std::collections::HashMap<Arc<str>, Arc<str>>,
    /// Response body
    pub body: Option<serde_json::Value>,
    /// Response time in milliseconds
    pub response_time_ms: u64,
    /// Success indicator
    pub success: bool,
}

/// Create a webhook integration configuration
pub fn create_webhook_integration(
    name: impl Into<Arc<str>>,
    endpoint: impl Into<Arc<str>>,
) -> IntegrationConfig {
    IntegrationConfig {
        integration_type: IntegrationType::Webhook,
        name: name.into(),
        endpoint: endpoint.into(),
        ..Default::default()
    }
}

/// Create a REST API integration configuration
pub fn create_rest_api_integration(
    name: impl Into<Arc<str>>,
    endpoint: impl Into<Arc<str>>,
    auth_token: Option<Arc<str>>,
) -> IntegrationConfig {
    IntegrationConfig {
        integration_type: IntegrationType::RestApi,
        name: name.into(),
        endpoint: endpoint.into(),
        auth_token,
        ..Default::default()
    }
}

/// Create a plugin integration configuration
pub fn create_plugin_integration(
    name: impl Into<Arc<str>>,
    plugin_id: impl Into<Arc<str>>,
) -> IntegrationConfig {
    IntegrationConfig {
        integration_type: IntegrationType::Plugin,
        name: name.into(),
        endpoint: plugin_id.into(),
        ..Default::default()
    }
}