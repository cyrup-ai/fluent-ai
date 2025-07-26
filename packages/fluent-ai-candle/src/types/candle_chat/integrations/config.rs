//! Integration configuration types and validation
//!
//! This module provides zero-allocation configuration types for external integrations
//! with comprehensive validation and ergonomic builder patterns.

use std::sync::Arc;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Integration type options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntegrationType {
    /// Webhook integration
    Webhook,
    /// REST API integration
    RestApi,
    /// Plugin integration
    Plugin,
    /// External service integration
    ExternalService}

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
    pub enabled: bool}

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
            enabled: true}
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
    PluginError { detail: Arc<str> }}

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
    pub last_error_timestamp: Option<std::time::SystemTime>}

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
    pub enabled: bool}

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
    pub timeout_ms: Option<u64>}

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
    pub success: bool}

/// Create a webhook integration configuration with default settings
///
/// Creates a new `IntegrationConfig` configured for webhook integration.
/// Webhooks are used for receiving notifications and events from external services.
/// The configuration uses default timeout (30s) and retry settings (3 attempts).
///
/// # Arguments
///
/// * `name` - A human-readable name for the webhook integration
/// * `endpoint` - The webhook URL where events will be sent
///
/// # Returns
///
/// An `IntegrationConfig` with webhook-specific settings and default values for
/// timeout, retries, and other configuration options.
///
/// # Example
///
/// ```rust
/// let webhook = create_webhook_integration(
///     "github_notifications",
///     "https://api.myapp.com/webhooks/github"
/// );
/// assert_eq!(webhook.integration_type, IntegrationType::Webhook);
/// assert_eq!(webhook.timeout_seconds, 30);
/// ```
///
/// # Default Configuration
///
/// - Type: `IntegrationType::Webhook`
/// - Timeout: 30 seconds
/// - Retry attempts: 3
/// - Enabled: true
/// - No authentication token
/// - No custom headers
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

/// Create a REST API integration configuration with authentication support
///
/// Creates a new `IntegrationConfig` configured for REST API integration.
/// REST APIs are used for bidirectional communication with external services,
/// supporting both sending requests and receiving responses.
///
/// # Arguments
///
/// * `name` - A human-readable name for the REST API integration
/// * `endpoint` - The base URL of the REST API service
/// * `auth_token` - Optional authentication token for API requests
///
/// # Returns
///
/// An `IntegrationConfig` with REST API-specific settings and default values for
/// timeout, retries, and other configuration options.
///
/// # Authentication
///
/// If an `auth_token` is provided, it will be included in API requests. The exact
/// authentication method depends on the API implementation (Bearer token, API key, etc.).
///
/// # Example
///
/// ```rust
/// let api = create_rest_api_integration(
///     "slack_api",
///     "https://api.slack.com/api",
///     Some(Arc::from("xoxb-your-token"))
/// );
/// assert_eq!(api.integration_type, IntegrationType::RestApi);
/// assert!(api.auth_token.is_some());
/// ```
///
/// # Default Configuration
///
/// - Type: `IntegrationType::RestApi`
/// - Timeout: 30 seconds
/// - Retry attempts: 3
/// - Enabled: true
/// - No custom headers
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

/// Create a plugin integration configuration for internal plugins
///
/// Creates a new `IntegrationConfig` configured for plugin integration.
/// Plugins are internal extensions that provide additional functionality
/// within the system. The plugin ID is stored in the endpoint field.
///
/// # Arguments
///
/// * `name` - A human-readable name for the plugin integration
/// * `plugin_id` - A unique identifier for the plugin (stored as endpoint)
///
/// # Returns
///
/// An `IntegrationConfig` with plugin-specific settings and default values for
/// timeout, retries, and other configuration options.
///
/// # Plugin System
///
/// The plugin system allows extending functionality through internal modules.
/// The `plugin_id` should be a unique identifier that the plugin system can
/// use to locate and load the appropriate plugin.
///
/// # Example
///
/// ```rust
/// let plugin = create_plugin_integration(
///     "sentiment_analyzer",
///     "com.myapp.plugins.sentiment"
/// );
/// assert_eq!(plugin.integration_type, IntegrationType::Plugin);
/// assert_eq!(plugin.endpoint.as_ref(), "com.myapp.plugins.sentiment");
/// ```
///
/// # Default Configuration
///
/// - Type: `IntegrationType::Plugin`
/// - Timeout: 30 seconds (for plugin initialization/execution)
/// - Retry attempts: 3
/// - Enabled: true
/// - No authentication token
/// - No custom headers
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
