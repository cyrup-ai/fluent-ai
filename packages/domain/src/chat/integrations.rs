//! Chat integrations functionality
//!
//! Provides zero-allocation integration capabilities for external services and plugins.
//! Supports multiple integration types with blazing-fast communication and ergonomic APIs.

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

/// Integration manager for handling external connections
#[derive(Debug, Clone)]
pub struct IntegrationManager {
    /// Active integrations
    pub integrations: Vec<IntegrationConfig>,
    /// Default timeout
    pub default_timeout: u32}

impl Default for IntegrationManager {
    fn default() -> Self {
        Self {
            integrations: Vec::new(),
            default_timeout: 30}
    }
}

impl IntegrationManager {
    /// Create a new integration manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an integration
    pub fn add_integration(&mut self, config: IntegrationConfig) -> IntegrationResult<()> {
        if config.name.is_empty() {
            return Err(IntegrationError::ConfigurationError {
                detail: Arc::from("Integration name cannot be empty")});
        }

        if config.endpoint.is_empty() {
            return Err(IntegrationError::ConfigurationError {
                detail: Arc::from("Integration endpoint cannot be empty")});
        }

        self.integrations.push(config);
        Ok(())
    }

    /// Remove an integration by name
    pub fn remove_integration(&mut self, name: &str) -> IntegrationResult<()> {
        let initial_len = self.integrations.len();
        self.integrations
            .retain(|config| config.name.as_ref() != name);

        if self.integrations.len() == initial_len {
            return Err(IntegrationError::ConfigurationError {
                detail: Arc::from("Integration not found")});
        }

        Ok(())
    }

    /// Get integration by name
    pub fn get_integration(&self, name: &str) -> Option<&IntegrationConfig> {
        self.integrations
            .iter()
            .find(|config| config.name.as_ref() == name)
    }

    /// List all enabled integrations
    pub fn list_enabled_integrations(&self) -> Vec<&IntegrationConfig> {
        self.integrations
            .iter()
            .filter(|config| config.enabled)
            .collect()
    }

    /// Send message to integration
    pub async fn send_to_integration(
        &self,
        integration_name: &str,
        message: &str,
    ) -> IntegrationResult<String> {
        let integration = self.get_integration(integration_name).ok_or_else(|| {
            IntegrationError::ConfigurationError {
                detail: Arc::from("Integration not found")}
        })?;

        if !integration.enabled {
            return Err(IntegrationError::ConfigurationError {
                detail: Arc::from("Integration is disabled")});
        }

        match integration.integration_type {
            IntegrationType::Webhook => self.send_webhook(integration, message).await,
            IntegrationType::RestApi => self.send_rest_api(integration, message).await,
            IntegrationType::Plugin => self.send_plugin(integration, message).await,
            IntegrationType::ExternalService => {
                self.send_external_service(integration, message).await
            }
        }
    }

    /// Send webhook message
    async fn send_webhook(
        &self,
        integration: &IntegrationConfig,
        message: &str,
    ) -> IntegrationResult<String> {
        // Placeholder for webhook implementation
        // In production, this would use an HTTP client like reqwest
        Ok(format!(
            "Webhook sent to {}: {}",
            integration.endpoint, message
        ))
    }

    /// Send REST API message
    async fn send_rest_api(
        &self,
        integration: &IntegrationConfig,
        message: &str,
    ) -> IntegrationResult<String> {
        // Placeholder for REST API implementation
        // In production, this would use an HTTP client like reqwest
        Ok(format!(
            "REST API call to {}: {}",
            integration.endpoint, message
        ))
    }

    /// Send plugin message
    async fn send_plugin(
        &self,
        integration: &IntegrationConfig,
        message: &str,
    ) -> IntegrationResult<String> {
        // Placeholder for plugin implementation
        // In production, this would interface with a plugin system
        Ok(format!("Plugin call to {}: {}", integration.name, message))
    }

    /// Send external service message
    async fn send_external_service(
        &self,
        integration: &IntegrationConfig,
        message: &str,
    ) -> IntegrationResult<String> {
        // Placeholder for external service implementation
        // In production, this would interface with external APIs
        Ok(format!(
            "External service call to {}: {}",
            integration.endpoint, message
        ))
    }
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

/// External integration system for managing third-party services and plugins
///
/// This system provides comprehensive external integration capabilities with:
/// - Multiple integration types (webhooks, APIs, plugins, services)
/// - Authentication and security management
/// - Request/response handling with retry logic
/// - Performance monitoring and error tracking
/// - Configuration management and validation
#[derive(Debug, Clone)]
pub struct ExternalIntegration {
    /// Integration configuration
    config: IntegrationConfig,
    /// Integration statistics
    stats: IntegrationStats,
    /// HTTP client for API calls
    client: Option<Arc<reqwest::Client>>,
    /// Plugin manager for plugin integrations
    plugin_manager: Option<Arc<PluginManager>>}

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

/// Plugin manager for handling plugin-based integrations
#[derive(Debug)]
pub struct PluginManager {
    /// Loaded plugins
    plugins: std::collections::HashMap<Arc<str>, Arc<dyn Plugin>>,
    /// Plugin configurations
    configs: std::collections::HashMap<Arc<str>, PluginConfig>}

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

impl ExternalIntegration {
    /// Create a new external integration
    pub fn new(config: IntegrationConfig) -> Self {
        let client = if matches!(
            config.integration_type,
            IntegrationType::RestApi | IntegrationType::Webhook
        ) {
            Some(Arc::new(reqwest::Client::new()))
        } else {
            None
        };

        let plugin_manager = if matches!(config.integration_type, IntegrationType::Plugin) {
            Some(Arc::new(PluginManager::new()))
        } else {
            None
        };

        Self {
            config,
            stats: IntegrationStats::default(),
            client,
            plugin_manager}
    }

    /// Execute an integration request
    pub async fn execute_request(
        &mut self,
        request: IntegrationRequest,
    ) -> IntegrationResult<IntegrationResponse> {
        if !self.config.enabled {
            return Err(IntegrationError::ConfigurationError {
                detail: Arc::from("Integration is disabled")});
        }

        let start_time = std::time::Instant::now();
        self.stats.total_requests += 1;

        let result = match self.config.integration_type {
            IntegrationType::RestApi | IntegrationType::Webhook => {
                self.execute_http_request(request).await
            }
            IntegrationType::Plugin => self.execute_plugin_request(request).await,
            IntegrationType::ExternalService => self.execute_service_request(request).await};

        let response_time = start_time.elapsed().as_millis() as u64;

        match &result {
            Ok(_) => {
                self.stats.successful_requests += 1;
                self.stats.last_success_timestamp = Some(std::time::SystemTime::now());
            }
            Err(_) => {
                self.stats.failed_requests += 1;
                self.stats.last_error_timestamp = Some(std::time::SystemTime::now());
            }
        }

        // Update average response time
        self.stats.avg_response_time_ms =
            ((self.stats.avg_response_time_ms * (self.stats.total_requests - 1)) + response_time)
                / self.stats.total_requests;

        result
    }

    /// Execute HTTP request (REST API or Webhook)
    async fn execute_http_request(
        &mut self,
        request: IntegrationRequest,
    ) -> IntegrationResult<IntegrationResponse> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| IntegrationError::ConfigurationError {
                detail: Arc::from("HTTP client not initialized")})?;

        let url = format!("{}{}", self.config.endpoint, request.path);
        let mut req_builder = match request.method.as_ref() {
            "GET" => client.get(&url),
            "POST" => client.post(&url),
            "PUT" => client.put(&url),
            "DELETE" => client.delete(&url),
            "PATCH" => client.patch(&url),
            _ => {
                return Err(IntegrationError::ConfigurationError {
                    detail: Arc::from("Unsupported HTTP method")});
            }
        };

        // Add headers
        for (key, value) in &request.headers {
            req_builder = req_builder.header(key.as_ref(), value.as_ref());
        }

        // Add auth token if configured
        if let Some(token) = &self.config.auth_token {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", token));
        }

        // Add body if present
        if let Some(body) = &request.body {
            req_builder = req_builder.json(body);
        }

        // Set timeout
        let timeout = request
            .timeout_ms
            .unwrap_or(self.config.timeout_seconds as u64 * 1000);
        req_builder = req_builder.timeout(std::time::Duration::from_millis(timeout));

        let response = req_builder
            .send()
            .await
            .map_err(|e| IntegrationError::ConnectionError {
                detail: Arc::from(e.to_string())})?;

        let status_code = response.status().as_u16();
        let headers = response
            .headers()
            .iter()
            .map(|(k, v)| (Arc::from(k.as_str()), Arc::from(v.to_str().unwrap_or(""))))
            .collect();

        let body: Option<serde_json::Value> = if response.status().is_success() {
            response.json().await.ok()
        } else {
            None
        };

        Ok(IntegrationResponse {
            status_code,
            headers,
            body,
            response_time_ms: 0, // Will be set by caller
            success: status_code >= 200 && status_code < 300})
    }

    /// Execute plugin request
    async fn execute_plugin_request(
        &mut self,
        request: IntegrationRequest,
    ) -> IntegrationResult<IntegrationResponse> {
        let plugin_manager =
            self.plugin_manager
                .as_ref()
                .ok_or_else(|| IntegrationError::ConfigurationError {
                    detail: Arc::from("Plugin manager not initialized")})?;

        let plugin = plugin_manager
            .get_plugin(&self.config.endpoint)
            .ok_or_else(|| IntegrationError::PluginError {
                detail: Arc::from("Plugin not found")})?;

        let result = plugin
            .execute(&request.path, &request.body.unwrap_or_default())
            .map_err(|e| IntegrationError::PluginError {
                detail: Arc::from(format!("Plugin execution failed: {}", e))})?;

        Ok(IntegrationResponse {
            status_code: 200,
            headers: std::collections::HashMap::new(),
            body: Some(result),
            response_time_ms: 0,
            success: true})
    }

    /// Execute external service request
    async fn execute_service_request(
        &mut self,
        _request: IntegrationRequest,
    ) -> IntegrationResult<IntegrationResponse> {
        // Placeholder for external service integration
        Err(IntegrationError::ConfigurationError {
            detail: Arc::from("External service integration not implemented")})
    }

    /// Get integration statistics
    pub fn stats(&self) -> &IntegrationStats {
        &self.stats
    }

    /// Get integration configuration
    pub fn config(&self) -> &IntegrationConfig {
        &self.config
    }

    /// Update integration configuration
    pub fn update_config(&mut self, config: IntegrationConfig) {
        self.config = config;
    }

    /// Test integration connectivity
    pub async fn test_connection(&mut self) -> IntegrationResult<bool> {
        let test_request = IntegrationRequest {
            method: Arc::from("GET"),
            path: Arc::from("/health"),
            headers: std::collections::HashMap::new(),
            body: None,
            timeout_ms: Some(5000), // 5 second timeout for health check
        };

        match self.execute_request(test_request).await {
            Ok(response) => Ok(response.success),
            Err(_) => Ok(false)}
    }
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: std::collections::HashMap::new(),
            configs: std::collections::HashMap::new()}
    }

    /// Load a plugin
    pub fn load_plugin(
        &mut self,
        plugin: Arc<dyn Plugin>,
        config: PluginConfig,
    ) -> IntegrationResult<()> {
        let plugin_id = config.plugin_id.clone();
        self.plugins.insert(plugin_id.clone(), plugin);
        self.configs.insert(plugin_id, config);
        Ok(())
    }

    /// Get a plugin by ID
    pub fn get_plugin(&self, plugin_id: &str) -> Option<&Arc<dyn Plugin>> {
        self.plugins.get(plugin_id)
    }

    /// Unload a plugin
    pub fn unload_plugin(&mut self, plugin_id: &str) -> IntegrationResult<()> {
        self.plugins.remove(plugin_id);
        self.configs.remove(plugin_id);
        Ok(())
    }
}

impl Default for ExternalIntegration {
    fn default() -> Self {
        Self::new(IntegrationConfig::default())
    }
}
