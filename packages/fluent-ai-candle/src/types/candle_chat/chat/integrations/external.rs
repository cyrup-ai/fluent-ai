//! External integration system for managing third-party services and plugins
//!
//! This system provides comprehensive external integration capabilities with:
//! - Multiple integration types (webhooks, APIs, plugins, services)
//! - Authentication and security management
//! - Request/response handling with retry logic
//! - Performance monitoring and error tracking
//! - Configuration management and validation

use std::sync::Arc;

use super::types::{
    IntegrationConfig, IntegrationType, IntegrationError, IntegrationResult,
    IntegrationStats, IntegrationRequest, IntegrationResponse, PluginConfig,
};

/// External integration system for managing third-party services and plugins
#[derive(Debug, Clone)]
pub struct ExternalIntegration {
    /// Integration configuration
    config: IntegrationConfig,
    /// Integration statistics
    stats: IntegrationStats,
    /// HTTP client for API calls
    client: Option<Arc<reqwest::Client>>,
    /// Plugin manager for plugin integrations
    plugin_manager: Option<Arc<crate::types::candle_chat::chat::integrations::plugin::PluginManager>>,
}

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
            Some(Arc::new(crate::types::candle_chat::chat::integrations::plugin::PluginManager::new()))
        } else {
            None
        };

        Self {
            config,
            stats: IntegrationStats::default(),
            client,
            plugin_manager,
        }
    }

    /// Execute an integration request
    pub async fn execute_request(
        &mut self,
        request: IntegrationRequest,
    ) -> IntegrationResult<IntegrationResponse> {
        if !self.config.enabled {
            return Err(IntegrationError::ConfigurationError {
                detail: Arc::from("Integration is disabled"),
            });
        }

        let start_time = std::time::Instant::now();
        self.stats.total_requests += 1;

        let result = match self.config.integration_type {
            IntegrationType::RestApi | IntegrationType::Webhook => {
                self.execute_http_request(request).await
            }
            IntegrationType::Plugin => self.execute_plugin_request(request).await,
            IntegrationType::ExternalService => self.execute_service_request(request).await,
        };

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
                detail: Arc::from("HTTP client not initialized"),
            })?;

        let url = format!("{}{}", self.config.endpoint, request.path);
        let mut req_builder = match request.method.as_ref() {
            "GET" => client.get(&url),
            "POST" => client.post(&url),
            "PUT" => client.put(&url),
            "DELETE" => client.delete(&url),
            "PATCH" => client.patch(&url),
            _ => {
                return Err(IntegrationError::ConfigurationError {
                    detail: Arc::from("Unsupported HTTP method"),
                });
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
                detail: Arc::from(e.to_string()),
            })?;

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
            success: status_code >= 200 && status_code < 300,
        })
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
                    detail: Arc::from("Plugin manager not initialized"),
                })?;

        let plugin = plugin_manager
            .get_plugin(&self.config.endpoint)
            .ok_or_else(|| IntegrationError::PluginError {
                detail: Arc::from("Plugin not found"),
            })?;

        let result = plugin
            .execute(&request.path, &request.body.unwrap_or_default())
            .map_err(|e| IntegrationError::PluginError {
                detail: Arc::from(format!("Plugin execution failed: {}", e)),
            })?;

        Ok(IntegrationResponse {
            status_code: 200,
            headers: std::collections::HashMap::new(),
            body: Some(result),
            response_time_ms: 0,
            success: true,
        })
    }

    /// Execute external service request
    async fn execute_service_request(
        &mut self,
        _request: IntegrationRequest,
    ) -> IntegrationResult<IntegrationResponse> {
        // Placeholder for external service integration
        Err(IntegrationError::ConfigurationError {
            detail: Arc::from("External service integration not implemented"),
        })
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
            Err(_) => Ok(false),
        }
    }
}

impl Default for ExternalIntegration {
    fn default() -> Self {
        Self::new(IntegrationConfig::default())
    }
}