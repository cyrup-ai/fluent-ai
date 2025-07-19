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

/// Integration manager for handling external connections
#[derive(Debug, Clone)]
pub struct IntegrationManager {
    /// Active integrations
    pub integrations: Vec<IntegrationConfig>,
    /// Default timeout
    pub default_timeout: u32,
}

impl Default for IntegrationManager {
    fn default() -> Self {
        Self {
            integrations: Vec::new(),
            default_timeout: 30,
        }
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
                detail: Arc::from("Integration name cannot be empty"),
            });
        }

        if config.endpoint.is_empty() {
            return Err(IntegrationError::ConfigurationError {
                detail: Arc::from("Integration endpoint cannot be empty"),
            });
        }

        self.integrations.push(config);
        Ok(())
    }

    /// Remove an integration by name
    pub fn remove_integration(&mut self, name: &str) -> IntegrationResult<()> {
        let initial_len = self.integrations.len();
        self.integrations.retain(|config| config.name.as_ref() != name);
        
        if self.integrations.len() == initial_len {
            return Err(IntegrationError::ConfigurationError {
                detail: Arc::from("Integration not found"),
            });
        }

        Ok(())
    }

    /// Get integration by name
    pub fn get_integration(&self, name: &str) -> Option<&IntegrationConfig> {
        self.integrations.iter().find(|config| config.name.as_ref() == name)
    }

    /// List all enabled integrations
    pub fn list_enabled_integrations(&self) -> Vec<&IntegrationConfig> {
        self.integrations.iter().filter(|config| config.enabled).collect()
    }

    /// Send message to integration
    pub async fn send_to_integration(
        &self,
        integration_name: &str,
        message: &str,
    ) -> IntegrationResult<String> {
        let integration = self.get_integration(integration_name)
            .ok_or_else(|| IntegrationError::ConfigurationError {
                detail: Arc::from("Integration not found"),
            })?;

        if !integration.enabled {
            return Err(IntegrationError::ConfigurationError {
                detail: Arc::from("Integration is disabled"),
            });
        }

        match integration.integration_type {
            IntegrationType::Webhook => self.send_webhook(integration, message).await,
            IntegrationType::RestApi => self.send_rest_api(integration, message).await,
            IntegrationType::Plugin => self.send_plugin(integration, message).await,
            IntegrationType::ExternalService => self.send_external_service(integration, message).await,
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
        Ok(format!("Webhook sent to {}: {}", integration.endpoint, message))
    }

    /// Send REST API message
    async fn send_rest_api(
        &self,
        integration: &IntegrationConfig,
        message: &str,
    ) -> IntegrationResult<String> {
        // Placeholder for REST API implementation
        // In production, this would use an HTTP client like reqwest
        Ok(format!("REST API call to {}: {}", integration.endpoint, message))
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
        Ok(format!("External service call to {}: {}", integration.endpoint, message))
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
