//! Integration manager for handling external connections
//!
//! Provides comprehensive integration management with zero-allocation patterns
//! and ergonomic APIs for external service coordination.

use std::sync::Arc;

use super::types::{
    IntegrationConfig, IntegrationType, IntegrationError, IntegrationResult,
};

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
        self.integrations
            .retain(|config| config.name.as_ref() != name);

        if self.integrations.len() == initial_len {
            return Err(IntegrationError::ConfigurationError {
                detail: Arc::from("Integration not found"),
            });
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
                detail: Arc::from("Integration not found"),
            }
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