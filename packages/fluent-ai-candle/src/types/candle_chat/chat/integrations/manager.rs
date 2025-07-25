//! Integration manager for handling external connections
//!
//! Provides comprehensive integration management with zero-allocation patterns
//! and ergonomic APIs for external service coordination.

use std::sync::Arc;

use fluent_ai_async::AsyncStream;

use super::types::{
    IntegrationConfig, IntegrationType, IntegrationError, IntegrationResult};

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
    pub fn send_to_integration(
        &self,
        integration_name: &str,
        message: &str,
    ) -> AsyncStream<String> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let integration_name = integration_name.to_string();
        let message = message.to_string();
        let manager = self.clone();
        
        AsyncStream::with_channel(move |sender| {
            let integration = match manager.get_integration(&integration_name) {
                Some(integration) => integration,
                None => {
                    handle_error!(
                        IntegrationError::ConfigurationError {
                            detail: Arc::from("Integration not found")},
                        "integration lookup failed"
                    );
                }
            };

            if !integration.enabled {
                handle_error!(
                    IntegrationError::ConfigurationError {
                        detail: Arc::from("Integration is disabled")},
                    "integration disabled"
                );
            }

            let result_stream = match integration.integration_type {
                IntegrationType::Webhook => manager.send_webhook(integration, &message),
                IntegrationType::RestApi => manager.send_rest_api(integration, &message),
                IntegrationType::Plugin => manager.send_plugin(integration, &message),
                IntegrationType::ExternalService => {
                    manager.send_external_service(integration, &message)
                }
            };
            
            result_stream.on_chunk(move |response| {
                emit!(sender, response);
            });
        })
    }

    /// Send webhook message
    fn send_webhook(
        &self,
        integration: &IntegrationConfig,
        message: &str,
    ) -> AsyncStream<String> {
        use fluent_ai_async::{AsyncStream, emit};
        
        let endpoint = integration.endpoint.clone();
        let message = message.to_string();
        
        AsyncStream::with_channel(move |sender| {
            // Placeholder for webhook implementation
            // In production, this would use fluent_ai_http3 streaming
            let response = format!("Webhook sent to {}: {}", endpoint, message);
            emit!(sender, response);
        })
    }

    /// Send REST API message
    fn send_rest_api(
        &self,
        integration: &IntegrationConfig,
        message: &str,
    ) -> AsyncStream<String> {
        use fluent_ai_async::{AsyncStream, emit};
        
        let endpoint = integration.endpoint.clone();
        let message = message.to_string();
        
        AsyncStream::with_channel(move |sender| {
            // Placeholder for REST API implementation
            // In production, this would use fluent_ai_http3 streaming
            let response = format!("REST API call to {}: {}", endpoint, message);
            emit!(sender, response);
        })
    }

    /// Send plugin message
    fn send_plugin(
        &self,
        integration: &IntegrationConfig,
        message: &str,
    ) -> AsyncStream<String> {
        use fluent_ai_async::{AsyncStream, emit};
        
        let name = integration.name.clone();
        let message = message.to_string();
        
        AsyncStream::with_channel(move |sender| {
            // Placeholder for plugin implementation
            // In production, this would interface with a plugin system
            let response = format!("Plugin call to {}: {}", name, message);
            emit!(sender, response);
        })
    }

    /// Send external service message
    fn send_external_service(
        &self,
        integration: &IntegrationConfig,
        message: &str,
    ) -> AsyncStream<String> {
        use fluent_ai_async::{AsyncStream, emit};
        
        let endpoint = integration.endpoint.clone();
        let message = message.to_string();
        
        AsyncStream::with_channel(move |sender| {
            // Placeholder for external service implementation
            // In production, this would interface with external APIs
            let response = format!("External service call to {}: {}", endpoint, message);
            emit!(sender, response);
        })
    }
}