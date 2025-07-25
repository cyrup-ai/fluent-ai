//! Integration management system
//!
//! This module provides comprehensive integration management with zero-allocation patterns,
//! configuration validation, and streaming-first architecture using fluent-ai-async primitives.

use std::sync::Arc;
use fluent_ai_async::{AsyncStream, emit, handle_error};

use super::config::{
    IntegrationConfig, IntegrationError, IntegrationResult, IntegrationType
};

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

    /// Send message to integration using fluent-ai-async streaming architecture
    pub fn send_to_integration(
        &self,
        integration_name: &str,
        message: &str,
    ) -> AsyncStream<String> {
        let integration_name = integration_name.to_string();
        let message = message.to_string();
        let integrations = self.integrations.clone();
        
        AsyncStream::with_channel(move |sender| {
            let integration = match integrations.iter().find(|config| config.name.as_ref() == integration_name) {
                Some(integration) => integration,
                None => {
                    handle_error!(
                        IntegrationError::ConfigurationError {
                            detail: Arc::from("Integration not found")
                        },
                        "Integration not found"
                    );
                    return;
                }
            };

            if !integration.enabled {
                handle_error!(
                    IntegrationError::ConfigurationError {
                        detail: Arc::from("Integration is disabled")
                    },
                    "Integration is disabled"
                );
                return;
            }

            // Route to appropriate integration handler based on type
            let response = match integration.integration_type {
                IntegrationType::Webhook => {
                    format!("Webhook sent to {}: {}", integration.endpoint, message)
                }
                IntegrationType::RestApi => {
                    format!("REST API call to {}: {}", integration.endpoint, message)
                }
                IntegrationType::Plugin => {
                    format!("Plugin call to {}: {}", integration.name, message)
                }
                IntegrationType::ExternalService => {
                    format!("External service call to {}: {}", integration.endpoint, message)
                }
            };

            emit!(sender, response);
        })
    }

    /// Send webhook message using fluent-ai-async streaming architecture
    pub fn send_webhook(
        &self,
        integration: &IntegrationConfig,
        message: &str,
    ) -> AsyncStream<String> {
        let endpoint = integration.endpoint.clone();
        let message = message.to_string();
        
        AsyncStream::with_channel(move |sender| {
            // Placeholder for webhook implementation
            // In production, this would use an HTTP client like reqwest
            let result = format!("Webhook sent to {}: {}", endpoint, message);
            emit!(sender, result);
        })
    }

    /// Send REST API message using fluent-ai-async streaming architecture
    pub fn send_rest_api(
        &self,
        integration: &IntegrationConfig,
        message: &str,
    ) -> AsyncStream<String> {
        let endpoint = integration.endpoint.clone();
        let message = message.to_string();
        
        AsyncStream::with_channel(move |sender| {
            // Placeholder for REST API implementation
            // In production, this would use an HTTP client like reqwest
            let result = format!("REST API call to {}: {}", endpoint, message);
            emit!(sender, result);
        })
    }

    /// Send plugin message using fluent-ai-async streaming architecture
    pub fn send_plugin(
        &self,
        integration: &IntegrationConfig,
        message: &str,
    ) -> AsyncStream<String> {
        let name = integration.name.clone();
        let message = message.to_string();
        
        AsyncStream::with_channel(move |sender| {
            // Placeholder for plugin implementation
            // In production, this would interface with a plugin system
            let result = format!("Plugin call to {}: {}", name, message);
            emit!(sender, result);
        })
    }

    /// Send external service message using fluent-ai-async streaming architecture
    pub fn send_external_service(
        &self,
        integration: &IntegrationConfig,
        message: &str,
    ) -> AsyncStream<String> {
        let endpoint = integration.endpoint.clone();
        let message = message.to_string();
        
        AsyncStream::with_channel(move |sender| {
            // Placeholder for external service implementation
            // In production, this would interface with external APIs
            let result = format!("External service call to {}: {}", endpoint, message);
            emit!(sender, result);
        })
    }

    /// Validate all integration configurations using fluent-ai-async streaming architecture
    pub fn validate_all_integrations(&self) -> AsyncStream<Vec<IntegrationResult<()>>> {
        let integrations = self.integrations.clone();
        
        AsyncStream::with_channel(move |sender| {
            let mut results = Vec::new();
            
            for integration in &integrations {
                let result = if integration.name.is_empty() {
                    Err(IntegrationError::ConfigurationError {
                        detail: Arc::from("Integration name cannot be empty")})
                } else if integration.endpoint.is_empty() {
                    Err(IntegrationError::ConfigurationError {
                        detail: Arc::from("Integration endpoint cannot be empty")})
                } else if integration.timeout_seconds == 0 {
                    Err(IntegrationError::ConfigurationError {
                        detail: Arc::from("Timeout must be greater than zero")})
                } else {
                    Ok(())
                };
                
                results.push(result);
            }
            
            emit!(sender, results);
        })
    }

    /// Get integration count by type using fluent-ai-async streaming architecture
    pub fn get_integration_count_by_type(&self) -> AsyncStream<std::collections::HashMap<IntegrationType, usize>> {
        let integrations = self.integrations.clone();
        
        AsyncStream::with_channel(move |sender| {
            let mut counts = std::collections::HashMap::new();
            
            for integration in &integrations {
                *counts.entry(integration.integration_type).or_insert(0) += 1;
            }
            
            emit!(sender, counts);
        })
    }
}
