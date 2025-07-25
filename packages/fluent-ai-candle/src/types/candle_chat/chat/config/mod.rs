//! Configuration management system for chat features
//!
//! This module provides a comprehensive configuration management system with atomic updates,
//! validation, persistence, and change notifications using zero-allocation patterns and
//! lock-free operations for blazing-fast performance.
//!
//! # Architecture
//!
//! The configuration system is decomposed into focused modules:
//! - [`model`] - Model configuration for AI providers and parameters
//! - [`core`] - Core chat configuration types and structures
//! - [`events`] - Configuration change events and notification system
//! - [`validation`] - Configuration validation and error handling
//! - [`manager`] - Configuration manager with atomic operations
//! - [`builder`] - Fluent builder patterns for configuration creation
//!
//! # Example Usage
//!
//! ```rust
//! use fluent_ai_candle::types::candle_chat::chat::config::*;
//!
//! // Create a configuration using the builder pattern
//! let config = ConfigurationBuilder::new()
//!     .name("My Chat Bot")
//!     .personality(|p| {
//!         p.personality_type("professional")
//!             .tone("friendly")
//!             .creativity_level(0.7)
//!     })
//!     .model(|m| {
//!         m.provider("openai")
//!             .model_name("gpt-4")
//!             .temperature(0.7)
//!             .max_tokens(2048)
//!     })
//!     .build();
//!
//! // Create a configuration manager
//! let manager = ConfigurationManager::with_config(config);
//!
//! // Subscribe to configuration changes
//! let mut changes = manager.subscribe_to_changes();
//!
//! // Update configuration and process changes
//! // ...
//! ```
//!
//! # Quick Start Templates
//!
//! ```rust
//! // Professional assistant
//! let professional = ConfigurationBuilder::professional();
//!
//! // Casual friend
//! let casual = ConfigurationBuilder::casual();
//!
//! // Creative partner
//! let creative = ConfigurationBuilder::creative();
//! ```

// Decomposed modules - claude92 completed decomposition
pub mod config_core;
pub mod validation;
pub mod config_manager;
pub mod config_builder;

// Legacy module structure for compatibility
pub mod builder {
    pub use super::config_builder::*;
}
pub mod model;
pub mod core;
pub mod events;
pub mod manager;
pub mod presets;

// Re-export commonly used types
pub use model::{
    ModelConfig, ModelRetryConfig, ModelPerformanceConfig, ModelValidator
};

pub use core::{
    ChatConfig, PersonalityConfig, IntegrationConfig,
    LanguageHandlingConfig, DisplayConfig, ApiIntegrationConfig, PluginConfig
};

pub use config_core::{
    BehaviorConfig, UIConfig
};

pub use events::{
    ConfigurationChangeEvent, ConfigurationChangeType,
    ConfigurationPersistence, BackupConfig, EncryptionConfig,
    ErrorSeverity
};

pub use validation::{
    ConfigurationValidationError, ConfigurationValidationResult, ConfigurationValidator,
    PersonalityValidator, BehaviorValidator, UIValidator, IntegrationValidator
};

pub use manager::{
    ConfigurationManager, ConfigurationStatistics
};

// Configuration builders are imported from main config module when needed

pub use presets::{
    professional, casual, creative, technical, customer_support,
    gaming_companion, educational_tutor, therapy_assistant
};

/// Create a default configuration manager
pub fn create_default_manager() -> ConfigurationManager {
    ConfigurationManager::new()
}

/// Create a configuration manager with a specific configuration
pub fn create_manager_with_config(config: ChatConfig) -> ConfigurationManager {
    ConfigurationManager::with_config(config)
}

/// Validate a configuration using all available validators
pub fn validate_configuration(config: &ChatConfig) -> ConfigurationValidationResult<()> {
    use std::sync::Arc;
    let mut all_errors: Vec<ConfigurationValidationError> = Vec::new();

    // Use individual validators
    let model_validator = ModelValidator;
    if let Err(error) = model_validator.validate(&config.model) {
        all_errors.push(ConfigurationValidationError::SchemaValidation { 
            detail: Arc::from(error) 
        });
    }

    let personality_validator = PersonalityValidator;
    if let Err(error) = personality_validator.validate(config) {
        all_errors.push(error);
    }

    let behavior_validator = BehaviorValidator;
    if let Err(error) = behavior_validator.validate(config) {
        all_errors.push(error);
    }

    let ui_validator = UIValidator;
    if let Err(error) = ui_validator.validate(config) {
        all_errors.push(error);
    }

    let integration_validator = IntegrationValidator;
    if let Err(error) = integration_validator.validate(config) {
        all_errors.push(error);
    }

    if all_errors.is_empty() {
        Ok(())
    } else {
        // Return the first error to match ConfigurationValidationResult<()>
        Err(all_errors.into_iter().next().unwrap())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_configuration_builder() {
        let config = ConfigurationBuilder::new()
            .name("Test Config")
            .description("A test configuration")
            .model(|m| m.provider("openai").model_name("gpt-4").temperature(0.7))
            .personality(|p| p.name("TestBot").personality_type("professional"))
            .build();

        assert_eq!(config.name.as_ref(), "Test Config");
        assert_eq!(config.description.as_ref().unwrap().as_ref(), "A test configuration");
        assert_eq!(config.model.provider.as_ref(), "openai");
        assert_eq!(config.model.model_name.as_ref(), "gpt-4");
        assert_eq!(config.model.temperature, 0.7);
        assert_eq!(config.personality.name.as_ref(), "TestBot");
    }

    #[test]
    fn test_configuration_manager() {
        let manager = ConfigurationManager::new();
        let config = manager.get_config();
        
        assert_eq!(config.name.as_ref(), "Default Chat Configuration");
        assert_eq!(manager.get_version(), 1);
    }

    #[test]
    fn test_validation() {
        let valid_config = professional();
        assert!(validate_configuration(&valid_config).is_ok());

        let mut invalid_config = ConfigurationBuilder::new().build();
        invalid_config.personality.formality_level = 2.0; // Invalid: should be 0.0-1.0
        
        assert!(validate_configuration(&invalid_config).is_err());
    }

    #[test]
    fn test_presets() {
        let professional = professional();
        assert_eq!(professional.personality.personality_type.as_ref(), "professional");

        let casual = casual();
        assert_eq!(casual.personality.personality_type.as_ref(), "casual");

        let creative = creative();
        assert_eq!(creative.personality.personality_type.as_ref(), "creative");

        let technical = technical();
        assert_eq!(technical.personality.personality_type.as_ref(), "technical");
    }
}