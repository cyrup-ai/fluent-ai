//! Configuration validation logic for chat features
//!
//! This module provides comprehensive validation for all configuration types including
//! personality, behavior, UI, and integration configurations with detailed error reporting
//! and validation rules using zero-allocation patterns.

use std::sync::Arc;

use crate::types::candle_chat::chat::config_core::{ChatConfig, PersonalityConfig, BehaviorConfig, UIConfig, IntegrationConfig};

/// Configuration validation error
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConfigurationValidationError {
    #[error("Invalid personality configuration: {detail}")]
    InvalidPersonality { detail: Arc<str> },
    #[error("Invalid behavior configuration: {detail}")]
    InvalidBehavior { detail: Arc<str> },
    #[error("Invalid UI configuration: {detail}")]
    InvalidUI { detail: Arc<str> },
    #[error("Invalid integration configuration: {detail}")]
    InvalidIntegration { detail: Arc<str> },
    #[error("Configuration conflict: {detail}")]
    Conflict { detail: Arc<str> },
    #[error("Schema validation failed: {detail}")]
    SchemaValidation { detail: Arc<str> },
    #[error("Range validation failed: {field} must be between {min} and {max}")]
    RangeValidation { field: Arc<str>, min: f32, max: f32 },
    #[error("Required field missing: {field}")]
    RequiredField { field: Arc<str> },
}

/// Configuration validation result
pub type ConfigurationValidationResult<T> = Result<T, ConfigurationValidationError>;

/// Configuration validator trait
pub trait ConfigurationValidator {
    /// Validate configuration section
    fn validate(&self, config: &ChatConfig) -> ConfigurationValidationResult<()>;
    /// Get validator name
    fn name(&self) -> &str;
    /// Get validation priority (lower = higher priority)
    fn priority(&self) -> u8;
}

/// Personality configuration validator
pub struct PersonalityValidator;

impl ConfigurationValidator for PersonalityValidator {
    fn validate(&self, config: &ChatConfig) -> ConfigurationValidationResult<()> {
        let personality = &config.personality;

        // Validate creativity range
        if !(0.0..=1.0).contains(&personality.creativity) {
            return Err(ConfigurationValidationError::RangeValidation {
                field: Arc::from("creativity"),
                min: 0.0,
                max: 1.0,
            });
        }

        // Validate formality range
        if !(0.0..=1.0).contains(&personality.formality) {
            return Err(ConfigurationValidationError::RangeValidation {
                field: Arc::from("formality"),
                min: 0.0,
                max: 1.0,
            });
        }

        // Validate humor range
        if !(0.0..=1.0).contains(&personality.humor) {
            return Err(ConfigurationValidationError::RangeValidation {
                field: Arc::from("humor"),
                min: 0.0,
                max: 1.0,
            });
        }

        // Validate empathy range
        if !(0.0..=1.0).contains(&personality.empathy) {
            return Err(ConfigurationValidationError::RangeValidation {
                field: Arc::from("empathy"),
                min: 0.0,
                max: 1.0,
            });
        }

        // Validate expertise level
        let valid_expertise = ["beginner", "intermediate", "advanced", "expert"];
        if !valid_expertise.contains(&personality.expertise_level.as_ref()) {
            return Err(ConfigurationValidationError::InvalidPersonality {
                detail: Arc::from("Invalid expertise level"),
            });
        }

        // Validate tone
        let valid_tones = ["formal", "casual", "friendly", "professional", "neutral"];
        if !valid_tones.contains(&personality.tone.as_ref()) {
            return Err(ConfigurationValidationError::InvalidPersonality {
                detail: Arc::from("Invalid tone"),
            });
        }

        // Validate verbosity
        let valid_verbosity = ["concise", "balanced", "detailed"];
        if !valid_verbosity.contains(&personality.verbosity.as_ref()) {
            return Err(ConfigurationValidationError::InvalidPersonality {
                detail: Arc::from("Invalid verbosity level"),
            });
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "personality"
    }

    fn priority(&self) -> u8 {
        1
    }
}

/// Behavior configuration validator
pub struct BehaviorValidator;

impl ConfigurationValidator for BehaviorValidator {
    fn validate(&self, config: &ChatConfig) -> ConfigurationValidationResult<()> {
        let behavior = &config.behavior;

        // Validate proactivity range
        if !(0.0..=1.0).contains(&behavior.proactivity) {
            return Err(ConfigurationValidationError::RangeValidation {
                field: Arc::from("proactivity"),
                min: 0.0,
                max: 1.0,
            });
        }

        // Validate question frequency range
        if !(0.0..=1.0).contains(&behavior.question_frequency) {
            return Err(ConfigurationValidationError::RangeValidation {
                field: Arc::from("question_frequency"),
                min: 0.0,
                max: 1.0,
            });
        }

        // Validate context awareness range
        if !(0.0..=1.0).contains(&behavior.context_awareness) {
            return Err(ConfigurationValidationError::RangeValidation {
                field: Arc::from("context_awareness"),
                min: 0.0,
                max: 1.0,
            });
        }

        // Validate memory retention range
        if !(0.0..=1.0).contains(&behavior.memory_retention) {
            return Err(ConfigurationValidationError::RangeValidation {
                field: Arc::from("memory_retention"),
                min: 0.0,
                max: 1.0,
            });
        }

        // Validate conversation flow
        let valid_flows = ["natural", "structured", "adaptive", "guided"];
        if !valid_flows.contains(&behavior.conversation_flow.as_ref()) {
            return Err(ConfigurationValidationError::InvalidBehavior {
                detail: Arc::from("Invalid conversation flow"),
            });
        }

        // Validate follow-up behavior
        let valid_followups = ["contextual", "consistent", "adaptive", "minimal"];
        if !valid_followups.contains(&behavior.follow_up_behavior.as_ref()) {
            return Err(ConfigurationValidationError::InvalidBehavior {
                detail: Arc::from("Invalid follow-up behavior"),
            });
        }

        // Validate error handling
        let valid_error_handling = ["graceful", "verbose", "silent", "strict"];
        if !valid_error_handling.contains(&behavior.error_handling.as_ref()) {
            return Err(ConfigurationValidationError::InvalidBehavior {
                detail: Arc::from("Invalid error handling approach"),
            });
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "behavior"
    }

    fn priority(&self) -> u8 {
        2
    }
}

/// UI configuration validator
pub struct UIValidator;

impl ConfigurationValidator for UIValidator {
    fn validate(&self, config: &ChatConfig) -> ConfigurationValidationResult<()> {
        let ui = &config.ui;

        // Validate theme
        let valid_themes = ["light", "dark", "auto", "system", "custom"];
        if !valid_themes.contains(&ui.theme.as_ref()) {
            return Err(ConfigurationValidationError::InvalidUI {
                detail: Arc::from("Invalid theme"),
            });
        }

        // Validate layout
        let valid_layouts = ["standard", "compact", "wide", "mobile", "adaptive"];
        if !valid_layouts.contains(&ui.layout.as_ref()) {
            return Err(ConfigurationValidationError::InvalidUI {
                detail: Arc::from("Invalid layout"),
            });
        }

        // Validate color scheme
        let valid_color_schemes = ["adaptive", "high_contrast", "colorblind", "custom"];
        if !valid_color_schemes.contains(&ui.color_scheme.as_ref()) {
            return Err(ConfigurationValidationError::InvalidUI {
                detail: Arc::from("Invalid color scheme"),
            });
        }

        // Validate display density
        let valid_densities = ["compact", "comfortable", "spacious"];
        if !valid_densities.contains(&ui.display_density.as_ref()) {
            return Err(ConfigurationValidationError::InvalidUI {
                detail: Arc::from("Invalid display density"),
            });
        }

        // Validate animations
        let valid_animations = ["none", "minimal", "smooth", "rich"];
        if !valid_animations.contains(&ui.animations.as_ref()) {
            return Err(ConfigurationValidationError::InvalidUI {
                detail: Arc::from("Invalid animation setting"),
            });
        }

        // Validate animation speed range
        if !(0.0..=2.0).contains(&ui.animation_speed) {
            return Err(ConfigurationValidationError::RangeValidation {
                field: Arc::from("animation_speed"),
                min: 0.0,
                max: 2.0,
            });
        }

        // Validate sound volume range
        if !(0.0..=1.0).contains(&ui.sound_volume) {
            return Err(ConfigurationValidationError::RangeValidation {
                field: Arc::from("sound_volume"),
                min: 0.0,
                max: 1.0,
            });
        }

        // Validate font size
        if ui.font_size < 8 || ui.font_size > 72 {
            return Err(ConfigurationValidationError::RangeValidation {
                field: Arc::from("font_size"),
                min: 8.0,
                max: 72.0,
            });
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "ui"
    }

    fn priority(&self) -> u8 {
        3
    }
}

/// Integration configuration validator
pub struct IntegrationValidator;

impl ConfigurationValidator for IntegrationValidator {
    fn validate(&self, config: &ChatConfig) -> ConfigurationValidationResult<()> {
        let integration = &config.integration;

        // Validate external services
        let valid_services = ["mcp", "tools", "plugins", "apis", "webhooks"];
        for service in &integration.external_services {
            if !valid_services.contains(&service.as_ref()) {
                return Err(ConfigurationValidationError::InvalidIntegration {
                    detail: Arc::from(format!("Invalid external service: {}", service)),
                });
            }
        }

        // Validate API configurations
        let valid_apis = ["rest", "graphql", "websocket", "grpc"];
        for api in &integration.api_configurations {
            if !valid_apis.contains(&api.as_ref()) {
                return Err(ConfigurationValidationError::InvalidIntegration {
                    detail: Arc::from(format!("Invalid API configuration: {}", api)),
                });
            }
        }

        // Validate authentication methods
        let valid_auth = ["token", "oauth", "apikey", "basic", "jwt"];
        for auth in &integration.authentication {
            if !valid_auth.contains(&auth.as_ref()) {
                return Err(ConfigurationValidationError::InvalidIntegration {
                    detail: Arc::from(format!("Invalid authentication method: {}", auth)),
                });
            }
        }

        // Validate API rate limit
        if integration.api_rate_limit == 0 || integration.api_rate_limit > 10000 {
            return Err(ConfigurationValidationError::RangeValidation {
                field: Arc::from("api_rate_limit"),
                min: 1.0,
                max: 10000.0,
            });
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "integration"
    }

    fn priority(&self) -> u8 {
        4
    }
}

/// Composite validator that runs all validation rules
pub struct CompositeValidator {
    validators: Vec<Box<dyn ConfigurationValidator + Send + Sync>>,
}

impl CompositeValidator {
    /// Create a new composite validator with all standard validators
    pub fn new() -> Self {
        Self {
            validators: vec![
                Box::new(PersonalityValidator),
                Box::new(BehaviorValidator),
                Box::new(UIValidator),
                Box::new(IntegrationValidator),
            ],
        }
    }

    /// Add a custom validator
    pub fn add_validator(&mut self, validator: Box<dyn ConfigurationValidator + Send + Sync>) {
        self.validators.push(validator);
        // Sort by priority
        self.validators.sort_by_key(|v| v.priority());
    }

    /// Validate configuration with all registered validators
    pub fn validate_all(&self, config: &ChatConfig) -> Vec<ConfigurationValidationError> {
        let mut errors = Vec::new();
        
        for validator in &self.validators {
            if let Err(error) = validator.validate(config) {
                errors.push(error);
            }
        }
        
        errors
    }

    /// Validate configuration and return first error
    pub fn validate_first(&self, config: &ChatConfig) -> ConfigurationValidationResult<()> {
        for validator in &self.validators {
            validator.validate(config)?;
        }
        Ok(())
    }
}

impl Default for CompositeValidator {
    fn default() -> Self {
        Self::new()
    }
}