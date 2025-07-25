//! Configuration validators
//!
//! Validation logic for different configuration sections with type-safe
//! validation rules and zero-allocation patterns.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use super::model_config::{ModelConfig, ValidationResult};
use super::types::{ConfigSection, ValidationSeverity};

/// Trait for configuration validation
pub trait ConfigurationValidator: Send + Sync {
    /// Validate configuration changes
    fn validate(&self, old_value: &serde_json::Value, new_value: &serde_json::Value) -> ValidationResult;
    
    /// Get validator name
    fn name(&self) -> &str;
    
    /// Get validation rules
    fn rules(&self) -> Vec<ValidationRule>;
}

/// Validation rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule type
    pub rule_type: ValidationRuleType,
    /// Rule parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Rule severity
    pub severity: ValidationSeverity,
}

/// Type of validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRuleType {
    /// Range validation
    Range,
    /// Pattern validation
    Pattern,
    /// Required field validation
    Required,
    /// Type validation
    Type,
    /// Custom validation
    Custom(String),
}

/// Personality configuration validator
pub struct PersonalityValidator;

impl ConfigurationValidator for PersonalityValidator {
    fn validate(&self, _old_value: &serde_json::Value, new_value: &serde_json::Value) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate personality levels are between 0.0 and 1.0
        if let Some(obj) = new_value.as_object() {
            for (key, value) in obj {
                if key.ends_with("_level") {
                    if let Some(level) = value.as_f64() {
                        if level < 0.0 || level > 1.0 {
                            errors.push(format!("{} must be between 0.0 and 1.0", key));
                        }
                    }
                }
            }
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            config_hash: "personality_hash".to_string(),
        }
    }

    fn name(&self) -> &str {
        "PersonalityValidator"
    }

    fn rules(&self) -> Vec<ValidationRule> {
        vec![
            ValidationRule {
                name: "level_range".to_string(),
                description: "Personality levels must be between 0.0 and 1.0".to_string(),
                rule_type: ValidationRuleType::Range,
                parameters: HashMap::new(),
                severity: ValidationSeverity::Error,
            }
        ]
    }
}

/// Behavior configuration validator
pub struct BehaviorValidator;

impl ConfigurationValidator for BehaviorValidator {
    fn validate(&self, _old_value: &serde_json::Value, new_value: &serde_json::Value) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate response delay is reasonable
        if let Some(obj) = new_value.as_object() {
            if let Some(delay) = obj.get("response_delay_ms").and_then(|v| v.as_u64()) {
                if delay > 10000 {
                    warnings.push("Response delay over 10 seconds may impact user experience".to_string());
                }
            }
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            config_hash: "behavior_hash".to_string(),
        }
    }

    fn name(&self) -> &str {
        "BehaviorValidator"
    }

    fn rules(&self) -> Vec<ValidationRule> {
        vec![
            ValidationRule {
                name: "response_delay".to_string(),
                description: "Response delay should be reasonable".to_string(),
                rule_type: ValidationRuleType::Range,
                parameters: HashMap::new(),
                severity: ValidationSeverity::Warning,
            }
        ]
    }
}

/// UI configuration validator
pub struct UIValidator;

impl ConfigurationValidator for UIValidator {
    fn validate(&self, _old_value: &serde_json::Value, new_value: &serde_json::Value) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate font size is reasonable
        if let Some(obj) = new_value.as_object() {
            if let Some(theme) = obj.get("theme").and_then(|v| v.as_object()) {
                if let Some(fonts) = theme.get("fonts").and_then(|v| v.as_object()) {
                    if let Some(size) = fonts.get("font_size").and_then(|v| v.as_u64()) {
                        if size < 8 || size > 72 {
                            errors.push("Font size must be between 8 and 72 pixels".to_string());
                        }
                    }
                }
            }
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            config_hash: "ui_hash".to_string(),
        }
    }

    fn name(&self) -> &str {
        "UIValidator"
    }

    fn rules(&self) -> Vec<ValidationRule> {
        vec![
            ValidationRule {
                name: "font_size_range".to_string(),
                description: "Font size must be between 8 and 72 pixels".to_string(),
                rule_type: ValidationRuleType::Range,
                parameters: HashMap::new(),
                severity: ValidationSeverity::Error,
            }
        ]
    }
}

/// Integration configuration validator
pub struct IntegrationValidator;

impl ConfigurationValidator for IntegrationValidator {
    fn validate(&self, _old_value: &serde_json::Value, new_value: &serde_json::Value) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate webhook URLs
        if let Some(obj) = new_value.as_object() {
            if let Some(webhooks) = obj.get("webhooks").and_then(|v| v.as_array()) {
                for webhook in webhooks {
                    if let Some(webhook_obj) = webhook.as_object() {
                        if let Some(url) = webhook_obj.get("url").and_then(|v| v.as_str()) {
                            if !url.starts_with("https://") {
                                warnings.push("Webhook URLs should use HTTPS for security".to_string());
                            }
                        }
                    }
                }
            }
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            config_hash: "integration_hash".to_string(),
        }
    }

    fn name(&self) -> &str {
        "IntegrationValidator"
    }

    fn rules(&self) -> Vec<ValidationRule> {
        vec![
            ValidationRule {
                name: "webhook_security".to_string(),
                description: "Webhook URLs should use HTTPS".to_string(),
                rule_type: ValidationRuleType::Pattern,
                parameters: HashMap::new(),
                severity: ValidationSeverity::Warning,
            }
        ]
    }
}