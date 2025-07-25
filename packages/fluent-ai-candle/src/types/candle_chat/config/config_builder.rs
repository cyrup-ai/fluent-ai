//! Configuration builder utilities for fluent configuration creation
//!
//! This module provides builder patterns for creating and modifying
//! configurations with validation and zero-allocation patterns.

use std::collections::HashMap;
use std::sync::Arc;

use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};

use super::model_config::{ModelConfig, ModelParameters, ModelRetryConfig, ModelPerformanceConfig, ValidationResult};
use super::chat_core::{ChatConfig, PersonalityConfig};
use super::behavior::BehaviorConfig;
use super::ui::UIConfig;
use super::integration::IntegrationConfig;
use super::config_manager::ConfigurationManager;

/// Handle errors in streaming context without panicking
macro_rules! handle_error {
    ($error:expr, $context:literal) => {
        eprintln!("Streaming error in {}: {}", $context, $error);
        // Continue processing instead of returning error
    };
}

/// Main configuration builder for creating complete configurations
pub struct ConfigurationBuilder {
    /// Model configuration being built
    pub model_config: ModelConfig,
    /// Chat configuration being built
    pub chat_config: ChatConfig,
    /// Validation errors
    pub validation_errors: Vec<String>,
    /// Build warnings
    pub warnings: Vec<String>,
    /// Custom build options
    pub build_options: HashMap<String, serde_json::Value>,
}

impl ConfigurationBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            model_config: ModelConfig::default(),
            chat_config: ChatConfig::default(),
            validation_errors: Vec::new(),
            warnings: Vec::new(),
            build_options: HashMap::new(),
        }
    }

    /// Set model configuration
    pub fn with_model(mut self, model_config: ModelConfig) -> Self {
        self.model_config = model_config;
        self
    }

    /// Set chat configuration
    pub fn with_chat(mut self, chat_config: ChatConfig) -> Self {
        self.chat_config = chat_config;
        self
    }

    /// Add build option
    pub fn with_option(mut self, key: String, value: serde_json::Value) -> Self {
        self.build_options.insert(key, value);
        self
    }

    /// Build the complete configuration (streaming)
    pub fn build(self) -> AsyncStream<ConfigurationManager> {
        AsyncStream::with_channel(move |sender| {
            let mut manager = ConfigurationManager::new();
            manager.model_config = self.model_config;
            manager.chat_config = self.chat_config;
            
            let _ = sender.send(manager);
        })
    }

    /// Validate the configuration being built (streaming)
    pub fn validate(&self) -> AsyncStream<ValidationResult> {
        let model_config = self.model_config.clone();
        
        AsyncStream::with_channel(move |sender| {
            let mut errors = Vec::new();
            let mut warnings = Vec::new();

            // Validate model configuration
            if model_config.model_name.is_empty() {
                errors.push("Model name is required".to_string());
            }

            if model_config.temperature < 0.0 || model_config.temperature > 2.0 {
                warnings.push("Temperature outside recommended range (0.0-2.0)".to_string());
            }

            let result = ValidationResult {
                is_valid: errors.is_empty(),
                errors,
                warnings,
                config_hash: "builder_hash".to_string(),
            };

            let _ = sender.send(result);
        })
    }
}

impl Default for ConfigurationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for personality configuration
pub struct PersonalityConfigBuilder {
    /// Personality configuration being built
    pub config: PersonalityConfig,
    /// Builder state
    pub builder_state: BuilderState,
}

/// Builder state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuilderState {
    /// Fields that have been set
    pub fields_set: Vec<String>,
    /// Validation status
    pub is_valid: bool,
    /// Build progress
    pub progress: f32,
    /// Last modified timestamp
    pub last_modified: chrono::DateTime<chrono::Utc>,
}

impl PersonalityConfigBuilder {
    /// Create a new personality config builder
    pub fn new() -> Self {
        Self {
            config: PersonalityConfig::default(),
            builder_state: BuilderState {
                fields_set: Vec::new(),
                is_valid: true,
                progress: 0.0,
                last_modified: chrono::Utc::now(),
            },
        }
    }

    /// Set personality name
    pub fn with_name(mut self, name: Arc<str>) -> Self {
        self.config.name = name;
        self.builder_state.fields_set.push("name".to_string());
        self.update_progress();
        self
    }

    /// Set personality description
    pub fn with_description(mut self, description: String) -> Self {
        self.config.description = Some(description);
        self.builder_state.fields_set.push("description".to_string());
        self.update_progress();
        self
    }

    /// Set response style
    pub fn with_response_style(mut self, style: super::chat_core::ResponseStyle) -> Self {
        self.config.response_style = style;
        self.builder_state.fields_set.push("response_style".to_string());
        self.update_progress();
        self
    }

    /// Set formality level
    pub fn with_formality(mut self, level: f32) -> Self {
        self.config.formality_level = level.clamp(0.0, 1.0);
        self.builder_state.fields_set.push("formality_level".to_string());
        self.update_progress();
        self
    }

    /// Set enthusiasm level
    pub fn with_enthusiasm(mut self, level: f32) -> Self {
        self.config.enthusiasm_level = level.clamp(0.0, 1.0);
        self.builder_state.fields_set.push("enthusiasm_level".to_string());
        self.update_progress();
        self
    }

    /// Set helpfulness level
    pub fn with_helpfulness(mut self, level: f32) -> Self {
        self.config.helpfulness_level = level.clamp(0.0, 1.0);
        self.builder_state.fields_set.push("helpfulness_level".to_string());
        self.update_progress();
        self
    }

    /// Set creativity level
    pub fn with_creativity(mut self, level: f32) -> Self {
        self.config.creativity_level = level.clamp(0.0, 1.0);
        self.builder_state.fields_set.push("creativity_level".to_string());
        self.update_progress();
        self
    }

    /// Set verbosity level
    pub fn with_verbosity(mut self, level: f32) -> Self {
        self.config.verbosity_level = level.clamp(0.0, 1.0);
        self.builder_state.fields_set.push("verbosity_level".to_string());
        self.update_progress();
        self
    }

    /// Set humor level
    pub fn with_humor(mut self, level: f32) -> Self {
        self.config.humor_level = level.clamp(0.0, 1.0);
        self.builder_state.fields_set.push("humor_level".to_string());
        self.update_progress();
        self
    }

    /// Set empathy level
    pub fn with_empathy(mut self, level: f32) -> Self {
        self.config.empathy_level = level.clamp(0.0, 1.0);
        self.builder_state.fields_set.push("empathy_level".to_string());
        self.update_progress();
        self
    }

    /// Add custom trait
    pub fn with_custom_trait(mut self, name: String, value: f32) -> Self {
        self.config.custom_traits.insert(name.clone(), value.clamp(0.0, 1.0));
        self.builder_state.fields_set.push(format!("custom_trait_{}", name));
        self.update_progress();
        self
    }

    /// Add personality prompt
    pub fn with_prompt(mut self, prompt: String) -> Self {
        self.config.prompts.push(prompt);
        self.builder_state.fields_set.push("prompt".to_string());
        self.update_progress();
        self
    }

    /// Add personality example
    pub fn with_example(mut self, example: super::chat_core::PersonalityExample) -> Self {
        self.config.examples.push(example);
        self.builder_state.fields_set.push("example".to_string());
        self.update_progress();
        self
    }

    /// Update build progress
    fn update_progress(&mut self) {
        let total_fields = 12; // Total configurable fields
        self.builder_state.progress = (self.builder_state.fields_set.len() as f32) / (total_fields as f32);
        self.builder_state.last_modified = chrono::Utc::now();
    }

    /// Build the personality configuration (streaming)
    pub fn build(self) -> AsyncStream<PersonalityConfig> {
        AsyncStream::with_channel(move |sender| {
            let _ = sender.send(self.config);
        })
    }

    /// Validate the personality configuration (streaming)
    pub fn validate(&self) -> AsyncStream<ValidationResult> {
        let config = self.config.clone();
        
        AsyncStream::with_channel(move |sender| {
            let mut errors = Vec::new();
            let mut warnings = Vec::new();

            // Validate personality name
            if config.name.is_empty() {
                errors.push("Personality name cannot be empty".to_string());
            }

            // Validate levels are in range
            let levels = [
                ("formality", config.formality_level),
                ("enthusiasm", config.enthusiasm_level),
                ("helpfulness", config.helpfulness_level),
                ("creativity", config.creativity_level),
                ("verbosity", config.verbosity_level),
                ("humor", config.humor_level),
                ("empathy", config.empathy_level),
            ];

            for (name, level) in levels {
                if level < 0.0 || level > 1.0 {
                    errors.push(format!("{} level must be between 0.0 and 1.0", name));
                }
            }

            // Check for balanced personality
            let avg_level = (config.formality_level + config.enthusiasm_level + 
                           config.helpfulness_level + config.creativity_level +
                           config.verbosity_level + config.humor_level + 
                           config.empathy_level) / 7.0;
            
            if avg_level < 0.2 {
                warnings.push("Personality levels are very low, may result in bland responses".to_string());
            } else if avg_level > 0.9 {
                warnings.push("Personality levels are very high, may result in overwhelming responses".to_string());
            }

            let result = ValidationResult {
                is_valid: errors.is_empty(),
                errors,
                warnings,
                config_hash: format!("personality_{}", config.name),
            };

            let _ = sender.send(result);
        })
    }

    /// Get build progress
    pub fn get_progress(&self) -> f32 {
        self.builder_state.progress
    }

    /// Get builder state
    pub fn get_state(&self) -> &BuilderState {
        &self.builder_state
    }

    /// Reset builder to defaults
    pub fn reset(mut self) -> Self {
        self.config = PersonalityConfig::default();
        self.builder_state = BuilderState {
            fields_set: Vec::new(),
            is_valid: true,
            progress: 0.0,
            last_modified: chrono::Utc::now(),
        };
        self
    }

    /// Create preset personality configurations (streaming)
    pub fn create_preset(preset_name: &str) -> AsyncStream<PersonalityConfig> {
        let preset_name = preset_name.to_string();
        
        AsyncStream::with_channel(move |sender| {
            let config = match preset_name.as_str() {
                "professional" => PersonalityConfig {
                    name: Arc::from("Professional Assistant"),
                    description: Some("A professional and business-oriented assistant".to_string()),
                    response_style: super::chat_core::ResponseStyle::Professional,
                    formality_level: 0.9,
                    enthusiasm_level: 0.4,
                    helpfulness_level: 0.9,
                    creativity_level: 0.3,
                    verbosity_level: 0.6,
                    humor_level: 0.1,
                    empathy_level: 0.5,
                    ..Default::default()
                },
                "friendly" => PersonalityConfig {
                    name: Arc::from("Friendly Assistant"),
                    description: Some("A warm and approachable assistant".to_string()),
                    response_style: super::chat_core::ResponseStyle::Casual,
                    formality_level: 0.3,
                    enthusiasm_level: 0.8,
                    helpfulness_level: 0.9,
                    creativity_level: 0.6,
                    verbosity_level: 0.5,
                    humor_level: 0.6,
                    empathy_level: 0.9,
                    ..Default::default()
                },
                "technical" => PersonalityConfig {
                    name: Arc::from("Technical Expert"),
                    description: Some("A precise and technically-focused assistant".to_string()),
                    response_style: super::chat_core::ResponseStyle::Technical,
                    formality_level: 0.7,
                    enthusiasm_level: 0.5,
                    helpfulness_level: 0.9,
                    creativity_level: 0.4,
                    verbosity_level: 0.8,
                    humor_level: 0.2,
                    empathy_level: 0.4,
                    ..Default::default()
                },
                "creative" => PersonalityConfig {
                    name: Arc::from("Creative Assistant"),
                    description: Some("An imaginative and artistic assistant".to_string()),
                    response_style: super::chat_core::ResponseStyle::Creative,
                    formality_level: 0.2,
                    enthusiasm_level: 0.9,
                    helpfulness_level: 0.8,
                    creativity_level: 0.9,
                    verbosity_level: 0.7,
                    humor_level: 0.8,
                    empathy_level: 0.8,
                    ..Default::default()
                },
                _ => PersonalityConfig::default(),
            };

            let _ = sender.send(config);
        })
    }
}

impl Default for PersonalityConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for model configuration
pub struct ModelConfigBuilder {
    /// Model configuration being built
    pub config: ModelConfig,
    /// Builder state
    pub builder_state: BuilderState,
}

impl ModelConfigBuilder {
    /// Create a new model config builder
    pub fn new() -> Self {
        Self {
            config: ModelConfig::default(),
            builder_state: BuilderState {
                fields_set: Vec::new(),
                is_valid: true,
                progress: 0.0,
                last_modified: chrono::Utc::now(),
            },
        }
    }

    /// Set model name
    pub fn with_model_name(mut self, name: Arc<str>) -> Self {
        self.config.model_name = name;
        self.builder_state.fields_set.push("model_name".to_string());
        self.update_progress();
        self
    }

    /// Set provider
    pub fn with_provider(mut self, provider: Arc<str>) -> Self {
        self.config.provider = provider;
        self.builder_state.fields_set.push("provider".to_string());
        self.update_progress();
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = temperature.clamp(0.0, 2.0);
        self.builder_state.fields_set.push("temperature".to_string());
        self.update_progress();
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self.builder_state.fields_set.push("max_tokens".to_string());
        self.update_progress();
        self
    }

    /// Set streaming enabled
    pub fn with_streaming(mut self, enabled: bool) -> Self {
        self.config.enable_streaming = enabled;
        self.builder_state.fields_set.push("enable_streaming".to_string());
        self.update_progress();
        self
    }

    /// Update build progress
    fn update_progress(&mut self) {
        let total_fields = 10; // Total configurable fields
        self.builder_state.progress = (self.builder_state.fields_set.len() as f32) / (total_fields as f32);
        self.builder_state.last_modified = chrono::Utc::now();
    }

    /// Build the model configuration (streaming)
    pub fn build(self) -> AsyncStream<ModelConfig> {
        AsyncStream::with_channel(move |sender| {
            let _ = sender.send(self.config);
        })
    }
}

impl Default for ModelConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for behavior configuration
pub struct BehaviorConfigBuilder {
    /// Behavior configuration being built
    pub config: BehaviorConfig,
    /// Builder state
    pub builder_state: BuilderState,
}

impl BehaviorConfigBuilder {
    /// Create a new behavior config builder
    pub fn new() -> Self {
        Self {
            config: BehaviorConfig::default(),
            builder_state: BuilderState {
                fields_set: Vec::new(),
                is_valid: true,
                progress: 0.0,
                last_modified: chrono::Utc::now(),
            },
        }
    }

    /// Enable suggestions
    pub fn with_suggestions(mut self, enabled: bool) -> Self {
        self.config.enable_suggestions = enabled;
        self.builder_state.fields_set.push("enable_suggestions".to_string());
        self.update_progress();
        self
    }

    /// Enable context awareness
    pub fn with_context_awareness(mut self, enabled: bool) -> Self {
        self.config.enable_context_awareness = enabled;
        self.builder_state.fields_set.push("enable_context_awareness".to_string());
        self.update_progress();
        self
    }

    /// Set response delay
    pub fn with_response_delay(mut self, delay_ms: u64) -> Self {
        self.config.response_delay_ms = delay_ms;
        self.builder_state.fields_set.push("response_delay_ms".to_string());
        self.update_progress();
        self
    }

    /// Update build progress
    fn update_progress(&mut self) {
        let total_fields = 8; // Total configurable fields
        self.builder_state.progress = (self.builder_state.fields_set.len() as f32) / (total_fields as f32);
        self.builder_state.last_modified = chrono::Utc::now();
    }

    /// Build the behavior configuration (streaming)
    pub fn build(self) -> AsyncStream<BehaviorConfig> {
        AsyncStream::with_channel(move |sender| {
            let _ = sender.send(self.config);
        })
    }
}

impl Default for BehaviorConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}