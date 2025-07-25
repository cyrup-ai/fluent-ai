//! Configuration builders for ergonomic configuration creation
//!
//! This module provides fluent builder patterns for creating configuration objects
//! with type-safe construction, validation, and sensible defaults using zero-allocation
//! patterns and blazing-fast performance.

use std::sync::Arc;
use std::time::Duration;

use super::core::{PersonalityConfig, IntegrationConfig, ChatConfig};
use super::config_core::{BehaviorConfig, UIConfig};
use super::model::ModelConfig;

/// Configuration builder for ergonomic configuration creation
pub struct ConfigurationBuilder {
    config: ChatConfig}

impl ConfigurationBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: ChatConfig::default()}
    }

    /// Set configuration name
    pub fn name(mut self, name: impl Into<Arc<str>>) -> Self {
        self.config.name = name.into();
        self
    }

    /// Set maximum conversation length
    pub fn max_conversation_length(mut self, length: u32) -> Self {
        self.config.behavior.max_conversation_length = length;
        self
    }

    /// Enable or disable conversation memory
    pub fn enable_memory(mut self, enable: bool) -> Self {
        self.config.behavior.enable_memory = enable;
        self
    }

    /// Set memory retention period
    pub fn memory_duration(mut self, duration: Duration) -> Self {
        self.config.behavior.memory_duration = duration;
        self
    }

    /// Set response delay in milliseconds
    pub fn response_delay_ms(mut self, delay: u64) -> Self {
        self.config.behavior.response_delay_ms = delay;
        self
    }

    /// Set personality configuration
    pub fn personality(mut self, personality: PersonalityConfig) -> Self {
        self.config.personality = personality;
        self
    }

    /// Configure personality with a closure
    pub fn personality_with<F>(mut self, f: F) -> Self 
    where F: FnOnce(PersonalityConfigBuilder) -> PersonalityConfigBuilder
    {
        let builder = PersonalityConfigBuilder::new();
        self.config.personality = f(builder).build();
        self
    }

    /// Set behavior configuration
    pub fn behavior(mut self, behavior: BehaviorConfig) -> Self {
        self.config.behavior = behavior;
        self
    }

    /// Configure behavior with a closure
    pub fn behavior_with<F>(mut self, f: F) -> Self 
    where F: FnOnce(BehaviorConfigBuilder) -> BehaviorConfigBuilder
    {
        let builder = BehaviorConfigBuilder::new();
        self.config.behavior = f(builder).build();
        self
    }

    /// Set UI configuration
    pub fn ui(mut self, ui: UIConfig) -> Self {
        self.config.ui = ui;
        self
    }

    /// Configure UI with a closure
    pub fn ui_with<F>(mut self, f: F) -> Self 
    where F: FnOnce(UIConfigBuilder) -> UIConfigBuilder
    {
        let builder = UIConfigBuilder::new();
        self.config.ui = f(builder).build();
        self
    }

    /// Set integration configuration
    pub fn integration(mut self, integration: IntegrationConfig) -> Self {
        self.config.integrations = integration;
        self
    }

    /// Configure integrations with a closure
    pub fn integration_with<F>(mut self, f: F) -> Self 
    where F: FnOnce(IntegrationConfigBuilder) -> IntegrationConfigBuilder
    {
        let builder = IntegrationConfigBuilder::new();
        self.config.integrations = f(builder).build();
        self
    }

    /// Set model configuration
    pub fn model(mut self, model: ModelConfig) -> Self {
        self.config.model = model;
        self
    }

    /// Configure model with a closure
    pub fn model_with<F>(mut self, f: F) -> Self 
    where F: FnOnce(ModelConfigBuilder) -> ModelConfigBuilder
    {
        let builder = ModelConfigBuilder::new();
        self.config.model = f(builder).build();
        self
    }

    /// Build the configuration
    pub fn build(self) -> ChatConfig {
        self.config
    }
}

impl Default for ConfigurationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Personality configuration builder
pub struct PersonalityConfigBuilder {
    config: PersonalityConfig}

impl PersonalityConfigBuilder {
    /// Create a new personality configuration builder
    pub fn new() -> Self {
        Self {
            config: PersonalityConfig::default()}
    }

    /// Set personality type
    pub fn personality_type(mut self, personality_type: impl Into<Arc<str>>) -> Self {
        self.config.personality_type = personality_type.into();
        self
    }

    /// Set response style
    pub fn response_style(mut self, style: impl Into<Arc<str>>) -> Self {
        self.config.response_style = style.into();
        self
    }

    /// Set tone
    pub fn tone(mut self, tone: impl Into<Arc<str>>) -> Self {
        self.config.tone = tone.into();
        self
    }

    /// Set personality prompt
    pub fn custom_instructions(mut self, instructions: impl Into<Arc<str>>) -> Self {
        self.config.personality_prompt = Some(instructions.into());
        self
    }

    /// Set creativity level (0.0-1.0)
    pub fn creativity(mut self, creativity: f32) -> Self {
        self.config.creativity_level = creativity.clamp(0.0, 1.0);
        self
    }

    /// Set formality level (0.0-1.0)
    pub fn formality(mut self, formality: f32) -> Self {
        self.config.formality_level = formality.clamp(0.0, 1.0);
        self
    }

    /// Set humor level (0.0-1.0)
    pub fn humor(mut self, humor: f32) -> Self {
        self.config.humor_level = humor.clamp(0.0, 1.0);
        self
    }

    /// Set empathy level (0.0-1.0)
    pub fn empathy(mut self, empathy: f32) -> Self {
        self.config.empathy_level = empathy.clamp(0.0, 1.0);
        self
    }

    /// Add personality trait
    pub fn trait_name(mut self, trait_name: impl Into<Arc<str>>) -> Self {
        self.config.custom_traits.push(trait_name.into());
        self
    }

    /// Add multiple traits
    pub fn traits(mut self, traits: Vec<impl Into<Arc<str>>>) -> Self {
        for trait_name in traits {
            self.config.custom_traits.push(trait_name.into());
        }
        self
    }

    /// Build the personality configuration
    pub fn build(self) -> PersonalityConfig {
        self.config
    }
}

impl Default for PersonalityConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Behavior configuration builder
pub struct BehaviorConfigBuilder {
    config: BehaviorConfig}

impl BehaviorConfigBuilder {
    /// Create a new behavior configuration builder
    pub fn new() -> Self {
        Self {
            config: BehaviorConfig::default()}
    }

    /// Set response delay in milliseconds
    pub fn response_delay_ms(mut self, delay: u64) -> Self {
        self.config.response_delay_ms = delay;
        self
    }

    /// Enable or disable typing indicators
    pub fn enable_typing_indicators(mut self, enable: bool) -> Self {
        self.config.enable_typing_indicators = enable;
        self
    }

    /// Enable or disable memory
    pub fn enable_memory(mut self, enable: bool) -> Self {
        self.config.enable_memory = enable;
        self
    }

    /// Set memory duration
    pub fn memory_duration(mut self, duration: std::time::Duration) -> Self {
        self.config.memory_duration = duration;
        self
    }

    /// Set maximum conversation length
    pub fn max_conversation_length(mut self, length: u32) -> Self {
        self.config.max_conversation_length = length;
        self
    }

    /// Enable auto-save conversations
    pub fn auto_save_conversations(mut self, enable: bool) -> Self {
        self.config.auto_save_conversations = enable;
        self
    }

    /// Set typing speed (characters per second)
    pub fn typing_speed_cps(mut self, speed: f32) -> Self {
        self.config.typing_speed_cps = speed;
        self
    }

    /// Enable message reactions
    pub fn enable_reactions(mut self, enable: bool) -> Self {
        self.config.enable_reactions = enable;
        self
    }

    /// Set content filtering level
    pub fn content_filtering(mut self, level: impl Into<Arc<str>>) -> Self {
        self.config.content_filtering = level.into();
        self
    }

    /// Build the behavior configuration
    pub fn build(self) -> BehaviorConfig {
        self.config
    }
}

impl Default for BehaviorConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// UI configuration builder
pub struct UIConfigBuilder {
    config: UIConfig}

impl UIConfigBuilder {
    /// Create a new UI configuration builder
    pub fn new() -> Self {
        Self {
            config: UIConfig::default()}
    }

    /// Set theme
    pub fn theme(mut self, theme: impl Into<Arc<str>>) -> Self {
        self.config.theme = theme.into();
        self
    }

    /// Set font size
    pub fn font_size(mut self, size: u32) -> Self {
        self.config.font_size = size.clamp(8, 72);
        self
    }

    /// Enable or disable animations
    pub fn enable_animations(mut self, enable: bool) -> Self {
        self.config.enable_animations = enable;
        self
    }

    /// Set animation speed (0.0-2.0)
    pub fn animation_speed(mut self, speed: f32) -> Self {
        self.config.animation_speed = speed.clamp(0.0, 2.0);
        self
    }

    /// Set layout type
    pub fn layout(mut self, layout: impl Into<Arc<str>>) -> Self {
        self.config.layout = layout.into();
        self
    }

    /// Set color scheme
    pub fn color_scheme(mut self, scheme: impl Into<Arc<str>>) -> Self {
        self.config.color_scheme = scheme.into();
        self
    }

    /// Build the UI configuration
    pub fn build(self) -> UIConfig {
        self.config
    }
}

impl Default for UIConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Integration configuration builder
pub struct IntegrationConfigBuilder {
    config: IntegrationConfig}

impl IntegrationConfigBuilder {
    /// Create a new integration configuration builder
    pub fn new() -> Self {
        Self {
            config: IntegrationConfig::default()}
    }

    /// Enable or disable webhooks
    pub fn enable_webhooks(mut self, enable: bool) -> Self {
        self.config.enable_webhooks = enable;
        self
    }

    /// Add webhook URL
    pub fn webhook_url(mut self, url: impl Into<Arc<str>>) -> Self {
        self.config.webhook_urls.push(url.into());
        self
    }

    /// Add API integration
    pub fn api_integration(mut self, api_integration: super::core::ApiIntegrationConfig) -> Self {
        self.config.api_integrations.push(api_integration);
        self
    }

    /// Add plugin
    pub fn plugin(mut self, plugin: super::core::PluginConfig) -> Self {
        self.config.plugins.push(plugin);
        self
    }

    /// Build the integration configuration
    pub fn build(self) -> IntegrationConfig {
        self.config
    }
}

impl Default for IntegrationConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Model configuration builder
pub struct ModelConfigBuilder {
    config: ModelConfig}

impl ModelConfigBuilder {
    /// Create a new model configuration builder
    pub fn new() -> Self {
        Self {
            config: ModelConfig::default()}
    }

    /// Set provider and model
    pub fn provider_and_model(mut self, provider: impl Into<Arc<str>>, model: impl Into<Arc<str>>) -> Self {
        self.config.provider = provider.into();
        self.config.model_name = model.into();
        self
    }

    /// Set temperature (0.0-2.0)
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = temperature.clamp(0.0, 2.0);
        self
    }

    /// Set max tokens
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.config.max_tokens = Some(tokens);
        self
    }

    /// Set system prompt
    pub fn system_prompt(mut self, prompt: impl Into<Arc<str>>) -> Self {
        self.config.system_prompt = Some(prompt.into());
        self
    }

    /// Build the model configuration
    pub fn build(self) -> ModelConfig {
        self.config
    }
}

impl Default for ModelConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}