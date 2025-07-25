//! Configuration builders for ergonomic configuration creation
//!
//! This module provides fluent builder patterns for creating configuration objects
//! with type-safe construction, validation, and sensible defaults using zero-allocation
//! patterns and blazing-fast performance.

use std::sync::Arc;
use std::time::Duration;

use super::config_core::{
    ChatConfig, PersonalityConfig, BehaviorConfig, UIConfig, IntegrationConfig,
    ModelConfig, ModelRetryConfig, ModelPerformanceConfig,
};

/// Configuration builder for ergonomic configuration creation
pub struct ConfigurationBuilder {
    config: ChatConfig,
}

impl ConfigurationBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: ChatConfig::default(),
        }
    }

    /// Set maximum message length
    pub fn max_message_length(mut self, length: usize) -> Self {
        self.config.max_message_length = length;
        self
    }

    /// Enable or disable message history
    pub fn enable_history(mut self, enable: bool) -> Self {
        self.config.enable_history = enable;
        self
    }

    /// Set history retention period
    pub fn history_retention(mut self, duration: Duration) -> Self {
        self.config.history_retention = duration;
        self
    }

    /// Enable or disable streaming responses
    pub fn enable_streaming(mut self, enable: bool) -> Self {
        self.config.enable_streaming = enable;
        self
    }

    /// Set personality configuration
    pub fn personality(mut self, personality: PersonalityConfig) -> Self {
        self.config.personality = personality;
        self
    }

    /// Set behavior configuration
    pub fn behavior(mut self, behavior: BehaviorConfig) -> Self {
        self.config.behavior = behavior;
        self
    }

    /// Set UI configuration
    pub fn ui(mut self, ui: UIConfig) -> Self {
        self.config.ui = ui;
        self
    }

    /// Set integration configuration
    pub fn integration(mut self, integration: IntegrationConfig) -> Self {
        self.config.integration = integration;
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
    config: PersonalityConfig,
}

impl PersonalityConfigBuilder {
    /// Create a new personality configuration builder
    pub fn new() -> Self {
        Self {
            config: PersonalityConfig::default(),
        }
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

    /// Set custom instructions
    pub fn custom_instructions(mut self, instructions: impl Into<Arc<str>>) -> Self {
        self.config.custom_instructions = Some(instructions.into());
        self
    }

    /// Set creativity level (0.0-1.0)
    pub fn creativity(mut self, creativity: f64) -> Self {
        self.config.creativity = creativity.clamp(0.0, 1.0);
        self
    }

    /// Set formality level (0.0-1.0)
    pub fn formality(mut self, formality: f64) -> Self {
        self.config.formality = formality.clamp(0.0, 1.0);
        self
    }

    /// Set humor level (0.0-1.0)
    pub fn humor(mut self, humor: f64) -> Self {
        self.config.humor = humor.clamp(0.0, 1.0);
        self
    }

    /// Set empathy level (0.0-1.0)
    pub fn empathy(mut self, empathy: f64) -> Self {
        self.config.empathy = empathy.clamp(0.0, 1.0);
        self
    }

    /// Set expertise level
    pub fn expertise(mut self, expertise: impl Into<Arc<str>>) -> Self {
        self.config.expertise_level = expertise.into();
        self
    }

    /// Set verbosity level
    pub fn verbosity(mut self, verbosity: impl Into<Arc<str>>) -> Self {
        self.config.verbosity = verbosity.into();
        self
    }

    /// Add personality trait
    pub fn trait_name(mut self, trait_name: impl Into<Arc<str>>) -> Self {
        self.config.traits.push(trait_name.into());
        self
    }

    /// Add multiple traits
    pub fn traits(mut self, traits: Vec<impl Into<Arc<str>>>) -> Self {
        for trait_name in traits {
            self.config.traits.push(trait_name.into());
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
    config: BehaviorConfig,
}

impl BehaviorConfigBuilder {
    /// Create a new behavior configuration builder
    pub fn new() -> Self {
        Self {
            config: BehaviorConfig::default(),
        }
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

    /// Enable or disable auto-correction
    pub fn enable_auto_correction(mut self, enable: bool) -> Self {
        self.config.enable_auto_correction = enable;
        self
    }

    /// Set context awareness level (0.0-1.0)
    pub fn context_awareness(mut self, level: f64) -> Self {
        self.config.context_awareness = level.clamp(0.0, 1.0);
        self
    }

    /// Set proactivity level (0.0-1.0)
    pub fn proactivity(mut self, level: f64) -> Self {
        self.config.proactivity = level.clamp(0.0, 1.0);
        self
    }

    /// Set conversation flow type
    pub fn conversation_flow(mut self, flow: impl Into<Arc<str>>) -> Self {
        self.config.conversation_flow = flow.into();
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
    config: UIConfig,
}

impl UIConfigBuilder {
    /// Create a new UI configuration builder
    pub fn new() -> Self {
        Self {
            config: UIConfig::default(),
        }
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
    config: IntegrationConfig,
}

impl IntegrationConfigBuilder {
    /// Create a new integration configuration builder
    pub fn new() -> Self {
        Self {
            config: IntegrationConfig::default(),
        }
    }

    /// Enable or disable plugins
    pub fn enable_plugins(mut self, enable: bool) -> Self {
        self.config.enable_plugins = enable;
        self
    }

    /// Set plugin directory
    pub fn plugin_directory(mut self, directory: impl Into<Arc<str>>) -> Self {
        self.config.plugin_directory = Some(directory.into());
        self
    }

    /// Enable or disable webhooks
    pub fn enable_webhooks(mut self, enable: bool) -> Self {
        self.config.enable_webhooks = enable;
        self
    }

    /// Set webhook URL
    pub fn webhook_url(mut self, url: impl Into<Arc<str>>) -> Self {
        self.config.webhook_url = Some(url.into());
        self
    }

    /// Set API rate limit
    pub fn api_rate_limit(mut self, limit: u32) -> Self {
        self.config.api_rate_limit = limit;
        self
    }

    /// Add external service
    pub fn external_service(mut self, service: impl Into<Arc<str>>) -> Self {
        self.config.external_services.push(service.into());
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
    config: ModelConfig,
}

impl ModelConfigBuilder {
    /// Create a new model configuration builder
    pub fn new() -> Self {
        Self {
            config: ModelConfig::default(),
        }
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