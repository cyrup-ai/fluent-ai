//! Detailed builder implementations for configuration types
//!
//! This module contains the detailed builder implementations for various
//! configuration types to keep the main builder module focused.

use std::sync::Arc;
use std::time::Duration;

use super::core::{
    PersonalityConfig, BehaviorConfig, UIConfig, 
    LanguageHandlingConfig, DisplayConfig
};
use super::model::{ModelConfig, ModelRetryConfig, ModelPerformanceConfig};

/// Model configuration builder
#[derive(Debug, Clone)]
pub struct ModelConfigBuilder {
    config: ModelConfig}

impl ModelConfigBuilder {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }

    pub fn provider(mut self, provider: impl Into<Arc<str>>) -> Self {
        self.config.provider = provider.into();
        self
    }

    pub fn model_name(mut self, model_name: impl Into<Arc<str>>) -> Self {
        self.config.model_name = model_name.into();
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = temperature;
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    pub fn system_prompt(mut self, prompt: impl Into<Arc<str>>) -> Self {
        self.config.system_prompt = Some(prompt.into());
        self
    }

    pub fn build(self) -> ModelConfig {
        self.config
    }
}

/// Personality configuration builder
#[derive(Debug, Clone)]
pub struct PersonalityConfigBuilder {
    config: PersonalityConfig}

impl PersonalityConfigBuilder {
    pub fn new(config: PersonalityConfig) -> Self {
        Self { config }
    }

    pub fn name(mut self, name: impl Into<Arc<str>>) -> Self {
        self.config.name = name.into();
        self
    }

    pub fn personality_type(mut self, personality_type: impl Into<Arc<str>>) -> Self {
        self.config.personality_type = personality_type.into();
        self
    }

    pub fn response_style(mut self, style: impl Into<Arc<str>>) -> Self {
        self.config.response_style = style.into();
        self
    }

    pub fn tone(mut self, tone: impl Into<Arc<str>>) -> Self {
        self.config.tone = tone.into();
        self
    }

    pub fn formality_level(mut self, level: f32) -> Self {
        self.config.formality_level = level;
        self
    }

    pub fn creativity_level(mut self, level: f32) -> Self {
        self.config.creativity_level = level;
        self
    }

    pub fn empathy_level(mut self, level: f32) -> Self {
        self.config.empathy_level = level;
        self
    }

    pub fn humor_level(mut self, level: f32) -> Self {
        self.config.humor_level = level;
        self
    }

    pub fn add_trait(mut self, trait_name: impl Into<Arc<str>>) -> Self {
        self.config.custom_traits.push(trait_name.into());
        self
    }

    pub fn personality_prompt(mut self, prompt: impl Into<Arc<str>>) -> Self {
        self.config.personality_prompt = Some(prompt.into());
        self
    }

    pub fn build(self) -> PersonalityConfig {
        self.config
    }
}

impl Default for PersonalityConfigBuilder {
    fn default() -> Self {
        Self::new(PersonalityConfig::default())
    }
}

/// Behavior configuration builder
#[derive(Debug, Clone)]
pub struct BehaviorConfigBuilder {
    config: BehaviorConfig}

impl BehaviorConfigBuilder {
    pub fn new(config: BehaviorConfig) -> Self {
        Self { config }
    }

    pub fn enable_memory(mut self, enable: bool) -> Self {
        self.config.enable_memory = enable;
        self
    }

    pub fn memory_duration(mut self, duration: Duration) -> Self {
        self.config.memory_duration = duration;
        self
    }

    pub fn max_conversation_length(mut self, length: u32) -> Self {
        self.config.max_conversation_length = length;
        self
    }

    pub fn auto_save_conversations(mut self, auto_save: bool) -> Self {
        self.config.auto_save_conversations = auto_save;
        self
    }

    pub fn response_delay_ms(mut self, delay: u64) -> Self {
        self.config.response_delay_ms = delay;
        self
    }

    pub fn enable_typing_indicators(mut self, enable: bool) -> Self {
        self.config.enable_typing_indicators = enable;
        self
    }

    pub fn typing_speed_cps(mut self, speed: f32) -> Self {
        self.config.typing_speed_cps = speed;
        self
    }

    pub fn content_filtering(mut self, level: impl Into<Arc<str>>) -> Self {
        self.config.content_filtering = level.into();
        self
    }

    pub fn preferred_language(mut self, language: impl Into<Arc<str>>) -> Self {
        self.config.language_handling.preferred_language = language.into();
        self
    }

    pub fn build(self) -> BehaviorConfig {
        self.config
    }
}

/// UI configuration builder
#[derive(Debug, Clone)]
pub struct UIConfigBuilder {
    config: UIConfig}

impl UIConfigBuilder {
    pub fn new(config: UIConfig) -> Self {
        Self { config }
    }

    pub fn theme(mut self, theme: impl Into<Arc<str>>) -> Self {
        self.config.theme = theme.into();
        self
    }

    pub fn font_size_multiplier(mut self, multiplier: f32) -> Self {
        self.config.font_size_multiplier = multiplier;
        self
    }

    pub fn enable_animations(mut self, enable: bool) -> Self {
        self.config.enable_animations = enable;
        self
    }

    pub fn animation_speed(mut self, speed: f32) -> Self {
        self.config.animation_speed = speed;
        self
    }

    pub fn enable_sound_effects(mut self, enable: bool) -> Self {
        self.config.enable_sound_effects = enable;
        self
    }

    pub fn sound_volume(mut self, volume: f32) -> Self {
        self.config.sound_volume = volume;
        self
    }

    pub fn show_timestamps(mut self, show: bool) -> Self {
        self.config.display.show_timestamps = show;
        self
    }

    pub fn show_avatars(mut self, show: bool) -> Self {
        self.config.display.show_avatars = show;
        self
    }

    pub fn message_bubble_style(mut self, style: impl Into<Arc<str>>) -> Self {
        self.config.display.message_bubble_style = style.into();
        self
    }

    pub fn enable_markdown(mut self, enable: bool) -> Self {
        self.config.display.enable_markdown = enable;
        self
    }

    pub fn build(self) -> UIConfig {
        self.config
    }
}