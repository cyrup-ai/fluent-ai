//! Configuration builder for fluent configuration creation
//!
//! This module provides builder patterns for creating configurations
//! with validation and fluent APIs.

use std::sync::Arc;

use uuid::Uuid;

use super::core::{
    ChatConfig, PersonalityConfig, BehaviorConfig, UIConfig, IntegrationConfig,
    LanguageHandlingConfig, DisplayConfig, ApiIntegrationConfig, PluginConfig
};
use super::model::{ModelConfig, ModelRetryConfig, ModelPerformanceConfig};
use super::builders::{
    ModelConfigBuilder, PersonalityConfigBuilder, BehaviorConfigBuilder, UIConfigBuilder
};

/// Configuration builder for fluent configuration creation
#[derive(Debug, Clone)]
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

    /// Set configuration name
    pub fn name(mut self, name: impl Into<Arc<str>>) -> Self {
        self.config.name = name.into();
        self
    }

    /// Set configuration description
    pub fn description(mut self, description: impl Into<Arc<str>>) -> Self {
        self.config.description = Some(description.into());
        self
    }

    /// Configure model settings
    pub fn model<F>(mut self, configure: F) -> Self
    where
        F: FnOnce(ModelConfigBuilder) -> ModelConfigBuilder,
    {
        let builder = ModelConfigBuilder::new(self.config.model);
        self.config.model = configure(builder).build();
        self
    }

    /// Configure personality settings
    pub fn personality<F>(mut self, configure: F) -> Self
    where
        F: FnOnce(PersonalityConfigBuilder) -> PersonalityConfigBuilder,
    {
        let builder = PersonalityConfigBuilder::new(self.config.personality);
        self.config.personality = configure(builder).build();
        self
    }

    /// Configure behavior settings
    pub fn behavior<F>(mut self, configure: F) -> Self
    where
        F: FnOnce(BehaviorConfigBuilder) -> BehaviorConfigBuilder,
    {
        let builder = BehaviorConfigBuilder::new(self.config.behavior);
        self.config.behavior = configure(builder).build();
        self
    }

    /// Configure UI settings
    pub fn ui<F>(mut self, configure: F) -> Self
    where
        F: FnOnce(UIConfigBuilder) -> UIConfigBuilder,
    {
        let builder = UIConfigBuilder::new(self.config.ui);
        self.config.ui = configure(builder).build();
        self
    }

    /// Build the final configuration
    pub fn build(self) -> ChatConfig {
        self.config
    }
}

impl Default for ConfigurationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Builder implementations are re-exported from the builders module
pub use super::builders::{
    ModelConfigBuilder, PersonalityConfigBuilder, BehaviorConfigBuilder, UIConfigBuilder
};

