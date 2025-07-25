//! Configuration module with decomposed submodules
//!
//! This module provides comprehensive configuration management for candle_chat
//! with zero-allocation streaming patterns and lock-free operations.

pub mod model_config;
pub mod config_manager;
pub mod config_builder;

// Decomposed chat configuration modules
pub mod chat_core;
pub mod behavior;
pub mod ui;
pub mod integration;

// Re-export core configuration types
pub use model_config::{
    ModelConfig, ModelParameters, ModelRetryConfig, ModelPerformanceConfig,
    CompressionType, ValidationResult, ConfigUpdate, UpdateResult, ConfigSummary
};

// Re-export decomposed chat configuration types
pub use chat_core::{
    ChatConfig, SessionConfig, PersonalityConfig, ResponseStyle, PersonalityExample
};

pub use behavior::{
    BehaviorConfig, TypingIndicatorConfig, ErrorHandlingConfig, ErrorLoggingLevel,
    ContentFilteringConfig, ContentFilter, FilterAction, ConversationFlowConfig, BehaviorRule
};

pub use ui::{
    UIConfig, ThemeConfig, FontConfig, FontWeight, LayoutConfig, SidebarPosition,
    AnimationConfig, EasingFunction, AccessibilityConfig
};

pub use integration::{
    IntegrationConfig, WebhookConfig, WebhookAuth, PluginConfig,
    HistoryConfig, SecurityConfig, SessionSecurityConfig, PerformanceConfig
};

// Re-export configuration management types
pub use config_manager::{
    ConfigurationManager, ConfigurationChangeEvent, ConfigurationPersistence,
    ConfigSection, ChangeSource, ValidationStatus, RollbackInfo, RollbackComplexity,
    BackupConfiguration, EncryptionSettings, CompressionSettings, VersionControlSettings,
    KeySource, ConfigurationValidator, ValidationRule, ValidationRuleType, ValidationSeverity,
    ConfigurationWatcher, PersonalityValidator, BehaviorValidator, UIValidator, IntegrationValidator,
    LoadResult, SaveResult, ConfigurationStatistics
};

// Re-export configuration builder types
pub use config_builder::{
    ConfigurationBuilder, PersonalityConfigBuilder, ModelConfigBuilder, BehaviorConfigBuilder,
    BuilderState
};