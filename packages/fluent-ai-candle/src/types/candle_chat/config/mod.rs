//! Configuration module with decomposed submodules
//!
//! This module provides comprehensive configuration management for candle_chat
//! with zero-allocation streaming patterns and lock-free operations.

pub mod model_config;
pub mod chat_config;
pub mod config_manager;
pub mod config_builder;

// Re-export core configuration types
pub use model_config::{
    ModelConfig, ModelParameters, ModelRetryConfig, ModelPerformanceConfig,
    CompressionType, ValidationResult, ConfigUpdate, UpdateResult, ConfigSummary
};

// Re-export chat configuration types
pub use chat_config::{
    ChatConfig, SessionConfig, PersonalityConfig, BehaviorConfig, UIConfig, IntegrationConfig,
    ResponseStyle, PersonalityExample, TypingIndicatorConfig, ErrorHandlingConfig,
    ContentFilteringConfig, ConversationFlowConfig, BehaviorRule, ThemeConfig, FontConfig,
    LayoutConfig, AnimationConfig, AccessibilityConfig, WebhookConfig, PluginConfig,
    HistoryConfig, SecurityConfig, PerformanceConfig, FontWeight, SidebarPosition,
    EasingFunction, WebhookAuth, ErrorLoggingLevel, ContentFilter, FilterAction
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