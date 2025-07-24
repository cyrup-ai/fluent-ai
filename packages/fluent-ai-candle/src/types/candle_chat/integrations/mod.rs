//! Integrations module with decomposed submodules
//!
//! This module provides comprehensive integration management for candle_chat
//! with zero-allocation streaming patterns and lock-free operations.

pub mod integration_types;
pub mod integration_manager;
pub mod plugin_manager;

// Re-export core integration types
pub use integration_types::{
    IntegrationConfig, IntegrationType, AuthConfig, AuthType, TokenRefreshConfig,
    RateLimitConfig, RateLimitStrategy, RateLimitRule, TimeoutConfig, RetryConfig,
    IntegrationRequest, IntegrationResponse, IntegrationStats, ExternalIntegration,
    IntegrationStatus, IntegrationCapability, HttpMethod
};

// Re-export integration management types
pub use integration_manager::{
    IntegrationManager, ManagerConfig, ManagerStatistics, IntegrationEventHandler,
    IntegrationEvent, HealthCheckResult, IntegrationHealthStatus, LogLevel,
    create_webhook_integration, create_rest_api_integration, create_plugin_integration
};

// Re-export plugin management types
pub use plugin_manager::{
    PluginManager, PluginConfig, PluginDependency, DependencyType, PluginPermission,
    LoadedPlugin, PluginStatus, PluginPerformanceMetrics, PluginRegistry, PluginMetadata,
    RegistryConfig, PluginManagerConfig, PluginManagerStats, PluginEventListener,
    PluginEvent, PluginExecutionResult, RegistryUpdateResult
};