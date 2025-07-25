//! Chat integrations module
//!
//! Provides comprehensive external integration capabilities with zero-allocation
//! patterns and ergonomic APIs for webhook, REST API, plugin, and service integrations.

pub mod types;
pub mod manager;
pub mod external;
pub mod plugin;

// Re-export commonly used types for ergonomic API
pub use types::{
    IntegrationType,
    IntegrationConfig,
    IntegrationError,
    IntegrationResult,
    IntegrationStats,
    IntegrationRequest,
    IntegrationResponse,
    Plugin,
    PluginConfig,
    create_webhook_integration,
    create_rest_api_integration,
    create_plugin_integration,
};

pub use manager::IntegrationManager;
pub use external::ExternalIntegration;
pub use plugin::PluginManager;