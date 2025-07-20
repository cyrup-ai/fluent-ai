//! Middleware components for cross-cutting concerns in the fluent-ai application layer
//!
//! This module contains middleware implementations that handle:
//! - Command processing and validation
//! - Performance monitoring and metrics
//! - Security and authorization
//! - Caching and optimization
//!
//! Middleware belongs in the application orchestration layer (fluent-ai) as it handles
//! cross-cutting concerns that coordinate between domain entities and external services.

pub mod caching;
pub mod command;
pub mod performance;
pub mod security;

// Re-export commonly used middleware types
pub use caching::CachingMiddleware;
pub use command::{CommandMiddleware, MiddlewareChain};
pub use performance::PerformanceMiddleware;
pub use security::SecurityMiddleware;
