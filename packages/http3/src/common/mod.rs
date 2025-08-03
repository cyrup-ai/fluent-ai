//! Common Utilities for HTTP3 Operations
//!
//! Shared utilities for cross-cutting concerns across all HTTP operations:
//! headers, authentication, retry logic, caching, and metrics.

pub mod auth;
pub mod auth_method;
pub mod cache;
pub mod content_types;
pub mod headers;
pub mod metrics;
pub mod retry;

// Re-export commonly used types and utilities
pub use auth::{ApiKey, AuthProvider, BasicAuth, BearerToken};
pub use auth_method::AuthMethod;
pub use cache::{CacheConfig, CacheEntry, CacheKey, CacheStats, ResponseCache};
pub use content_types::ContentTypes;
pub use headers::HeaderManager;
pub use metrics::{MetricsCollector, OperationMetrics, RequestMetrics};
pub use retry::{
    GLOBAL_RETRY_STATS, GlobalRetryStats, HttpRetryExecutor, RetryExecutor, RetryPolicy,
    RetryResult, RetryStats, execute_with_aggressive_retry, execute_with_conservative_retry,
    execute_with_default_retry, execute_without_retry, with_retry,
};

/// Common result type for utility operations
pub type UtilityResult<T> = Result<T, UtilityError>;

/// Unified error type for utility operations
#[derive(Debug, Clone)]
pub enum UtilityError {
    /// Header validation or manipulation error
    HeaderError,
    /// Authentication error
    AuthError,
    /// Retry logic error
    RetryError,
    /// Cache operation error
    CacheError,
    /// Metrics collection error
    MetricsError,
}

impl std::fmt::Display for UtilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UtilityError::HeaderError => write!(f, "Header operation failed"),
            UtilityError::AuthError => write!(f, "Authentication failed"),
            UtilityError::RetryError => write!(f, "Retry operation failed"),
            UtilityError::CacheError => write!(f, "Cache operation failed"),
            UtilityError::MetricsError => write!(f, "Metrics collection failed"),
        }
    }
}

impl std::error::Error for UtilityError {}
