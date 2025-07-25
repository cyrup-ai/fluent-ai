//! Common Utilities for HTTP3 Operations
//!
//! Shared utilities for cross-cutting concerns across all HTTP operations:
//! headers, authentication, retry logic, caching, and metrics.

pub mod auth;
pub mod cache;
pub mod headers;
pub mod metrics;
pub mod retry;

// Re-export commonly used types and utilities
pub use auth::{ApiKey, AuthProvider, BasicAuth, BearerToken};
pub use cache::{CacheConfig, CacheEntry, CacheKey, CacheStats, ResponseCache};
pub use headers::HeaderManager;
pub use metrics::{MetricsCollector, OperationMetrics, RequestMetrics};
pub use retry::{HttpRetryExecutor, RetryExecutor, RetryPolicy, RetryResult, RetryStats};

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
    MetricsError}

impl std::fmt::Display for UtilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UtilityError::HeaderError => write!(f, "Header operation failed"),
            UtilityError::AuthError => write!(f, "Authentication failed"),
            UtilityError::RetryError => write!(f, "Retry operation failed"),
            UtilityError::CacheError => write!(f, "Cache operation failed"),
            UtilityError::MetricsError => write!(f, "Metrics collection failed")}
    }
}

impl std::error::Error for UtilityError {}
