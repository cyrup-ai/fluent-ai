//! Rate limiting module decomposition
//!
//! This module provides the decomposed rate limiting functionality split into
//! logical modules for better maintainability and adherence to the 300-line limit.

pub mod algorithms;
pub mod distributed;
pub mod limiter;

// Re-export key types and functions for backward compatibility
pub use algorithms::{
    AlgorithmState, HybridAlgorithm, RateLimitAlgorithm, RateLimiter, SlidingWindow, TokenBucket,
};
pub use distributed::{
    DistributedRateLimitManager, DistributedRateLimitState, DistributedRateLimitSummary,
    EndpointRateConfig,
};
pub use limiter::{
    AdvancedRateLimitManager, RateLimitAlgorithmType, RateLimitConfig, RateLimitStats,
    RateLimitStatsSnapshot, SlidingWindowConfig, TokenBucketConfig,
};
