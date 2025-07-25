//! Comprehensive resilience patterns for embedding operations
//!
//! This module provides enterprise-grade resilience capabilities including:
//! - Advanced circuit breaker patterns with adaptive thresholds
//! - Multi-tier failure detection and recovery strategies
//! - Exponential backoff with jitter for retry logic
//! - Resource isolation and cascading failure prevention
//! - Lock-free state management for zero-contention access

pub mod circuit_breaker;

// Re-export core types
pub use circuit_breaker::{
    AdaptiveThresholdCalculator, CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError,
    CircuitBreakerRegistry, CircuitMetricsSnapshot, CircuitState, CircuitStateChange,
    ErrorCategory, ErrorEntry, FailureType, RegistryStats};
