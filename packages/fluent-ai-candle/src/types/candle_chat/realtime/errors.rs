//! Real-time system error types
//!
//! This module defines all error types used throughout the real-time system,
//! providing comprehensive error handling with detailed context information.

use thiserror::Error;

/// Real-time system errors
#[derive(Debug, Error)]
pub enum RealTimeError {
    #[error("Backpressure exceeded: current size {current_size}, limit {limit}")]
    BackpressureExceeded { current_size: usize, limit: usize },
    
    #[error("Connection not found: {user_id}:{session_id}")]
    ConnectionNotFound { user_id: String, session_id: String },
    
    #[error("Subscription failed: {reason}")]
    SubscriptionFailed { reason: String },
    
    #[error("Message delivery failed: {reason}")]
    MessageDeliveryFailed { reason: String },
    
    #[error("System timeout: {operation}")]
    SystemTimeout { operation: String },
    
    #[error("Invalid message format: {details}")]
    InvalidMessageFormat { details: String },
    
    #[error("Rate limit exceeded: {current_rate}/{limit}")]
    RateLimitExceeded { current_rate: usize, limit: usize },
    
    #[error("System overload: {resource}")]
    SystemOverload { resource: String },
    
    #[error("Internal error: {0}")]
    InternalError(String),
    
    #[error("Subscriber not found: {0}")]
    SubscriberNotFound(String),
}

impl RealTimeError {
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::BackpressureExceeded { .. }
                | Self::SystemTimeout { .. }
                | Self::RateLimitExceeded { .. }
                | Self::SystemOverload { .. }
        )
    }

    /// Check if error indicates a temporary condition
    pub fn is_temporary(&self) -> bool {
        matches!(
            self,
            Self::BackpressureExceeded { .. }
                | Self::SystemTimeout { .. }
                | Self::SystemOverload { .. }
        )
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::InternalError(_) | Self::SystemOverload { .. } => ErrorSeverity::Critical,
            Self::ConnectionNotFound { .. } | Self::SubscriberNotFound(_) => ErrorSeverity::High,
            Self::MessageDeliveryFailed { .. } 
            | Self::SubscriptionFailed { .. } 
            | Self::InvalidMessageFormat { .. } => ErrorSeverity::Medium,
            Self::BackpressureExceeded { .. } 
            | Self::SystemTimeout { .. } 
            | Self::RateLimitExceeded { .. } => ErrorSeverity::Low,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

impl ErrorSeverity {
    /// Check if error requires immediate attention
    pub fn requires_immediate_action(&self) -> bool {
        matches!(self, Self::Critical | Self::High)
    }
}

impl Default for ErrorSeverity {
    fn default() -> Self {
        Self::Medium
    }
}

/// Specialized result type for real-time operations
pub type RealTimeResult<T> = Result<T, RealTimeError>;