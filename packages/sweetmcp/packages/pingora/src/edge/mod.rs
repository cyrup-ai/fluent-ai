//! Edge service module decomposition
//!
//! This module provides the decomposed EdgeService functionality split into
//! logical modules for better maintainability and adherence to the 300-line limit.

pub mod auth;
pub mod core;
pub mod routing;

// Re-export key types and functions for backward compatibility
pub use core::{EdgeService, EdgeServiceBuilder, EdgeServiceError, ServiceStats};

pub use auth::{AuthConfig, AuthContext, AuthHandler, AuthMethod, AuthResult, UserClaims};
pub use routing::{RoutingContext, RoutingHandler, RoutingStrategy};
