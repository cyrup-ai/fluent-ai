//! Authentication and authorization module
//!
//! This module provides comprehensive authentication and authorization for edge
//! requests with zero allocation patterns and blazing-fast performance.

pub mod core;
pub mod validation;

// Re-export core types for ergonomic use
pub use core::{AuthConfig, AuthContext, AuthHandler, AuthMethod, UserClaims};

// Re-export validation functionality
pub use validation::*;
