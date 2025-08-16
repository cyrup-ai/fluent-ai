//! Proxy matching system with zero-allocation patterns
//!
//! Modular organization of proxy matcher functionality including
//! pattern matching, builder configuration, and system integration.

pub mod builder;
pub mod implementation;
pub mod intercept;
pub mod public_interface;
pub mod system_integration;
pub mod types;

// Re-export main types for convenience
pub use builder::MatcherBuilder;
pub use implementation::{Intercept, Matcher as MatcherImpl, Via};
// Re-export key functions
pub use intercept::{Intercept as InterceptConfig, Via as ProxyVia};
pub use public_interface::*;
pub use system_integration::SystemProxy;
pub use types::{Intercepted, Matcher};
