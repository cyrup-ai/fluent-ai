//! Builder patterns for fluent-ai components
//!
//! This module contains all builder patterns moved from the domain package
//! to maintain architectural separation: domain contains only value objects,
//! fluent-ai contains all builders for construction and configuration.

pub mod agent_role;
pub mod chat;
pub mod completion;
pub mod model;

// Re-export all builders for convenience
pub use agent_role::*;
pub use chat::*;
pub use completion::*;
pub use model::*;