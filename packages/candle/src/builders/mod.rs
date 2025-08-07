//! Builder patterns for fluent-ai components
//!
//! This module contains all builder patterns moved from the domain package
//! to maintain architectural separation: domain contains only value objects,
//! fluent-ai contains all builders for construction and configuration.

pub mod agent_role;
pub mod completion;

// Real workflow builders
pub mod workflow;
pub mod memory_workflow;

// Re-export main builder types for public API
pub use agent_role::{CandleFluentAi, CandleAgentRoleBuilder, CandleAgentBuilder};