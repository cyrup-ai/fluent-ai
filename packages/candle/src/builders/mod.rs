//! Builder patterns for candle components
//!
//! This module contains candle-specific builder patterns following fluent-ai
//! architecture but with candle prefixes. NO trait objects allowed - only
//! impl Trait patterns for zero allocation.

pub mod agent_role;
pub mod completion;

// Re-export main builder types for public API
pub use agent_role::{CandleAgentBuilder, CandleAgentRoleBuilder, CandleFluentAi};
