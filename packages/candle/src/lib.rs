//! Fluent AI Candle Library
//!
//! This crate provides Candle ML framework integration for AI services.
//! All Candle-prefixed domain types, builders, and providers are defined here
//! to ensure complete independence from the main fluent-ai packages.

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![forbid(unsafe_code)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

// Candle-specific modules (minimal set for core functionality)
/// Candle domain types (replaces fluent_ai_domain dependency)
pub mod domain;
/// Candle builders for zero-allocation construction patterns
pub mod builders;  
/// Candle model providers for local inference - temporarily disabled
// pub mod providers;

// Essential Candle re-exports for public API (minimal set)
// Domain types will be added as they become available

// Re-export main builders for public API
pub use builders::{CandleFluentAi, CandleAgentRoleBuilder, CandleAgentBuilder};

// Re-export main providers - temporarily disabled due to domain dependencies
// pub use providers::CandleKimiK2Provider;

// Streaming primitives from fluent-ai-async (kept as-is per requirements)
pub use fluent_ai_async::{AsyncStream, AsyncStreamSender, AsyncTask, spawn_task};

// Alias for backward compatibility - people expect async_task module
pub use fluent_ai_async as async_task;
pub use fluent_ai_async::spawn_task as spawn_async;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_architecture_md_syntax_works() {
        // Test that ARCHITECTURE.md builder pattern still works after all fixes
        let _agent = CandleFluentAi::agent_role("test-agent")
            .temperature(0.7)
            .max_tokens(1000)
            .system_prompt("You are a helpful assistant")
            .into_agent();
        
        // If this compiles, the ARCHITECTURE.md syntax is working! ✅
    }
}
