//! Fluent AI Candle Library
//! 
//! This crate provides Candle ML framework integration for AI services.
//! All Candle-prefixed domain types, builders, and providers are defined here
//! to ensure complete independence from the main fluent-ai packages.

#![allow(missing_docs)]
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
/// Candle model providers for local inference
pub mod providers;
/// Chat functionality and loop control
pub mod chat;
/// Core engine for completion processing (temporarily disabled)
// pub mod engine;

// Essential Candle re-exports for public API (minimal set)
// Domain types will be added as they become available

// Prelude - All types needed for ARCHITECTURE.md syntax
pub mod prelude {
    pub use crate::builders::{
        CandleFluentAi, CandleAgentRoleBuilder, CandleAgentBuilder,
    };
    pub use crate::domain::chat::message::CandleMessageChunk;
    pub use crate::builders::agent_role::CandleChatLoop;
    
    pub use crate::domain::{
        agent::CandleAgent,
        chat::message::types::CandleMessageRole,
        context::{CandleContext, FinishReason, provider::{CandleFile, CandleFiles, CandleDirectory, CandleGithub}},
        tool::{CandleExecToText, core::CandlePerplexity},
    };
    
    pub use crate::providers::{CandleKimiK2Provider, CandleKimiK2Config};
    
    // Placeholder types for ARCHITECTURE.md completeness
    pub struct CandleModels;
    pub struct CandleLibrary;
    
    impl CandleLibrary {
        pub fn named(_name: &str) -> Self {
            Self
        }
    }
    
    impl CandleModels {
        pub const KIMI_K2: Self = Self;
    }
    
    // Re-export tool implementation that provides static methods
    pub use crate::domain::tool::core::CandleToolImpl as CandleTool;
    
    pub use fluent_ai_async::AsyncStream;
    
    // Helper function for ARCHITECTURE.md example
    pub fn process_turn() -> CandleChatLoop {
        CandleChatLoop::Reprompt("continue".to_string())
    }
}

// Re-export everything from prelude at root level for convenience
pub use prelude::*;

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