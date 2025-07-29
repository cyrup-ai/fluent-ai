//! Fluent AI Candle Domain Library
//!
//! This crate provides Candle-prefixed domain types and traits for AI services.
//! All domain logic, message types, and business objects are defined here with Candle prefixes
//! to ensure complete independence from the main fluent-ai domain package.

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![forbid(unsafe_code)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

// Core modules (minimal set for testing)
// Core modules (minimal set to get to 0 errors)
// Most modules temporarily disabled to fix critical issues first
pub mod collections;

// Re-export HashMap from hashbrown for domain consistency
pub use hashbrown::HashMap;

// Alias for backward compatibility - people expect async_task module
pub use fluent_ai_async as async_task;
// Re-export from cyrup_sugars for convenience with Candle prefixes
pub use cyrup_sugars::{ByteSize, OneOrMany};
pub use fluent_ai_async::spawn_task as spawn_async; // Alias for backward compatibility

// Streaming primitives from fluent-ai-async (kept as-is per requirements)
pub use fluent_ai_async::{AsyncStream, AsyncStreamSender, AsyncTask, NotResult, spawn_task};

// Use ZeroOneOrMany from cyrup_sugars directly
pub use cyrup_sugars::ZeroOneOrMany as CandleZeroOneOrMany;

// Re-export only from minimal working modules
// Most re-exports temporarily disabled until import issues resolved
