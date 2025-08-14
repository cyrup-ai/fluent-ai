//! Fluent AI Candle Domain Library
//!
//! This crate provides Candle-prefixed domain types and traits for AI services.
//! All domain logic, message types, and business objects are defined here with Candle prefixes
//! to ensure complete independence from the main fluent-ai domain package.

#![allow(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![forbid(unsafe_code)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

// Core modules and submodules
pub mod additional_types;
pub mod agent;
pub mod chat;
pub mod collections;
pub mod completion;
pub mod concurrency;
pub mod context;
pub mod core;
pub mod embedding;
pub mod engine;
pub mod error;
/// HTTP client and response handling for AI providers
pub mod http;
/// Image processing and vision model support
pub mod image;
pub mod init;
pub mod memory;
pub mod model;
/// Prompt construction and templating
pub mod prompt;
pub mod tool;
pub mod util;
pub mod voice;
// Use ZeroOneOrMany from cyrup_sugars directly
pub use cyrup_sugars::ZeroOneOrMany as CandleZeroOneOrMany;
// Re-export from cyrup_sugars for convenience with Candle prefixes
pub use cyrup_sugars::{ByteSize, OneOrMany};
// Alias for backward compatibility - people expect async_task module
pub use fluent_ai_async as async_task;
pub use fluent_ai_async::spawn_task as spawn_async; // Alias for backward compatibility
// Streaming primitives from fluent-ai-async (kept as-is per requirements)
pub use fluent_ai_async::{spawn_task, AsyncStream, AsyncStreamSender, AsyncTask, NotResult};
/// Agent workflow and execution patterns - removed fake implementation
// Re-export HashMap from hashbrown for domain consistency
pub use hashbrown::HashMap;

// Re-export only from minimal working modules
// Most re-exports temporarily disabled until import issues resolved
