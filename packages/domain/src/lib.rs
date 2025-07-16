//! Fluent AI Domain Library
//!
//! This crate provides core domain types and traits for AI services.
//! All domain logic, message types, and business objects are defined here.

// Re-export cyrup_sugars for convenience
pub use cyrup_sugars::{OneOrMany, ZeroOneOrMany};

// Use std HashMap instead
pub use std::collections::HashMap;

// Define our own async task types
pub type AsyncTask<T> = tokio::task::JoinHandle<T>;

pub fn spawn_async<F, T>(future: F) -> AsyncTask<T>
where
    F: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    tokio::spawn(future)
}

// Create async_task module for compatibility
pub mod async_task {
    pub use super::{AsyncTask, spawn_async};
    
    // Define NotResult trait for compatibility
    pub trait NotResult {}
    impl<T> NotResult for T where T: Send + 'static {}
    
    // Error handlers module
    pub mod error_handlers {
        pub fn default_error_handler<T: std::fmt::Debug>(_error: T) {
            // Default error handler implementation
        }
    }
}

// Re-export streaming types
pub use futures::stream::Stream as AsyncStream;

// Domain modules
pub mod agent;
pub mod agent_role;
pub mod audio;
pub mod chunk;
pub mod completion;
pub mod context;
pub mod conversation;
pub mod chat;
pub mod engine;
pub mod document;
pub mod embedding;
pub mod extractor;
pub mod image;
pub mod library;
pub mod loader;
pub mod mcp;
pub mod mcp_tool;
pub mod memory;
pub mod memory_ops;
pub mod memory_workflow;
pub mod message;
pub mod model;
pub mod model_info_provider;
pub mod prompt;
pub mod provider;
pub mod tool;
pub mod tool_v2;
pub mod workflow;

// Re-export all types for convenience
pub use agent::*;
pub use agent_role::*;
pub use audio::*;
pub use chunk::*;
pub use completion::*;
pub use context::*;
pub use conversation::*;
pub use document::*;
pub use embedding::*;
pub use extractor::*;
pub use image::*;
pub use library::*;
pub use loader::*;
pub use mcp::*;
pub use mcp_tool::*;
pub use memory::*;
pub use memory_ops::*;
pub use memory_workflow::*;
pub use message::*;
pub use model::*;
pub use model_info_provider::*;
pub use prompt::*;
pub use provider::*;
pub use tool::*;
pub use tool_v2::*;
pub use workflow::*;