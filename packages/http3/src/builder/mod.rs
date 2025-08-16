//! Http3Builder module with decomposed components
//!
//! Provides the main Http3Builder struct and related types for constructing
//! HTTP requests with zero allocation and elegant fluent interface.

pub mod builder_core;
pub mod configuration;
pub mod content_type;
pub mod state_types;
pub mod trait_impls;

// Re-export all public types and traits
pub use builder_core::Http3Builder;
// Import trait implementations to make them available
use configuration::*;
pub use content_type::ContentType;
pub use state_types::{BodyNotSet, BodySet, JsonPathStreaming};
use trait_impls::*;
