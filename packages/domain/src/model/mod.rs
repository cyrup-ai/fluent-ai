//! Model system core module
//!
//! This module provides the core abstractions and types for AI model management,
//! including traits, information, registry, and error handling.

pub mod capabilities;
pub mod error;
pub mod info;
pub mod registry;
pub mod resolver;
pub mod traits;

// Re-export commonly used types
pub use capabilities::*;
pub use error::{ModelError, Result};
pub use info::ModelInfo;
pub use registry::ModelRegistry;
pub use resolver::*;
pub use traits::*;
