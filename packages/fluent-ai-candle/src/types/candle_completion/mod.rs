//! Candle completion module - one concept per file
//!
//! Contains all completion-related types with proper file organization

// Individual concept modules
pub mod compact_completion_response;
pub mod completion_params;
pub mod completion_response;
pub mod constants;
pub mod core;
pub mod error;
pub mod model_params;
pub mod request;
pub mod streaming;
pub mod tool_definition;

// Re-export main types for convenience - organized by concept
pub use core::*;

pub use compact_completion_response::*;
pub use completion_params::*;
pub use completion_response::*;
pub use constants::*;
pub use error::*;
pub use model_params::*;
pub use request::*;
pub use streaming::*;
pub use tool_definition::*;
