//! Output format handling for streaming responses
//!
//! This module has been decomposed into focused submodules for better organization.
//! All functionality has been preserved while improving maintainability.
//!
//! ## Decomposed Structure
//! - `formats/types`: Core format types and StreamingFormatter struct
//! - `formats/json`: JSON and WebSocket format implementations  
//! - `formats/text`: Plain text, SSE, and raw format implementations
//! - `formats/utils`: Utility functions, buffer management, and performance tools
//!
//! ## Migration Guide
//! All exports remain the same, but now come from focused submodules:
//! ```rust
//! // This still works exactly the same:
//! use crate::streaming::formats::{OutputFormat, StreamingFormatter};
//! 
//! let mut formatter = StreamingFormatter::new(OutputFormat::Json);
//! let formatted = formatter.format_response(&response)?;
//! ```

// Re-export all functionality from the decomposed modules
pub use self::formats::*;

// Include the decomposed formats module
pub mod formats;