//! Command parsing and validation logic
//!
//! This module has been decomposed into focused submodules for better organization.
//! All functionality has been preserved while improving maintainability.

// Decomposed parsing modules
pub mod core;
pub mod errors;
pub mod registry;
pub mod validators;

// Re-export for backward compatibility
pub use core::CommandParser;
pub use errors::{ParseError, ParseResult};
pub use registry::{CommandInfo, ParameterInfo, ParameterType};
pub use validators::validate_command;
