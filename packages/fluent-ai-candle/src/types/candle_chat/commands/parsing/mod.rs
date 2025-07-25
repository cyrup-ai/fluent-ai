//! Command parsing and validation logic
//!
//! Provides zero-allocation command parsing with comprehensive validation and error handling.
//! Uses blazing-fast parsing algorithms with ergonomic APIs and production-ready error messages.

// Core parsing components
pub mod parser;
pub mod command_parsers;
pub mod registration;
pub mod errors;
pub mod registry;
pub mod validators;

// Re-export commonly used types
pub use parser::CommandParser;
pub use errors::{ParseError, ParseResult};
pub use registry::{CommandInfo, ParameterInfo, ParameterType};
pub use validators::validate_command;