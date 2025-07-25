//! Command parsing and validation modules
//!
//! This module provides comprehensive command parsing with zero-allocation patterns,
//! blazing-fast lexical analysis, comprehensive validation, and production-ready
//! error handling.

pub mod builtin_commands;
pub mod error_handling;
pub mod lexer;
pub mod parser_core;
pub mod validation;

// Re-export main types and functions for convenience
pub use builtin_commands::BuiltinCommands;
pub use error_handling::{ParseError, ParseResult};
pub use lexer::CommandLexer;
pub use parser_core::CommandParser;
pub use validation::CommandValidator;