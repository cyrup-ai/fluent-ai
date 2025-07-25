//! Command parsing and validation logic - decomposed module
//!
//! Provides zero-allocation command parsing with comprehensive validation and error handling.
//! Uses blazing-fast parsing algorithms with ergonomic APIs and production-ready error messages.
//!
//! This module has been decomposed into focused submodules for better maintainability:
//! - error_handling: ParseError types and error handling
//! - lexer: Tokenization and basic command recognition  
//! - parser_core: Main CommandParser implementation
//! - validation: Parameter validation and constraint checking

// Re-export all functionality from the decomposed modules
pub use self::parsing::{
    error_handling::{ParseError, ParseResult},
    lexer::CommandLexer,
    parser_core::CommandParser,
    validation::CommandValidator,
};

// Import the decomposed modules
pub mod parsing;