//! Command parsing error handling and type definitions
//!
//! Provides comprehensive error types for command parsing with owned strings
//! for zero-allocation patterns and blazing-fast error handling.

use thiserror::Error;

/// Command parsing errors with owned strings
#[derive(Error, Debug, Clone)]
pub enum ParseError {
    #[error("Invalid command syntax: {detail}")]
    InvalidSyntax { detail: String },

    #[error("Missing required parameter: {parameter}")]
    MissingParameter { parameter: String },

    #[error("Invalid parameter value: {parameter} = {value}")]
    InvalidParameterValue { parameter: String, value: String },

    #[error("Unknown parameter: {parameter}")]
    UnknownParameter { parameter: String },

    #[error("Parameter type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String }}

/// Result type for parsing operations
pub type ParseResult<T> = Result<T, ParseError>;