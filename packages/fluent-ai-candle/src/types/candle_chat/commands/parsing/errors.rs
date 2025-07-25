//! Command parsing errors and result types
//!
//! Provides comprehensive error handling for command parsing operations with detailed
//! error messages and proper error propagation patterns.

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
    TypeMismatch { expected: String, actual: String },
}

/// Result type for parsing operations
pub type ParseResult<T> = Result<T, ParseError>;