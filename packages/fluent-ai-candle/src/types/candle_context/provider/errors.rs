//! Error Types and Validation for Context Provider System
//!
//! Comprehensive error handling with zero allocations and production-ready error types.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Comprehensive error types for context operations with zero allocations
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum ContextError {
    #[error("Context not found: {0}")]
    ContextNotFound(String),
    #[error("Invalid path: {0}")]
    InvalidPath(String),
    #[error("IO error: {0}")]
    IoError(String),
    #[error("Pattern error: {0}")]
    PatternError(String),
    #[error("Memory integration error: {0}")]
    MemoryError(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Performance threshold exceeded: {0}")]
    PerformanceThresholdExceeded(String),
    #[error("Provider unavailable: {0}")]
    ProviderUnavailable(String)}

/// Provider-specific error types
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum ProviderError {
    #[error("File provider error: {0}")]
    FileProvider(String),
    #[error("Directory provider error: {0}")]
    DirectoryProvider(String),
    #[error("GitHub provider error: {0}")]
    GithubProvider(String),
    #[error("Embedding provider error: {0}")]
    EmbeddingProvider(String)}

/// Validation error types with semantic meaning
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum ValidationError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Path validation failed: {0}")]
    PathValidation(String),
    #[error("Pattern validation failed: {0}")]
    PatternValidation(String),
    #[error("Size limit exceeded: {0}")]
    SizeLimitExceeded(String)}

impl From<std::io::Error> for ContextError {
    fn from(error: std::io::Error) -> Self {
        ContextError::IoError(error.to_string())
    }
}

impl From<ValidationError> for ContextError {
    fn from(error: ValidationError) -> Self {
        ContextError::ValidationError(error.to_string())
    }
}

impl From<ProviderError> for ContextError {
    fn from(error: ProviderError) -> Self {
        match error {
            ProviderError::FileProvider(msg) => ContextError::InvalidPath(msg),
            ProviderError::DirectoryProvider(msg) => ContextError::InvalidPath(msg),
            ProviderError::GithubProvider(msg) => ContextError::ProviderUnavailable(msg),
            ProviderError::EmbeddingProvider(msg) => ContextError::MemoryError(msg)}
    }
}