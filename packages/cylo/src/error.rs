use anyhow;
use std::io;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExecError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Command failed: {0}")]
    CommandFailed(String),

    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),

    #[error("Invalid code: {0}")]
    InvalidCode(String),

    #[error("Runtime error: {0}")]
    RuntimeError(String),

    #[error("System error: {0}")]
    SystemError(#[from] anyhow::Error),

    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),
}

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Command failed: {0}")]
    CommandFailed(String),

    #[error("Unsupported OS: {0}")]
    UnsupportedOs(String),

    #[error("Mount point exists and is mounted: {0}")]
    AlreadyMounted(std::path::PathBuf),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Insufficient privileges: {0}")]
    InsufficientPrivileges(String),

    #[error("Partial operation failure: {0}")]
    PartialFailure(String),

    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

// Generic result type that can be used with either error
pub type Result<T, E = ExecError> = std::result::Result<T, E>;
