// ============================================================================
// File: src/loaders/file.rs
// ----------------------------------------------------------------------------
// Common file loading errors used across PDF and EPUB loaders.
// ============================================================================

use glob::{GlobError, PatternError};
use std::io;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FileLoaderError {
    #[error("Pattern error: {0}")]
    PatternError(#[from] PatternError),

    #[error("Glob error: {0}")]
    GlobError(#[from] GlobError),

    #[error("I/O error: {0}")]
    IoError(#[from] io::Error),
}
