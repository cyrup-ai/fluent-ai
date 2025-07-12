// ============================================================================
// File: src/loaders/epub/errors.rs
// ----------------------------------------------------------------------------
// Exhaustive, zero-surprise error enumeration for every public surface of the
// EPUB loader stack.  All variants carry enough context to be actionable
// without leaking internal implementation details.
// ============================================================================

use std::error::Error;

use epub::doc::DocError;
use thiserror::Error;

use crate::loaders::file::FileLoaderError;

/// Fatal condition surfaced by the EPUB loading pipeline.
#[derive(Error, Debug)]
#[non_exhaustive] // allow graceful forward-compat
pub enum EpubLoaderError {
    /// Failure surfaced by the `epub` crate (I/O or malformed archive).
    #[error("epub: {0}")]
    Epub(#[from] DocError),

    /// Underlying filesystem / globbing failure.
    #[error("file-loader: {0}")]
    File(#[from] FileLoaderError),

    /// Post-processing transformer failed (e.g. XML stripping).
    #[error("text-processor: {0}")]
    TextProcessor(#[from] Box<dyn Error + Send + Sync>),
}
