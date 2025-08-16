//! HTTP Client Builder implementation
//!
//! This module contains the decomposed ClientBuilder functionality organized
//! into focused modules for maintainability and clarity.

mod compression;
mod cookies;
mod headers;
mod protocols;
mod timeouts;
mod types;

// Re-export the main ClientBuilder type
pub use types::ClientBuilder;
