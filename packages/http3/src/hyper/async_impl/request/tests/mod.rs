//! Request tests module with logical separation of concerns
//!
//! This module provides a decomposed test suite for HTTP request functionality,
//! organized into focused test modules for maintainability and clarity.

mod auth_tests;
mod basic_tests;
mod body_tests;
mod builder_tests;
mod header_tests;

// Re-export all test modules for compatibility
pub use auth_tests::*;
pub use basic_tests::*;
pub use body_tests::*;
pub use builder_tests::*;
pub use header_tests::*;
