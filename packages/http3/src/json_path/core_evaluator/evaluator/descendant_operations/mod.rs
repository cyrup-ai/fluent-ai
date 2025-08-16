//! Descendant operations for JSONPath recursive descent processing
//!
//! Handles recursive descent (..) operations and descendant collection
//! with RFC 9535 compliance.

mod core;
mod advanced;
mod utilities;
#[cfg(test)]
mod tests;

// Re-export the main struct for public API compatibility
pub use core::DescendantOperations;