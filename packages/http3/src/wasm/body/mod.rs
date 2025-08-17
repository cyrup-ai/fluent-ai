//! WASM Body implementation with logical separation of concerns
//!
//! This module provides a decomposed implementation of the Body type for WASM targets,
//! organized into logical modules for maintainability and clarity.

mod body_impl;
mod conversions;
mod single_impl;
mod types;

#[cfg(test)]
mod tests;

// Re-export the main types for public API compatibility
// Re-export implementations through the types
pub use body_impl::*;
pub use conversions::*;
pub use single_impl::*;
#[cfg(test)]
pub use tests::*;
pub use types::Body;
pub(crate) use types::{Inner, Single};
