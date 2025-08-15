//! HTTP/3 connector builder module
//! 
//! This module has been decomposed into logical submodules for better maintainability.
//! See the builder/ subdirectory for the actual implementation.

#[path = "builder/mod.rs"]
mod builder_impl;

pub use builder_impl::*;