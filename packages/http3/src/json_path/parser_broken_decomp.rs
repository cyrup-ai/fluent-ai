//! JSONPath expression compilation module
//! 
//! This module has been decomposed into logical submodules for better maintainability.
//! See the parser_broken_decomp/ subdirectory for the actual implementation.

#[path = "parser_broken_decomp/mod.rs"]
mod parser_broken_decomp_impl;

pub use parser_broken_decomp_impl::*;