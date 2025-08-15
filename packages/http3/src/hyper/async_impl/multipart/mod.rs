//! Multipart/form-data implementation for HTTP requests
//! 
//! Zero-allocation, production-quality multipart handling with comprehensive streaming support.

mod types;
mod form;
mod part;

#[cfg(test)]
mod tests;

// Re-export main types
pub use types::{Form, Part};

// Internal types for module organization
pub(crate) use types::{FormParts, PartMetadata, PartProps, PercentEncoding, gen_boundary};