//! Module organization and re-exports for Response core functionality
//!
//! This module organizes the decomposed Response implementation into logical components.

pub mod accessors;
pub mod body_operations;
pub mod error_handling;
pub mod static_constructors;
pub mod status_checks;
pub mod trait_impls;
pub mod types;

// Re-export the main Response type
pub use static_constructors::ResponseBuilder;
pub use types::Response;
