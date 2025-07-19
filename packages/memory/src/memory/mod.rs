//! Memory module that provides the core memory functionality

// New hierarchical module structure
pub mod manager;
pub mod ops;
pub mod primitives;
pub mod schema;
pub mod systems;

#[cfg(test)]
pub mod tests;

// Re-export main types to maintain backward compatibility
pub use manager::*;
pub use ops::*;
pub use primitives::*;
pub use schema::*;
pub use systems::*;
