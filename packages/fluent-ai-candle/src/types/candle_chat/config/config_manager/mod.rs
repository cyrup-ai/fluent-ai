//! Configuration management and persistence
//!
//! This module provides configuration management capabilities with
//! validation, persistence, and change tracking using zero-allocation patterns.

pub mod types;
pub mod validators; 
pub mod manager;

pub use types::*;
pub use validators::*;
pub use manager::*;