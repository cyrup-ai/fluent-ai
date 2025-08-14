//! Model system core module
//!
//! This module provides the core abstractions and types for AI model management,
//! including traits, information, registry, and error handling.

pub mod capabilities;
pub mod error;
pub mod info;
pub mod registry;
pub mod resolver;
pub mod traits;
pub mod usage;
pub mod validation;

// Generated modules - created by build system
pub mod models;
pub mod providers;

// Re-export commonly used Candle types
pub use capabilities::*;
pub use error::{CandleModelError, CandleResult};
pub use info::CandleModelInfo;
pub use registry::CandleModelRegistry;
pub use resolver::*;
pub use traits::*;
pub use usage::CandleUsage;
pub use validation::*;

// Re-export generated types (commented out until build system generates actual content)
// pub use providers::*;
// pub use models::*;
