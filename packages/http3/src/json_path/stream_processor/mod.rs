pub mod core;
pub mod error_recovery;
pub mod processing;
pub mod stats;
pub mod types;

// Re-export main types and functions for backward compatibility
pub use core::*;

// Re-export internal types needed by other modules
pub(crate) use error_recovery::*;
pub use processing::*;
pub(crate) use stats::*;
pub use types::{
    CircuitState, ErrorRecoveryState, JsonStreamProcessor, ProcessorStats, ProcessorStatsSnapshot,
};
