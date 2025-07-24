//! Constants for completion functionality

use std::ops::RangeInclusive;

/// Temperature range for generation (0.0 to 2.0)
pub const TEMPERATURE_RANGE: RangeInclusive<f64> = 0.0..=2.0;

/// Maximum tokens for a single completion
pub const MAX_TOKENS: u64 = 8192;

/// Maximum chunk size for streaming
pub const MAX_CHUNK_SIZE: usize = 4096;
