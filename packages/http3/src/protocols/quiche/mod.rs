//! Quiche QUIC protocol implementation
//!
//! This module provides Quiche-specific QUIC functionality including connection management,
//! packet handling, and Quiche library integration.

pub mod chunks;
pub mod streaming;

pub use chunks::*;
// Re-export chunk types for compatibility
pub use chunks::{QuicheReadableChunk, QuicheWriteResult};
pub use streaming::*;
