//! HTTP/2 protocol implementation
//!
//! This module provides HTTP/2 specific functionality including connection management,
//! stream handling, and protocol-specific optimizations.

pub mod chunks;
pub mod connection;
pub mod frame_processor;

pub use chunks::*;
pub use connection::{H2Connection, H2Stream};
