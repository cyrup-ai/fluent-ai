//! HTTP response body decoder with zero-allocation streaming decompression
//! Blazing-fast content encoding support (gzip, brotli, zstd, deflate) using fluent_ai_async patterns
//!
//! This module is decomposed into logical submodules:
//! - `accepts`: Accept-Encoding header handling and compression format detection
//! - `types`: Decoder type enumeration and content encoding mapping
//! - `core`: Main decoder implementation with streaming decompression
//! - `wrappers`: MessageChunk wrapper types for Body/Stream trait integration
//! - `compression`: Feature-gated compression algorithm implementations

pub(super) mod accepts;
pub(super) mod compression;
pub(super) mod core;
pub(super) mod types;
pub(super) mod wrappers;

// Re-export all public types for backward compatibility
pub use accepts::{get_accept_encoding, Accepts};
pub use core::Decoder;
pub use types::DecoderType;
pub use wrappers::{
    create_decoder_body, create_decoder_stream, DecoderBodyWrapper, DecoderStreamWrapper,
};

// Import ResponseBody type for module consistency
use super::body::ResponseBody;