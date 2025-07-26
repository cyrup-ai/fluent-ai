//! Streaming UTF-8 decoder for incremental processing of byte streams
//!
//! This module provides a streaming UTF-8 decoder that can handle partial
//! sequences across chunk boundaries, making it ideal for network protocols
//! or file formats where UTF-8 text may be split across multiple chunks.
//!
//! # Features
//!
//! - **Incremental Decoding**: Handle UTF-8 sequences split across chunks
//! - **Configurable Validation**: Toggle strict UTF-8 validation
//! - **Error Recovery**: Graceful handling of invalid sequences
//! - **Zero-copy Operations**: Minimize allocations where possible
//! - **Statistics**: Track decoding metrics and performance
//!
//! # Example
//!
//! ```no_run
//! use fluent_ai_candle::streaming::decoder::{StreamingDecoder, DecoderConfig};
//!
//! let config = DecoderConfig {
//!     validate_utf8: true,
//!     enable_incremental: true,
//!     ..Default::default()
//! };
//! 
//! let mut decoder = StreamingDecoder::new(config);
//! 
//! // Process chunks of data
//! let chunk1 = [0xE2, 0x82]; // First part of '€' (U+20AC)
//! let chunk2 = [0xAC];        // Second part of '€'
//! 
//! // First chunk (partial sequence)
//! let result = decoder.decode(&chunk1);
//! assert!(result.unwrap().is_empty()); // No complete characters yet
//! 
//! // Second chunk (completes the sequence)
//! let result = decoder.decode(&chunk2);
//! assert_eq!(result.unwrap(), "€");
//! ```

#![deny(missing_docs)]
#![warn(rust_2018_idioms)]
#![deny(unsafe_code)]
#![deny(trivial_casts)]
#![deny(trivial_numeric_casts)]
#![deny(unused_import_braces)]
#![deny(unused_qualifications)]
#![deny(unused_results)]
#![deny(unreachable_pub)]
#![deny(unused_must_use)]
#![deny(clippy::all)]

mod core;
mod error;
mod state;
mod stats;
mod validation;

// Re-exports
pub use self::core::{DecoderConfig, StreamingDecoder};
pub use self::error::DecoderError;
pub use self::state::DecoderState;
pub use self::stats::DecoderStats;
pub use self::validation::{
    decode_codepoint, expected_sequence_length, is_ascii, is_continuation_byte, is_valid_start_byte,
    validate_ascii, validate_utf8_sequence};

/// A specialized `Result` type for decoder operations
pub type Result<T> = std::result::Result<T, DecoderError>;
