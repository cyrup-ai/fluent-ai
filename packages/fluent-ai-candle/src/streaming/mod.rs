//! Real-time token streaming with zero-allocation performance
//!
//! This module provides blazing-fast token streaming organized by concept:
//! - Zero allocation in hot paths with bounded buffers
//! - Lock-free atomic flow control and backpressure handling  
//! - Sub-100μs token transmission latency
//! - Bounded 32KB memory usage with graceful degradation
//! - Production-ready error handling without unwrap/expect

pub mod constants;
pub mod decoder;
pub mod flow_control;
pub mod formats;
pub mod streaming_config;
pub mod streaming_metrics;
pub mod token_chunk;
pub mod token_metadata;
pub mod token_sender;
pub mod token_stream;

// Re-export core types for ergonomic access
pub use constants::{StreamingError, StreamingTokenResponse, *};
pub use streaming_config::{FlushPolicy, StreamingConfig};
pub use streaming_metrics::StreamingMetrics;
pub use token_chunk::TokenChunk;
pub use token_metadata::TokenMetadata;
pub use token_sender::TokenStreamSender;
pub use token_stream::{create_token_stream, TokenOutputStream};

// Re-export specific items from modules instead of the modules themselves
pub use decoder::*;
pub use flow_control::*;
pub use formats::*;