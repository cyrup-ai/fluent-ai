//! HTTP/3 Type Definitions
//!
//! This module provides core type definitions for HTTP/3 streaming
//! with zero-allocation patterns and fluent_ai_async integration.

pub mod chunks;
pub mod h2_chunks;
pub mod h3_chunks;
pub mod quiche_chunks;

// Re-export the canonical HttpResponseChunk
pub use chunks::HttpResponseChunk;
pub use chunks::{headers_from_iter, merge_headers};
pub use h2_chunks::{H2ConnectionChunk, H2DataChunk, H2RequestChunk, H2SendResult};
pub use quiche_chunks::{
    QuicheConnectionChunk, QuichePacketChunk, QuicheReadableChunk, QuicheStreamChunk,
    QuicheWriteResult,
};
