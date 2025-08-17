//! Core streaming foundation for HTTP/2 and HTTP/3
//!
//! Zero-allocation, lock-free streaming using ONLY fluent_ai_async patterns.
//! Foundation layer for all H2/H3 operations using AsyncStream::with_channel and emit!.

pub mod connection;
pub mod frames;
pub mod h2;
pub mod h3;
pub mod pipeline;
pub mod protocol;
pub mod transport;

pub use connection::*;
pub use frames::*;
pub use h2::*;
pub use h3::*;
pub use pipeline::*;
pub use protocol::*;
pub use transport::*;
