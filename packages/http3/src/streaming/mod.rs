//! Core streaming foundation for HTTP/2 and HTTP/3
//!
//! Zero-allocation, lock-free streaming using ONLY fluent_ai_async patterns.
//! Foundation layer for all H2/H3 operations using AsyncStream::with_channel and emit!.

pub mod chunks;
pub mod client;
pub mod frames;
pub mod pipeline;
pub mod request;
pub mod resolver;
pub mod response;
pub mod stream;

pub use chunks::*;
pub use frames::*;
pub use pipeline::*;
pub use response::*;
pub use stream::*;
