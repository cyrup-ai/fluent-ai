//! Direct protocol implementations using fluent_ai_async AsyncStream
//!
//! NO middleware, NO Futures, NO abstractions - pure streaming protocols

pub mod connection;
pub mod frames;
pub mod h2;
pub mod h3;
pub mod quiche;
pub mod transport;
pub mod wire;

// Re-export protocol types
pub use connection::{Connection, ConnectionManager};
pub use h2::{H2Connection, H2Stream};
pub use h3::{H3Connection, H3Stream};
pub use quiche::{QuicheConnectionChunk, QuichePacketChunk, QuicheStreamChunk};
pub use transport::{TransportConnection, TransportManager, TransportType};
pub use wire::{H2FrameParser, H3FrameParser};
