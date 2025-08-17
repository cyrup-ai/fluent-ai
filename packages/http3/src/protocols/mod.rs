//! Direct protocol implementations using fluent_ai_async AsyncStream
//! 
//! NO middleware, NO Futures, NO abstractions - pure streaming protocols

pub mod h2;
pub mod h3;
pub mod quiche;

// Re-export protocol chunk types
pub use h2::H2Chunk;
pub use h3::H3Chunk;
pub use quiche::QuicheChunk;