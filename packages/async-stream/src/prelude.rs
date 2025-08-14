//! Prelude module for fluent_ai_async
//!
//! Re-exports all essential types and traits for ergonomic usage.
//! Users should import this with `use fluent_ai_async::prelude::*;`

// Re-export our core types
// Re-export common std types that are frequently used with streams
pub use std::{
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

// Re-export cyrup_sugars essentials for ChunkHandler pattern
pub use cyrup_sugars::prelude::{ChunkHandler, MessageChunk};

pub use crate::{
    AsyncStream, AsyncStreamBuilder, AsyncStreamIterator, AsyncStreamSender, AsyncTask, channel,
    channel_with_capacity, emit, spawn_stream, spawn_stream_with_capacity, spawn_task,
    unbounded_channel,
};
