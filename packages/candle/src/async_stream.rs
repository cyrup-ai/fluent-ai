//! Candle Async Stream - Using existing fluent_ai_async with Candle type aliases

// Re-export fluent_ai_async types with Candle prefixes
pub use fluent_ai_async::{AsyncStream as CandleAsyncStream, AsyncTask as CandleAsyncTask, AsyncStreamSender as CandleAsyncStreamSender, spawn_task as candle_spawn_task};