//! Chat search module with decomposed submodules
//!
//! This module provides comprehensive chat search functionality with clear
//! separation of concerns across focused submodules.

pub mod types;
pub mod index;
pub mod tagging;
pub mod export;
pub mod manager;
pub mod searcher;
pub mod query;
pub mod ranking;

// Re-export core types for convenience
pub use types::*;
pub use index::*;
pub use tagging::*;
pub use export::*;
pub use manager::*;
pub use searcher::*;
pub use query::*;
pub use ranking::*;

use fluent_ai_async::AsyncStream;

/// Stream collection trait to provide .collect() method for future-like behavior
pub trait StreamCollect<T> {
    fn collect_sync(self) -> AsyncStream<Vec<T>>;
}

impl<T> StreamCollect<T> for AsyncStream<T>
where
    T: Send + 'static,
{
    fn collect_sync(self) -> AsyncStream<Vec<T>> {
        AsyncStream::with_channel(move |sender| {
            // AsyncStream doesn't implement Iterator - use proper streaming pattern
            let results = Vec::new();
            // For now, send empty results - this would need proper stream collection logic
            let _ = sender.send(results);
        })
    }
}

/// Handle errors in streaming context without panicking
macro_rules! handle_error {
    ($error:expr, $context:literal) => {
        eprintln!("Streaming error in {}: {}", $context, $error)
        // Continue processing instead of returning error
    };
}

pub(crate) use handle_error;