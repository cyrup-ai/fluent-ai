//! Search statistics and monitoring functionality
//!
//! This module provides statistics collection and monitoring for search
//! operations with streaming support.

use fluent_ai_async::AsyncStream;

use super::super::types::SearchStatistics;
use super::types::ChatSearchIndex;

impl ChatSearchIndex {
    /// Get search statistics (streaming)
    pub fn get_statistics_stream(&self) -> AsyncStream<SearchStatistics> {
        let self_clone = self.clone();

        AsyncStream::with_channel(move |sender| {
            if let Ok(stats) = self_clone.statistics.try_read() {
                let _ = sender.send(stats.clone());
            }
        })
    }
}