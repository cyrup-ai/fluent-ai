//! Search index core types and structures
//!
//! Contains the main ChatSearchIndex struct and core functionality for
//! managing inverted indexes, document stores, and search statistics.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use atomic_counter::{AtomicCounter, ConsistentCounter};
use crossbeam_skiplist::SkipMap;
use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::types::candle_chat::message::types::CandleMessage;
use super::super::types::{SearchStatistics, TermFrequency};

/// Index entry for search operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    /// Document ID
    pub doc_id: Arc<str>,
    /// Term frequency score
    pub term_frequency: f32,
    /// Token positions in document
    pub positions: Vec<usize>,
}

/// Handle errors in streaming context without panicking
macro_rules! handle_error {
    ($error:expr, $context:literal) => {
        eprintln!("Streaming error in {}: {}", $context, $error)
        // Continue processing instead of returning error
    };
}

/// Chat search index with SIMD optimization
pub struct ChatSearchIndex {
    /// Inverted index: term -> documents containing term
    pub(super) inverted_index: SkipMap<Arc<str>, Vec<IndexEntry>>,
    /// Document store: doc_id -> message
    pub(super) document_store: SkipMap<Arc<str>, CandleMessage>,
    /// Term frequencies for TF-IDF calculation
    pub(super) term_frequencies: SkipMap<Arc<str>, TermFrequency>,
    /// Document count
    pub(super) document_count: Arc<AtomicUsize>,
    /// Query counter
    pub(super) query_counter: Arc<ConsistentCounter>,
    /// Index update counter
    pub(super) index_update_counter: Arc<ConsistentCounter>,
    /// Search statistics
    pub(super) statistics: Arc<RwLock<SearchStatistics>>,
    /// SIMD processing threshold
    pub(super) simd_threshold: Arc<AtomicUsize>,
}

impl Clone for ChatSearchIndex {
    fn clone(&self) -> Self {
        // Create a new empty instance since SkipMap doesn't implement Clone
        Self::new()
    }
}

impl std::fmt::Debug for ChatSearchIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatSearchIndex")
            .field(
                "inverted_index",
                &format!("SkipMap with {} entries", self.inverted_index.len()),
            )
            .field(
                "document_store",
                &format!("SkipMap with {} entries", self.document_store.len()),
            )
            .field(
                "term_frequencies",
                &format!("SkipMap with {} entries", self.term_frequencies.len()),
            )
            .field(
                "document_count",
                &self.document_count.load(Ordering::Relaxed),
            )
            .field("query_counter", &"ConsistentCounter")
            .field("index_update_counter", &"ConsistentCounter")
            .field("statistics", &"Arc<RwLock<SearchStatistics>>")
            .field(
                "simd_threshold",
                &self.simd_threshold.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl ChatSearchIndex {
    /// Create a new search index
    pub fn new() -> Self {
        Self {
            inverted_index: SkipMap::new(),
            document_store: SkipMap::new(),
            term_frequencies: SkipMap::new(),
            document_count: Arc::new(AtomicUsize::new(0)),
            query_counter: Arc::new(ConsistentCounter::new(0)),
            index_update_counter: Arc::new(ConsistentCounter::new(0)),
            statistics: Arc::new(RwLock::new(SearchStatistics::default())),
            simd_threshold: Arc::new(AtomicUsize::new(8)), // Process 8 terms at once with SIMD
        }
    }

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

impl Default for ChatSearchIndex {
    fn default() -> Self {
        Self::new()
    }
}