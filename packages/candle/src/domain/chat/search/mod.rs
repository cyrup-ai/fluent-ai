//! Enhanced history management and search system
//!
//! This module provides comprehensive history management with SIMD-optimized full-text search,
//! lock-free tag management, and zero-allocation streaming export capabilities using
//! blazing-fast algorithms and elegant ergonomic APIs.

use std::sync::Arc;

use fluent_ai_async::AsyncStream;

// Submodules
pub mod types;
pub mod index;
pub mod algorithms;
pub mod query;
pub mod ranking;
pub mod export;

// Re-export public types
pub use types::*;
pub use index::ChatSearchIndex;
pub use query::QueryProcessor;
pub use ranking::ResultRanker;
pub use export::{SearchExporter, HistoryExporter};

// Additional types needed for domain compatibility
pub use crate::chat::search::tagger::ConversationTagger;
pub use crate::chat::search::manager::EnhancedHistoryManager;
pub use crate::chat::search::types::HistoryManagerStatistics;

use crate::domain::chat::message::CandleSearchChatMessage as SearchChatMessage;

/// Main search interface combining all components
pub struct ChatSearcher {
    /// Search index
    index: Arc<ChatSearchIndex>,
    /// Query processor
    query_processor: QueryProcessor,
    /// Result ranker
    ranker: ResultRanker,
    /// Result exporter
    exporter: SearchExporter,
}

impl ChatSearcher {
    /// Create a new chat searcher
    pub fn new(index: Arc<ChatSearchIndex>) -> Self {
        Self {
            index,
            query_processor: QueryProcessor::new(),
            ranker: ResultRanker::new(),
            exporter: SearchExporter::new(),
        }
    }

    /// Search messages with SIMD optimization (streaming individual results)
    pub fn search_stream(&self, query: SearchQuery) -> AsyncStream<SearchResult> {
        let self_clone = self.clone();
        let query_terms = query.terms.clone();
        let query_operator = query.operator.clone();
        let query_fuzzy_matching = query.fuzzy_matching;

        AsyncStream::with_channel(move |sender| {
            let results = match query_operator {
                QueryOperator::And => self_clone.index
                    .search_and_stream(&query_terms, query_fuzzy_matching)
                    .collect(),
                QueryOperator::Or => self_clone.index
                    .search_or_stream(&query_terms, query_fuzzy_matching)
                    .collect(),
                _ => Vec::new(), // Other operators not implemented in simplified version
            };

            // Stream results
            for result in results {
                let _ = sender.send(result);
            }
        })
    }

    /// Search messages (blocking, collects all results)
    pub fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>, SearchError> {
        let stream = self.search_stream(query);
        Ok(stream.collect())
    }

    /// Add message to search index
    pub fn add_message(&self, message: SearchChatMessage) -> Result<(), SearchError> {
        self.index.add_message(message)
    }

    /// Add message to search index (streaming)
    pub fn add_message_stream(&self, message: SearchChatMessage) -> AsyncStream<()> {
        self.index.add_message_stream(message)
    }

    /// Export search results
    pub fn export_results(
        &self,
        results: Vec<SearchResult>,
        options: Option<ExportOptions>,
    ) -> AsyncStream<String> {
        self.exporter.export_stream(results, options)
    }

    /// Get search statistics
    pub fn get_statistics(&self) -> SearchStatistics {
        self.index.get_statistics()
    }
}

impl Clone for ChatSearcher {
    fn clone(&self) -> Self {
        Self {
            index: Arc::clone(&self.index),
            query_processor: self.query_processor.clone(),
            ranker: self.ranker.clone(),
            exporter: self.exporter.clone(),
        }
    }
}

impl Default for ChatSearcher {
    fn default() -> Self {
        Self::new(Arc::new(ChatSearchIndex::new()))
    }
}