//! Enhanced history management and search system
//!
//! This module provides comprehensive history management with SIMD-optimized full-text search,
//! lock-free tag management, and zero-allocation streaming export capabilities using
//! blazing-fast algorithms and elegant ergonomic APIs.

use std::sync::Arc;

use fluent_ai_async::AsyncStream;

// Submodules
pub mod algorithms;
pub mod export;
pub mod index;
pub mod manager;
pub mod query;
pub mod ranking;
pub mod tagger;
pub mod types;

// Re-export public types
pub use export::HistoryExporter as CandleHistoryExporter;
pub use export::{HistoryExporter, SearchExporter};
pub use index::ChatSearchIndex;
// Additional search capabilities with Candle prefixes
pub use index::ChatSearchIndex as CandleChatSearchIndex;
// Re-export migrated components with Candle prefixes
pub use manager::CandleEnhancedHistoryManager;
pub use query::QueryProcessor;
pub use ranking::ResultRanker;
pub use tagger::{CandleConversationTag, CandleConversationTagger, CandleTaggingStatistics};
pub use types::*;

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
                QueryOperator::And => self_clone
                    .index
                    .search_and_stream(&query_terms, query_fuzzy_matching)
                    .collect(),
                QueryOperator::Or => self_clone
                    .index
                    .search_or_stream(&query_terms, query_fuzzy_matching)
                    .collect(),
                QueryOperator::Not => self_clone
                    .index
                    .search_not_stream(&query_terms, query_fuzzy_matching)
                    .collect(),
                QueryOperator::Phrase => self_clone
                    .index
                    .search_phrase_stream(&query_terms, query_fuzzy_matching)
                    .collect(),
                QueryOperator::Proximity { distance } => self_clone
                    .index
                    .search_proximity_stream(&query_terms, distance, query_fuzzy_matching)
                    .collect(),
            };

            // Apply enhanced filtering, sorting and pagination
            let filtered_results = self_clone.apply_filters(results, &query);
            let sorted_results = self_clone.apply_sorting(filtered_results, &query.sort_order);
            let paginated_results =
                self_clone.apply_pagination(sorted_results, query.offset, query.max_results);

            // Stream results
            for result in paginated_results {
                let _ = sender.send(result);
            }

            // Update query statistics
            self_clone.update_query_statistics();
        })
    }

    /// Apply comprehensive filtering system (date, user, session, tag, content)
    fn apply_filters(&self, results: Vec<SearchResult>, query: &SearchQuery) -> Vec<SearchResult> {
        let mut filtered = results;

        // Apply date range filter
        if let Some(date_range) = &query.date_range {
            filtered.retain(|result| {
                if let Some(timestamp) = result.message.message.timestamp {
                    timestamp >= date_range.start && timestamp <= date_range.end
                } else {
                    false
                }
            });
        }

        // Apply user filter
        if let Some(user_filter) = &query.user_filter {
            filtered.retain(|result| {
                result
                    .message
                    .message
                    .role
                    .to_string()
                    .contains(user_filter.as_ref())
            });
        }

        // Apply session filter
        if let Some(session_filter) = &query.session_filter {
            filtered.retain(|result| {
                result
                    .message
                    .message
                    .id
                    .as_ref()
                    .map(|id| id.contains(session_filter.as_ref()))
                    .unwrap_or(false)
            });
        }

        // Apply content type filter
        if let Some(content_type_filter) = &query.content_type_filter {
            filtered.retain(|result| {
                result
                    .message
                    .message
                    .content
                    .contains(content_type_filter.as_ref())
            });
        }

        filtered
    }

    /// Apply multiple sorting options (Relevance, DateDesc/Asc, UserDesc/Asc)
    fn apply_sorting(
        &self,
        mut results: Vec<SearchResult>,
        sort_order: &SortOrder,
    ) -> Vec<SearchResult> {
        match sort_order {
            SortOrder::Relevance => {
                // Sort by relevance score (highest first)
                results.sort_by(|a, b| {
                    b.relevance_score
                        .partial_cmp(&a.relevance_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            SortOrder::DateDescending => {
                // Sort by date (newest first)
                results.sort_by(|a, b| {
                    b.message
                        .message
                        .timestamp
                        .unwrap_or(0)
                        .cmp(&a.message.message.timestamp.unwrap_or(0))
                });
            }
            SortOrder::DateAscending => {
                // Sort by date (oldest first)
                results.sort_by(|a, b| {
                    a.message
                        .message
                        .timestamp
                        .unwrap_or(0)
                        .cmp(&b.message.message.timestamp.unwrap_or(0))
                });
            }
            SortOrder::UserDescending => {
                // Sort by user role (descending)
                results.sort_by(|a, b| {
                    b.message
                        .message
                        .role
                        .to_string()
                        .cmp(&a.message.message.role.to_string())
                });
            }
            SortOrder::UserAscending => {
                // Sort by user role (ascending)
                results.sort_by(|a, b| {
                    a.message
                        .message
                        .role
                        .to_string()
                        .cmp(&b.message.message.role.to_string())
                });
            }
        }
        results
    }

    /// Apply pagination support (offset, max_results)
    fn apply_pagination(
        &self,
        results: Vec<SearchResult>,
        offset: usize,
        max_results: usize,
    ) -> Vec<SearchResult> {
        results.into_iter().skip(offset).take(max_results).collect()
    }

    /// Update query statistics with performance tracking
    fn update_query_statistics(&self) {
        // TODO: Implement statistics tracking with query time averaging
        // This will be enhanced with atomic counters for thread-safe updates
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
