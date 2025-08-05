//! Enhanced history management and search system
//!
//! This module provides comprehensive history management with SIMD-optimized full-text search,
//! lock-free tag management, and zero-allocation streaming export capabilities using
//! blazing-fast algorithms and elegant ergonomic APIs.

use std::sync::Arc;
use std::time::Instant;
use fluent_ai_async::AsyncStream;

// Submodules
pub mod types;
pub mod index;
pub mod algorithms;
pub mod query;
pub mod ranking;
pub mod export;
pub mod tagger;
pub mod manager;

// Re-export public types
pub use types::*;
pub use index::ChatSearchIndex;
pub use query::QueryProcessor;
pub use ranking::ResultRanker;
pub use tagger::ConversationTagger;
pub use export::HistoryExporter;
pub use manager::EnhancedHistoryManager;
pub use export::SearchExporter;

use crate::chat::message::SearchChatMessage;

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

    /// Create searcher with custom components
    pub fn with_components(
        index: Arc<ChatSearchIndex>,
        query_processor: QueryProcessor,
        ranker: ResultRanker,
        exporter: SearchExporter,
    ) -> Self {
        Self {
            index,
            query_processor,
            ranker,
            exporter,
        }
    }

    /// Search messages with SIMD optimization (streaming individual results)
    pub fn search_stream(&self, query: SearchQuery) -> AsyncStream<SearchResult> {
        let self_clone = self.clone();
        let query_terms = query.terms.clone();
        let query_operator = query.operator.clone();
        let query_fuzzy_matching = query.fuzzy_matching;
        let query_sort_order = query.sort_order.clone();
        let query_offset = query.offset;
        let query_max_results = query.max_results;

        AsyncStream::with_channel(move |sender| {
            let start_time = Instant::now();
            self_clone.index.query_counter.inc();

            let results = match query_operator {
                QueryOperator::And => self_clone.index
                    .search_and_stream(&query_terms, query_fuzzy_matching)
                    .collect(),
                QueryOperator::Or => self_clone.index
                    .search_or_stream(&query_terms, query_fuzzy_matching)
                    .collect(),
                QueryOperator::Not => self_clone.index
                    .search_not_stream(&query_terms, query_fuzzy_matching)
                    .collect(),
                QueryOperator::Phrase => self_clone.index
                    .search_phrase_stream(&query_terms, query_fuzzy_matching)
                    .collect(),
                QueryOperator::Proximity { distance } => self_clone.index
                    .search_proximity_stream(&query_terms, distance, query_fuzzy_matching)
                    .collect(),
            };

            // Apply filters and sorting
            let filtered_results = self_clone.apply_filters(results, &query);
            let sorted_results = self_clone.apply_sorting(filtered_results, &query_sort_order);
            
            // Apply pagination
            let paginated_results = self_clone.apply_pagination(sorted_results, query_offset, query_max_results);

            // Stream results
            for result in paginated_results {
                let _ = sender.send(result);
            }

            // Update statistics
            let query_time = start_time.elapsed().as_millis() as f64;
            self_clone.update_query_statistics(query_time);
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

    /// Clear search index
    pub fn clear_index(&self) -> Result<(), SearchError> {
        self.index.clear()
    }

    /// Get query processor
    pub fn query_processor(&self) -> &QueryProcessor {
        &self.query_processor
    }

    /// Get query processor (mutable)
    pub fn query_processor_mut(&mut self) -> &mut QueryProcessor {
        &mut self.query_processor
    }

    /// Get result ranker
    pub fn ranker(&self) -> &ResultRanker {
        &self.ranker
    }

    /// Get result ranker (mutable)
    pub fn ranker_mut(&mut self) -> &mut ResultRanker {
        &mut self.ranker
    }

    /// Get exporter
    pub fn exporter(&self) -> &SearchExporter {
        &self.exporter
    }

    /// Get exporter (mutable)
    pub fn exporter_mut(&mut self) -> &mut SearchExporter {
        &mut self.exporter
    }

    /// Apply search filters
    fn apply_filters(&self, results: Vec<SearchResult>, query: &SearchQuery) -> Vec<SearchResult> {
        let mut filtered = results;

        // Apply date range filter
        if let Some(ref date_range) = query.date_range {
            filtered.retain(|result| {
                if let Some(timestamp) = result.message.message.timestamp {
                    timestamp >= date_range.start && timestamp <= date_range.end
                } else {
                    false
                }
            });
        }

        // Apply user filter
        if let Some(ref user_filter) = query.user_filter {
            filtered.retain(|result| {
                result.message.message.content
                    .to_lowercase()
                    .contains(&user_filter.to_lowercase())
            });
        }

        // Apply session filter
        if let Some(ref session_filter) = query.session_filter {
            filtered.retain(|result| {
                result.message.message.id
                    .as_ref()
                    .map(|id| id.contains(session_filter.as_ref()))
                    .unwrap_or(false)
            });
        }

        // Apply tag filter
        if let Some(ref tag_filter) = query.tag_filter {
            filtered.retain(|result| {
                tag_filter.iter().any(|tag| {
                    result.tags.contains(tag)
                })
            });
        }

        // Apply content type filter
        if let Some(ref content_type_filter) = query.content_type_filter {
            filtered.retain(|result| {
                result.message.message.content
                    .to_lowercase()
                    .contains(&content_type_filter.to_lowercase())
            });
        }

        filtered
    }

    /// Apply sorting to results
    fn apply_sorting(&self, mut results: Vec<SearchResult>, sort_order: &SortOrder) -> Vec<SearchResult> {
        match sort_order {
            SortOrder::Relevance => {
                results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
            }
            SortOrder::DateDescending => {
                results.sort_by(|a, b| {
                    let a_time = a.message.message.timestamp.unwrap_or(0);
                    let b_time = b.message.message.timestamp.unwrap_or(0);
                    b_time.cmp(&a_time)
                });
            }
            SortOrder::DateAscending => {
                results.sort_by(|a, b| {
                    let a_time = a.message.message.timestamp.unwrap_or(0);
                    let b_time = b.message.message.timestamp.unwrap_or(0);
                    a_time.cmp(&b_time)
                });
            }
            SortOrder::UserAscending => {
                results.sort_by(|a, b| {
                    a.message.message.content.cmp(&b.message.message.content)
                });
            }
            SortOrder::UserDescending => {
                results.sort_by(|a, b| {
                    b.message.message.content.cmp(&a.message.message.content)
                });
            }
        }
        
        results
    }

    /// Apply pagination to results
    fn apply_pagination(&self, results: Vec<SearchResult>, offset: usize, max_results: usize) -> Vec<SearchResult> {
        results
            .into_iter()
            .skip(offset)
            .take(max_results)
            .collect()
    }

    /// Update query statistics
    fn update_query_statistics(&self, query_time_ms: f64) {
        if let Ok(mut stats) = self.index.statistics.try_write() {
            stats.total_queries += 1;
            stats.average_query_time = 
                (stats.average_query_time * (stats.total_queries - 1) as f64 + query_time_ms) / 
                stats.total_queries as f64;
        }
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

impl std::fmt::Debug for ChatSearcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatSearcher")
            .field("index", &self.index)
            .field("query_processor", &self.query_processor)
            .field("ranker", &self.ranker)
            .field("exporter", &self.exporter)
            .finish()
    }
}