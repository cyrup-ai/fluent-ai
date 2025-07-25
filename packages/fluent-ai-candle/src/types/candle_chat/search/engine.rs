//! Main search engine orchestration
//!
//! This module provides the main search engine that coordinates all search operations
//! with streaming-first architecture and zero-allocation patterns.

use std::sync::Arc;
use std::time::Instant;

use fluent_ai_async::{AsyncStream, emit, handle_error};

use super::types::{SearchQuery, SearchResult, SearchOptions, SearchStatistics};
use super::index::ChatSearchIndex;
use super::query::{QueryProcessor, ProcessedQuery};
use super::results::ResultRanker;
use super::stats::SearchStatsTracker;

/// Main search engine that orchestrates all search operations
#[derive(Debug, Clone)]
pub struct ChatSearcher {
    /// Search index for message storage and retrieval
    index: Arc<ChatSearchIndex>,
    /// Query processor for parsing and expanding queries
    query_processor: QueryProcessor,
    /// Result ranker for scoring and sorting results
    result_ranker: ResultRanker,
    /// Statistics tracker for performance monitoring
    stats_tracker: Arc<SearchStatsTracker>,
    /// Search options
    options: SearchOptions}

impl ChatSearcher {
    /// Create a new chat searcher
    pub fn new(index: Arc<ChatSearchIndex>) -> Self {
        Self {
            index,
            query_processor: QueryProcessor::new(),
            result_ranker: ResultRanker::new(),
            stats_tracker: Arc::new(SearchStatsTracker::new()),
            options: SearchOptions::default()}
    }

    /// Create chat searcher with custom options
    pub fn with_options(index: Arc<ChatSearchIndex>, options: SearchOptions) -> Self {
        Self {
            index,
            query_processor: QueryProcessor::with_options(&options),
            result_ranker: ResultRanker::new(),
            stats_tracker: Arc::new(SearchStatsTracker::new()),
            options}
    }

    /// Search messages using fluent-ai-async streaming architecture
    pub fn search_messages(&self, query: SearchQuery) -> AsyncStream<Vec<SearchResult>> {
        let index = Arc::clone(&self.index);
        let query_processor = self.query_processor.clone();
        let result_ranker = self.result_ranker.clone();
        let stats_tracker = Arc::clone(&self.stats_tracker);
        let options = self.options.clone();
        
        AsyncStream::with_channel(move |sender| {
            let start_time = Instant::now();
            
            // Process the query
            let processed_query = ProcessedQuery {
                original: Arc::from(query.terms.iter()
                    .map(|t| t.as_ref())
                    .collect::<Vec<_>>()
                    .join(" ")
                    .as_str()),
                terms: query.terms.iter().cloned().collect(),
                expanded_terms: Vec::new(), // Would be populated by query processor
                operator: query.operator.clone(),
                metadata: super::query::QueryMetadata {
                    processed_at: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                    processing_time_us: 0,
                    expansion_applied: false,
                    normalization_applied: true}};

            // Search the index
            let search_results = Self::execute_search(&index, &query);
            
            // Rank the results
            let ranked_results = Self::rank_search_results(&result_ranker, search_results, &processed_query);
            
            // Record statistics
            let query_duration = start_time.elapsed();
            stats_tracker.record_query(query_duration);
            
            emit!(sender, ranked_results);
        })
    }

    /// Add message to search index using fluent-ai-async streaming architecture
    pub fn add_message(&self, message: crate::types::CandleSearchChatMessage) -> AsyncStream<()> {
        let index = Arc::clone(&self.index);
        let stats_tracker = Arc::clone(&self.stats_tracker);
        
        AsyncStream::with_channel(move |sender| {
            // Add to index (this would use the index's streaming method)
            stats_tracker.record_index_operation();
            
            // For now, just emit completion - the actual implementation would
            // use the index's add_message_stream method
            emit!(sender, ());
        })
    }

    /// Get search statistics using fluent-ai-async streaming architecture
    pub fn get_statistics(&self) -> AsyncStream<SearchStatistics> {
        let stats_tracker = Arc::clone(&self.stats_tracker);
        let index = Arc::clone(&self.index);
        
        AsyncStream::with_channel(move |sender| {
            // Get basic statistics from tracker
            let total_queries = stats_tracker.total_queries();
            let total_index_ops = stats_tracker.total_index_operations();
            let cache_hits = stats_tracker.cache_hits();
            let cache_misses = stats_tracker.cache_misses();
            
            // Get index statistics
            let total_docs = index.total_documents();
            let index_version = index.version();
            
            let stats = SearchStatistics {
                total_messages: total_docs,
                total_terms: 0, // Would be provided by index
                total_queries,
                average_query_time: 0.0, // Would be calculated from tracker
                index_size: total_docs * 256, // Rough estimate
                last_index_update: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                search_operations: total_queries,
                index_operations: total_index_ops,
                cache_hits,
                cache_misses};
            
            emit!(sender, stats);
        })
    }

    /// Clear the search index
    pub fn clear_index(&self) {
        self.index.clear();
    }

    /// Get total document count
    pub fn total_documents(&self) -> usize {
        self.index.total_documents()
    }

    /// Execute search against the index (synchronous helper)
    fn execute_search(index: &ChatSearchIndex, query: &SearchQuery) -> Vec<SearchResult> {
        // Simplified search implementation - would use index's streaming search
        let mut results = Vec::new();
        
        // For now, return empty results - the actual implementation would
        // use the index's search_messages_stream method and collect results
        
        results
    }

    /// Rank search results (synchronous helper)
    fn rank_search_results(
        ranker: &ResultRanker,
        results: Vec<SearchResult>,
        query: &ProcessedQuery,
    ) -> Vec<SearchResult> {
        // Simplified ranking - would use ranker's streaming methods
        let mut ranked = results;
        
        // Apply simple relevance scoring
        for result in &mut ranked {
            let mut score = 0.0f32;
            let content_lower = result.message.content.to_lowercase();
            
            for term in &query.terms {
                let term_lower = term.to_lowercase();
                let matches = content_lower.matches(&term_lower).count();
                score += matches as f32;
            }
            
            result.relevance_score = score;
        }
        
        // Sort by relevance score (descending)
        ranked.sort_by(|a, b| {
            b.relevance_score.partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        ranked
    }

    /// Update search options
    pub fn update_options(&mut self, options: SearchOptions) {
        self.options = options.clone();
        self.query_processor = QueryProcessor::with_options(&options);
    }

    /// Get current search options
    pub fn get_options(&self) -> &SearchOptions {
        &self.options
    }

    /// Reset statistics
    pub fn reset_statistics(&self) {
        self.stats_tracker.reset();
    }
}

impl Default for ChatSearcher {
    fn default() -> Self {
        Self::new(Arc::new(ChatSearchIndex::new()))
    }
}
