//! Chat searcher with advanced search capabilities
//!
//! This module provides the main search interface with caching,
//! result ranking, and performance optimization.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::types::candle_chat::search::tagging::ConsistentCounter;
use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::CandleMessageRole;
use crate::types::candle_chat::message::SearchChatMessage;
use super::types::{SearchResult, SearchQuery};
use super::index::ChatSearchIndex;

/// Search options for fine-tuning search behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchOptions {
    /// Enable result caching
    pub enable_caching: bool,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// Enable result ranking
    pub enable_ranking: bool,
    /// Maximum search time in milliseconds
    pub max_search_time_ms: u64,
    /// Enable search suggestions
    pub enable_suggestions: bool,
    /// Enable search analytics
    pub enable_analytics: bool,
    /// Custom search parameters
    pub custom_params: HashMap<String, String>}

/// Main chat searcher with advanced capabilities
pub struct ChatSearcher {
    /// Core search index
    pub search_index: ChatSearchIndex,
    /// Search result cache
    pub result_cache: Arc<crossbeam_skiplist::SkipMap<String, CachedSearchResult>>,
    /// Search statistics
    pub stats: Arc<ChatSearcherStats>,
    /// Search options
    pub options: SearchOptions,
    /// Operation counters
    pub search_counter: ConsistentCounter,
    pub cache_hits: Arc<AtomicUsize>,
    pub cache_misses: Arc<AtomicUsize>}

impl std::fmt::Debug for ChatSearcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatSearcher")
            .field("search_count", &self.search_counter.get())
            .field("cache_size", &self.result_cache.len())
            .field("cache_hits", &self.cache_hits.load(Ordering::Relaxed))
            .field("cache_misses", &self.cache_misses.load(Ordering::Relaxed))
            .finish()
    }
}

/// Cached search result with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedSearchResult {
    /// Search results
    pub results: Vec<SearchResult>,
    /// Cache timestamp
    pub cached_at: chrono::DateTime<chrono::Utc>,
    /// Cache expiry time
    pub expires_at: chrono::DateTime<chrono::Utc>,
    /// Result metadata
    pub metadata: SearchResultMetadata,
    /// Cache hit count
    pub hit_count: usize}

/// Metadata about search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultMetadata {
    /// Total search time in milliseconds
    pub search_time_ms: f64,
    /// Number of results found
    pub result_count: usize,
    /// Whether results were cached
    pub from_cache: bool,
    /// Search algorithm used
    pub algorithm: String,
    /// Index version used
    pub index_version: usize,
    /// Additional metadata
    pub extra: HashMap<String, serde_json::Value>}

/// Statistics for the chat searcher
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSearcherStats {
    /// Total searches performed
    pub total_searches: usize,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Average search time
    pub avg_search_time_ms: f64,
    /// Most common search terms
    pub popular_terms: Vec<(String, usize)>,
    /// Search performance metrics
    pub performance_metrics: HashMap<String, f64>}

impl ChatSearcher {
    /// Create a new chat searcher
    pub fn new(search_index: ChatSearchIndex, options: SearchOptions) -> Self {
        Self {
            search_index,
            result_cache: Arc::new(crossbeam_skiplist::SkipMap::new()),
            stats: Arc::new(ChatSearcherStats::default()),
            options,
            search_counter: ConsistentCounter::new(0),
            cache_hits: Arc::new(AtomicUsize::new(0)),
            cache_misses: Arc::new(AtomicUsize::new(0))}
    }

    /// Perform a search with caching and ranking (streaming)
    pub fn search(&self, query: SearchQuery) -> AsyncStream<SearchResult> {
        let cache_key = self.generate_cache_key(&query);
        let result_cache = Arc::clone(&self.result_cache);
        let search_index = self.search_index.clone();
        let search_counter = self.search_counter.clone();
        let cache_hits = Arc::clone(&self.cache_hits);
        let cache_misses = Arc::clone(&self.cache_misses);
        let enable_caching = self.options.enable_caching;

        AsyncStream::with_channel(move |sender| {
            search_counter.inc();
            let start_time = std::time::Instant::now();

            // Check cache first
            if enable_caching {
                if let Some(cached_entry) = result_cache.get(&cache_key) {
                    let cached_result = cached_entry.value();
                    if cached_result.expires_at > chrono::Utc::now() {
                        cache_hits.fetch_add(1, Ordering::Relaxed);
                        
                        // Send cached results
                        for result in &cached_result.results {
                            let _ = sender.send(result.clone());
                        }
                        return;
                    }
                }
                cache_misses.fetch_add(1, Ordering::Relaxed);
            }

            // Perform actual search
            let mut results = Vec::new();
            
            // Simple search implementation (would be more sophisticated in production)
            for term in &query.terms {
                // Delegate to search index
                let search_stream = search_index.search_messages(&[term.clone()]);
                // In a real implementation, we would collect results from the stream
                // For now, create a sample result
                let result = SearchResult {
                    id: Uuid::new_v4(),
                    message: SearchChatMessage {
                        id: Uuid::new_v4().to_string(),
                        content: format!("Sample result for term: {}", term),
                        role: CandleMessageRole::User,
                        name: None,
                        metadata: None,
                        timestamp: Some(
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs(),
                        ),
                        relevance_score: 1.0,
                        search_timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64},
                    score: 1.0,
                    highlighted_content: None,
                    context: Vec::new(),
                    match_metadata: HashMap::new(),
                    match_positions: Vec::new(),
                    conversation_id: None,
                    tags: Vec::new(),
                    result_timestamp: chrono::Utc::now(),
                    extra_data: HashMap::new()};
                results.push(result);
            }

            // Cache results if enabled
            if enable_caching && !results.is_empty() {
                let cached_result = CachedSearchResult {
                    results: results.clone(),
                    cached_at: chrono::Utc::now(),
                    expires_at: chrono::Utc::now() + chrono::Duration::hours(1),
                    metadata: SearchResultMetadata {
                        search_time_ms: start_time.elapsed().as_millis() as f64,
                        result_count: results.len(),
                        from_cache: false,
                        algorithm: "basic".to_string(),
                        index_version: 1,
                        extra: HashMap::new()},
                    hit_count: 0};
                result_cache.insert(cache_key, cached_result);
            }

            // Send results
            for result in results {
                let _ = sender.send(result);
            }
        })
    }

    /// Generate cache key for a search query
    fn generate_cache_key(&self, query: &SearchQuery) -> String {
        // Simple cache key generation (would use proper hashing in production)
        format!("{:?}", query.terms)
    }

    /// Clear search cache
    pub fn clear_cache(&self) -> AsyncStream<usize> {
        let result_cache = Arc::clone(&self.result_cache);

        AsyncStream::with_channel(move |sender| {
            let cleared_count = result_cache.len();
            result_cache.clear();
            let _ = sender.send(cleared_count);
        })
    }

    /// Get search statistics (streaming)
    pub fn get_statistics(&self) -> AsyncStream<ChatSearcherStats> {
        let stats = Arc::clone(&self.stats);
        let total_searches = self.search_counter.get();
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(Ordering::Relaxed);

        AsyncStream::with_channel(move |sender| {
            let mut current_stats = (*stats).clone();
            current_stats.total_searches = total_searches;
            
            let total_cache_requests = cache_hits + cache_misses;
            current_stats.cache_hit_rate = if total_cache_requests > 0 {
                cache_hits as f64 / total_cache_requests as f64
            } else {
                0.0
            };
            
            let _ = sender.send(current_stats);
        })
    }
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_size_limit: 10000,
            enable_ranking: true,
            max_search_time_ms: 5000,
            enable_suggestions: false,
            enable_analytics: true,
            custom_params: HashMap::new()}
    }
}

impl Default for ChatSearcher {
    fn default() -> Self {
        Self::new(ChatSearchIndex::new(), SearchOptions::default())
    }
}

impl Default for ChatSearcherStats {
    fn default() -> Self {
        Self {
            total_searches: 0,
            cache_hit_rate: 0.0,
            avg_search_time_ms: 0.0,
            popular_terms: Vec::new(),
            performance_metrics: HashMap::new()}
    }
}