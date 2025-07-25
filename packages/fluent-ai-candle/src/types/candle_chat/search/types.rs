//! Core search types and query structures
//!
//! This module defines the fundamental data structures for search queries,
//! results, and configuration options used throughout the search system.

use std::sync::Arc;
use serde::{Deserialize, Serialize};
use fluent_ai_async::AsyncStream;
use crate::types::candle_chat::message::types::CandleMessage;

/// Search query with advanced filtering options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Search terms
    pub terms: Vec<Arc<str>>,
    /// Boolean operator (AND, OR, NOT)
    pub operator: QueryOperator,
    /// Date range filter
    pub date_range: Option<DateRange>,
    /// User filter
    pub user_filter: Option<Arc<str>>,
    /// Session filter
    pub session_filter: Option<Arc<str>>,
    /// Tag filter
    pub tag_filter: Option<Vec<Arc<str>>>,
    /// Content type filter
    pub content_type_filter: Option<Arc<str>>,
    /// Fuzzy matching enabled
    pub fuzzy_matching: bool,
    /// Maximum results
    pub max_results: usize,
    /// Result offset for pagination
    pub offset: usize,
    /// Sort order
    pub sort_order: SortOrder,
}

/// Query operator enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryOperator {
    /// All terms must match
    And,
    /// Any term must match
    Or,
    /// Terms must not match
    Not,
    /// Exact phrase match
    Phrase,
    /// Proximity search
    Proximity { distance: u32 },
}

/// Date range filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    /// Start timestamp
    pub start: u64,
    /// End timestamp
    pub end: u64,
}

/// Sort order enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortOrder {
    /// Sort by relevance score (default)
    Relevance,
    /// Sort by date (newest first)
    DateDescending,
    /// Sort by date (oldest first)
    DateAscending,
    /// Sort by user alphabetically
    UserAscending,
    /// Sort by user reverse alphabetically
    UserDescending,
}

/// Search result with relevance scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Message that matched
    pub message: CandleMessage,
    /// Relevance score (0.0-1.0)
    pub relevance_score: f32,
    /// Matching terms
    pub matching_terms: Vec<Arc<str>>,
    /// Highlighted content
    pub highlighted_content: Option<Arc<str>>,
    /// Associated tags
    pub tags: Vec<Arc<str>>,
    /// Context messages (before/after)
    pub context: Vec<CandleMessage>,
    /// Match positions in the content
    pub match_positions: Vec<MatchPosition>,
    /// Search metadata
    pub metadata: Option<SearchResultMetadata>,
}

/// Search statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchStatistics {
    /// Total messages indexed
    pub total_messages: usize,
    /// Total unique terms
    pub total_terms: usize,
    /// Total search queries
    pub total_queries: usize,
    /// Average query time in milliseconds
    pub average_query_time: f64,
    /// Index size in bytes
    pub index_size: usize,
    /// Last index update timestamp
    pub last_index_update: u64,
}

/// Match position in content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchPosition {
    /// Start position in content
    pub start: usize,
    /// End position in content
    pub end: usize,
    /// Match type
    pub match_type: MatchType,
}

/// Type of match found
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatchType {
    /// Exact term match
    Exact,
    /// Fuzzy match
    Fuzzy,
    /// Phrase match
    Phrase,
    /// Proximity match
    Proximity,
}

/// Search result metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultMetadata {
    /// Query processing time in milliseconds
    pub query_time_ms: f64,
    /// Number of documents scanned
    pub documents_scanned: usize,
    /// Index version used
    pub index_version: u64,
    /// Search algorithm used
    pub algorithm: String,
}

/// Search options for fine-tuning behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchOptions {
    /// Enable SIMD optimization
    pub enable_simd: bool,
    /// Minimum relevance score threshold
    pub min_relevance: f32,
    /// Include context messages
    pub include_context: bool,
    /// Context window size (messages before/after)
    pub context_size: usize,
    /// Enable result highlighting
    pub enable_highlighting: bool,
    /// Maximum highlights per result
    pub max_highlights: usize,
    /// Enable fuzzy matching
    pub enable_fuzzy: bool,
    /// Fuzzy matching threshold (0.0-1.0)
    pub fuzzy_threshold: f32,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            enable_simd: true,
            min_relevance: 0.1,
            include_context: true,
            context_size: 2,
            enable_highlighting: true,
            max_highlights: 10,
            enable_fuzzy: false,
            fuzzy_threshold: 0.8,
        }
    }
}

/// Search scope enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchScope {
    /// Search all messages
    All,
    /// Search current session only
    CurrentSession,
    /// Search specific user's messages
    User(Arc<str>),
    /// Search specific time range
    TimeRange { start: u64, end: u64 },
    /// Search messages with specific tags
    Tagged(Vec<Arc<str>>),
}

/// Term frequency data structure for TF-IDF calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermFrequency {
    /// Term string
    pub term: Arc<str>,
    /// Frequency count
    pub frequency: u32,
    /// Document frequency
    pub document_frequency: u32,
    /// TF-IDF score
    pub tf_idf_score: f64,
}

impl TermFrequency {
    /// Calculate TF-IDF score
    pub fn calculate_tfidf(&self) -> f32 {
        self.tf_idf_score as f32
    }
}

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