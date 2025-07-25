//! Core search types and utilities for candle_chat search functionality
//!
//! This module provides the fundamental data structures used throughout
//! the candle_chat search system with zero-allocation patterns.

use std::sync::Arc;

use fluent_ai_async::{AsyncStream, handle_error};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::CandleSearchChatMessage;

/// Search query structure for candle_chat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Search terms
    pub terms: Vec<Arc<str>>,
    /// Role filter
    pub role_filter: Option<crate::types::CandleMessageRole>,
    /// Date range filter
    pub date_range: Option<DateRange>,
    /// Result limit
    pub limit: Option<usize>,
    /// Enable fuzzy matching
    pub fuzzy: bool,
    /// Minimum similarity score
    pub min_similarity: f32,
    /// Highlight matches
    pub highlight: bool,
    /// Include context
    pub include_context: bool,
    /// Context window size
    pub context_size: usize,
    /// Search content only
    pub content_only: bool,
    /// Metadata filters
    pub metadata_filters: HashMap<String, String>,
    /// Case sensitive matching
    pub case_sensitive: bool,
    /// Whole word matching
    pub whole_words: bool,
    /// Regex matching
    pub regex: bool,
    /// Sort order
    pub sort_by: SortOrder,
    /// Group by conversation
    pub group_by_conversation: bool,
    /// Include deleted messages
    pub include_deleted: bool,
    /// Language filter
    pub language_filter: Option<String>,
    /// Minimum message length
    pub min_length: Option<usize>,
    /// Maximum message length
    pub max_length: Option<usize>}

/// Date range for filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    /// Start date (inclusive)
    pub start: Option<chrono::DateTime<chrono::Utc>>,
    /// End date (inclusive)
    pub end: Option<chrono::DateTime<chrono::Utc>>,
    /// Relative time specification
    pub relative: Option<String>,
    /// Timezone
    pub timezone: Option<String>,
    /// Include time in matching
    pub include_time: bool,
    /// Custom date format
    pub date_format: Option<String>}

/// Sort order options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortOrder {
    /// Sort by relevance
    Relevance,
    /// Sort by date (newest first)
    DateDesc,
    /// Sort by date (oldest first)
    DateAsc,
    /// Sort by length (longest first)
    LengthDesc,
    /// Sort by length (shortest first)
    LengthAsc,
    /// Sort by conversation
    Conversation,
    /// Sort by role
    Role,
    /// Custom sort field
    Custom(String)}

/// Search result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Result ID
    pub id: Uuid,
    /// Matching message
    pub message: CandleSearchChatMessage,
    /// Relevance score
    pub score: f32,
    /// Highlighted content
    pub highlighted_content: Option<String>,
    /// Context messages
    pub context: Vec<CandleSearchChatMessage>,
    /// Match metadata
    pub match_metadata: HashMap<String, String>,
    /// Match positions
    pub match_positions: Vec<MatchPosition>,
    /// Conversation ID
    pub conversation_id: Option<Uuid>,
    /// Associated tags
    pub tags: Vec<String>,
    /// Result timestamp
    pub result_timestamp: chrono::DateTime<chrono::Utc>,
    /// Extra data
    pub extra_data: HashMap<String, serde_json::Value>}

/// Position of a match within content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchPosition {
    /// Start position
    pub start: usize,
    /// End position
    pub end: usize,
    /// Matched term
    pub term: String,
    /// Match type
    pub match_type: MatchType,
    /// Confidence score
    pub confidence: f32}

/// Type of match
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatchType {
    /// Exact match
    Exact,
    /// Fuzzy match
    Fuzzy,
    /// Regex match
    Regex,
    /// Stemmed match
    Stemmed,
    /// Synonym match
    Synonym,
    /// Phonetic match
    Phonetic}

/// Search statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchStatistics {
    /// Total messages indexed
    pub total_messages: usize,
    /// Unique terms count
    pub unique_terms: usize,
    /// Average search time
    pub avg_search_time_ms: f64,
    /// Total searches performed
    pub total_searches: usize,
    /// Cache hit rate
    pub cache_hit_rate: f32,
    /// Index size in bytes
    pub index_size_bytes: usize,
    /// Memory usage
    pub memory_usage_bytes: usize,
    /// Last update timestamp
    pub last_update: chrono::DateTime<chrono::Utc>,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>}

/// Term frequency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermFrequency {
    /// The term
    pub term: Arc<str>,
    /// Frequency count
    pub frequency: usize,
    /// Document frequency
    pub document_frequency: usize,
    /// IDF score
    pub idf_score: f64,
    /// Term weight
    pub weight: f32}

impl TermFrequency {
    /// Create new term frequency
    pub fn new(term: Arc<str>, frequency: usize, document_frequency: usize, total_documents: usize) -> Self {
        let idf_score = if document_frequency > 0 {
            (total_documents as f64 / document_frequency as f64).ln()
        } else {
            0.0
        };
        
        Self {
            term,
            frequency,
            document_frequency,
            idf_score,
            weight: 1.0}
    }
}

impl Default for SearchQuery {
    fn default() -> Self {
        Self {
            terms: Vec::new(),
            role_filter: None,
            date_range: None,
            limit: Some(100),
            fuzzy: false,
            min_similarity: 0.7,
            highlight: true,
            include_context: false,
            context_size: 2,
            content_only: true,
            metadata_filters: HashMap::new(),
            case_sensitive: false,
            whole_words: false,
            regex: false,
            sort_by: SortOrder::Relevance,
            group_by_conversation: false,
            include_deleted: false,
            language_filter: None,
            min_length: None,
            max_length: None}
    }
}

impl Default for SortOrder {
    fn default() -> Self {
        SortOrder::Relevance
    }
}

impl Default for SearchStatistics {
    fn default() -> Self {
        Self {
            total_messages: 0,
            unique_terms: 0,
            avg_search_time_ms: 0.0,
            total_searches: 0,
            cache_hit_rate: 0.0,
            index_size_bytes: 0,
            memory_usage_bytes: 0,
            last_update: chrono::Utc::now(),
            performance_metrics: HashMap::new()}
    }
}