//! Core types and traits for chat search functionality
//!
//! This module provides the fundamental data structures and traits used throughout
//! the chat search system, including search queries, results, and streaming utilities.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::candle_chat::chat::MessageRole;
use crate::types::candle_chat::chat::message::SearchChatMessage;

/// Handle errors in streaming context without panicking
macro_rules! handle_error {
    ($error:expr, $context:literal) => {
        eprintln!("Streaming error in {}: {}", $context, $error);
        // Continue processing instead of returning error
    };
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

/// Represents a search query with various filtering options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// The search terms to look for
    pub terms: Vec<Arc<str>>,
    /// Optional role filter
    pub role_filter: Option<MessageRole>,
    /// Optional date range filter
    pub date_range: Option<DateRange>,
    /// Maximum number of results to return
    pub limit: Option<usize>,
    /// Whether to use fuzzy matching
    pub fuzzy: bool,
    /// Minimum similarity score for fuzzy matching (0.0 to 1.0)
    pub min_similarity: f32,
    /// Whether to highlight matches in results
    pub highlight: bool,
    /// Whether to include message context
    pub include_context: bool,
    /// Context window size (messages before/after)
    pub context_size: usize,
    /// Whether to search in message content only or include metadata
    pub content_only: bool,
    /// Custom metadata filters
    pub metadata_filters: HashMap<String, String>,
    /// Whether to use case-sensitive matching
    pub case_sensitive: bool,
    /// Whether to use whole word matching
    pub whole_words: bool,
    /// Whether to use regex matching
    pub regex: bool,
    /// Sort order for results
    pub sort_by: SortOrder,
    /// Whether to group results by conversation
    pub group_by_conversation: bool,
    /// Whether to include deleted messages
    pub include_deleted: bool,
    /// Language filter (ISO 639-1 codes)
    pub language_filter: Option<String>,
    /// Minimum message length filter
    pub min_length: Option<usize>,
    /// Maximum message length filter
    pub max_length: Option<usize>,
}

/// Date range filter for search queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    /// Start of the date range (inclusive)
    pub start: Option<chrono::DateTime<chrono::Utc>>,
    /// End of the date range (inclusive)
    pub end: Option<chrono::DateTime<chrono::Utc>>,
    /// Relative time specifications (e.g., "last 7 days")
    pub relative: Option<String>,
    /// Timezone for date interpretation
    pub timezone: Option<String>,
    /// Whether to include time in matching
    pub include_time: bool,
    /// Custom date format for parsing
    pub date_format: Option<String>,
}

/// Sort order options for search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortOrder {
    /// Sort by relevance score (default)
    Relevance,
    /// Sort by timestamp (newest first)
    DateDesc,
    /// Sort by timestamp (oldest first)
    DateAsc,
    /// Sort by message length (longest first)
    LengthDesc,
    /// Sort by message length (shortest first)
    LengthAsc,
    /// Sort by conversation ID
    Conversation,
    /// Sort by user/role
    Role,
    /// Custom sort field
    Custom(String),
}

/// Represents a single search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Unique identifier for the result
    pub id: Uuid,
    /// The matching message
    pub message: SearchChatMessage,
    /// Relevance score (0.0 to 1.0)
    pub score: f32,
    /// Highlighted content with matches emphasized
    pub highlighted_content: Option<String>,
    /// Context messages (before and after)
    pub context: Vec<SearchChatMessage>,
    /// Metadata about the match
    pub match_metadata: HashMap<String, String>,
    /// Position of matches within the content
    pub match_positions: Vec<MatchPosition>,
    /// Conversation ID this message belongs to
    pub conversation_id: Option<Uuid>,
    /// Tags associated with this result
    pub tags: Vec<String>,
    /// Timestamp when this result was generated
    pub result_timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional result-specific data
    pub extra_data: HashMap<String, serde_json::Value>,
}

/// Position information for a match within content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchPosition {
    /// Start position in the content
    pub start: usize,
    /// End position in the content
    pub end: usize,
    /// The matched term
    pub term: String,
    /// Match type (exact, fuzzy, regex, etc.)
    pub match_type: MatchType,
    /// Confidence score for this match
    pub confidence: f32,
}

/// Type of match found
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatchType {
    /// Exact string match
    Exact,
    /// Fuzzy/approximate match
    Fuzzy,
    /// Regular expression match
    Regex,
    /// Stemmed word match
    Stemmed,
    /// Synonym match
    Synonym,
    /// Phonetic match
    Phonetic,
}

/// Statistics about search operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchStatistics {
    /// Total number of messages indexed
    pub total_messages: usize,
    /// Total number of unique terms
    pub unique_terms: usize,
    /// Average search time in milliseconds
    pub avg_search_time_ms: f64,
    /// Total number of searches performed
    pub total_searches: usize,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f32,
    /// Index size in bytes
    pub index_size_bytes: usize,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Last index update timestamp
    pub last_update: chrono::DateTime<chrono::Utc>,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
}

/// Term frequency information for search indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermFrequency {
    /// The term
    pub term: Arc<str>,
    /// Frequency count
    pub frequency: usize,
    /// Document frequency (number of messages containing this term)
    pub document_frequency: usize,
    /// Inverse document frequency score
    pub idf_score: f64,
    /// Term importance weight
    pub weight: f32,
}

impl TermFrequency {
    /// Create a new term frequency entry
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
            weight: 1.0,
        }
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
            max_length: None,
        }
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
            performance_metrics: HashMap::new(),
        }
    }
}