//! Search system type definitions and data structures
//!
//! This module contains all the core type definitions, enums, and data structures
//! used throughout the domain search system, following the single responsibility principle.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::domain::chat::message::CandleSearchChatMessage as SearchChatMessage;

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
    Proximity {
        /// Distance value for proximity-based ranking
        distance: u32,
    },
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
    pub message: SearchChatMessage,
    /// Relevance score (0.0-1.0)
    pub relevance_score: f32,
    /// Matching terms
    pub matching_terms: Vec<Arc<str>>,
    /// Highlighted content
    pub highlighted_content: Option<Arc<str>>,
    /// Associated tags
    pub tags: Vec<Arc<str>>,
    /// Context messages (before/after)
    pub context: Vec<SearchChatMessage>,
    /// Match positions in the content
    pub match_positions: Vec<MatchPosition>,
    /// Search metadata
    pub metadata: Option<SearchResultMetadata>,
}

/// Match position in content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchPosition {
    /// Start position in characters
    pub start: usize,
    /// End position in characters
    pub end: usize,
    /// Matched term
    pub term: Arc<str>,
}

/// Search result metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultMetadata {
    /// Query processing time
    pub query_time_ms: f64,
    /// Index version used
    pub index_version: u32,
    /// Total matches before filtering
    pub total_matches: usize,
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

/// Term frequency and document frequency for TF-IDF calculation
#[derive(Debug, Clone)]
pub struct TermFrequency {
    /// Term frequency in document
    pub tf: f32,
    /// Document frequency (how many docs contain this term)
    pub df: u32,
    /// Total number of documents
    pub total_docs: u32,
}

impl TermFrequency {
    /// Calculate TF-IDF score
    pub fn calculate_tfidf(&self) -> f32 {
        let tf = self.tf;
        let idf = ((self.total_docs as f32) / (self.df as f32)).ln();
        tf * idf
    }
}

/// Inverted index entry
#[derive(Debug, Clone)]
pub struct IndexEntry {
    /// Document ID (message ID)
    pub doc_id: Arc<str>,
    /// Term frequency in document
    pub term_frequency: f32,
    /// Positions of term in document
    pub positions: Vec<usize>,
}

/// Search error types
#[derive(Debug, Clone)]
pub enum SearchError {
    /// Index operation failed
    IndexError { reason: Arc<str> },
    /// Query parsing failed
    QueryError { reason: Arc<str> },
    /// Search execution failed
    SearchError { reason: Arc<str> },
    /// Export operation failed
    ExportError { reason: Arc<str> },
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchError::IndexError { reason } => write!(f, "Index error: {}", reason),
            SearchError::QueryError { reason } => write!(f, "Query error: {}", reason),
            SearchError::SearchError { reason } => write!(f, "Search error: {}", reason),
            SearchError::ExportError { reason } => write!(f, "Export error: {}", reason),
        }
    }
}

impl std::error::Error for SearchError {}

/// Processed query with metadata
#[derive(Debug, Clone)]
pub struct ProcessedQuery {
    /// Original query string
    pub original: Arc<str>,
    /// Processed terms
    pub terms: Vec<Arc<str>>,
    /// Expanded terms from synonyms
    pub expanded_terms: Vec<Arc<str>>,
    /// Query operator
    pub operator: QueryOperator,
    /// Processing metadata
    pub metadata: QueryMetadata,
}

/// Query metadata
#[derive(Debug, Clone)]
pub struct QueryMetadata {
    /// Processing timestamp
    pub processed_at: u64,
    /// Processing time in microseconds
    pub processing_time_us: u64,
    /// Expansion applied
    pub expansion_applied: bool,
    /// Normalization applied
    pub normalization_applied: bool,
}

/// Search options for query processing
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Enable query expansion
    pub enable_query_expansion: bool,
    /// Expansion dictionary
    pub expansion_dictionary: HashMap<Arc<str>, Vec<Arc<str>>>,
    /// Enable fuzzy matching
    pub enable_fuzzy_matching: bool,
    /// Maximum edit distance for fuzzy matching
    pub max_edit_distance: u8,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            enable_query_expansion: false,
            expansion_dictionary: HashMap::new(),
            enable_fuzzy_matching: false,
            max_edit_distance: 2,
        }
    }
}

/// Ranking algorithm types
#[derive(Debug, Clone)]
pub enum RankingAlgorithm {
    /// TF-IDF based ranking
    TfIdf,
    /// BM25 ranking algorithm
    Bm25,
    /// Custom scoring function
    Custom,
}

/// Export format enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// XML format
    Xml,
    /// Plain text format
    Text,
}

/// Export options
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Export format
    pub format: ExportFormat,
    /// Include metadata
    pub include_metadata: bool,
    /// Include context messages
    pub include_context: bool,
    /// Maximum results to export
    pub max_results: Option<usize>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::Json,
            include_metadata: true,
            include_context: false,
            max_results: None,
        }
    }
}
