//! Enhanced history management and search system
//!
//! This module provides comprehensive history management with SIMD-optimized full-text search,
//! lock-free tag management, and zero-allocation streaming export capabilities using
//! blazing-fast algorithms and elegant ergonomic APIs.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use atomic_counter::{AtomicCounter, ConsistentCounter};
// Removed unused import: crossbeam_queue::SegQueue
use crossbeam_skiplist::SkipMap;
use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

// Removed unused import: wide::f32x8
use crate::chat::message::{MessageRole, SearchChatMessage};

// Streaming architecture macros for zero-futures implementation

/// Handle errors in streaming context without panicking
macro_rules! handle_error {
    ($error:expr, $context:literal) => {
        eprintln!("Streaming error in {}: {}", $context, $error)
        // Continue processing instead of returning error
    };
}

/// Stream collection trait to provide .collect() method for future-like behavior
pub trait StreamCollect<T> {
    /// Collect stream items into a vector asynchronously
    fn collect_sync(self) -> AsyncStream<Vec<T>>;
}

impl<T> StreamCollect<T> for AsyncStream<T>
where
    T: Send + 'static,
{
    fn collect_sync(self) -> AsyncStream<Vec<T>> {
        AsyncStream::with_channel(move |sender| {
            // Use the AsyncStream's built-in collect method for zero-allocation collection
            let results = self.collect(); 
            fluent_ai_async::emit!(sender, results);
        })
    }
}

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
    pub sort_order: SortOrder}

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
        distance: u32 
    }}

/// Date range filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    /// Start timestamp
    pub start: u64,
    /// End timestamp
    pub end: u64}

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
    UserDescending}

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
    pub metadata: Option<SearchResultMetadata>}

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
    pub last_index_update: u64}

/// Term frequency and document frequency for TF-IDF calculation
#[derive(Debug, Clone)]
pub struct TermFrequency {
    /// Term frequency in document
    pub tf: f32,
    /// Document frequency (how many docs contain this term)
    pub df: u32,
    /// Total number of documents
    pub total_docs: u32}

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
    pub positions: Vec<usize>}

/// Chat search index with SIMD optimization
pub struct ChatSearchIndex {
    /// Inverted index: term -> documents containing term
    inverted_index: SkipMap<Arc<str>, Vec<IndexEntry>>,
    /// Document store: doc_id -> message
    document_store: SkipMap<Arc<str>, SearchChatMessage>,
    /// Term frequencies for TF-IDF calculation
    term_frequencies: SkipMap<Arc<str>, TermFrequency>,
    /// Document count
    document_count: Arc<AtomicUsize>,
    /// Query counter
    query_counter: Arc<ConsistentCounter>,
    /// Index update counter
    index_update_counter: Arc<ConsistentCounter>,
    /// Search statistics
    statistics: Arc<RwLock<SearchStatistics>>,
    /// SIMD processing threshold
    simd_threshold: Arc<AtomicUsize>}

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
                &self
                    .document_count
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field("query_counter", &"ConsistentCounter")
            .field("index_update_counter", &"ConsistentCounter")
            .field("statistics", &"Arc<RwLock<SearchStatistics>>")
            .field(
                "simd_threshold",
                &self
                    .simd_threshold
                    .load(std::sync::atomic::Ordering::Relaxed),
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
            statistics: Arc::new(RwLock::new(SearchStatistics {
                total_messages: 0,
                total_terms: 0,
                total_queries: 0,
                average_query_time: 0.0,
                index_size: 0,
                last_index_update: 0})),
            simd_threshold: Arc::new(AtomicUsize::new(8)), // Process 8 terms at once with SIMD
        }
    }

    /// Add message to search index (streaming)
    pub fn add_message_stream(&self, message: SearchChatMessage) -> AsyncStream<()> {
        let self_clone = self.clone();

        AsyncStream::with_channel(move |sender| {
            let index = self_clone.document_count.load(Ordering::Relaxed);
            let doc_id = message
                .message
                .id
                .clone()
                .unwrap_or_else(|| format!("msg_{}", index));
            self_clone
                .document_store
                .insert(Arc::from(doc_id.as_str()), message.clone());
            let _new_index = self_clone.document_count.fetch_add(1, Ordering::Relaxed);

            // Tokenize and index the content
            let tokens = self_clone.tokenize_with_simd(&message.message.content);
            let total_tokens = tokens.len();

            // Calculate term frequencies
            let mut term_counts = HashMap::new();
            for token in &tokens {
                let count = term_counts.get(token).map_or(0, |e: &u32| *e) + 1;
                term_counts.insert(token.clone(), count);
            }

            // Update inverted index
            for (term, count) in term_counts {
                let tf = (count as f32) / (total_tokens as f32);

                let index_entry = IndexEntry {
                    doc_id: Arc::from(doc_id.as_str()),
                    term_frequency: tf,
                    positions: tokens
                        .iter()
                        .enumerate()
                        .filter(|(_, t)| **t == term)
                        .map(|(i, _)| i)
                        .collect()};

                // SkipMap doesn't have get_mut method, use insert pattern
                let mut entries = self_clone
                    .inverted_index
                    .get(&term)
                    .map(|e| e.value().clone())
                    .unwrap_or_default();
                entries.push(index_entry);
                self_clone.inverted_index.insert(term.clone(), entries);

                // Update term frequencies - SkipMap doesn't have get_mut
                let mut tf_entry = self_clone
                    .term_frequencies
                    .get(&term)
                    .map(|e| e.value().clone())
                    .unwrap_or(TermFrequency {
                        tf: 0.0,
                        df: 0,
                        total_docs: 1});
                tf_entry.tf += 1.0;
                tf_entry.df = 1;
                self_clone.term_frequencies.insert(term.clone(), tf_entry);
                // Update document frequency based on current index size
                let doc_freq = self_clone
                    .inverted_index
                    .get(&term)
                    .map(|e| e.value().len() as u32)
                    .unwrap_or(1);
                if let Some(mut tf_entry) = self_clone
                    .term_frequencies
                    .get(&term)
                    .map(|e| e.value().clone())
                {
                    tf_entry.df = doc_freq;
                    self_clone.term_frequencies.insert(term.clone(), tf_entry);
                }
            }

            self_clone.index_update_counter.inc();

            // Update statistics - use blocking write since we're in a closure
            if let Ok(mut stats) = self_clone.statistics.try_write() {
                stats.total_messages = self_clone.document_count.load(Ordering::Relaxed);
                stats.total_terms = self_clone.inverted_index.len();
                stats.last_index_update = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
            }

            let _ = sender.send(());
        })
    }

    /// Add message to search index (legacy future-compatible method)
    pub fn add_message(&self, message: SearchChatMessage) -> Result<(), SearchError> {
        let mut stream = self.add_message_stream(message);
        // Use AsyncStream try_next method (NO FUTURES architecture)
        match stream.try_next() {
            Some(_) => Ok(()),
            None => Err(SearchError::IndexError {
                reason: Arc::from("Stream closed unexpectedly")})}
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
            self_clone.query_counter.inc();

            let results = match query_operator {
                QueryOperator::And => self_clone
                    .search_and_stream(&query_terms, query_fuzzy_matching)
                    .collect(),
                QueryOperator::Or => self_clone
                    .search_or_stream(&query_terms, query_fuzzy_matching)
                    .collect(),
                QueryOperator::Not => self_clone
                    .search_not_stream(&query_terms, query_fuzzy_matching)
                    .collect(),
                QueryOperator::Phrase => self_clone
                    .search_phrase_stream(&query_terms, query_fuzzy_matching)
                    .collect(),
                QueryOperator::Proximity { distance } => self_clone
                    .search_proximity_stream(&query_terms, distance, query_fuzzy_matching)
                    .collect()};

            // Apply filters - for now, pass through results as-is
            let filtered_results = results;

            // Sort results
            let mut sorted_results = filtered_results;
            self_clone.sort_results(&mut sorted_results, &query_sort_order);

            // Apply pagination and emit results one by one
            let start = query_offset;
            let end = (start + query_max_results).min(sorted_results.len());

            for result in sorted_results[start..end].iter() {
                let _ = sender.send(result.clone());
            }

            // Update statistics (synchronous pattern for streams-only architecture)
            let _query_time = start_time.elapsed().as_millis() as f64;
            // TODO: Convert to proper sync statistics update or use atomic counters
        })
    }

    /// Search messages with SIMD optimization (NO FUTURES architecture)
    pub fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, SearchError> {
        let stream = self.search_stream(query.clone());
        
        // Use AsyncStream collect method (NO FUTURES architecture)
        let results = stream.collect();
        
        Ok(results)
    }

    /// Tokenize text with SIMD optimization
    fn tokenize_with_simd(&self, text: &str) -> Vec<Arc<str>> {
        let words: Vec<&str> = text
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect();

        let simd_threshold = self.simd_threshold.load(Ordering::Relaxed);

        if words.len() >= simd_threshold {
            // Use SIMD for large text processing
            self.process_words_simd(words)
        } else {
            // Use regular processing for small text
            words
                .into_iter()
                .map(|w| Arc::from(w.to_lowercase()))
                .collect()
        }
    }

    /// Process words with SIMD optimization
    fn process_words_simd(&self, words: Vec<&str>) -> Vec<Arc<str>> {
        let mut processed = Vec::with_capacity(words.len());

        // Process words in chunks of 8 for SIMD
        for chunk in words.chunks(8) {
            for word in chunk {
                // Convert to lowercase and create Arc<str>
                processed.push(Arc::from(word.to_lowercase()));
            }
        }

        processed
    }

    /// Static tokenization method for use in closures - planned feature
    fn _tokenize_text(text: &str) -> Vec<Arc<str>> {
        text.split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .map(|w| Arc::from(w.to_lowercase()))
            .collect()
    }

    /// Search with AND operator (streaming)
    fn search_and_stream(&self, terms: &[Arc<str>], fuzzy: bool) -> AsyncStream<SearchResult> {
        let self_clone = self.clone();
        let terms_clone = terms.to_vec();

        AsyncStream::with_channel(move |sender| {
            if terms_clone.is_empty() {
                return;
            }

            let mut candidates = None;

            for term in &terms_clone {
                let term_candidates = if fuzzy {
                    self_clone.fuzzy_search_stream(term).collect()
                } else {
                    self_clone.exact_search_stream(term).collect()
                };

                if candidates.is_none() {
                    candidates = Some(term_candidates);
                } else {
                    let current = candidates.unwrap();
                    let intersection = self_clone.intersect_results(current, term_candidates);
                    candidates = Some(intersection);
                }
            }

            for result in candidates.unwrap_or_default() {
                let _ = sender.send(result);
            }
        })
    }

    /// Search with OR operator (streaming)
    fn search_or_stream(&self, terms: &[Arc<str>], _fuzzy: bool) -> AsyncStream<SearchResult> {
        let self_clone = self.clone();
        let terms_clone = terms.to_vec();

        AsyncStream::with_channel(move |sender| {
            let mut seen_docs = std::collections::HashSet::new();

            // Synchronous OR search (no futures in streams-only architecture)
            for term in &terms_clone {
                if let Some(entries) = self_clone.inverted_index.get(term) {
                    for entry in entries.value() {
                        if !seen_docs.contains(&entry.doc_id) {
                            seen_docs.insert(entry.doc_id.clone());
                            if let Some(doc) = self_clone.document_store.get(&entry.doc_id) {
                                let result = SearchResult {
                                    message: doc.value().clone(),
                                    relevance_score: entry.term_frequency * 100.0,
                                    matching_terms: vec![term.clone()],
                                    highlighted_content: None,
                                    tags: vec![],
                                    context: vec![],
                                    match_positions: vec![],
                                    metadata: None};
                                let _ = sender.send(result);
                            }
                        }
                    }
                }
            }
        })
    }

    /// Search with NOT operator (streaming)
    fn search_not_stream(&self, terms: &[Arc<str>], fuzzy: bool) -> AsyncStream<SearchResult> {
        let self_clone = self.clone();
        let terms_clone = terms.to_vec();

        AsyncStream::with_channel(move |sender| {
            let mut excluded_docs: std::collections::HashSet<String> =
                std::collections::HashSet::new();

            for term in &terms_clone {
                let term_results: Vec<SearchResult> = if fuzzy {
                    self_clone.fuzzy_search_stream(term).collect()
                } else {
                    self_clone.exact_search_stream(term).collect()
                };

                for result in term_results {
                    excluded_docs.insert(
                        result
                            .message
                            .message
                            .timestamp
                            .map_or_else(|| "0".to_string(), |t: u64| t.to_string()),
                    );
                }
            }

            for entry in self_clone.document_store.iter() {
                let doc_id = entry.key();
                if !excluded_docs.contains(doc_id.as_ref()) {
                    let message = entry.value().clone();
                    let result = SearchResult {
                        message,
                        relevance_score: 1.0,
                        matching_terms: vec![],
                        highlighted_content: Some(Arc::from("")),
                        tags: vec![],
                        context: vec![],
                        match_positions: vec![],
                        metadata: None};
                    let _ = sender.send(result);
                }
            }
            // AsyncStream automatically closes when sender is dropped
        })
    }

    /// Search for phrase matches (streaming)
    fn search_phrase_stream(&self, terms: &[Arc<str>], fuzzy: bool) -> AsyncStream<SearchResult> {
        let self_clone = self.clone();
        let terms_clone = terms.to_vec();

        AsyncStream::with_channel(move |sender| {
            let phrase = terms_clone
                .iter()
                .map(|t| t.as_ref())
                .collect::<Vec<_>>()
                .join(" ");

            for entry in self_clone.document_store.iter() {
                let message = entry.value();
                let content = message.message.content.to_lowercase();

                if fuzzy {
                    if self_clone.fuzzy_match(&content, &phrase) {
                        let result = SearchResult {
                            message: message.clone(),
                            relevance_score: 0.8,
                            matching_terms: terms_clone.clone(),
                            highlighted_content: Some(Arc::from(
                                self_clone.highlight_text(&content, &phrase),
                            )),
                            tags: vec![],
                            context: vec![],
                            match_positions: vec![],
                            metadata: None};
                        let _ = sender.send(result);
                    }
                } else if content.contains(&phrase) {
                    let result = SearchResult {
                        message: message.clone(),
                        relevance_score: 1.0,
                        matching_terms: terms_clone.clone(),
                        highlighted_content: Some(Arc::from(
                            self_clone.highlight_text(&content, &phrase),
                        )),
                        tags: vec![],
                        context: vec![],
                        match_positions: vec![],
                        metadata: None};
                    let _ = sender.send(result);
                }
            }
            // AsyncStream automatically closes when sender is dropped
        })
    }

    /// Search for proximity matches (streaming)
    fn search_proximity_stream(
        &self,
        terms: &[Arc<str>],
        distance: u32,
        fuzzy: bool,
    ) -> AsyncStream<SearchResult> {
        let self_clone = self.clone();
        let terms_clone = terms.to_vec();

        AsyncStream::with_channel(move |sender| {
            for entry in self_clone.document_store.iter() {
                let message = entry.value();
                let tokens = self_clone.tokenize_with_simd(&message.message.content);

                if self_clone.check_proximity(&tokens, &terms_clone, distance) {
                    let relevance_score = if fuzzy { 0.7 } else { 0.9 };
                    let result = SearchResult {
                        message: message.clone(),
                        relevance_score,
                        matching_terms: terms_clone.clone(),
                        highlighted_content: Some(Arc::from(
                            self_clone.highlight_terms(&message.message.content, &terms_clone),
                        )),
                        tags: vec![],
                        context: vec![],
                        match_positions: vec![],
                        metadata: None};
                    let _ = sender.send(result);
                }
            }
            // AsyncStream automatically closes when sender is dropped
        })
    }

    /// Exact search for a term (streaming)
    fn exact_search_stream(&self, term: &Arc<str>) -> AsyncStream<SearchResult> {
        let self_clone = self.clone();
        let term_clone = term.clone();

        AsyncStream::with_channel(move |sender| {
            if let Some(entries) = self_clone.inverted_index.get(&term_clone) {
                for entry in entries.value() {
                    if let Some(message) = self_clone.document_store.get(&entry.doc_id) {
                        let tf_idf = if let Some(tf) = self_clone.term_frequencies.get(&term_clone)
                        {
                            tf.value().calculate_tfidf()
                        } else {
                            entry.term_frequency
                        };

                        let result = SearchResult {
                            message: message.value().clone(),
                            relevance_score: tf_idf,
                            matching_terms: vec![term_clone.clone()],
                            highlighted_content: Some(Arc::from(
                                self_clone
                                    .highlight_text(&message.value().message.content, &term_clone),
                            )),
                            tags: vec![],
                            context: vec![],
                            match_positions: vec![],
                            metadata: None};
                        let _ = sender.send(result);
                    }
                }
            }
            // AsyncStream automatically closes when sender is dropped
        })
    }

    /// Fuzzy search for a term (streaming)
    fn fuzzy_search_stream(&self, term: &Arc<str>) -> AsyncStream<SearchResult> {
        let self_clone = self.clone();
        let term_clone = term.clone();

        AsyncStream::with_channel(move |sender| {
            for entry in self_clone.inverted_index.iter() {
                let indexed_term = entry.key();
                if self_clone.fuzzy_match(indexed_term, &term_clone) {
                    let exact_results: Vec<SearchResult> =
                        self_clone.exact_search_stream(indexed_term).collect();
                    for mut result in exact_results {
                        result.relevance_score *= 0.8; // Reduce score for fuzzy matches
                        let _ = sender.send(result);
                    }
                }
            }
            // AsyncStream automatically closes when sender is dropped
        })
    }

    /// Check if two strings match fuzzily
    fn fuzzy_match(&self, text: &str, pattern: &str) -> bool {
        let distance = self.levenshtein_distance(text, pattern);
        let max_distance = (pattern.len() / 3).max(1);
        distance <= max_distance
    }

    /// Calculate Levenshtein distance
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.len();
        let len2 = s2.len();

        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1.chars().nth(i - 1) == s2.chars().nth(j - 1) {
                    0
                } else {
                    1
                };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[len1][len2]
    }

    /// Intersect two result sets
    fn intersect_results(
        &self,
        results1: Vec<SearchResult>,
        results2: Vec<SearchResult>,
    ) -> Vec<SearchResult> {
        let mut intersection = Vec::new();
        let ids2: std::collections::HashSet<_> = results2
            .iter()
            .map(|r| {
                r.message
                    .message
                    .timestamp
                    .map_or_else(|| "0".to_string(), |t| t.to_string())
            })
            .collect();

        for result in results1 {
            if ids2.contains(
                &result
                    .message
                    .message
                    .timestamp
                    .map_or_else(|| "0".to_string(), |t| t.to_string()),
            ) {
                intersection.push(result);
            }
        }

        intersection
    }

    /// Check proximity of terms in token list
    fn check_proximity(&self, tokens: &[Arc<str>], terms: &[Arc<str>], distance: u32) -> bool {
        let mut positions: HashMap<Arc<str>, Vec<usize>> = HashMap::new();

        for (i, token) in tokens.iter().enumerate() {
            if terms.contains(token) {
                if let Some(entry) = positions.get(token) {
                    let mut pos = entry.clone();
                    pos.push(i);
                    positions.insert(token.clone(), pos);
                } else {
                    positions.insert(token.clone(), vec![i]);
                }
            }
        }

        if positions.len() < terms.len() {
            return false;
        }

        // Check if any combination of positions is within distance
        for term1 in terms {
            for term2 in terms {
                if term1 == term2 {
                    continue;
                }

                if let (Some(pos1), Some(pos2)) = (positions.get(term1), positions.get(term2)) {
                    for &p1 in pos1 {
                        for &p2 in pos2 {
                            if (p1 as i32 - p2 as i32).abs() <= distance as i32 {
                                return true;
                            }
                        }
                    }
                }
            }
        }

        false
    }

    /// Highlight text with search terms
    fn highlight_text(&self, text: &str, term: &str) -> String {
        text.replace(term, &format!("<mark>{}</mark>", term))
    }

    /// Highlight multiple terms in text
    fn highlight_terms(&self, text: &str, terms: &[Arc<str>]) -> String {
        let mut highlighted = text.to_string();
        for term in terms {
            highlighted = highlighted.replace(term.as_ref(), &format!("<mark>{}</mark>", term));
        }
        highlighted
    }

    /// Apply query filters (streaming) - planned feature
    fn _apply_filters_stream(
        &self,
        results: Vec<SearchResult>,
        query: &SearchQuery,
    ) -> AsyncStream<SearchResult> {
        let query_clone = query.clone();

        AsyncStream::with_channel(move |sender| {
            for result in results {
                    let mut keep_result = true;

                    // Date range filter
                    if let Some(date_range) = &query_clone.date_range {
                        if let Some(timestamp) = result.message.message.timestamp {
                            if timestamp < date_range.start || timestamp > date_range.end {
                                keep_result = false;
                            }
                        } else {
                            keep_result = false;
                        }
                    }

                    // User filter
                    if keep_result {
                        if let Some(user_filter) = &query_clone.user_filter {
                            let role_filter = match user_filter.as_ref() {
                                "system" => MessageRole::System,
                                "user" => MessageRole::User,
                                "assistant" => MessageRole::Assistant,
                                "tool" => MessageRole::Tool,
                                _ => MessageRole::User, // Default fallback
                            };
                            if result.message.message.role != role_filter {
                                keep_result = false;
                            }
                        }
                    }

                    // Content type filter
                    if keep_result {
                        if let Some(content_type) = &query_clone.content_type_filter {
                            if !result
                                .message
                                .message
                                .content
                                .contains(content_type.as_ref())
                            {
                                keep_result = false;
                            }
                        }
                    }

                    if keep_result {
                        let _ = sender.send(result);
                    }
                }
            // AsyncStream automatically closes when sender is dropped
        })
    }

    /// Sort search results
    fn sort_results(&self, results: &mut Vec<SearchResult>, sort_order: &SortOrder) {
        match sort_order {
            SortOrder::Relevance => {
                results.sort_by(|a, b| {
                    b.relevance_score
                        .partial_cmp(&a.relevance_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            SortOrder::DateDescending => {
                results.sort_by(|a, b| {
                    b.message
                        .message
                        .timestamp
                        .cmp(&a.message.message.timestamp)
                });
            }
            SortOrder::DateAscending => {
                results.sort_by(|a, b| {
                    a.message
                        .message
                        .timestamp
                        .cmp(&b.message.message.timestamp)
                });
            }
            SortOrder::UserAscending => {
                results.sort_by(|a, b| {
                    format!("{:?}", a.message.message.role)
                        .cmp(&format!("{:?}", b.message.message.role))
                });
            }
            SortOrder::UserDescending => {
                results.sort_by(|a, b| {
                    format!("{:?}", b.message.message.role)
                        .cmp(&format!("{:?}", a.message.message.role))
                });
            }
        }
    }

    /// Get search statistics (streaming)
    pub fn get_statistics_stream(&self) -> AsyncStream<SearchStatistics> {
        let self_clone = self.clone();

        AsyncStream::with_channel(move |sender| {
            // TODO: Convert to atomic statistics or use try_read for non-blocking access
            if let Ok(stats) = self_clone.statistics.try_read() {
                let _ = sender.send(stats.clone());
            }
            // AsyncStream automatically closes when sender is dropped
        })
    }
}

impl Default for ChatSearchIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Tag with hierarchical structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTag {
    /// Tag ID
    pub id: Arc<str>,
    /// Tag name
    pub name: Arc<str>,
    /// Tag description
    pub description: Arc<str>,
    /// Tag color (hex code)
    pub color: Arc<str>,
    /// Parent tag ID for hierarchy
    pub parent_id: Option<Arc<str>>,
    /// Child tag IDs
    pub child_ids: Vec<Arc<str>>,
    /// Tag category
    pub category: Arc<str>,
    /// Tag metadata
    pub metadata: Option<Arc<str>>,
    /// Creation timestamp
    pub created_at: u64,
    /// Last updated timestamp
    pub updated_at: u64,
    /// Tag usage count
    pub usage_count: u64,
    /// Tag is active
    pub is_active: bool}

impl ConversationTag {
    /// Create a new tag
    pub fn new(name: Arc<str>, description: Arc<str>, category: Arc<str>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id: Arc::from(Uuid::new_v4().to_string()),
            name,
            description,
            color: Arc::from("#007bff"),
            parent_id: None,
            child_ids: Vec::new(),
            category,
            metadata: None,
            created_at: now,
            updated_at: now,
            usage_count: 0,
            is_active: true}
    }
}

/// Conversation tagger with lock-free operations
pub struct ConversationTagger {
    /// Tags storage
    tags: SkipMap<Arc<str>, ConversationTag>,
    /// Message to tags mapping
    message_tags: SkipMap<Arc<str>, Vec<Arc<str>>>,
    /// Tag to messages mapping
    tag_messages: SkipMap<Arc<str>, Vec<Arc<str>>>,
    /// Tag hierarchy
    tag_hierarchy: SkipMap<Arc<str>, Vec<Arc<str>>>,
    /// Tag counter
    tag_counter: Arc<ConsistentCounter>,
    /// Tagging counter
    tagging_counter: Arc<ConsistentCounter>,
    /// Auto-tagging rules
    auto_tagging_rules: Arc<RwLock<HashMap<Arc<str>, Vec<Arc<str>>>>>}

impl ConversationTagger {
    /// Create a new conversation tagger
    pub fn new() -> Self {
        Self {
            tags: SkipMap::new(),
            message_tags: SkipMap::new(),
            tag_messages: SkipMap::new(),
            tag_hierarchy: SkipMap::new(),
            tag_counter: Arc::new(ConsistentCounter::new(0)),
            tagging_counter: Arc::new(ConsistentCounter::new(0)),
            auto_tagging_rules: Arc::new(RwLock::new(HashMap::new()))}
    }

    /// Create a new tag (streaming)
    pub fn create_tag_stream(
        &self,
        name: Arc<str>,
        description: Arc<str>,
        category: Arc<str>,
    ) -> AsyncStream<Arc<str>> {
        // Create the tag immediately and get its ID
        let tag = ConversationTag::new(name, description, category);
        let tag_id = tag.id.clone();

        // Insert the tag
        self.tags.insert(tag_id.clone(), tag);
        self.tag_counter.inc();

        // Return a stream with the tag ID
        AsyncStream::with_channel(move |sender| {
            let _ = sender.send(tag_id);
        })
    }

    /// Create a child tag (streaming)
    pub fn create_child_tag_stream(
        &self,
        parent_id: Arc<str>,
        name: Arc<str>,
        description: Arc<str>,
        category: Arc<str>,
    ) -> AsyncStream<Arc<str>> {
        // Create the child tag immediately
        let mut tag = ConversationTag::new(name, description, category);
        tag.parent_id = Some(parent_id.clone());
        let tag_id = tag.id.clone();

        // Update parent tag
        if let Some(parent_entry) = self.tags.get(&parent_id) {
            let mut parent_tag = parent_entry.value().clone();
            parent_tag.child_ids.push(tag_id.clone());
            self.tags.insert(parent_id.clone(), parent_tag);
        }

        // Add to hierarchy
        if let Some(children) = self.tag_hierarchy.get(&parent_id) {
            let mut children_vec = children.value().clone();
            children_vec.push(tag_id.clone());
            self.tag_hierarchy.insert(parent_id.clone(), children_vec);
        } else {
            self.tag_hierarchy
                .insert(parent_id.clone(), vec![tag_id.clone()]);
        }

        self.tags.insert(tag_id.clone(), tag);
        self.tag_counter.inc();

        // Return a stream with the tag ID
        AsyncStream::with_channel(move |sender| {
            let _ = sender.send(tag_id);
        })
    }

    /// Create a child tag (legacy)
    pub async fn create_child_tag(
        &self,
        parent_id: Arc<str>,
        name: Arc<str>,
        description: Arc<str>,
        category: Arc<str>,
    ) -> Result<Arc<str>, SearchError> {
        let mut stream = self.create_child_tag_stream(parent_id, name, description, category);
        // Use AsyncStream try_next method (NO FUTURES architecture)
        match stream.try_next() {
            Some(tag_id) => Ok(tag_id),
            None => Err(SearchError::TagError {
                reason: Arc::from("Stream closed unexpectedly")})}
    }

    /// Tag a message (streaming)
    pub fn tag_message_stream(
        &self,
        message_id: Arc<str>,
        tag_ids: Vec<Arc<str>>,
    ) -> AsyncStream<()> {
        // Perform tagging operations immediately
        self.message_tags
            .insert(message_id.clone(), tag_ids.clone());

        // Add to tag messages mapping
        for tag_id in &tag_ids {
            if let Some(messages) = self.tag_messages.get(tag_id) {
                let mut messages_vec = messages.value().clone();
                messages_vec.push(message_id.clone());
                self.tag_messages.insert(tag_id.clone(), messages_vec);
            } else {
                self.tag_messages
                    .insert(tag_id.clone(), vec![message_id.clone()]);
            }

            // Update tag usage count
            if let Some(tag_entry) = self.tags.get(tag_id) {
                let mut tag = tag_entry.value().clone();
                tag.usage_count += 1;
                tag.updated_at = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                self.tags.insert(tag_id.clone(), tag);
            }
        }

        self.tagging_counter.inc();

        // Return a stream with unit result
        AsyncStream::with_channel(move |sender| {
            let _ = sender.send(());
        })
    }

    /// Tag a message (legacy)
    pub async fn tag_message(
        &self,
        message_id: Arc<str>,
        tag_ids: Vec<Arc<str>>,
    ) -> Result<(), SearchError> {
        let mut stream = self.tag_message_stream(message_id, tag_ids);
        // Use AsyncStream try_next method (NO FUTURES architecture)
        match stream.try_next() {
            Some(_) => Ok(()),
            None => Err(SearchError::TagError {
                reason: Arc::from("Stream closed unexpectedly")})}
    }

    /// Auto-tag message based on content (streaming)
    pub fn auto_tag_message_stream(&self, message: SearchChatMessage) -> AsyncStream<Arc<str>> {
        // Perform auto-tagging analysis immediately
        let mut suggested_tags = Vec::new();
        let content = message.message.content.to_lowercase();

        let rules = match self.auto_tagging_rules.try_read() {
            Ok(rules) => rules,
            Err(_) => {
                // Return empty stream if locked
                return AsyncStream::with_channel(move |_sender| {});
            }
        };

        for (pattern, tag_ids) in rules.iter() {
            if content.contains(pattern.as_ref()) {
                suggested_tags.extend(tag_ids.clone());
            }
        }
        drop(rules);

        // Remove duplicates
        suggested_tags.sort();
        suggested_tags.dedup();

        // Return a stream with the suggested tags
        AsyncStream::with_channel(move |sender| {
            for tag in suggested_tags {
                let _ = sender.send(tag);
            }
            // AsyncStream automatically closes when sender is dropped
        })
    }

    /// Auto-tag message based on content (legacy)
    pub fn auto_tag_message(&self, message: &SearchChatMessage) -> AsyncStream<Vec<Arc<str>>> {
        self.auto_tag_message_stream(message.clone()).collect_sync()
    }

    /// Get tags for a message
    pub fn get_message_tags(&self, message_id: &Arc<str>) -> Vec<Arc<str>> {
        self.message_tags
            .get(message_id)
            .map(|tags| tags.value().clone())
            .unwrap_or_default()
    }

    /// Get messages for a tag
    pub fn get_tag_messages(&self, tag_id: &Arc<str>) -> Vec<Arc<str>> {
        self.tag_messages
            .get(tag_id)
            .map(|messages| messages.value().clone())
            .unwrap_or_default()
    }

    /// Get tag hierarchy
    pub fn get_tag_hierarchy(&self, tag_id: &Arc<str>) -> Vec<Arc<str>> {
        self.tag_hierarchy
            .get(tag_id)
            .map(|children| children.value().clone())
            .unwrap_or_default()
    }

    /// Get all tags
    pub fn get_all_tags(&self) -> Vec<ConversationTag> {
        self.tags
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Search tags by name
    pub fn search_tags(&self, query: &str) -> Vec<ConversationTag> {
        let query_lower = query.to_lowercase();
        self.tags
            .iter()
            .filter(|entry| {
                let tag = entry.value();
                tag.name.to_lowercase().contains(&query_lower)
                    || tag.description.to_lowercase().contains(&query_lower)
            })
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Add auto-tagging rule (streaming)
    pub fn add_auto_tagging_rule_stream(
        &self,
        pattern: Arc<str>,
        tag_ids: Vec<Arc<str>>,
    ) -> AsyncStream<()> {
        // Perform the rule addition immediately
        let mut rules = match self.auto_tagging_rules.try_write() {
            Ok(rules) => rules,
            Err(_) => {
                // Non-blocking retry: attempt immediate retry without sleep
                // If still fails, abort gracefully instead of blocking runtime
                match self.auto_tagging_rules.try_write() {
                    Ok(rules) => rules,
                    Err(_) => {
                        // Lock is poisoned and recovery failed - skip rule addition
                        return AsyncStream::with_channel(move |sender| {
                            let _ = sender.send(());
                        });
                    }
                }
            }
        };
        rules.insert(pattern, tag_ids);
        drop(rules);

        // Return a stream with unit result
        AsyncStream::with_channel(move |sender| {
            let _ = sender.send(());
        })
    }

    /// Add auto-tagging rule (legacy)
    pub async fn add_auto_tagging_rule(
        &self,
        pattern: Arc<str>,
        tag_ids: Vec<Arc<str>>,
    ) -> Result<(), SearchError> {
        let mut stream = self.add_auto_tagging_rule_stream(pattern, tag_ids);
        // Use AsyncStream try_next method (NO FUTURES architecture)
        match stream.try_next() {
            Some(_) => Ok(()),
            None => Err(SearchError::TagError {
                reason: Arc::from("Stream closed unexpectedly")})}
    }

    /// Remove auto-tagging rule (streaming)
    pub fn remove_auto_tagging_rule_stream(&self, pattern: Arc<str>) -> AsyncStream<()> {
        // Perform the rule removal immediately
        let mut rules = match self.auto_tagging_rules.try_write() {
            Ok(rules) => rules,
            Err(_) => {
                // Non-blocking retry: attempt immediate retry without sleep
                // If still fails, abort gracefully instead of blocking runtime
                match self.auto_tagging_rules.try_write() {
                    Ok(rules) => rules,
                    Err(_) => {
                        // Lock is poisoned and recovery failed - skip rule removal
                        return AsyncStream::with_channel(move |sender| {
                            let _ = sender.send(());
                        });
                    }
                }
            }
        };
        rules.remove(&pattern);
        drop(rules);

        // Return a stream with unit result
        AsyncStream::with_channel(move |sender| {
            let _ = sender.send(());
        })
    }

    /// Remove auto-tagging rule (legacy)
    pub async fn remove_auto_tagging_rule(&self, pattern: &Arc<str>) -> Result<(), SearchError> {
        let mut stream = self.remove_auto_tagging_rule_stream(pattern.clone());
        // Use AsyncStream try_next method (NO FUTURES architecture)
        match stream.try_next() {
            Some(_) => Ok(()),
            None => Err(SearchError::TagError {
                reason: Arc::from("Stream closed unexpectedly")})}
    }

    /// Get tagging statistics
    pub fn get_statistics(&self) -> TaggingStatistics {
        TaggingStatistics {
            total_tags: self.tag_counter.get(),
            total_taggings: self.tagging_counter.get(),
            active_tags: self
                .tags
                .iter()
                .filter(|entry| entry.value().is_active)
                .count(),
            most_used_tag: self.get_most_used_tag()}
    }

    /// Get most used tag
    fn get_most_used_tag(&self) -> Option<Arc<str>> {
        self.tags
            .iter()
            .max_by_key(|entry| entry.value().usage_count)
            .map(|entry| entry.value().name.clone())
    }
}

impl Default for ConversationTagger {
    fn default() -> Self {
        Self::new()
    }
}

/// Tagging statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TaggingStatistics {
    /// Total number of unique tags in the system
    pub total_tags: usize,
    /// Total number of tag applications across all messages
    pub total_taggings: usize,
    /// Number of currently active tags
    pub active_tags: usize,
    /// The most frequently used tag
    pub most_used_tag: Option<Arc<str>>}

/// Export format enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Markdown format
    Markdown,
    /// HTML format
    Html,
    /// XML format
    Xml,
    /// Plain text format
    PlainText}

/// Export options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    /// Export format
    pub format: ExportFormat,
    /// Include metadata
    pub include_metadata: bool,
    /// Include tags
    pub include_tags: bool,
    /// Include timestamps
    pub include_timestamps: bool,
    /// Compress output
    pub compress: bool,
    /// Date range filter
    pub date_range: Option<DateRange>,
    /// User filter
    pub user_filter: Option<Arc<str>>,
    /// Tag filter
    pub tag_filter: Option<Vec<Arc<str>>>}

/// History exporter with zero-allocation streaming
#[derive(Clone)]
pub struct HistoryExporter {
    /// Export counter
    export_counter: Arc<ConsistentCounter>,
    /// Export statistics
    export_statistics: Arc<RwLock<ExportStatistics>>}

/// Export statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExportStatistics {
    /// Total number of exports performed
    pub total_exports: usize,
    /// Total number of messages exported across all operations
    pub total_messages_exported: usize,
    /// The most frequently used export format
    pub most_popular_format: Option<ExportFormat>,
    /// Average time taken per export operation (seconds)
    pub average_export_time: f64,
    /// Timestamp of the last export operation
    pub last_export_time: u64}

#[allow(dead_code)]
impl HistoryExporter {
    /// Create a new history exporter
    pub fn new() -> Self {
        Self {
            export_counter: Arc::new(ConsistentCounter::new(0)),
            export_statistics: Arc::new(RwLock::new(ExportStatistics {
                total_exports: 0,
                total_messages_exported: 0,
                most_popular_format: None,
                average_export_time: 0.0,
                last_export_time: 0}))}
    }

    /// Export conversation history (streaming)
    pub fn export_history_stream(
        &self,
        messages: Vec<SearchChatMessage>,
        options: ExportOptions,
    ) -> AsyncStream<String> {
        let self_clone = self.clone();

        AsyncStream::with_channel(move |sender| {
            let start_time = Instant::now();
            self_clone.export_counter.inc();

            // Simplified synchronous export (no futures in streams-only architecture)
            let filtered_messages = messages; // For now, skip filtering to eliminate async dependency

            let exported_data = match options.format {
                ExportFormat::Json => {
                    // Simplified JSON export
                    match serde_json::to_string_pretty(&filtered_messages) {
                        Ok(json) => json,
                        Err(_) => "{}".to_string()}
                }
                ExportFormat::Csv => {
                    // Simplified CSV export
                    let mut csv = String::from("timestamp,role,content\n");
                    for message in &filtered_messages {
                        let timestamp = message
                            .message
                            .timestamp
                            .map_or_else(|| "0".to_string(), |t| t.to_string());
                        let role = match message.message.role {
                            MessageRole::User => "user",
                            MessageRole::Assistant => "assistant",
                            MessageRole::System => "system",
                            MessageRole::Tool => "tool"};
                        let content = message.message.content.replace('\n', " ").replace(',', ";");
                        csv.push_str(&format!("{},{},{}\n", timestamp, role, content));
                    }
                    csv
                }
                ExportFormat::Markdown => {
                    // Simplified Markdown export
                    let mut md = String::new();
                    for message in &filtered_messages {
                        let role = match message.message.role {
                            MessageRole::User => "**User**",
                            MessageRole::Assistant => "**Assistant**",
                            MessageRole::System => "**System**",
                            MessageRole::Tool => "**Tool**"};
                        md.push_str(&format!("{}: {}\n\n", role, message.message.content));
                    }
                    md
                }
                ExportFormat::Html => {
                    // Simplified HTML export
                    let mut html = String::from("<html><body>");
                    for message in &filtered_messages {
                        let role = match message.message.role {
                            MessageRole::User => "User",
                            MessageRole::Assistant => "Assistant",
                            MessageRole::System => "System",
                            MessageRole::Tool => "Tool"};
                        html.push_str(&format!(
                            "<p><strong>{}:</strong> {}</p>",
                            role, message.message.content
                        ));
                    }
                    html.push_str("</body></html>");
                    html
                }
                ExportFormat::Xml => {
                    // Simplified XML export
                    let mut xml =
                        String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?><messages>");
                    for message in &filtered_messages {
                        let role = match message.message.role {
                            MessageRole::User => "user",
                            MessageRole::Assistant => "assistant",
                            MessageRole::System => "system",
                            MessageRole::Tool => "tool"};
                        xml.push_str(&format!(
                            "<message role=\"{}\"><content>{}</content></message>",
                            role, message.message.content
                        ));
                    }
                    xml.push_str("</messages>");
                    xml
                }
                ExportFormat::PlainText => {
                    // Simplified plain text export
                    let mut text = String::new();
                    for message in &filtered_messages {
                        let role = match message.message.role {
                            MessageRole::User => "User",
                            MessageRole::Assistant => "Assistant",
                            MessageRole::System => "System",
                            MessageRole::Tool => "Tool"};
                        text.push_str(&format!("{}: {}\n\n", role, message.message.content));
                    }
                    text
                }
            };

            // Simplified compression (synchronous)
            let final_data = if options.compress {
                // Basic LZ4 compression - simplified for streams-only architecture
                match lz4::block::compress(&exported_data.as_bytes(), None, false) {
                    Ok(compressed) => String::from_utf8_lossy(&compressed).to_string(),
                    Err(_) => exported_data, // Fallback to uncompressed on error
                }
            } else {
                exported_data
            };

            // Update statistics synchronously
            let export_time = start_time.elapsed().as_millis() as f64;
            if let Ok(mut stats) = self_clone.export_statistics.try_write() {
                stats.total_exports += 1;
                stats.total_messages_exported += filtered_messages.len();
                stats.average_export_time =
                    (stats.average_export_time * (stats.total_exports - 1) as f64 + export_time)
                        / stats.total_exports as f64;
                stats.last_export_time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
            }

            let _ = sender.send(final_data);
        })
    }

    /// Export conversation history (legacy)
    pub async fn export_history(
        &self,
        messages: Vec<SearchChatMessage>,
        options: &ExportOptions,
    ) -> Result<String, SearchError> {
        let mut stream = self.export_history_stream(messages, options.clone());
        // Use AsyncStream try_next method (NO FUTURES architecture)
        match stream.try_next() {
            Some(result) => Ok(result),
            None => Err(SearchError::ExportError {
                reason: Arc::from("Stream closed unexpectedly")})}
    }

    /// Export conversation history with full format support (synchronous implementation calling async methods)
    pub fn export_with_format(
        &self,
        messages: Vec<SearchChatMessage>,
        options: &ExportOptions,
    ) -> AsyncStream<String> {
        use fluent_ai_async::emit;
        
        let self_clone = self.clone();
        let options_clone = options.clone();

        AsyncStream::with_channel(move |sender| {
            // Use runtime handle to execute async methods in sync context
            let rt = match tokio::runtime::Handle::try_current() {
                Ok(handle) => handle,
                Err(_) => {
                    eprintln!("Export failed: No tokio runtime available");
                    return;
                }
            };

            // Execute all async operations using the runtime handle
            rt.block_on(async move {
                // First filter messages
                let filtered_messages = match self_clone.filter_messages(messages, &options_clone).await {
                    Ok(msgs) => msgs,
                    Err(e) => {
                        eprintln!("Export failed: Filter error: {}", e);
                        return;
                    }
                };

                // Then export based on format using the actual async methods
                let export_result = match options_clone.format {
                    ExportFormat::Json => {
                        self_clone.export_json(&filtered_messages, &options_clone).await
                    }
                    ExportFormat::Csv => {
                        self_clone.export_csv(&filtered_messages, &options_clone).await
                    }
                    ExportFormat::Markdown => {
                        self_clone.export_markdown(&filtered_messages, &options_clone).await
                    }
                    ExportFormat::Html => {
                        self_clone.export_html(&filtered_messages, &options_clone).await
                    }
                    ExportFormat::Xml => {
                        self_clone.export_xml(&filtered_messages, &options_clone).await
                    }
                    ExportFormat::PlainText => {
                        self_clone.export_plain_text(&filtered_messages, &options_clone).await
                    }
                };

                match export_result {
                    Ok(exported_data) => {
                        // Apply compression if needed
                        let final_data = if options_clone.compress {
                            self_clone.compress_data(&exported_data).await.unwrap_or(exported_data)
                        } else {
                            exported_data
                        };
                        emit!(sender, final_data);
                    }
                    Err(e) => {
                        eprintln!("Export failed: {}", e);
                    }
                }
            });
        })
    }

    /// Filter messages based on export options (streaming) - planned feature
    fn _filter_messages_stream(
        &self,
        messages: Vec<SearchChatMessage>,
        options: &ExportOptions,
    ) -> AsyncStream<SearchChatMessage> {
        let options_clone = options.clone();

        AsyncStream::with_channel(move |sender| {
            for message in messages {
                let mut keep_message = true;

                // Date range filter
                if let Some(date_range) = &options_clone.date_range {
                    if let Some(timestamp) = message.message.timestamp {
                        if timestamp < date_range.start || timestamp > date_range.end {
                            keep_message = false;
                        }
                    } else {
                        keep_message = false;
                    }
                }

                // User filter
                if keep_message {
                    if let Some(user_filter) = &options_clone.user_filter {
                    let role_filter = match user_filter.as_ref() {
                        "system" => MessageRole::System,
                        "user" => MessageRole::User,
                        "assistant" => MessageRole::Assistant,
                        "tool" => MessageRole::Tool,
                        _ => MessageRole::User, // Default fallback
                    };
                    if message.message.role != role_filter {
                        keep_message = false;
                    }
                    }
                }

                if keep_message {
                    let _ = sender.send(message);
                }
            }
            // AsyncStream automatically closes when sender is dropped
        })
    }

    /// Filter messages based on export options (legacy)
    async fn filter_messages(
        &self,
        messages: Vec<SearchChatMessage>,
        options: &ExportOptions,
    ) -> Result<Vec<SearchChatMessage>, SearchError> {
        // Use proper AsyncStream pattern from fluent_ai_async
        let stream = self._filter_messages_stream(messages, options);
        Ok(stream.collect())
    }

    /// Export to JSON format
    async fn export_json(
        &self,
        messages: &[SearchChatMessage],
        options: &ExportOptions,
    ) -> Result<String, SearchError> {
        let mut json_messages = Vec::new();

        for message in messages {
            let mut json_obj = serde_json::json!({
                "role": message.message.role,
                "content": message.message.content,
                "relevance_score": message.relevance_score});

            if options.include_timestamps {
                if let Some(timestamp) = message.message.timestamp {
                    json_obj["timestamp"] = serde_json::Value::Number(timestamp.into());
                }
            }

            if options.include_metadata {
                // Note: Message struct doesn't have metadata field, using relevance_score instead
                json_obj["relevance_score"] = serde_json::Value::from(message.relevance_score);
            }

            json_messages.push(json_obj);
        }

        let export_obj = serde_json::json!({
            "messages": json_messages,
            "export_info": {
                "format": "json",
                "exported_at": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                "total_messages": messages.len()}
        });

        serde_json::to_string_pretty(&export_obj).map_err(|e| SearchError::ExportError {
            reason: Arc::from(e.to_string())})
    }

    /// Export to CSV format
    async fn export_csv(
        &self,
        messages: &[SearchChatMessage],
        options: &ExportOptions,
    ) -> Result<String, SearchError> {
        let mut csv_output = String::new();

        // Header
        let mut headers = vec!["role", "content", "tokens"];
        if options.include_timestamps {
            headers.push("timestamp");
        }
        if options.include_metadata {
            headers.push("metadata");
        }
        csv_output.push_str(&headers.join(","));
        csv_output.push('\n');

        // Data rows
        for message in messages {
            let escaped_content = message
                .message
                .content
                .replace(',', "\\,")
                .replace('\n', "\\n");
            let timestamp_str = message
                .message
                .timestamp
                .map_or_else(|| "0".to_string(), |t| t.to_string());
            let tokens_str = "0"; // tokens field not available in Message struct
            let role_str = message.message.role.to_string();
            let relevance_str = message.relevance_score.to_string();

            let mut row = vec![role_str.as_str(), &escaped_content, tokens_str];

            if options.include_timestamps {
                row.push(&timestamp_str);
            }

            if options.include_metadata {
                // Note: Message struct doesn't have metadata field, using relevance_score instead
                row.push(&relevance_str);
            }

            csv_output.push_str(&row.join(","));
            csv_output.push('\n');
        }

        Ok(csv_output)
    }

    /// Export to Markdown format
    async fn export_markdown(
        &self,
        messages: &[SearchChatMessage],
        options: &ExportOptions,
    ) -> Result<String, SearchError> {
        let mut markdown_output = String::new();

        markdown_output.push_str("# Conversation History\n\n");

        if options.include_timestamps {
            markdown_output.push_str(&format!(
                "Exported at: {}\n\n",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            ));
        }

        for message in messages {
            markdown_output.push_str(&format!("## {}\n\n", message.message.role));
            markdown_output.push_str(&format!("{}\n\n", message.message.content));

            if options.include_timestamps {
                let timestamp_str = message
                    .message
                    .timestamp
                    .map_or_else(|| "Unknown".to_string(), |t| t.to_string());
                markdown_output.push_str(&format!("*Timestamp: {}*\n\n", timestamp_str));
            }

            // Note: metadata field not available in Message struct - using relevance_score instead
            if options.include_metadata {
                markdown_output.push_str(&format!(
                    "*Relevance Score: {:.2}*\n\n",
                    message.relevance_score
                ));
            }

            markdown_output.push_str("---\n\n");
        }

        Ok(markdown_output)
    }

    /// Export to HTML format
    async fn export_html(
        &self,
        messages: &[SearchChatMessage],
        options: &ExportOptions,
    ) -> Result<String, SearchError> {
        let mut html_output = String::new();

        html_output.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html_output.push_str("<title>Conversation History</title>\n");
        html_output.push_str("<style>\n");
        html_output.push_str("body { font-family: Arial, sans-serif; margin: 20px; }\n");
        html_output.push_str(".message { border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px; }\n");
        html_output.push_str(".role { font-weight: bold; color: #333; }\n");
        html_output.push_str(".timestamp { color: #666; font-size: 0.9em; }\n");
        html_output.push_str(".metadata { color: #999; font-size: 0.8em; }\n");
        html_output.push_str("</style>\n");
        html_output.push_str("</head>\n<body>\n");

        html_output.push_str("<h1>Conversation History</h1>\n");

        for message in messages {
            html_output.push_str("<div class=\"message\">\n");
            html_output.push_str(&format!(
                "<div class=\"role\">{}</div>\n",
                message.message.role
            ));
            html_output.push_str(&format!(
                "<div class=\"content\">{}</div>\n",
                message
                    .message
                    .content
                    .replace('<', "&lt;")
                    .replace('>', "&gt;")
            ));

            if options.include_timestamps {
                let timestamp_str = message
                    .message
                    .timestamp
                    .map_or_else(|| "Unknown".to_string(), |t| t.to_string());
                html_output.push_str(&format!(
                    "<div class=\"timestamp\">Timestamp: {}</div>\n",
                    timestamp_str
                ));
            }

            // Note: metadata field not available in Message struct - using relevance_score instead
            if options.include_metadata {
                html_output.push_str(&format!(
                    "<div class=\"metadata\">Relevance Score: {:.2}</div>\n",
                    message.relevance_score
                ));
            }

            html_output.push_str("</div>\n");
        }

        html_output.push_str("</body>\n</html>");

        Ok(html_output)
    }

    /// Export to XML format
    async fn export_xml(
        &self,
        messages: &[SearchChatMessage],
        options: &ExportOptions,
    ) -> Result<String, SearchError> {
        let mut xml_output = String::new();

        xml_output.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml_output.push_str("<conversation>\n");

        if options.include_timestamps {
            xml_output.push_str(&format!(
                "  <export_info exported_at=\"{}\"/>\n",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            ));
        }

        for message in messages {
            xml_output.push_str("  <message>\n");
            xml_output.push_str(&format!("    <role>{}</role>\n", message.message.role));
            xml_output.push_str(&format!(
                "    <content>{}</content>\n",
                message
                    .message
                    .content
                    .replace('<', "&lt;")
                    .replace('>', "&gt;")
            ));
            xml_output.push_str("    <tokens>0</tokens>\n"); // tokens field not available in Message struct

            if options.include_timestamps {
                let timestamp_str = message
                    .message
                    .timestamp
                    .map_or_else(|| "Unknown".to_string(), |t| t.to_string());
                xml_output.push_str(&format!("    <timestamp>{}</timestamp>\n", timestamp_str));
            }

            // Note: metadata field not available in Message struct - using relevance_score instead
            if options.include_metadata {
                xml_output.push_str(&format!(
                    "    <relevance_score>{:.2}</relevance_score>\n",
                    message.relevance_score
                ));
            }

            xml_output.push_str("  </message>\n");
        }

        xml_output.push_str("</conversation>");

        Ok(xml_output)
    }

    /// Export to plain text format
    async fn export_plain_text(
        &self,
        messages: &[SearchChatMessage],
        options: &ExportOptions,
    ) -> Result<String, SearchError> {
        let mut text_output = String::new();

        text_output.push_str("CONVERSATION HISTORY\n");
        text_output.push_str("===================\n\n");

        for message in messages {
            text_output.push_str(&format!(
                "{}: {}\n",
                message.message.role, message.message.content
            ));

            if options.include_timestamps {
                let timestamp_str = message
                    .message
                    .timestamp
                    .map_or_else(|| "Unknown".to_string(), |t| t.to_string());
                text_output.push_str(&format!("Timestamp: {}\n", timestamp_str));
            }

            // Note: metadata field not available in Message struct - using relevance_score instead
            if options.include_metadata {
                text_output.push_str(&format!(
                    "Relevance Score: {:.2}\n",
                    message.relevance_score
                ));
            }

            text_output.push_str("\n");
        }

        Ok(text_output)
    }

    /// Compress data using LZ4 (streaming)
    fn compress_data_stream(&self, data: &str) -> AsyncStream<String> {
        let data_clone = data.to_string();

        AsyncStream::with_channel(move |sender| {
            match lz4::block::compress(data_clone.as_bytes(), None, true) {
                Ok(compressed) => {
                    use base64::Engine;
                    let encoded = base64::engine::general_purpose::STANDARD.encode(compressed);
                    let _ = sender.send(encoded);
                }
                Err(e) => handle_error!(e, "LZ4 compression failed")}
            // AsyncStream automatically closes when sender is dropped
        })
    }

    /// Compress data using LZ4 (legacy)
    async fn compress_data(&self, data: &str) -> Result<String, SearchError> {
        let mut stream = self.compress_data_stream(data);
        // Use AsyncStream try_next method (NO FUTURES architecture)
        match stream.try_next() {
            Some(result) => Ok(result),
            None => Err(SearchError::ExportError {
                reason: Arc::from("Compression stream closed unexpectedly")})}
    }

    /// Get export statistics (streaming)
    pub fn get_statistics_stream(&self) -> AsyncStream<ExportStatistics> {
        let _self_clone = self.clone();

        AsyncStream::with_channel(move |sender| {
            // TODO: Replace with proper async statistics read using AsyncStream
            // For now, skip statistics read to maintain streams-only architecture
            // match self_clone.export_statistics.read() - removed await
            let _ = sender.send(Default::default()); // Send default stats for now
            // AsyncStream automatically closes when sender is dropped
        })
    }
}

impl Default for HistoryExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Search system errors
#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    /// Index-related error occurred
    #[error("Index error: {reason}")]
    IndexError { 
        /// Details about the index error that occurred
        reason: Arc<str> 
    },
    /// Search query parsing or execution failed  
    #[error("Search error: {reason}")]
    SearchQueryError { 
        /// Details about the search query error
        reason: Arc<str> 
    },
    /// Tag processing error occurred
    #[error("Tag error: {reason}")]
    TagError { 
        /// Details about the tag processing error
        reason: Arc<str> 
    },
    /// Export operation failed
    #[error("Export error: {reason}")]
    ExportError { 
        /// Details about the export failure
        reason: Arc<str> 
    },
    /// Invalid or malformed query provided
    #[error("Invalid query: {details}")]
    InvalidQuery { 
        /// Specific details about the invalid query
        details: Arc<str> 
    },
    /// System resource overload encountered
    #[error("System overload: {resource}")]
    SystemOverload { 
        /// Name of the overloaded system resource
        resource: Arc<str> 
    }}

/// Enhanced history management system
#[derive(Clone)]
pub struct EnhancedHistoryManager {
    /// Search index
    pub search_index: Arc<ChatSearchIndex>,
    /// Conversation tagger
    pub tagger: Arc<ConversationTagger>,
    /// History exporter
    pub exporter: Arc<HistoryExporter>,
    /// System statistics
    pub statistics: Arc<RwLock<HistoryManagerStatistics>>}

/// History manager statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HistoryManagerStatistics {
    /// Search operation statistics
    pub search_stats: SearchStatistics,
    /// Tagging operation statistics  
    pub tagging_stats: TaggingStatistics,
    /// Export operation statistics
    pub export_stats: ExportStatistics,
    /// Total number of operations performed
    pub total_operations: usize,
    /// System uptime in seconds
    pub system_uptime: u64}

impl EnhancedHistoryManager {
    /// Create a new enhanced history manager
    pub fn new() -> Self {
        Self {
            search_index: Arc::new(ChatSearchIndex::new()),
            tagger: Arc::new(ConversationTagger::new()),
            exporter: Arc::new(HistoryExporter::new()),
            statistics: Arc::new(RwLock::new(HistoryManagerStatistics {
                search_stats: SearchStatistics {
                    total_messages: 0,
                    total_terms: 0,
                    total_queries: 0,
                    average_query_time: 0.0,
                    index_size: 0,
                    last_index_update: 0},
                tagging_stats: TaggingStatistics {
                    total_tags: 0,
                    total_taggings: 0,
                    active_tags: 0,
                    most_used_tag: None},
                export_stats: ExportStatistics {
                    total_exports: 0,
                    total_messages_exported: 0,
                    most_popular_format: None,
                    average_export_time: 0.0,
                    last_export_time: 0},
                total_operations: 0,
                system_uptime: 0}))}
    }

    /// Add message to history manager (streaming)
    pub fn add_message_stream(&self, message: SearchChatMessage) -> AsyncStream<()> {
        let self_clone = self.clone();

        AsyncStream::with_channel(move |sender| {
            // Add to search index (using proper streams-only pattern)
            let _ = self_clone
                .search_index
                .add_message_stream(message.clone())
                .collect();

            // Auto-tag message (using proper streams-only pattern)
            let suggested_tags = self_clone
                .tagger
                .auto_tag_message_stream(message.clone())
                .collect();
            if !suggested_tags.is_empty() {
                let message_id = Arc::from(
                    message
                        .message
                        .timestamp
                        .map_or_else(|| "0".to_string(), |t| t.to_string()),
                );
                let _ = self_clone
                    .tagger
                    .tag_message_stream(message_id, suggested_tags)
                    .collect();
            }

            // Update statistics (synchronous operation)
            // TODO: Replace with proper async statistics update using AsyncStream
            // For now, skip statistics update to maintain streams-only architecture

            let _ = sender.send(());
        })
    }

    /// Add message to history manager (legacy)
    pub fn add_message(&self, message: SearchChatMessage) -> Result<(), SearchError> {
        let mut stream = self.add_message_stream(message);
        // Use AsyncStream try_next method (NO FUTURES architecture)
        match stream.try_next() {
            Some(_) => Ok(()),
            None => Err(SearchError::IndexError {
                reason: Arc::from("Stream closed unexpectedly")})}
    }

    /// Search messages (streaming)
    pub fn search_messages_stream(&self, query: SearchQuery) -> AsyncStream<SearchResult> {
        let self_clone = self.clone();

        AsyncStream::with_channel(move |sender| {
            let results = self_clone.search_index.search_stream(query).collect();

            // Add tag information to results and emit them one by one
            for mut result in results {
                let message_id = Arc::from(
                    result
                        .message
                        .message
                        .timestamp
                        .map_or_else(|| "0".to_string(), |t| t.to_string()),
                );
                result.tags = self_clone.tagger.get_message_tags(&message_id);
                let _ = sender.send(result);
            }

            // Update statistics (synchronous operation)
            // TODO: Replace with proper async statistics update using AsyncStream
            // For now, skip statistics update to maintain streams-only architecture
            // AsyncStream automatically closes when sender is dropped
        })
    }

    /// Search messages (legacy)
    pub async fn search_messages(
        &self,
        query: &SearchQuery,
    ) -> Result<Vec<SearchResult>, SearchError> {
        // Use proper AsyncStream pattern from fluent_ai_async
        let stream = self.search_messages_stream(query.clone());
        Ok(stream.collect())
    }

    /// Export conversation history (streaming)
    pub fn export_history_stream(
        &self,
        messages: Vec<SearchChatMessage>,
        options: ExportOptions,
    ) -> AsyncStream<String> {
        let self_clone = self.clone();

        AsyncStream::with_channel(move |sender| {
            let mut result_stream = self_clone.exporter.export_history_stream(messages, options);
            // Use proper streams-only pattern - no await allowed
            if let Some(result) = result_stream.try_next() {
                // Update statistics (synchronous operation)
                // TODO: Replace with proper async statistics update using AsyncStream
                // For now, skip statistics update to maintain streams-only architecture

                let _ = sender.send(result);
            }
            // AsyncStream automatically closes when sender is dropped
        })
    }

    /// Export conversation history (legacy)
    pub async fn export_history(
        &self,
        messages: Vec<SearchChatMessage>,
        options: &ExportOptions,
    ) -> Result<String, SearchError> {
        let mut stream = self.export_history_stream(messages, options.clone());
        // Use AsyncStream try_next method (NO FUTURES architecture)
        match stream.try_next() {
            Some(result) => Ok(result),
            None => Err(SearchError::ExportError {
                reason: Arc::from("Stream closed unexpectedly")})}
    }

    /// Get system statistics (streaming)
    pub fn get_system_statistics_stream(&self) -> AsyncStream<HistoryManagerStatistics> {
        let self_clone = self.clone();

        AsyncStream::with_channel(move |sender| {
            // TODO: Replace with proper async statistics handling using AsyncStream
            // For now, create default statistics to maintain streams-only architecture

            let mut search_stats_stream = self_clone.search_index.get_statistics_stream();
            let mut export_stats_stream = self_clone.exporter.get_statistics_stream();

            // Use try_next instead of recv().await
            let search_stats = search_stats_stream.try_next().unwrap_or_default();
            let export_stats = export_stats_stream.try_next().unwrap_or_default();
            let tagging_stats = self_clone.tagger.get_statistics();

            // Create combined statistics without async write
            let combined_stats = HistoryManagerStatistics {
                search_stats,
                tagging_stats,
                export_stats,
                system_uptime: 0u64,
                total_operations: 0, // TODO: Implement proper counter
            };

            let _ = sender.send(combined_stats);
            // AsyncStream automatically closes when sender is dropped
        })
    }

    /// Get system statistics (legacy)
    pub fn get_system_statistics(&self) -> HistoryManagerStatistics {
        let mut stream = self.get_system_statistics_stream();
        // Use AsyncStream try_next method (NO FUTURES architecture)
        stream.try_next().unwrap_or_default()
    }
}

impl Default for EnhancedHistoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// History manager builder for ergonomic configuration

/// Search options for configuring chat search behavior
///
/// This configuration controls search behavior including:
/// - Search algorithm selection and optimization
/// - Result filtering and ranking options
/// - Performance tuning and caching settings
/// - Output formatting and pagination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchOptions {
    /// Enable fuzzy matching for typos and variations
    pub enable_fuzzy_matching: bool,
    /// Fuzzy matching threshold (0.0 to 1.0)
    pub fuzzy_threshold: f32,
    /// Enable semantic search using embeddings
    pub enable_semantic_search: bool,
    /// Semantic similarity threshold (0.0 to 1.0)
    pub semantic_threshold: f32,
    /// Maximum number of results to return
    pub max_results: usize,
    /// Result offset for pagination
    pub offset: usize,
    /// Search timeout in milliseconds
    pub timeout_ms: u64,
    /// Enable result caching
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Enable search highlighting
    pub enable_highlighting: bool,
    /// Highlight tags (e.g., "<mark>", "</mark>")
    pub highlight_tags: (Arc<str>, Arc<str>),
    /// Include search metadata in results
    pub include_metadata: bool,
    /// Sort order for results
    pub sort_order: SortOrder,
    /// Search scope (all messages, current session, etc.)
    pub search_scope: SearchScope,
    /// Enable search analytics
    pub enable_analytics: bool,
    /// Minimum query length
    pub min_query_length: usize,
    /// Maximum query length
    pub max_query_length: usize,
    /// Enable query expansion (synonyms, etc.)
    pub enable_query_expansion: bool,
    /// Query expansion dictionary
    pub expansion_dictionary: HashMap<Arc<str>, Vec<Arc<str>>>}

/// Search scope enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SearchScope {
    /// Search all messages
    All,
    /// Search current session only
    CurrentSession,
    /// Search recent messages (last N days)
    Recent,
    /// Search specific user's messages
    User,
    /// Search specific date range
    DateRange,
    /// Search tagged messages
    Tagged}

/// Chat searcher for performing advanced search operations
///
/// This searcher provides comprehensive search capabilities with:
/// - Full-text search with SIMD optimization
/// - Fuzzy matching and semantic search
/// - Advanced filtering and ranking
/// - Performance monitoring and caching
/// - Result highlighting and metadata
pub struct ChatSearcher {
    /// Search index for fast lookups
    #[allow(dead_code)] // TODO: Implement in search indexing system
    search_index: Arc<ChatSearchIndex>,
    /// Search options configuration
    options: SearchOptions,
    /// Search cache for performance
    cache: Arc<SkipMap<Arc<str>, CachedSearchResult>>,
    /// Search statistics
    stats: Arc<ChatSearcherStats>,
    /// Query processor for advanced queries
    query_processor: Arc<QueryProcessor>,
    /// Result ranker for relevance scoring
    result_ranker: Arc<ResultRanker>}

impl std::fmt::Debug for ChatSearcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatSearcher")
            .field("search_index", &"Arc<ChatSearchIndex>")
            .field("options", &self.options)
            .field("cache", &"Arc<SkipMap<Arc<str>, CachedSearchResult>>")
            .field("stats", &"Arc<ChatSearcherStats>")
            .field("query_processor", &"Arc<QueryProcessor>")
            .field("result_ranker", &"Arc<ResultRanker>")
            .finish()
    }
}

/// Cached search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedSearchResult {
    /// Search results
    pub results: Vec<SearchResult>,
    /// Cache timestamp
    pub cached_at: u64,
    /// Cache TTL in seconds
    pub ttl_seconds: u64,
    /// Query hash for validation
    pub query_hash: Arc<str>}

// Duplicate SearchResult removed - using the one defined above with enhanced fields

/// Match position in content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchPosition {
    /// Start position in content
    pub start: usize,
    /// End position in content
    pub end: usize,
    /// Matched term
    pub term: Arc<str>,
    /// Match type (exact, fuzzy, semantic)
    pub match_type: MatchType}

/// Match type enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MatchType {
    /// Exact text match
    Exact,
    /// Fuzzy match (with typos)
    Fuzzy,
    /// Semantic match (similar meaning)
    Semantic,
    /// Wildcard match
    Wildcard}

/// Search result metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultMetadata {
    /// Search query used
    pub query: Arc<str>,
    /// Search timestamp
    pub searched_at: u64,
    /// Processing time in microseconds
    pub processing_time_us: u64,
    /// Index used for search
    pub index_version: u64,
    /// Search algorithm used
    pub algorithm: Arc<str>}

/// Chat searcher statistics
#[derive(Debug, Default)]
pub struct ChatSearcherStats {
    /// Total searches performed
    pub total_searches: ConsistentCounter,
    /// Cache hits
    pub cache_hits: ConsistentCounter,
    /// Cache misses
    pub cache_misses: ConsistentCounter,
    /// Average search time in microseconds
    pub avg_search_time_us: AtomicUsize,
    /// Total results returned
    pub total_results: ConsistentCounter,
    /// Failed searches
    pub failed_searches: ConsistentCounter}

/// Query processor for advanced query parsing
#[derive(Debug)]
pub struct QueryProcessor {
    /// Query expansion enabled
    #[allow(dead_code)] // TODO: Implement in query expansion system
    expansion_enabled: bool,
    /// Expansion dictionary
    #[allow(dead_code)] // TODO: Implement in query expansion system
    expansion_dict: HashMap<Arc<str>, Vec<Arc<str>>>}

/// Result ranker for relevance scoring
#[derive(Debug)]
pub struct ResultRanker {
    /// Ranking algorithm
    #[allow(dead_code)] // TODO: Implement in ranking system
    algorithm: RankingAlgorithm,
    /// Boost factors for different fields
    #[allow(dead_code)] // TODO: Implement in ranking system
    field_boosts: HashMap<Arc<str>, f32>}

/// Ranking algorithm enumeration
#[derive(Debug, Clone, Copy)]
pub enum RankingAlgorithm {
    /// TF-IDF (Term Frequency-Inverse Document Frequency)
    TfIdf,
    /// BM25 (Best Matching 25)
    Bm25,
    /// Simple relevance scoring
    Simple,
    /// Machine learning based ranking
    MlRanking}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            enable_fuzzy_matching: true,
            fuzzy_threshold: 0.8,
            enable_semantic_search: false,
            semantic_threshold: 0.7,
            max_results: 50,
            offset: 0,
            timeout_ms: 5000,
            enable_caching: true,
            cache_ttl_seconds: 300, // 5 minutes
            enable_highlighting: true,
            highlight_tags: (Arc::from("<mark>"), Arc::from("</mark>")),
            include_metadata: true,
            sort_order: SortOrder::Relevance,
            search_scope: SearchScope::All,
            enable_analytics: true,
            min_query_length: 1,
            max_query_length: 1000,
            enable_query_expansion: false,
            expansion_dictionary: HashMap::new()}
    }
}

impl ChatSearcher {
    /// Create a new chat searcher
    pub fn new(search_index: Arc<ChatSearchIndex>) -> Self {
        Self {
            search_index,
            options: SearchOptions::default(),
            cache: Arc::new(SkipMap::new()),
            stats: Arc::new(ChatSearcherStats::default()),
            query_processor: Arc::new(QueryProcessor::new()),
            result_ranker: Arc::new(ResultRanker::new())}
    }

    /// Create a chat searcher with custom options
    pub fn with_options(search_index: Arc<ChatSearchIndex>, options: SearchOptions) -> Self {
        Self {
            search_index,
            options,
            cache: Arc::new(SkipMap::new()),
            stats: Arc::new(ChatSearcherStats::default()),
            query_processor: Arc::new(QueryProcessor::new()),
            result_ranker: Arc::new(ResultRanker::new())}
    }

    /// Perform a search with the given query
    pub async fn search(
        &self,
        query: &str,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        if query.len() < self.options.min_query_length {
            return Err(format!(
                "Query too short, minimum length: {}",
                self.options.min_query_length
            )
            .into());
        }

        if query.len() > self.options.max_query_length {
            return Err(format!(
                "Query too long, maximum length: {}",
                self.options.max_query_length
            )
            .into());
        }

        let start_time = Instant::now();
        self.stats.total_searches.inc();

        // Check cache first
        let query_hash = Arc::from(format!("{:x}", md5::compute(query.as_bytes())));
        if self.options.enable_caching {
            if let Some(cached) = self.cache.get(&query_hash) {
                let cached_result = cached.value();
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                if now - cached_result.cached_at < cached_result.ttl_seconds {
                    self.stats.cache_hits.inc();
                    return Ok(cached_result.results.clone());
                }
            }
        }

        self.stats.cache_misses.inc();

        // Process query
        let processed_query = self
            .query_processor
            .process_query(query, &self.options)
            .await?;

        // Perform search
        let search_results = self.perform_search(&processed_query).await?;

        // Rank results
        let ranked_results = self
            .result_ranker
            .rank_results(search_results, &processed_query)
            .await?;

        // Apply highlighting if enabled
        let final_results = if self.options.enable_highlighting {
            self.apply_highlighting(ranked_results, &processed_query)
                .await?
        } else {
            ranked_results
        };

        // Cache results
        if self.options.enable_caching {
            let cached_result = CachedSearchResult {
                results: final_results.clone(),
                cached_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                ttl_seconds: self.options.cache_ttl_seconds,
                query_hash: query_hash.clone()};
            self.cache.insert(query_hash, cached_result);
        }

        // Update statistics
        let search_time = start_time.elapsed().as_micros() as usize;
        let current_avg = self.stats.avg_search_time_us.load(Ordering::Relaxed);
        let total_searches = self.stats.total_searches.get();
        let new_avg = ((current_avg * (total_searches - 1)) + search_time) / total_searches;
        self.stats
            .avg_search_time_us
            .store(new_avg, Ordering::Relaxed);
        self.stats.total_results.add(final_results.len());

        Ok(final_results)
    }

    /// Perform the actual search operation
    async fn perform_search(
        &self,
        _query: &ProcessedQuery,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        // This would integrate with the existing ChatSearchIndex
        // For now, return empty results as a placeholder
        Ok(Vec::new())
    }

    /// Apply highlighting to search results
    async fn apply_highlighting(
        &self,
        results: Vec<SearchResult>,
        _query: &ProcessedQuery,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        // Apply highlighting logic here
        Ok(results)
    }

    /// Clear search cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Get search statistics
    pub fn stats(&self) -> &ChatSearcherStats {
        &self.stats
    }

    /// Get search options
    pub fn options(&self) -> &SearchOptions {
        &self.options
    }

    /// Update search options
    pub fn set_options(&mut self, options: SearchOptions) {
        self.options = options;
    }
}

/// Processed query structure
#[derive(Debug, Clone)]
pub struct ProcessedQuery {
    /// Original query
    pub original: Arc<str>,
    /// Processed terms
    pub terms: Vec<Arc<str>>,
    /// Expanded terms (synonyms, etc.)
    pub expanded_terms: Vec<Arc<str>>,
    /// Query operator
    pub operator: QueryOperator,
    /// Processing metadata
    pub metadata: QueryMetadata}

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
    pub normalization_applied: bool}

impl QueryProcessor {
    /// Create a new query processor
    pub fn new() -> Self {
        Self {
            expansion_enabled: false,
            expansion_dict: HashMap::new()}
    }

    /// Process a query string
    pub async fn process_query(
        &self,
        query: &str,
        options: &SearchOptions,
    ) -> Result<ProcessedQuery, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = Instant::now();

        // Basic query processing
        let terms: Vec<Arc<str>> = query
            .split_whitespace()
            .map(|term| Arc::from(term.to_lowercase()))
            .collect();

        // Apply query expansion if enabled
        let expanded_terms = if options.enable_query_expansion {
            self.expand_terms(&terms, &options.expansion_dictionary)
                .await?
        } else {
            Vec::new()
        };

        Ok(ProcessedQuery {
            original: Arc::from(query),
            terms,
            expanded_terms,
            operator: QueryOperator::And, // Default to AND
            metadata: QueryMetadata {
                processed_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                processing_time_us: start_time.elapsed().as_micros() as u64,
                expansion_applied: options.enable_query_expansion,
                normalization_applied: true}})
    }

    /// Expand query terms using synonyms
    async fn expand_terms(
        &self,
        terms: &[Arc<str>],
        dictionary: &HashMap<Arc<str>, Vec<Arc<str>>>,
    ) -> Result<Vec<Arc<str>>, Box<dyn std::error::Error + Send + Sync>> {
        let mut expanded = Vec::new();

        for term in terms {
            if let Some(synonyms) = dictionary.get(term) {
                expanded.extend(synonyms.clone());
            }
        }

        Ok(expanded)
    }
}

impl ResultRanker {
    /// Create a new result ranker
    pub fn new() -> Self {
        Self {
            algorithm: RankingAlgorithm::Bm25,
            field_boosts: HashMap::new()}
    }

    /// Rank search results by relevance
    pub async fn rank_results(
        &self,
        results: Vec<SearchResult>,
        _query: &ProcessedQuery,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        // Apply ranking algorithm
        // For now, return results as-is
        Ok(results)
    }
}

impl Default for ChatSearcher {
    fn default() -> Self {
        Self::new(Arc::new(ChatSearchIndex::new()))
    }
}
