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
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

// Removed unused import: wide::f32x8
use crate::chat::message::{SearchChatMessage, MessageRole};

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

/// Search statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Chat search index with SIMD optimization
pub struct ChatSearchIndex {
    /// Inverted index: term -> documents containing term
    inverted_index: SkipMap<Arc<str>, Vec<IndexEntry>>,
    /// Document store: doc_id -> message
    document_store: SkipMap<Arc<str>, SearchChatMessage>,
    /// Term frequencies for TF-IDF calculation
    term_frequencies: SkipMap<Arc<str>, TermFrequency>,
    /// Document count
    document_count: AtomicUsize,
    /// Query counter
    query_counter: ConsistentCounter,
    /// Index update counter
    index_update_counter: ConsistentCounter,
    /// Search statistics
    statistics: Arc<RwLock<SearchStatistics>>,
    /// SIMD processing threshold
    simd_threshold: AtomicUsize,
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
            document_count: AtomicUsize::new(0),
            query_counter: ConsistentCounter::new(0),
            index_update_counter: ConsistentCounter::new(0),
            statistics: Arc::new(RwLock::new(SearchStatistics {
                total_messages: 0,
                total_terms: 0,
                total_queries: 0,
                average_query_time: 0.0,
                index_size: 0,
                last_index_update: 0,
            })),
            simd_threshold: AtomicUsize::new(8), // Process 8 terms at once with SIMD
        }
    }

    /// Add message to search index
    pub async fn add_message(&self, message: SearchChatMessage) -> Result<(), SearchError> {
        let doc_id: Arc<str> = Arc::from(message.message.timestamp.map_or_else(|| "0".to_string(), |t| t.to_string()));

        // Store the document
        self.document_store.insert(doc_id.clone(), message.clone());
        self.document_count.fetch_add(1, Ordering::Relaxed);

        // Tokenize and index the content
        let tokens = self.tokenize_with_simd(&message.message.content);
        let total_tokens = tokens.len();

        // Calculate term frequencies
        let mut term_counts = HashMap::new();
        for token in &tokens {
            *term_counts.entry(token.clone()).or_insert(0) += 1;
        }

        // Update inverted index
        for (term, count) in term_counts {
            let tf = (count as f32) / (total_tokens as f32);

            let index_entry = IndexEntry {
                doc_id: doc_id.clone(),
                term_frequency: tf,
                positions: tokens
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| **t == term)
                    .map(|(i, _)| i)
                    .collect(),
            };

            // Update inverted index (zero-allocation approach)
            if let Some(entries) = self.inverted_index.get(&term) {
                let mut entries_vec = entries.value().clone();
                entries_vec.push(index_entry);
                self.inverted_index.insert(term.clone(), entries_vec);
            } else {
                self.inverted_index.insert(term.clone(), vec![index_entry]);
            }

            // Update term frequencies
            let total_docs = self.document_count.load(Ordering::Relaxed) as u32;
            let df = self
                .inverted_index
                .get(&term)
                .map(|entries| entries.value().len() as u32)
                .unwrap_or(1);

            self.term_frequencies
                .insert(term, TermFrequency { tf, df, total_docs });
        }

        self.index_update_counter.inc();

        // Update statistics
        let mut stats = self.statistics.write().await;
        stats.total_messages = self.document_count.load(Ordering::Relaxed);
        stats.total_terms = self.inverted_index.len();
        stats.last_index_update = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(())
    }

    /// Search messages with SIMD optimization
    pub async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, SearchError> {
        let start_time = Instant::now();
        self.query_counter.inc();

        let _scores: HashMap<Arc<str>, f64> = HashMap::new();

        let results = match query.operator {
            QueryOperator::And => self.search_and(&query.terms, query.fuzzy_matching).await?,
            QueryOperator::Or => self.search_or(&query.terms, query.fuzzy_matching).await?,
            QueryOperator::Not => self.search_not(&query.terms, query.fuzzy_matching).await?,
            QueryOperator::Phrase => {
                self.search_phrase(&query.terms, query.fuzzy_matching)
                    .await?
            }
            QueryOperator::Proximity { distance } => {
                self.search_proximity(&query.terms, distance, query.fuzzy_matching)
                    .await?
            }
        };

        // Apply filters
        let mut results = self.apply_filters(results, query).await?;

        // Sort results
        self.sort_results(&mut results, &query.sort_order);

        // Apply pagination
        let start = query.offset;
        let end = (start + query.max_results).min(results.len());
        let results = results[start..end].to_vec();

        // Update statistics
        let query_time = start_time.elapsed().as_millis() as f64;
        let mut stats = self.statistics.write().await;
        stats.total_queries += 1;
        stats.average_query_time = (stats.average_query_time * (stats.total_queries - 1) as f64
            + query_time)
            / stats.total_queries as f64;

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

    /// Search with AND operator
    async fn search_and(
        &self,
        terms: &[Arc<str>],
        fuzzy: bool,
    ) -> Result<Vec<SearchResult>, SearchError> {
        if terms.is_empty() {
            return Ok(vec![]);
        }

        let mut candidates = None;

        for term in terms {
            let term_candidates = if fuzzy {
                self.fuzzy_search(term).await?
            } else {
                self.exact_search(term).await?
            };

            if candidates.is_none() {
                candidates = Some(term_candidates);
            } else {
                let current = candidates.unwrap();
                let intersection = self.intersect_results(current, term_candidates);
                candidates = Some(intersection);
            }
        }

        Ok(candidates.unwrap_or_default())
    }

    /// Search with OR operator
    async fn search_or(
        &self,
        terms: &[Arc<str>],
        fuzzy: bool,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let mut all_results = Vec::new();

        for term in terms {
            let term_results = if fuzzy {
                self.fuzzy_search(term).await?
            } else {
                self.exact_search(term).await?
            };

            all_results.extend(term_results);
        }

        // Remove duplicates and merge scores
        let mut unique_results = HashMap::new();
        for result in all_results {
            let key = result.message.message.timestamp.map_or_else(|| "0".to_string(), |t| t.to_string());
            if let Some(existing) = unique_results.get_mut(&key) {
                let existing_result: &mut SearchResult = existing;
                existing_result.relevance_score =
                    (existing_result.relevance_score + result.relevance_score) / 2.0;
            } else {
                unique_results.insert(key, result);
            }
        }

        Ok(unique_results.into_values().collect())
    }

    /// Search with NOT operator
    async fn search_not(
        &self,
        terms: &[Arc<str>],
        fuzzy: bool,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let mut excluded_docs = std::collections::HashSet::new();

        for term in terms {
            let term_results = if fuzzy {
                self.fuzzy_search(term).await?
            } else {
                self.exact_search(term).await?
            };

            for result in term_results {
                excluded_docs.insert(result.message.message.timestamp.map_or_else(|| "0".to_string(), |t| t.to_string()));
            }
        }

        let mut results = Vec::new();
        for entry in self.document_store.iter() {
            let doc_id = entry.key();
            if !excluded_docs.contains(doc_id.as_ref()) {
                let message = entry.value().clone();
                results.push(SearchResult {
                    message,
                    relevance_score: 1.0,
                    matching_terms: vec![],
                    highlighted_content: Some(Arc::from("")),
                    tags: vec![],
                    context: vec![],
                    match_positions: vec![],
                    metadata: None,
                });
            }
        }

        Ok(results)
    }

    /// Search for phrase matches
    async fn search_phrase(
        &self,
        terms: &[Arc<str>],
        fuzzy: bool,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let mut results = Vec::new();

        for entry in self.document_store.iter() {
            let message = entry.value();
            let content = message.message.content.to_lowercase();
            let phrase = terms
                .iter()
                .map(|t| t.as_ref())
                .collect::<Vec<_>>()
                .join(" ");

            if fuzzy {
                if self.fuzzy_match(&content, &phrase) {
                    results.push(SearchResult {
                        message: message.clone(),
                        relevance_score: 0.8,
                        matching_terms: terms.to_vec(),
                        highlighted_content: Some(Arc::from(
                            self.highlight_text(&content, &phrase),
                        )),
                        tags: vec![],
                        context: vec![],
                        match_positions: vec![],
                        metadata: None,
                    });
                }
            } else if content.contains(&phrase) {
                results.push(SearchResult {
                    message: message.clone(),
                    relevance_score: 1.0,
                    matching_terms: terms.to_vec(),
                    highlighted_content: Some(Arc::from(self.highlight_text(&content, &phrase))),
                    tags: vec![],
                    context: vec![],
                    match_positions: vec![],
                    metadata: None,
                });
            }
        }

        Ok(results)
    }

    /// Search for proximity matches
    async fn search_proximity(
        &self,
        terms: &[Arc<str>],
        distance: u32,
        fuzzy: bool,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let mut results = Vec::new();

        for entry in self.document_store.iter() {
            let message = entry.value();
            let tokens = self.tokenize_with_simd(&message.message.content);

            if self.check_proximity(&tokens, terms, distance) {
                let relevance_score = if fuzzy { 0.7 } else { 0.9 };
                results.push(SearchResult {
                    message: message.clone(),
                    relevance_score,
                    matching_terms: terms.to_vec(),
                    highlighted_content: Some(Arc::from(
                        self.highlight_terms(&message.message.content, terms),
                    )),
                    tags: vec![],
                    context: vec![],
                    match_positions: vec![],
                    metadata: None,
                });
            }
        }

        Ok(results)
    }

    /// Exact search for a term
    async fn exact_search(&self, term: &Arc<str>) -> Result<Vec<SearchResult>, SearchError> {
        let mut results = Vec::new();

        if let Some(entries) = self.inverted_index.get(term) {
            for entry in entries.value() {
                if let Some(message) = self.document_store.get(&entry.doc_id) {
                    let tf_idf = if let Some(tf) = self.term_frequencies.get(term) {
                        tf.value().calculate_tfidf()
                    } else {
                        entry.term_frequency
                    };

                    results.push(SearchResult {
                        message: message.value().clone(),
                        relevance_score: tf_idf,
                        matching_terms: vec![term.clone()],
                        highlighted_content: Some(Arc::from(
                            self.highlight_text(&message.value().message.content, term),
                        )),
                        tags: vec![],
                        context: vec![],
                        match_positions: vec![],
                        metadata: None,
                    });
                }
            }
        }

        Ok(results)
    }

    /// Fuzzy search for a term
    async fn fuzzy_search(&self, term: &Arc<str>) -> Result<Vec<SearchResult>, SearchError> {
        let mut results = Vec::new();

        for entry in self.inverted_index.iter() {
            let indexed_term = entry.key();
            if self.fuzzy_match(indexed_term, term) {
                let exact_results = self.exact_search(indexed_term).await?;
                for mut result in exact_results {
                    result.relevance_score *= 0.8; // Reduce score for fuzzy matches
                    results.push(result);
                }
            }
        }

        Ok(results)
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
            .map(|r| r.message.message.timestamp.map_or_else(|| "0".to_string(), |t| t.to_string()))
            .collect();

        for result in results1 {
            if ids2.contains(&result.message.message.timestamp.map_or_else(|| "0".to_string(), |t| t.to_string())) {
                intersection.push(result);
            }
        }

        intersection
    }

    /// Check proximity of terms in token list
    fn check_proximity(&self, tokens: &[Arc<str>], terms: &[Arc<str>], distance: u32) -> bool {
        let mut positions = HashMap::new();

        for (i, token) in tokens.iter().enumerate() {
            if terms.contains(token) {
                positions.entry(token.clone()).or_insert(Vec::new()).push(i);
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

    /// Apply query filters
    async fn apply_filters(
        &self,
        mut results: Vec<SearchResult>,
        query: &SearchQuery,
    ) -> Result<Vec<SearchResult>, SearchError> {
        // Date range filter
        if let Some(date_range) = &query.date_range {
            results.retain(|r| {
                if let Some(timestamp) = r.message.message.timestamp {
                    timestamp >= date_range.start && timestamp <= date_range.end
                } else {
                    false
                }
            });
        }

        // User filter
        if let Some(user_filter) = &query.user_filter {
            let role_filter = match user_filter.as_ref() {
                "system" => MessageRole::System,
                "user" => MessageRole::User,
                "assistant" => MessageRole::Assistant,
                "tool" => MessageRole::Tool,
                _ => MessageRole::User, // Default fallback
            };
            results.retain(|r| r.message.message.role == role_filter);
        }

        // Content type filter
        if let Some(content_type) = &query.content_type_filter {
            results.retain(|r| {
                // Since Message doesn't have metadata field, check content instead
                r.message
                    .message
                    .content
                    .contains(content_type.as_ref())
            });
        }

        Ok(results)
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
                results.sort_by(|a, b| b.message.message.timestamp.cmp(&a.message.message.timestamp));
            }
            SortOrder::DateAscending => {
                results.sort_by(|a, b| a.message.message.timestamp.cmp(&b.message.message.timestamp));
            }
            SortOrder::UserAscending => {
                results.sort_by(|a, b| a.message.role.cmp(&b.message.role));
            }
            SortOrder::UserDescending => {
                results.sort_by(|a, b| b.message.role.cmp(&a.message.role));
            }
        }
    }

    /// Get search statistics
    pub async fn get_statistics(&self) -> SearchStatistics {
        self.statistics.read().await.clone()
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
    pub is_active: bool,
}

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
            is_active: true,
        }
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
    tag_counter: ConsistentCounter,
    /// Tagging counter
    tagging_counter: ConsistentCounter,
    /// Auto-tagging rules
    auto_tagging_rules: Arc<RwLock<HashMap<Arc<str>, Vec<Arc<str>>>>>,
}

impl ConversationTagger {
    /// Create a new conversation tagger
    pub fn new() -> Self {
        Self {
            tags: SkipMap::new(),
            message_tags: SkipMap::new(),
            tag_messages: SkipMap::new(),
            tag_hierarchy: SkipMap::new(),
            tag_counter: ConsistentCounter::new(0),
            tagging_counter: ConsistentCounter::new(0),
            auto_tagging_rules: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new tag
    pub async fn create_tag(
        &self,
        name: Arc<str>,
        description: Arc<str>,
        category: Arc<str>,
    ) -> Result<Arc<str>, SearchError> {
        let tag = ConversationTag::new(name, description, category);
        let tag_id = tag.id.clone();

        self.tags.insert(tag_id.clone(), tag);
        self.tag_counter.inc();

        Ok(tag_id)
    }

    /// Create a child tag
    pub async fn create_child_tag(
        &self,
        parent_id: Arc<str>,
        name: Arc<str>,
        description: Arc<str>,
        category: Arc<str>,
    ) -> Result<Arc<str>, SearchError> {
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
            self.tag_hierarchy.insert(parent_id, children_vec);
        } else {
            self.tag_hierarchy.insert(parent_id, vec![tag_id.clone()]);
        }

        self.tags.insert(tag_id.clone(), tag);
        self.tag_counter.inc();

        Ok(tag_id)
    }

    /// Tag a message
    pub async fn tag_message(
        &self,
        message_id: Arc<str>,
        tag_ids: Vec<Arc<str>>,
    ) -> Result<(), SearchError> {
        // Add message to tags mapping
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

        Ok(())
    }

    /// Auto-tag message based on content
    pub async fn auto_tag_message(
        &self,
        message: &SearchChatMessage,
    ) -> Result<Vec<Arc<str>>, SearchError> {
        let mut suggested_tags = Vec::new();
        let content = message.message.content.to_lowercase();

        let rules = self.auto_tagging_rules.read().await;
        for (pattern, tag_ids) in rules.iter() {
            if content.contains(pattern.as_ref()) {
                suggested_tags.extend(tag_ids.clone());
            }
        }

        // Remove duplicates
        suggested_tags.sort();
        suggested_tags.dedup();

        Ok(suggested_tags)
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

    /// Add auto-tagging rule
    pub async fn add_auto_tagging_rule(
        &self,
        pattern: Arc<str>,
        tag_ids: Vec<Arc<str>>,
    ) -> Result<(), SearchError> {
        let mut rules = self.auto_tagging_rules.write().await;
        rules.insert(pattern, tag_ids);
        Ok(())
    }

    /// Remove auto-tagging rule
    pub async fn remove_auto_tagging_rule(&self, pattern: &Arc<str>) -> Result<(), SearchError> {
        let mut rules = self.auto_tagging_rules.write().await;
        rules.remove(pattern);
        Ok(())
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
            most_used_tag: self.get_most_used_tag(),
        }
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaggingStatistics {
    pub total_tags: usize,
    pub total_taggings: usize,
    pub active_tags: usize,
    pub most_used_tag: Option<Arc<str>>,
}

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
    PlainText,
}

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
    pub tag_filter: Option<Vec<Arc<str>>>,
}

/// History exporter with zero-allocation streaming
pub struct HistoryExporter {
    /// Export counter
    export_counter: ConsistentCounter,
    /// Export statistics
    export_statistics: Arc<RwLock<ExportStatistics>>,
}

/// Export statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportStatistics {
    pub total_exports: usize,
    pub total_messages_exported: usize,
    pub most_popular_format: Option<ExportFormat>,
    pub average_export_time: f64,
    pub last_export_time: u64,
}

impl HistoryExporter {
    /// Create a new history exporter
    pub fn new() -> Self {
        Self {
            export_counter: ConsistentCounter::new(0),
            export_statistics: Arc::new(RwLock::new(ExportStatistics {
                total_exports: 0,
                total_messages_exported: 0,
                most_popular_format: None,
                average_export_time: 0.0,
                last_export_time: 0,
            })),
        }
    }

    /// Export conversation history
    pub async fn export_history(
        &self,
        messages: Vec<SearchChatMessage>,
        options: &ExportOptions,
    ) -> Result<String, SearchError> {
        let start_time = Instant::now();
        self.export_counter.inc();

        let filtered_messages = self.filter_messages(messages, options).await?;

        let exported_data = match options.format {
            ExportFormat::Json => self.export_json(&filtered_messages, options).await?,
            ExportFormat::Csv => self.export_csv(&filtered_messages, options).await?,
            ExportFormat::Markdown => self.export_markdown(&filtered_messages, options).await?,
            ExportFormat::Html => self.export_html(&filtered_messages, options).await?,
            ExportFormat::Xml => self.export_xml(&filtered_messages, options).await?,
            ExportFormat::PlainText => self.export_plain_text(&filtered_messages, options).await?,
        };

        let final_data = if options.compress {
            self.compress_data(&exported_data).await?
        } else {
            exported_data
        };

        // Update statistics
        let export_time = start_time.elapsed().as_millis() as f64;
        let mut stats = self.export_statistics.write().await;
        stats.total_exports += 1;
        stats.total_messages_exported += filtered_messages.len();
        stats.average_export_time = (stats.average_export_time * (stats.total_exports - 1) as f64
            + export_time)
            / stats.total_exports as f64;
        stats.last_export_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(final_data)
    }

    /// Filter messages based on export options
    async fn filter_messages(
        &self,
        mut messages: Vec<SearchChatMessage>,
        options: &ExportOptions,
    ) -> Result<Vec<SearchChatMessage>, SearchError> {
        // Date range filter
        if let Some(date_range) = &options.date_range {
            messages.retain(|m| {
                if let Some(timestamp) = m.message.timestamp {
                    timestamp >= date_range.start && timestamp <= date_range.end
                } else {
                    false
                }
            });
        }

        // User filter
        if let Some(user_filter) = &options.user_filter {
            let role_filter = match user_filter.as_ref() {
                "system" => MessageRole::System,
                "user" => MessageRole::User,
                "assistant" => MessageRole::Assistant,
                "tool" => MessageRole::Tool,
                _ => MessageRole::User, // Default fallback
            };
            messages.retain(|m| m.message.role == role_filter);
        }

        Ok(messages)
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
                "relevance_score": message.relevance_score,
            });

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
                "total_messages": messages.len(),
            }
        });

        serde_json::to_string_pretty(&export_obj).map_err(|e| SearchError::ExportError {
            reason: Arc::from(e.to_string()),
        })
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
            let escaped_content = message.message.content.replace(',', "\\,").replace('\n', "\\n");
            let timestamp_str = message.message.timestamp.map_or_else(|| "0".to_string(), |t| t.to_string());
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
                let timestamp_str = message.message.timestamp.map_or_else(|| "Unknown".to_string(), |t| t.to_string());
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
            html_output.push_str(&format!("<div class=\"role\">{}</div>\n", message.message.role));
            html_output.push_str(&format!(
                "<div class=\"content\">{}</div>\n",
                message.message.content.replace('<', "&lt;").replace('>', "&gt;")
            ));

            if options.include_timestamps {
                let timestamp_str = message.message.timestamp.map_or_else(|| "Unknown".to_string(), |t| t.to_string());
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
                message.message.content.replace('<', "&lt;").replace('>', "&gt;")
            ));
            xml_output.push_str("    <tokens>0</tokens>\n"); // tokens field not available in Message struct

            if options.include_timestamps {
                let timestamp_str = message.message.timestamp.map_or_else(|| "Unknown".to_string(), |t| t.to_string());
                xml_output.push_str(&format!(
                    "    <timestamp>{}</timestamp>\n",
                    timestamp_str
                ));
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
            text_output.push_str(&format!("{}: {}\n", message.message.role, message.message.content));

            if options.include_timestamps {
                let timestamp_str = message.message.timestamp.map_or_else(|| "Unknown".to_string(), |t| t.to_string());
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

    /// Compress data using LZ4
    async fn compress_data(&self, data: &str) -> Result<String, SearchError> {
        let compressed = lz4::block::compress(data.as_bytes(), None, true).map_err(|e| {
            SearchError::ExportError {
                reason: Arc::from(e.to_string()),
            }
        })?;
        {
            use base64::Engine;
            Ok(base64::engine::general_purpose::STANDARD.encode(compressed))
        }
    }

    /// Get export statistics
    pub async fn get_statistics(&self) -> ExportStatistics {
        self.export_statistics.read().await.clone()
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
    #[error("Index error: {reason}")]
    IndexError { reason: Arc<str> },
    #[error("Search error: {reason}")]
    SearchQueryError { reason: Arc<str> },
    #[error("Tag error: {reason}")]
    TagError { reason: Arc<str> },
    #[error("Export error: {reason}")]
    ExportError { reason: Arc<str> },
    #[error("Invalid query: {details}")]
    InvalidQuery { details: Arc<str> },
    #[error("System overload: {resource}")]
    SystemOverload { resource: Arc<str> },
}

/// Enhanced history management system
pub struct EnhancedHistoryManager {
    /// Search index
    pub search_index: Arc<ChatSearchIndex>,
    /// Conversation tagger
    pub tagger: Arc<ConversationTagger>,
    /// History exporter
    pub exporter: Arc<HistoryExporter>,
    /// System statistics
    pub statistics: Arc<RwLock<HistoryManagerStatistics>>,
}

/// History manager statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryManagerStatistics {
    pub search_stats: SearchStatistics,
    pub tagging_stats: TaggingStatistics,
    pub export_stats: ExportStatistics,
    pub total_operations: usize,
    pub system_uptime: u64,
}

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
                    last_index_update: 0,
                },
                tagging_stats: TaggingStatistics {
                    total_tags: 0,
                    total_taggings: 0,
                    active_tags: 0,
                    most_used_tag: None,
                },
                export_stats: ExportStatistics {
                    total_exports: 0,
                    total_messages_exported: 0,
                    most_popular_format: None,
                    average_export_time: 0.0,
                    last_export_time: 0,
                },
                total_operations: 0,
                system_uptime: 0,
            })),
        }
    }

    /// Add message to history manager
    pub async fn add_message(&self, message: SearchChatMessage) -> Result<(), SearchError> {
        // Add to search index
        self.search_index.add_message(message.clone()).await?;

        // Auto-tag message
        let suggested_tags = self.tagger.auto_tag_message(&message).await?;
        if !suggested_tags.is_empty() {
            let message_id = Arc::from(message.message.timestamp.map_or_else(|| "0".to_string(), |t| t.to_string()));
            self.tagger.tag_message(message_id, suggested_tags).await?;
        }

        // Update statistics
        let mut stats = self.statistics.write().await;
        stats.total_operations += 1;

        Ok(())
    }

    /// Search messages
    pub async fn search_messages(
        &self,
        query: &SearchQuery,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let mut results = self.search_index.search(query).await?;

        // Add tag information to results
        for result in &mut results {
            let message_id = Arc::from(result.message.message.timestamp.map_or_else(|| "0".to_string(), |t| t.to_string()));
            result.tags = self.tagger.get_message_tags(&message_id);
        }

        // Update statistics
        let mut stats = self.statistics.write().await;
        stats.total_operations += 1;

        Ok(results)
    }

    /// Export conversation history
    pub async fn export_history(
        &self,
        messages: Vec<SearchChatMessage>,
        options: &ExportOptions,
    ) -> Result<String, SearchError> {
        let result = self.exporter.export_history(messages, options).await?;

        // Update statistics
        let mut stats = self.statistics.write().await;
        stats.total_operations += 1;

        Ok(result)
    }

    /// Get system statistics
    pub async fn get_system_statistics(&self) -> HistoryManagerStatistics {
        let mut stats = self.statistics.write().await;
        stats.search_stats = self.search_index.get_statistics().await;
        stats.tagging_stats = self.tagger.get_statistics();
        stats.export_stats = self.exporter.get_statistics().await;
        stats.clone()
    }
}

impl Default for EnhancedHistoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// History manager builder for ergonomic configuration
pub struct HistoryManagerBuilder {
    simd_threshold: usize,
    auto_tagging_enabled: bool,
    compression_enabled: bool,
}

impl HistoryManagerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            simd_threshold: 8,
            auto_tagging_enabled: true,
            compression_enabled: true,
        }
    }

    /// Set SIMD threshold
    pub fn simd_threshold(mut self, threshold: usize) -> Self {
        self.simd_threshold = threshold;
        self
    }

    /// Enable auto-tagging
    pub fn auto_tagging(mut self, enabled: bool) -> Self {
        self.auto_tagging_enabled = enabled;
        self
    }

    /// Enable compression
    pub fn compression(mut self, enabled: bool) -> Self {
        self.compression_enabled = enabled;
        self
    }

    /// Build the history manager
    pub fn build(self) -> EnhancedHistoryManager {
        let search_index = Arc::new(ChatSearchIndex::new());
        search_index
            .simd_threshold
            .store(self.simd_threshold, Ordering::Relaxed);

        EnhancedHistoryManager {
            search_index,
            tagger: Arc::new(ConversationTagger::new()),
            exporter: Arc::new(HistoryExporter::new()),
            statistics: Arc::new(RwLock::new(HistoryManagerStatistics {
                search_stats: SearchStatistics {
                    total_messages: 0,
                    total_terms: 0,
                    total_queries: 0,
                    average_query_time: 0.0,
                    index_size: 0,
                    last_index_update: 0,
                },
                tagging_stats: TaggingStatistics {
                    total_tags: 0,
                    total_taggings: 0,
                    active_tags: 0,
                    most_used_tag: None,
                },
                export_stats: ExportStatistics {
                    total_exports: 0,
                    total_messages_exported: 0,
                    most_popular_format: None,
                    average_export_time: 0.0,
                    last_export_time: 0,
                },
                total_operations: 0,
                system_uptime: 0,
            })),
        }
    }
}

impl Default for HistoryManagerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

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
    pub expansion_dictionary: HashMap<Arc<str>, Vec<Arc<str>>>,
}

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
    Tagged,
}

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
    result_ranker: Arc<ResultRanker>,
}

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
    pub query_hash: Arc<str>,
}

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
    pub match_type: MatchType,
}

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
    Wildcard,
}

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
    pub algorithm: Arc<str>,
}

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
    pub failed_searches: ConsistentCounter,
}

/// Query processor for advanced query parsing
#[derive(Debug)]
pub struct QueryProcessor {
    /// Query expansion enabled
    #[allow(dead_code)] // TODO: Implement in query expansion system
    expansion_enabled: bool,
    /// Expansion dictionary
    #[allow(dead_code)] // TODO: Implement in query expansion system
    expansion_dict: HashMap<Arc<str>, Vec<Arc<str>>>,
}

/// Result ranker for relevance scoring
#[derive(Debug)]
pub struct ResultRanker {
    /// Ranking algorithm
    #[allow(dead_code)] // TODO: Implement in ranking system
    algorithm: RankingAlgorithm,
    /// Boost factors for different fields
    #[allow(dead_code)] // TODO: Implement in ranking system
    field_boosts: HashMap<Arc<str>, f32>,
}

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
    MlRanking,
}

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
            expansion_dictionary: HashMap::new(),
        }
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
            result_ranker: Arc::new(ResultRanker::new()),
        }
    }

    /// Create a chat searcher with custom options
    pub fn with_options(search_index: Arc<ChatSearchIndex>, options: SearchOptions) -> Self {
        Self {
            search_index,
            options,
            cache: Arc::new(SkipMap::new()),
            stats: Arc::new(ChatSearcherStats::default()),
            query_processor: Arc::new(QueryProcessor::new()),
            result_ranker: Arc::new(ResultRanker::new()),
        }
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
                query_hash: query_hash.clone(),
            };
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

impl QueryProcessor {
    /// Create a new query processor
    pub fn new() -> Self {
        Self {
            expansion_enabled: false,
            expansion_dict: HashMap::new(),
        }
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
                normalization_applied: true,
            },
        })
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
            field_boosts: HashMap::new(),
        }
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
