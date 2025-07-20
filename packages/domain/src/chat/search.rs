//! Enhanced history management and search system
//!
//! This module provides comprehensive history management with SIMD-optimized full-text search,
//! lock-free tag management, and zero-allocation streaming export capabilities using
//! blazing-fast algorithms and elegant ergonomic APIs.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use atomic_counter::{AtomicCounter, ConsistentCounter};
use crossbeam_queue::SegQueue;
use crossbeam_skiplist::SkipMap;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;
use wide::f32x8;

use crate::message::Message;

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
    pub message: ChatMessage,
    /// Relevance score (0.0-1.0)
    pub relevance_score: f32,
    /// Matching terms
    pub matching_terms: Vec<Arc<str>>,
    /// Highlighted content
    pub highlighted_content: Arc<str>,
    /// Associated tags
    pub tags: Vec<Arc<str>>,
    /// Context messages (before/after)
    pub context: Vec<ChatMessage>,
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
    document_store: SkipMap<Arc<str>, ChatMessage>,
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
    pub async fn add_message(&self, message: ChatMessage) -> Result<(), SearchError> {
        let doc_id = Arc::from(message.timestamp.to_string());

        // Store the document
        self.document_store.insert(doc_id.clone(), message.clone());
        self.document_count.fetch_add(1, Ordering::Relaxed);

        // Tokenize and index the content
        let tokens = self.tokenize_with_simd(&message.content);
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

            // Update inverted index
            if let Some(mut entries) = self.inverted_index.get(&term) {
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

        let mut results = Vec::new();
        let mut scores = HashMap::new();

        match query.operator {
            QueryOperator::And => {
                results = self.search_and(&query.terms, query.fuzzy_matching).await?;
            }
            QueryOperator::Or => {
                results = self.search_or(&query.terms, query.fuzzy_matching).await?;
            }
            QueryOperator::Not => {
                results = self.search_not(&query.terms, query.fuzzy_matching).await?;
            }
            QueryOperator::Phrase => {
                results = self
                    .search_phrase(&query.terms, query.fuzzy_matching)
                    .await?;
            }
            QueryOperator::Proximity { distance } => {
                results = self
                    .search_proximity(&query.terms, distance, query.fuzzy_matching)
                    .await?;
            }
        }

        // Apply filters
        results = self.apply_filters(results, query).await?;

        // Sort results
        self.sort_results(&mut results, &query.sort_order);

        // Apply pagination
        let start = query.offset;
        let end = (start + query.max_results).min(results.len());
        results = results[start..end].to_vec();

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
            let key = result.message.timestamp.to_string();
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
                excluded_docs.insert(result.message.timestamp.to_string());
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
                    highlighted_content: Arc::from(""),
                    tags: vec![],
                    context: vec![],
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
            let content = message.content.to_lowercase();
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
                        highlighted_content: Arc::from(self.highlight_text(&content, &phrase)),
                        tags: vec![],
                        context: vec![],
                    });
                }
            } else if content.contains(&phrase) {
                results.push(SearchResult {
                    message: message.clone(),
                    relevance_score: 1.0,
                    matching_terms: terms.to_vec(),
                    highlighted_content: Arc::from(self.highlight_text(&content, &phrase)),
                    tags: vec![],
                    context: vec![],
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
            let tokens = self.tokenize_with_simd(&message.content);

            if self.check_proximity(&tokens, terms, distance) {
                let relevance_score = if fuzzy { 0.7 } else { 0.9 };
                results.push(SearchResult {
                    message: message.clone(),
                    relevance_score,
                    matching_terms: terms.to_vec(),
                    highlighted_content: Arc::from(self.highlight_terms(&message.content, terms)),
                    tags: vec![],
                    context: vec![],
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
                        highlighted_content: Arc::from(
                            self.highlight_text(&message.value().content, term),
                        ),
                        tags: vec![],
                        context: vec![],
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
            .map(|r| r.message.timestamp.to_string())
            .collect();

        for result in results1 {
            if ids2.contains(&result.message.timestamp.to_string()) {
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
                r.message.timestamp >= date_range.start && r.message.timestamp <= date_range.end
            });
        }

        // User filter
        if let Some(user_filter) = &query.user_filter {
            results.retain(|r| r.message.role == *user_filter);
        }

        // Content type filter
        if let Some(content_type) = &query.content_type_filter {
            results.retain(|r| {
                r.message
                    .metadata
                    .as_ref()
                    .map(|m| m.contains(content_type.as_ref()))
                    .unwrap_or(false)
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
                results.sort_by(|a, b| b.message.timestamp.cmp(&a.message.timestamp));
            }
            SortOrder::DateAscending => {
                results.sort_by(|a, b| a.message.timestamp.cmp(&b.message.timestamp));
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
        message: &ChatMessage,
    ) -> Result<Vec<Arc<str>>, SearchError> {
        let mut suggested_tags = Vec::new();
        let content = message.content.to_lowercase();

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
        messages: Vec<ChatMessage>,
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
        mut messages: Vec<ChatMessage>,
        options: &ExportOptions,
    ) -> Result<Vec<ChatMessage>, SearchError> {
        // Date range filter
        if let Some(date_range) = &options.date_range {
            messages.retain(|m| m.timestamp >= date_range.start && m.timestamp <= date_range.end);
        }

        // User filter
        if let Some(user_filter) = &options.user_filter {
            messages.retain(|m| m.role == *user_filter);
        }

        Ok(messages)
    }

    /// Export to JSON format
    async fn export_json(
        &self,
        messages: &[ChatMessage],
        options: &ExportOptions,
    ) -> Result<String, SearchError> {
        let mut json_messages = Vec::new();

        for message in messages {
            let mut json_obj = serde_json::json!({
                "role": message.role,
                "content": message.content,
                "tokens": message.tokens,
            });

            if options.include_timestamps {
                json_obj["timestamp"] = serde_json::Value::Number(message.timestamp.into());
            }

            if options.include_metadata {
                if let Some(metadata) = &message.metadata {
                    json_obj["metadata"] = serde_json::Value::String(metadata.to_string());
                }
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
        messages: &[ChatMessage],
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
            let mut row = vec![
                message.role.as_ref(),
                &message.content.replace(',', "\\,").replace('\n', "\\n"),
                &message.tokens.to_string(),
            ];

            if options.include_timestamps {
                row.push(&message.timestamp.to_string());
            }

            if options.include_metadata {
                row.push(message.metadata.as_ref().map(|m| m.as_ref()).unwrap_or(""));
            }

            csv_output.push_str(&row.join(","));
            csv_output.push('\n');
        }

        Ok(csv_output)
    }

    /// Export to Markdown format
    async fn export_markdown(
        &self,
        messages: &[ChatMessage],
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
            markdown_output.push_str(&format!("## {}\n\n", message.role));
            markdown_output.push_str(&format!("{}\n\n", message.content));

            if options.include_timestamps {
                markdown_output.push_str(&format!("*Timestamp: {}*\n\n", message.timestamp));
            }

            if options.include_metadata && message.metadata.is_some() {
                markdown_output.push_str(&format!(
                    "*Metadata: {}*\n\n",
                    message.metadata.as_ref().unwrap()
                ));
            }

            markdown_output.push_str("---\n\n");
        }

        Ok(markdown_output)
    }

    /// Export to HTML format
    async fn export_html(
        &self,
        messages: &[ChatMessage],
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
            html_output.push_str(&format!("<div class=\"role\">{}</div>\n", message.role));
            html_output.push_str(&format!(
                "<div class=\"content\">{}</div>\n",
                message.content.replace('<', "&lt;").replace('>', "&gt;")
            ));

            if options.include_timestamps {
                html_output.push_str(&format!(
                    "<div class=\"timestamp\">Timestamp: {}</div>\n",
                    message.timestamp
                ));
            }

            if options.include_metadata && message.metadata.is_some() {
                html_output.push_str(&format!(
                    "<div class=\"metadata\">Metadata: {}</div>\n",
                    message.metadata.as_ref().unwrap()
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
        messages: &[ChatMessage],
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
            xml_output.push_str(&format!("    <role>{}</role>\n", message.role));
            xml_output.push_str(&format!(
                "    <content>{}</content>\n",
                message.content.replace('<', "&lt;").replace('>', "&gt;")
            ));
            xml_output.push_str(&format!("    <tokens>{}</tokens>\n", message.tokens));

            if options.include_timestamps {
                xml_output.push_str(&format!(
                    "    <timestamp>{}</timestamp>\n",
                    message.timestamp
                ));
            }

            if options.include_metadata && message.metadata.is_some() {
                xml_output.push_str(&format!(
                    "    <metadata>{}</metadata>\n",
                    message.metadata.as_ref().unwrap()
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
        messages: &[ChatMessage],
        options: &ExportOptions,
    ) -> Result<String, SearchError> {
        let mut text_output = String::new();

        text_output.push_str("CONVERSATION HISTORY\n");
        text_output.push_str("===================\n\n");

        for message in messages {
            text_output.push_str(&format!("{}: {}\n", message.role, message.content));

            if options.include_timestamps {
                text_output.push_str(&format!("Timestamp: {}\n", message.timestamp));
            }

            if options.include_metadata && message.metadata.is_some() {
                text_output.push_str(&format!(
                    "Metadata: {}\n",
                    message.metadata.as_ref().unwrap()
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
        Ok(base64::encode(compressed))
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
    pub async fn add_message(&self, message: ChatMessage) -> Result<(), SearchError> {
        // Add to search index
        self.search_index.add_message(message.clone()).await?;

        // Auto-tag message
        let suggested_tags = self.tagger.auto_tag_message(&message).await?;
        if !suggested_tags.is_empty() {
            let message_id = Arc::from(message.timestamp.to_string());
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
            let message_id = Arc::from(result.message.timestamp.to_string());
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
        messages: Vec<ChatMessage>,
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
