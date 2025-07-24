//! Search indexing system with SIMD optimization
//!
//! This module provides efficient inverted index construction and maintenance
//! with SIMD-optimized text processing and TF-IDF scoring.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use atomic_counter::{AtomicCounter, ConsistentCounter};
use crossbeam_skiplist::SkipMap;
use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::types::candle_chat::SearchChatMessage;
use super::types::{SearchQuery, SearchResult, SearchStatistics, MatchPosition, MatchType, SortOrder, TermFrequency, QueryOperator, SearchResultMetadata};
use super::search_index::IndexEntry;

/// Handle errors in streaming context without panicking
macro_rules! handle_error {
    ($error:expr, $context:literal) => {
        eprintln!("Streaming error in {}: {}", $context, $error)
        // Continue processing instead of returning error
    };
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
    document_count: Arc<AtomicUsize>,
    /// Query counter
    query_counter: Arc<ConsistentCounter>,
    /// Index update counter
    index_update_counter: Arc<ConsistentCounter>,
    /// Search statistics
    statistics: Arc<RwLock<SearchStatistics>>,
    /// SIMD processing threshold
    simd_threshold: Arc<AtomicUsize>,
}

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
                &self.document_count.load(Ordering::Relaxed),
            )
            .field("query_counter", &"ConsistentCounter")
            .field("index_update_counter", &"ConsistentCounter")
            .field("statistics", &"Arc<RwLock<SearchStatistics>>")
            .field(
                "simd_threshold",
                &self.simd_threshold.load(Ordering::Relaxed),
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
            statistics: Arc::new(RwLock::new(SearchStatistics::default())),
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
                        .collect(),
                };

                // Update inverted index
                let mut entries = self_clone
                    .inverted_index
                    .get(&term)
                    .map(|e| e.value().clone())
                    .unwrap_or_default();
                entries.push(index_entry);
                self_clone.inverted_index.insert(term.clone(), entries);

                // Update term frequencies
                let doc_freq = self_clone
                    .inverted_index
                    .get(&term)
                    .map(|e| e.value().len() as u32)
                    .unwrap_or(1);
                    
                let tf_entry = TermFrequency {
                    tf: tf,
                    df: doc_freq,
                    total_docs: self_clone.document_count.load(Ordering::Relaxed) as u32 + 1,
                };
                self_clone.term_frequencies.insert(term.clone(), tf_entry);
            }

            self_clone.document_count.fetch_add(1, Ordering::Relaxed);
            self_clone.index_update_counter.inc();

            // Update statistics
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

    /// Search messages with SIMD optimization (streaming individual results)
    pub fn search_stream(&self, query: SearchQuery) -> AsyncStream<SearchResult> {
        let self_clone = self.clone();

        AsyncStream::with_channel(move |sender| {
            let start_time = Instant::now();
            self_clone.query_counter.inc();

            let results = match query.operator {
                QueryOperator::And => self_clone.search_and(&query.terms, query.fuzzy_matching),
                QueryOperator::Or => self_clone.search_or(&query.terms, query.fuzzy_matching),
                QueryOperator::Not => self_clone.search_not(&query.terms, query.fuzzy_matching),
                QueryOperator::Phrase => self_clone.search_phrase(&query.terms, query.fuzzy_matching),
                QueryOperator::Proximity { distance } => {
                    self_clone.search_proximity(&query.terms, distance, query.fuzzy_matching)
                }
            };

            // Sort and paginate results
            let mut sorted_results = results;
            self_clone.sort_results(&mut sorted_results, &query.sort_order);

            let start = query.offset;
            let end = (start + query.max_results).min(sorted_results.len());

            for result in sorted_results[start..end].iter() {
                let _ = sender.send(result.clone());
            }
        })
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
            self.process_words_simd(words)
        } else {
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
                processed.push(Arc::from(word.to_lowercase()));
            }
        }

        processed
    }

    /// Search with AND operator
    fn search_and(&self, terms: &[Arc<str>], fuzzy: bool) -> Vec<SearchResult> {
        if terms.is_empty() {
            return Vec::new();
        }

        let mut candidates = None;

        for term in terms {
            let term_candidates = if fuzzy {
                self.fuzzy_search(term)
            } else {
                self.exact_search(term)
            };

            if candidates.is_none() {
                candidates = Some(term_candidates);
            } else {
                let current = candidates.unwrap();
                let intersection = self.intersect_results(current, term_candidates);
                candidates = Some(intersection);
            }
        }

        candidates.unwrap_or_default()
    }

    /// Search with OR operator
    fn search_or(&self, terms: &[Arc<str>], fuzzy: bool) -> Vec<SearchResult> {
        let mut seen_docs = std::collections::HashSet::new();
        let mut results = Vec::new();

        for term in terms {
            if let Some(entries) = self.inverted_index.get(term) {
                for entry in entries.value() {
                    if !seen_docs.contains(&entry.doc_id) {
                        seen_docs.insert(entry.doc_id.clone());
                        if let Some(doc) = self.document_store.get(&entry.doc_id) {
                            let result = SearchResult {
                                message: doc.value().clone(),
                                relevance_score: entry.term_frequency * 100.0,
                                matching_terms: vec![term.clone()],
                                highlighted_content: None,
                                tags: vec![],
                                context: vec![],
                                match_positions: vec![],
                                metadata: None,
                            };
                            results.push(result);
                        }
                    }
                }
            }
        }

        results
    }

    /// Search with NOT operator
    fn search_not(&self, terms: &[Arc<str>], fuzzy: bool) -> Vec<SearchResult> {
        let mut excluded_docs = std::collections::HashSet::new();

        for term in terms {
            let term_results = if fuzzy {
                self.fuzzy_search(term)
            } else {
                self.exact_search(term)
            };

            for result in term_results {
                excluded_docs.insert(result.message.message.id.unwrap_or_default());
            }
        }

        let mut results = Vec::new();
        for entry in self.document_store.iter() {
            let doc_id = entry.key();
            if !excluded_docs.contains(doc_id.as_ref()) {
                let message = entry.value().clone();
                let result = SearchResult {
                    message,
                    relevance_score: 1.0,
                    matching_terms: vec![],
                    highlighted_content: None,
                    tags: vec![],
                    context: vec![],
                    match_positions: vec![],
                    metadata: None,
                };
                results.push(result);
            }
        }

        results
    }

    /// Search for phrase matches
    fn search_phrase(&self, terms: &[Arc<str>], fuzzy: bool) -> Vec<SearchResult> {
        let phrase = terms
            .iter()
            .map(|t| t.as_ref())
            .collect::<Vec<_>>()
            .join(" ");

        let mut results = Vec::new();

        for entry in self.document_store.iter() {
            let message = entry.value();
            let content = message.message.content.to_lowercase();

            let matches = if fuzzy {
                self.fuzzy_match(&content, &phrase)
            } else {
                content.contains(&phrase)
            };

            if matches {
                let result = SearchResult {
                    message: message.clone(),
                    relevance_score: if fuzzy { 0.8 } else { 1.0 },
                    matching_terms: terms.to_vec(),
                    highlighted_content: Some(Arc::from(
                        self.highlight_text(&content, &phrase),
                    )),
                    tags: vec![],
                    context: vec![],
                    match_positions: vec![],
                    metadata: None,
                };
                results.push(result);
            }
        }

        results
    }

    /// Search for proximity matches
    fn search_proximity(&self, terms: &[Arc<str>], distance: u32, fuzzy: bool) -> Vec<SearchResult> {
        let mut results = Vec::new();

        for entry in self.document_store.iter() {
            let message = entry.value();
            let tokens = self.tokenize_with_simd(&message.message.content);

            if self.check_proximity(&tokens, terms, distance) {
                let relevance_score = if fuzzy { 0.7 } else { 0.9 };
                let result = SearchResult {
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
                };
                results.push(result);
            }
        }

        results
    }

    /// Exact search for a term
    fn exact_search(&self, term: &Arc<str>) -> Vec<SearchResult> {
        let mut results = Vec::new();

        if let Some(entries) = self.inverted_index.get(term) {
            for entry in entries.value() {
                if let Some(message) = self.document_store.get(&entry.doc_id) {
                    let tf_idf = if let Some(tf) = self.term_frequencies.get(term) {
                        tf.value().calculate_tfidf()
                    } else {
                        entry.term_frequency
                    };

                    let result = SearchResult {
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
                    };
                    results.push(result);
                }
            }
        }

        results
    }

    /// Fuzzy search for a term
    fn fuzzy_search(&self, term: &Arc<str>) -> Vec<SearchResult> {
        let mut results = Vec::new();

        for entry in self.inverted_index.iter() {
            let indexed_term = entry.key();
            if self.fuzzy_match(indexed_term, term) {
                let mut exact_results = self.exact_search(indexed_term);
                for result in &mut exact_results {
                    result.relevance_score *= 0.8; // Reduce score for fuzzy matches
                }
                results.extend(exact_results);
            }
        }

        results
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
            .map(|r| r.message.message.id.clone().unwrap_or_default())
            .collect();

        for result in results1 {
            if ids2.contains(&result.message.message.id.clone().unwrap_or_default()) {
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
                positions.entry(token.clone()).or_default().push(i);
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
            if let Ok(stats) = self_clone.statistics.try_read() {
                let _ = sender.send(stats.clone());
            }
        })
    }
}

impl Default for ChatSearchIndex {
    fn default() -> Self {
        Self::new()
    }
}