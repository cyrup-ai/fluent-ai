//! Core search operations and query processing
//!
//! This module provides the main search functionality including different query
//! operators (AND, OR, NOT, phrase, proximity) with SIMD optimization.

use std::sync::Arc;
use std::time::Instant;

use fluent_ai_async::AsyncStream;

use super::super::types::{SearchQuery, SearchResult, QueryOperator, SortOrder};
use super::types::{ChatSearchIndex, handle_error};

impl ChatSearchIndex {
    /// Search messages with SIMD optimization (streaming individual results)
    pub fn search_stream(&self, query: SearchQuery) -> AsyncStream<SearchResult> {
        let self_clone = self.clone();

        AsyncStream::with_channel(move |sender| {
            let _start_time = Instant::now();
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

    /// Search with AND operator
    pub(super) fn search_and(&self, terms: &[Arc<str>], fuzzy: bool) -> Vec<SearchResult> {
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
    pub(super) fn search_or(&self, terms: &[Arc<str>], fuzzy: bool) -> Vec<SearchResult> {
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
    pub(super) fn search_not(&self, terms: &[Arc<str>], fuzzy: bool) -> Vec<SearchResult> {
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
    pub(super) fn search_phrase(&self, terms: &[Arc<str>], fuzzy: bool) -> Vec<SearchResult> {
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
    pub(super) fn search_proximity(&self, terms: &[Arc<str>], distance: u32, fuzzy: bool) -> Vec<SearchResult> {
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

    /// Sort search results
    pub(super) fn sort_results(&self, results: &mut Vec<SearchResult>, sort_order: &SortOrder) {
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
}