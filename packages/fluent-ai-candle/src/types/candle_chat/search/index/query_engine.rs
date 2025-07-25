//! Search query execution engine
//!
//! Contains the main search execution logic for different query types,
//! including streaming search results and query optimization.

use std::sync::Arc;

use fluent_ai_async::AsyncStream;

use super::core::ChatSearchIndex;
use super::super::types::{SearchQuery, SearchResult, QueryOperator};

impl ChatSearchIndex {
    /// Search messages with SIMD optimization (streaming individual results)
    pub fn search_stream(&self, query: SearchQuery) -> AsyncStream<SearchResult> {
        let self_clone = self.clone();

        AsyncStream::with_channel(move |sender| {
            let _start_time = std::time::Instant::now();
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

    /// Search for documents containing all terms (AND operation)
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

    /// Search for documents containing any term (OR operation)
    pub(super) fn search_or(&self, terms: &[Arc<str>], fuzzy: bool) -> Vec<SearchResult> {
        let mut results = Vec::new();
        let mut seen_docs: std::collections::HashSet<String> = std::collections::HashSet::new();

        for term in terms {
            let term_results = if fuzzy {
                self.fuzzy_search(term)
            } else {
                self.exact_search(term)
            };

            for result in term_results {
                let doc_id = result.message.id.clone();
                if !seen_docs.contains(&doc_id) {
                    seen_docs.insert(doc_id);
                    results.push(result);
                }
            }
        }

        results
    }
}