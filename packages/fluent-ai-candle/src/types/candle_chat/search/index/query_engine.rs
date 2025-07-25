//! Search query execution engine
//!
//! Contains the main search execution logic for different query types,
//! including streaming search results and query optimization.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use fluent_ai_async::AsyncStream;

use crate::types::candle_chat::message::types::CandleMessage;
use super::core::ChatSearchIndex;
use super::super::types::{SearchQuery, SearchResult, QueryOperator, SearchResultMetadata};

impl ChatSearchIndex {
    /// Search messages with SIMD optimization (streaming individual results)
    pub fn search_stream(&self, query: SearchQuery) -> AsyncStream<SearchResult> {
        let self_clone = self.clone();

        AsyncStream::with_channel(move |sender| {
            let start_time = std::time::Instant::now();
            
            // Parse query terms
            let terms: Vec<Arc<str>> = query
                .query
                .to_lowercase()
                .split_whitespace()
                .map(|s| Arc::from(s))
                .collect();

            if terms.is_empty() {
                return;
            }

            // Execute search based on operator
            let mut results = match query.operator {
                QueryOperator::And => self_clone.search_and(&terms, query.fuzzy),
                QueryOperator::Or => self_clone.search_or(&terms, query.fuzzy),
                QueryOperator::Not => self_clone.search_not(&terms, query.fuzzy),
                QueryOperator::Phrase => self_clone.search_phrase(&terms, query.fuzzy),
                QueryOperator::Proximity(distance) => self_clone.search_proximity(&terms, distance, query.fuzzy),
            };

            // Apply sorting
            self_clone.sort_results(&mut results, query.sort_order);

            // Apply limit
            if let Some(limit) = query.limit {
                results.truncate(limit);
            }

            // Stream results
            for mut result in results {
                // Add metadata
                result.metadata = Some(SearchResultMetadata {
                    search_time_ms: start_time.elapsed().as_millis() as u64,
                    total_results: 1, // Will be updated by caller if needed
                    query_id: Arc::from(format!("query_{}", self_clone.query_counter.get())),
                });

                let _ = sender.send(result);
            }

            // Update statistics
            self_clone.query_counter.inc();
            if let Ok(mut stats) = self_clone.statistics.try_write() {
                stats.total_queries += 1;
                stats.average_query_time_ms = (stats.average_query_time_ms + start_time.elapsed().as_millis() as u64) / 2;
                stats.last_query_time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
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
        let mut seen_docs = std::collections::HashSet::new();

        for term in terms {
            let term_results = if fuzzy {
                self.fuzzy_search(term)
            } else {
                self.exact_search(term)
            };

            for result in term_results {
                let doc_id = result.message.message.id.as_ref().unwrap_or(&String::new()).clone();
                if !seen_docs.contains(&doc_id) {
                    seen_docs.insert(doc_id);
                    results.push(result);
                }
            }
        }

        results
    }
}