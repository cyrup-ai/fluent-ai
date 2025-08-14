//! SIMD-optimized search algorithms
//!
//! This module implements the core search algorithms with SIMD optimization
//! for high-performance text search and matching.

use std::sync::Arc;

use fluent_ai_async::AsyncStream;

use super::index::ChatSearchIndex;
use super::types::{MatchPosition, SearchResult, SearchResultMetadata};

impl ChatSearchIndex {
    /// Search with AND operator (all terms must match)
    pub fn search_and_stream(
        &self,
        terms: &[Arc<str>],
        _fuzzy_matching: bool,
    ) -> AsyncStream<SearchResult> {
        let self_clone = self.clone();
        let terms_clone = terms.to_vec();

        AsyncStream::with_channel(move |sender| {
            if terms_clone.is_empty() {
                return;
            }

            // Find documents that contain all terms
            let mut candidate_docs = None;

            for term in &terms_clone {
                if let Some(entries) = self_clone.inverted_index().get(&**term) {
                    let doc_ids: std::collections::HashSet<Arc<str>> = entries
                        .value()
                        .iter()
                        .map(|entry| entry.doc_id.clone())
                        .collect();

                    candidate_docs = match candidate_docs {
                        None => Some(doc_ids),
                        Some(existing) => Some(existing.intersection(&doc_ids).cloned().collect()),
                    };
                } else {
                    // Term not found - no results
                    return;
                }
            }

            if let Some(doc_ids) = candidate_docs {
                for doc_id in doc_ids {
                    if let Some(message) = self_clone.document_store().get(&doc_id) {
                        let result = SearchResult {
                            message: message.value().clone(),
                            relevance_score: self_clone
                                .calculate_relevance_score(&terms_clone, &doc_id),
                            matching_terms: terms_clone.clone(),
                            highlighted_content: None,
                            tags: Vec::new(),
                            context: Vec::new(),
                            match_positions: self_clone.find_match_positions(
                                &terms_clone,
                                &message.value().message.content,
                            ),
                            metadata: Some(SearchResultMetadata {
                                query_time_ms: 0.0,
                                index_version: 1,
                                total_matches: 1,
                            }),
                        };
                        let _ = sender.send(result);
                    }
                }
            }
        })
    }

    /// Search with OR operator (any term must match)
    pub fn search_or_stream(
        &self,
        terms: &[Arc<str>],
        _fuzzy_matching: bool,
    ) -> AsyncStream<SearchResult> {
        let self_clone = self.clone();
        let terms_clone = terms.to_vec();

        AsyncStream::with_channel(move |sender| {
            let mut all_docs = std::collections::HashSet::new();

            for term in &terms_clone {
                if let Some(entries) = self_clone.inverted_index().get(&**term) {
                    for entry in entries.value() {
                        all_docs.insert(entry.doc_id.clone());
                    }
                }
            }

            for doc_id in all_docs {
                if let Some(message) = self_clone.document_store().get(&doc_id) {
                    let matching_terms: Vec<Arc<str>> = terms_clone
                        .iter()
                        .filter(|term| {
                            self_clone
                                .inverted_index()
                                .get(&**term)
                                .map(|entries| entries.value().iter().any(|e| e.doc_id == doc_id))
                                .unwrap_or(false)
                        })
                        .cloned()
                        .collect();

                    let result = SearchResult {
                        message: message.value().clone(),
                        relevance_score: self_clone
                            .calculate_relevance_score(&matching_terms, &doc_id),
                        matching_terms,
                        highlighted_content: None,
                        tags: Vec::new(),
                        context: Vec::new(),
                        match_positions: self_clone
                            .find_match_positions(&terms_clone, &message.value().message.content),
                        metadata: Some(SearchResultMetadata {
                            query_time_ms: 0.0,
                            index_version: 1,
                            total_matches: 1,
                        }),
                    };
                    let _ = sender.send(result);
                }
            }
        })
    }

    /// Calculate relevance score using TF-IDF
    fn calculate_relevance_score(&self, terms: &[Arc<str>], doc_id: &Arc<str>) -> f32 {
        let mut score = 0.0;

        for term in terms {
            if let Some(tf_entry) = self.term_frequencies.get(&**term) {
                if let Some(entries) = self.inverted_index().get(&**term) {
                    for entry in entries.value() {
                        if entry.doc_id == *doc_id {
                            score += tf_entry.value().calculate_tfidf();
                            break;
                        }
                    }
                }
            }
        }

        score / terms.len() as f32
    }

    /// Find match positions in content
    fn find_match_positions(&self, terms: &[Arc<str>], content: &str) -> Vec<MatchPosition> {
        let mut positions = Vec::new();
        let content_lower = content.to_lowercase();

        for term in terms {
            let term_lower = term.to_lowercase();
            let mut start = 0;

            while let Some(pos) = content_lower[start..].find(&term_lower) {
                let actual_pos = start + pos;
                positions.push(MatchPosition {
                    start: actual_pos,
                    end: actual_pos + term.len(),
                    term: term.clone(),
                });
                start = actual_pos + 1;
            }
        }

        positions.sort_by_key(|p| p.start);
        positions
    }

    /// Search with NOT operator (terms must not match)
    pub fn search_not_stream(
        &self,
        terms: &[Arc<str>],
        _fuzzy_matching: bool,
    ) -> AsyncStream<SearchResult> {
        let self_clone = self.clone();
        let terms_clone = terms.to_vec();

        AsyncStream::with_channel(move |sender| {
            let mut excluded_docs = std::collections::HashSet::new();

            // Collect all documents that contain any of the terms
            for term in &terms_clone {
                if let Some(entries) = self_clone.inverted_index.get(term.as_ref()) {
                    for entry in entries.value() {
                        excluded_docs.insert(entry.doc_id.clone());
                    }
                }
            }

            // Return all documents that don't contain any of the terms
            for entry in self_clone.document_store.iter() {
                let doc_id = entry.key().clone();
                if !excluded_docs.contains(&doc_id) {
                    let result = SearchResult {
                        message: entry.value().clone(),
                        relevance_score: 1.0, // All non-matching documents have equal relevance
                        matching_terms: Vec::new(),
                        highlighted_content: None,
                        tags: Vec::new(),
                        context: Vec::new(),
                        match_positions: Vec::new(),
                        metadata: Some(SearchResultMetadata {
                            query_time_ms: 0.0,
                            index_version: 1,
                            total_matches: 1,
                        }),
                    };
                    let _ = sender.send(result);
                }
            }
        })
    }

    /// Search for exact phrase matching
    pub fn search_phrase_stream(
        &self,
        terms: &[Arc<str>],
        _fuzzy_matching: bool,
    ) -> AsyncStream<SearchResult> {
        let self_clone = self.clone();
        let terms_clone = terms.to_vec();

        AsyncStream::with_channel(move |sender| {
            if terms_clone.is_empty() {
                return;
            }

            // Build the phrase to search for
            let phrase: String = terms_clone
                .iter()
                .map(|t| t.as_ref())
                .collect::<Vec<_>>()
                .join(" ");

            // Search through all documents for exact phrase match
            for entry in self_clone.document_store.iter() {
                let message = entry.value();
                let content = &message.message.content;

                if content.to_lowercase().contains(&phrase.to_lowercase()) {
                    let result = SearchResult {
                        message: message.clone(),
                        relevance_score: 1.0, // Exact phrase matches have high relevance
                        matching_terms: terms_clone.clone(),
                        highlighted_content: None,
                        tags: Vec::new(),
                        context: Vec::new(),
                        match_positions: self_clone.find_phrase_positions(&phrase, content),
                        metadata: Some(SearchResultMetadata {
                            query_time_ms: 0.0,
                            index_version: 1,
                            total_matches: 1,
                        }),
                    };
                    let _ = sender.send(result);
                }
            }
        })
    }

    /// Search with proximity constraint (terms within specified distance)
    pub fn search_proximity_stream(
        &self,
        terms: &[Arc<str>],
        distance: u32,
        _fuzzy_matching: bool,
    ) -> AsyncStream<SearchResult> {
        let self_clone = self.clone();
        let terms_clone = terms.to_vec();

        AsyncStream::with_channel(move |sender| {
            if terms_clone.len() < 2 {
                // Proximity search requires at least 2 terms
                return;
            }

            for entry in self_clone.document_store.iter() {
                let message = entry.value();
                let content = &message.message.content;
                let tokens = self_clone.tokenize_with_simd(content);

                // Check if terms appear within the specified distance
                if self_clone.check_proximity(&terms_clone, &tokens, distance) {
                    let result = SearchResult {
                        message: message.clone(),
                        relevance_score: self_clone.calculate_proximity_score(
                            &terms_clone,
                            &tokens,
                            distance,
                        ),
                        matching_terms: terms_clone.clone(),
                        highlighted_content: None,
                        tags: Vec::new(),
                        context: Vec::new(),
                        match_positions: self_clone.find_match_positions(&terms_clone, content),
                        metadata: Some(SearchResultMetadata {
                            query_time_ms: 0.0,
                            index_version: 1,
                            total_matches: 1,
                        }),
                    };
                    let _ = sender.send(result);
                }
            }
        })
    }

    /// Find exact phrase positions in content
    fn find_phrase_positions(&self, phrase: &str, content: &str) -> Vec<MatchPosition> {
        let mut positions = Vec::new();
        let content_lower = content.to_lowercase();
        let phrase_lower = phrase.to_lowercase();
        let mut start = 0;

        while let Some(pos) = content_lower[start..].find(&phrase_lower) {
            let actual_pos = start + pos;
            positions.push(MatchPosition {
                start: actual_pos,
                end: actual_pos + phrase.len(),
                term: Arc::from(phrase),
            });
            start = actual_pos + 1;
        }

        positions
    }

    /// Check if terms appear within proximity distance
    fn check_proximity(&self, terms: &[Arc<str>], tokens: &[Arc<str>], distance: u32) -> bool {
        let mut term_positions: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();

        // Find positions of each term
        for (i, token) in tokens.iter().enumerate() {
            for term in terms {
                if token.to_lowercase() == term.to_lowercase() {
                    term_positions
                        .entry(term.to_string())
                        .or_insert_with(Vec::new)
                        .push(i);
                }
            }
        }

        // Check if we have all terms
        if term_positions.len() != terms.len() {
            return false;
        }

        // Check proximity for each combination
        let term_keys: Vec<String> = term_positions.keys().cloned().collect();
        for i in 0..term_keys.len() {
            for j in i + 1..term_keys.len() {
                let positions1 = &term_positions[&term_keys[i]];
                let positions2 = &term_positions[&term_keys[j]];

                let mut found_within_distance = false;
                for &pos1 in positions1 {
                    for &pos2 in positions2 {
                        if (pos1 as i32 - pos2 as i32).abs() <= distance as i32 {
                            found_within_distance = true;
                            break;
                        }
                    }
                    if found_within_distance {
                        break;
                    }
                }

                if !found_within_distance {
                    return false;
                }
            }
        }

        true
    }

    /// Calculate proximity-based relevance score
    fn calculate_proximity_score(
        &self,
        terms: &[Arc<str>],
        tokens: &[Arc<str>],
        distance: u32,
    ) -> f32 {
        let mut min_distance = distance;
        let mut term_positions: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();

        // Find positions of each term
        for (i, token) in tokens.iter().enumerate() {
            for term in terms {
                if token.to_lowercase() == term.to_lowercase() {
                    term_positions
                        .entry(term.to_string())
                        .or_insert_with(Vec::new)
                        .push(i);
                }
            }
        }

        // Find minimum distance between terms
        let term_keys: Vec<String> = term_positions.keys().cloned().collect();
        for i in 0..term_keys.len() {
            for j in i + 1..term_keys.len() {
                let positions1 = &term_positions[&term_keys[i]];
                let positions2 = &term_positions[&term_keys[j]];

                for &pos1 in positions1 {
                    for &pos2 in positions2 {
                        let dist = (pos1 as i32 - pos2 as i32).abs() as u32;
                        if dist < min_distance {
                            min_distance = dist;
                        }
                    }
                }
            }
        }

        // Score inversely proportional to distance
        1.0 - (min_distance as f32 / distance as f32)
    }
}
