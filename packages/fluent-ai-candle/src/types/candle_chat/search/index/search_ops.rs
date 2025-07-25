//! Search operations and algorithms
//!
//! Contains specific search algorithms including exact search, fuzzy search,
//! phrase search, proximity search, and result sorting.

use std::sync::Arc;

use super::core::ChatSearchIndex;
use super::super::types::{SearchResult, SortOrder};

impl ChatSearchIndex {
    /// Search for documents containing exact term
    pub(super) fn exact_search(&self, term: &Arc<str>) -> Vec<SearchResult> {
        let mut results = Vec::new();

        if let Some(entries) = self.inverted_index.get(term) {
            for entry in entries.value() {
                if let Some(message) = self.document_store.get(&entry.doc_id) {
                    let result = SearchResult {
                        message: message.value().clone(),
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

        results
    }

    /// Search for documents with fuzzy matching
    pub(super) fn fuzzy_search(&self, term: &Arc<str>) -> Vec<SearchResult> {
        let mut results = Vec::new();

        // Simple fuzzy matching - can be enhanced with more sophisticated algorithms
        for entry in self.inverted_index.iter() {
            let index_term = entry.key();
            if self.fuzzy_match(index_term.as_ref(), term.as_ref()) {
                for index_entry in entry.value() {
                    if let Some(message) = self.document_store.get(&index_entry.doc_id) {
                        let result = SearchResult {
                            message: message.value().clone(),
                            relevance_score: index_entry.term_frequency * 80.0, // Lower score for fuzzy
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
            
            // Find positions of all terms
            let mut term_positions = std::collections::HashMap::new();
            for (i, token) in tokens.iter().enumerate() {
                for term in terms {
                    let matches = if fuzzy {
                        self.fuzzy_match(token.as_ref(), term.as_ref())
                    } else {
                        token == term
                    };

                    if matches {
                        term_positions.entry(term.clone()).or_insert_with(Vec::new).push(i);
                    }
                }
            }

            // Check if all terms found and within proximity
            if term_positions.len() == terms.len() {
                let mut found_proximity = false;

                // Check all combinations of positions
                for positions_combo in self.cartesian_product(&term_positions) {
                    let min_pos = *positions_combo.iter().min().unwrap_or(&0);
                    let max_pos = *positions_combo.iter().max().unwrap_or(&0);
                    
                    if (max_pos - min_pos) <= distance as usize {
                        found_proximity = true;
                        break;
                    }
                }

                if found_proximity {
                    let result = SearchResult {
                        message: message.clone(),
                        relevance_score: if fuzzy { 0.7 } else { 0.9 },
                        matching_terms: terms.to_vec(),
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

        results
    }

    /// Simple fuzzy string matching using Levenshtein distance
    pub(super) fn fuzzy_match(&self, text: &str, pattern: &str) -> bool {
        let max_distance = (pattern.len() / 3).max(1);
        let distance = self.levenshtein_distance(text, pattern);
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


}