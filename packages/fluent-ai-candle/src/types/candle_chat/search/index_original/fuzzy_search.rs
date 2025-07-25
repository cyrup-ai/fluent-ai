//! Fuzzy search implementation with Levenshtein distance
//!
//! This module provides fuzzy matching capabilities for approximate string
//! matching with configurable distance thresholds.

use std::sync::Arc;
use std::collections::HashMap;

use super::super::types::SearchResult;
use super::types::ChatSearchIndex;

impl ChatSearchIndex {
    /// Exact search for a term
    pub(super) fn exact_search(&self, term: &Arc<str>) -> Vec<SearchResult> {
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
    pub(super) fn fuzzy_search(&self, term: &Arc<str>) -> Vec<SearchResult> {
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
    pub(super) fn fuzzy_match(&self, text: &str, pattern: &str) -> bool {
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
    pub(super) fn intersect_results(
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
    pub(super) fn check_proximity(&self, tokens: &[Arc<str>], terms: &[Arc<str>], distance: u32) -> bool {
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
    pub(super) fn highlight_text(&self, text: &str, term: &str) -> String {
        text.replace(term, &format!("<mark>{}</mark>", term))
    }

    /// Highlight multiple terms in text
    pub(super) fn highlight_terms(&self, text: &str, terms: &[Arc<str>]) -> String {
        let mut highlighted = text.to_string();
        for term in terms {
            highlighted = highlighted.replace(term.as_ref(), &format!("<mark>{}</mark>", term));
        }
        highlighted
    }
}