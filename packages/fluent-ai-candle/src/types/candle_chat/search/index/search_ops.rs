//! Search operations and algorithms
//!
//! Contains specific search algorithms including exact search, fuzzy search,
//! phrase search, proximity search, and result sorting.

use std::sync::Arc;

use super::core::ChatSearchIndex;
use super::super::types::SearchResult;

impl ChatSearchIndex {
    /// Search for documents containing exact term
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
                            self.highlight_text(&message.value().content, term),
                        )),
                        tags: vec![],
                        context: vec![],
                        match_positions: vec![],
                        metadata: None};
                    results.push(result);
                }
            }
        }

        results
    }

    /// Search for documents with fuzzy matching
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
                excluded_docs.insert(result.message.id.clone());
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
                    metadata: None};
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
            let content = message.content.to_lowercase();

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
                    metadata: None};
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
            let tokens = self.tokenize_with_simd(&message.content);

            if self.check_proximity(&tokens, terms, distance) {
                let relevance_score = if fuzzy { 0.7 } else { 0.9 };
                let result = SearchResult {
                    message: message.clone(),
                    relevance_score,
                    matching_terms: terms.to_vec(),
                    highlighted_content: Some(Arc::from(
                        self.highlight_terms(&message.content, terms),
                    )),
                    tags: vec![],
                    context: vec![],
                    match_positions: vec![],
                    metadata: None};
                results.push(result);
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