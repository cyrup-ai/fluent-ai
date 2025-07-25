//! Search utility functions
//!
//! Contains helper functions for text highlighting, result intersection,
//! and proximity checking.

use std::sync::Arc;

use super::core::ChatSearchIndex;
use super::super::types::SearchResult;

impl ChatSearchIndex {
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

    /// Intersect two result sets
    pub(super) fn intersect_results(
        &self,
        results1: Vec<SearchResult>,
        results2: Vec<SearchResult>,
    ) -> Vec<SearchResult> {
        let mut intersection = Vec::new();
        let ids2: std::collections::HashSet<_> = results2
            .iter()
            .map(|r| r.message.id.clone())
            .collect();

        for result in results1 {
            if ids2.contains(&result.message.id.clone()) {
                intersection.push(result);
            }
        }

        intersection
    }

    /// Check proximity of terms in token list
    pub(super) fn check_proximity(&self, tokens: &[Arc<str>], terms: &[Arc<str>], distance: u32) -> bool {
        let mut positions: std::collections::HashMap<Arc<str>, Vec<usize>> = std::collections::HashMap::new();

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

    /// Generate cartesian product of position vectors
    pub(super) fn cartesian_product(&self, term_positions: &std::collections::HashMap<Arc<str>, Vec<usize>>) -> Vec<Vec<usize>> {
        let mut result = vec![vec![]];
        
        for positions in term_positions.values() {
            let mut new_result = Vec::new();
            for combo in result {
                for &pos in positions {
                    let mut new_combo = combo.clone();
                    new_combo.push(pos);
                    new_result.push(new_combo);
                }
            }
            result = new_result;
        }

        result
    }
}