//! Search result ranking algorithms
//!
//! This module implements various ranking algorithms for search results
//! including TF-IDF, BM25, and custom scoring functions.

use std::collections::HashMap;

use super::types::{SearchResult, ProcessedQuery, RankingAlgorithm};

/// Result ranker for scoring and sorting search results
pub struct ResultRanker {
    /// Ranking algorithm to use
    algorithm: RankingAlgorithm,
    /// Field boost weights
    field_boosts: HashMap<String, f32>,
}

impl ResultRanker {
    /// Create a new result ranker
    pub fn new() -> Self {
        Self {
            algorithm: RankingAlgorithm::Bm25,
            field_boosts: HashMap::new(),
        }
    }

    /// Create ranker with specific algorithm
    pub fn with_algorithm(algorithm: RankingAlgorithm) -> Self {
        Self {
            algorithm,
            field_boosts: HashMap::new(),
        }
    }

    /// Set ranking algorithm
    pub fn set_algorithm(&mut self, algorithm: RankingAlgorithm) {
        self.algorithm = algorithm;
    }

    /// Add field boost weight
    pub fn add_field_boost(&mut self, field: String, boost: f32) {
        self.field_boosts.insert(field, boost);
    }

    /// Remove field boost
    pub fn remove_field_boost(&mut self, field: &str) {
        self.field_boosts.remove(field);
    }

    /// Rank search results by relevance (synchronous for streams-only architecture)
    pub fn rank_results_sync(
        &self,
        mut results: Vec<SearchResult>,
        query: &ProcessedQuery,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        // Apply ranking algorithm
        match self.algorithm {
            RankingAlgorithm::TfIdf => {
                self.apply_tfidf_ranking(&mut results, query)?;
            }
            RankingAlgorithm::Bm25 => {
                self.apply_bm25_ranking(&mut results, query)?;
            }
            RankingAlgorithm::Custom => {
                self.apply_custom_ranking(&mut results, query)?;
            }
        }

        // Sort by relevance score (descending)
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    /// Apply TF-IDF ranking
    fn apply_tfidf_ranking(
        &self,
        results: &mut [SearchResult],
        query: &ProcessedQuery,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let total_docs = results.len() as f32;
        
        // Calculate document frequencies for query terms
        let mut doc_frequencies = HashMap::new();
        for term in &query.terms {
            let df = results.iter()
                .filter(|result| {
                    result.message.message.content
                        .to_lowercase()
                        .contains(&term.to_lowercase())
                })
                .count() as f32;
            doc_frequencies.insert(term.clone(), df.max(1.0));
        }

        // Calculate TF-IDF scores
        for result in results.iter_mut() {
            let mut score = 0.0;
            let content = &result.message.message.content;
            let tokens = self.tokenize_simple(content);
            let doc_length = tokens.len() as f32;

            for term in &query.terms {
                // Calculate term frequency (TF)
                let tf = tokens.iter()
                    .filter(|token| token.to_lowercase() == term.to_lowercase())
                    .count() as f32;
                
                if tf > 0.0 {
                    // Calculate inverse document frequency (IDF)
                    let df = doc_frequencies.get(term).unwrap_or(&1.0);
                    let idf = (total_docs / df).ln();
                    
                    // TF-IDF formula: (tf / doc_length) * idf
                    let tf_normalized = tf / doc_length;
                    score += tf_normalized * idf;
                }
            }

            // Apply field boosts
            score = self.apply_field_boosts(score, result);
            
            result.relevance_score = score;
        }

        Ok(())
    }

    /// Apply BM25 ranking
    fn apply_bm25_ranking(
        &self,
        results: &mut [SearchResult],
        query: &ProcessedQuery,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let k1 = 1.2; // Controls term frequency scaling
        let b = 0.75; // Controls document length normalization
        
        let total_docs = results.len() as f32;
        
        // Calculate average document length
        let avg_doc_length = results.iter()
            .map(|result| self.tokenize_simple(&result.message.message.content).len())
            .sum::<usize>() as f32 / total_docs;

        // Calculate document frequencies for query terms
        let mut doc_frequencies = HashMap::new();
        for term in &query.terms {
            let df = results.iter()
                .filter(|result| {
                    result.message.message.content
                        .to_lowercase()
                        .contains(&term.to_lowercase())
                })
                .count() as f32;
            doc_frequencies.insert(term.clone(), df.max(1.0));
        }

        // Calculate BM25 scores
        for result in results.iter_mut() {
            let mut score = 0.0;
            let content = &result.message.message.content;
            let tokens = self.tokenize_simple(content);
            let doc_length = tokens.len() as f32;

            for term in &query.terms {
                // Calculate term frequency (TF)
                let tf = tokens.iter()
                    .filter(|token| token.to_lowercase() == term.to_lowercase())
                    .count() as f32;
                
                if tf > 0.0 {
                    // Calculate inverse document frequency (IDF)
                    let df = doc_frequencies.get(term).unwrap_or(&1.0);
                    let idf = ((total_docs - df + 0.5) / (df + 0.5)).ln();
                    
                    // BM25 formula
                    let tf_component = (tf * (k1 + 1.0)) / 
                        (tf + k1 * (1.0 - b + b * (doc_length / avg_doc_length)));
                    
                    score += idf * tf_component;
                }
            }

            // Apply field boosts
            score = self.apply_field_boosts(score, result);
            
            result.relevance_score = score;
        }

        Ok(())
    }

    /// Apply custom ranking algorithm
    fn apply_custom_ranking(
        &self,
        results: &mut [SearchResult],
        query: &ProcessedQuery,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Custom scoring that combines multiple factors
        for result in results.iter_mut() {
            let mut score = 0.0;
            let content = &result.message.message.content;
            let content_lower = content.to_lowercase();

            // Factor 1: Term matching (basic relevance)
            let matching_terms = query.terms.iter()
                .filter(|term| content_lower.contains(&term.to_lowercase()))
                .count() as f32;
            let term_coverage = matching_terms / query.terms.len() as f32;
            score += term_coverage * 10.0;

            // Factor 2: Exact phrase bonus
            let query_phrase = query.terms.iter()
                .map(|t| t.as_ref())
                .collect::<Vec<_>>()
                .join(" ");
            if content_lower.contains(&query_phrase.to_lowercase()) {
                score += 5.0;
            }

            // Factor 3: Term position bonus (earlier matches score higher)
            for term in &query.terms {
                if let Some(pos) = content_lower.find(&term.to_lowercase()) {
                    let position_bonus = 1.0 / (1.0 + pos as f32 / content.len() as f32);
                    score += position_bonus * 2.0;
                }
            }

            // Factor 4: Document length penalty (prefer concise, relevant content)
            let length_penalty = 1.0 / (1.0 + content.len() as f32 / 1000.0);
            score *= length_penalty;

            // Factor 5: Message role bonus
            match result.message.message.role {
                crate::chat::message::MessageRole::User => score *= 1.2,
                crate::chat::message::MessageRole::Assistant => score *= 1.0,
                crate::chat::message::MessageRole::System => score *= 0.8,
                crate::chat::message::MessageRole::Tool => score *= 0.9,
            }

            // Apply field boosts
            score = self.apply_field_boosts(score, result);
            
            result.relevance_score = score;
        }

        Ok(())
    }

    /// Apply field boost weights to score
    fn apply_field_boosts(&self, mut score: f32, result: &SearchResult) -> f32 {
        // Apply content boost if configured
        if let Some(&boost) = self.field_boosts.get("content") {
            score *= boost;
        }

        // Apply role-based boosts if configured
        let role_str = match result.message.message.role {
            crate::chat::message::MessageRole::User => "user",
            crate::chat::message::MessageRole::Assistant => "assistant", 
            crate::chat::message::MessageRole::System => "system",
            crate::chat::message::MessageRole::Tool => "tool",
        };
        
        if let Some(&boost) = self.field_boosts.get(role_str) {
            score *= boost;
        }

        score
    }

    /// Simple tokenization helper
    fn tokenize_simple(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|word| {
                word.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|token| !token.is_empty())
            .collect()
    }

    /// Get current algorithm
    pub fn get_algorithm(&self) -> &RankingAlgorithm {
        &self.algorithm
    }

    /// Get field boosts
    pub fn get_field_boosts(&self) -> &HashMap<String, f32> {
        &self.field_boosts
    }

    /// Clear all field boosts
    pub fn clear_field_boosts(&mut self) {
        self.field_boosts.clear();
    }

    /// Get boost for specific field
    pub fn get_field_boost(&self, field: &str) -> Option<f32> {
        self.field_boosts.get(field).copied()
    }
}

impl Default for ResultRanker {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ResultRanker {
    fn clone(&self) -> Self {
        Self {
            algorithm: self.algorithm.clone(),
            field_boosts: self.field_boosts.clone(),
        }
    }
}

impl std::fmt::Debug for ResultRanker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResultRanker")
            .field("algorithm", &self.algorithm)
            .field("field_boosts", &self.field_boosts)
            .finish()
    }
}