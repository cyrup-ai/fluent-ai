//! Result ranking and scoring functionality
//!
//! This module provides advanced result ranking algorithms including
//! relevance scoring, result filtering, and performance optimization.

use std::collections::HashMap;
use std::sync::Arc;

use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};

use super::types::{SearchResult, MatchPosition, MatchType};

/// Result ranker for scoring and sorting search results
pub struct ResultRanker {
    /// Ranking algorithm configuration
    pub config: RankingConfig,
    /// Performance statistics
    pub stats: HashMap<String, f64>,
}

/// Configuration for ranking algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingConfig {
    /// Weight for term frequency scoring
    pub tf_weight: f32,
    /// Weight for inverse document frequency
    pub idf_weight: f32,
    /// Weight for recency scoring
    pub recency_weight: f32,
    /// Weight for exact match bonus
    pub exact_match_bonus: f32,
    /// Weight for phrase match bonus
    pub phrase_match_bonus: f32,
    /// Maximum results to return
    pub max_results: usize,
    /// Minimum score threshold
    pub min_score_threshold: f32,
}

impl ResultRanker {
    /// Create a new result ranker
    pub fn new(config: RankingConfig) -> Self {
        Self {
            config,
            stats: HashMap::new(),
        }
    }

    /// Rank and score search results (streaming)
    pub fn rank_results(&self, results: Vec<SearchResult>) -> AsyncStream<SearchResult> {
        let config = self.config.clone();
        
        AsyncStream::with_channel(move |sender| {
            let mut scored_results = results;
            
            // Apply scoring
            for result in &mut scored_results {
                result.score = Self::calculate_relevance_score(result, &config);
            }
            
            // Sort by score (descending)
            scored_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            
            // Apply filters
            scored_results.retain(|r| r.score >= config.min_score_threshold);
            
            // Limit results
            scored_results.truncate(config.max_results);
            
            // Send ranked results
            for result in scored_results {
                let _ = sender.send(result);
            }
        })
    }

    /// Calculate relevance score for a search result
    fn calculate_relevance_score(result: &SearchResult, config: &RankingConfig) -> f32 {
        let mut score = 0.0;
        
        // Base content relevance (simplified)
        score += Self::calculate_content_relevance(&result.message.content) * config.tf_weight;
        
        // Recency bonus
        let age_hours = chrono::Utc::now()
            .signed_duration_since(result.message.timestamp)
            .num_hours() as f32;
        let recency_score = (1.0 / (1.0 + age_hours / 24.0)).max(0.1);
        score += recency_score * config.recency_weight;
        
        // Match type bonuses
        for match_pos in &result.match_positions {
            match match_pos.match_type {
                MatchType::Exact => score += config.exact_match_bonus,
                MatchType::Fuzzy => score += config.exact_match_bonus * 0.8,
                MatchType::Regex => score += config.exact_match_bonus * 0.9,
                MatchType::Stemmed => score += config.exact_match_bonus * 0.7,
                MatchType::Synonym => score += config.exact_match_bonus * 0.6,
                MatchType::Phonetic => score += config.exact_match_bonus * 0.5,
            }
        }
        
        score.max(0.0).min(10.0) // Clamp between 0 and 10
    }

    /// Calculate content relevance score
    fn calculate_content_relevance(content: &str) -> f32 {
        // Simple relevance calculation based on content length and word count
        let word_count = content.split_whitespace().count() as f32;
        let char_count = content.len() as f32;
        
        // Prefer moderate length content
        let length_score = if word_count < 10.0 {
            word_count / 10.0
        } else if word_count > 100.0 {
            1.0 - ((word_count - 100.0) / 200.0).min(0.8)
        } else {
            1.0
        };
        
        length_score * (1.0 + (char_count / 1000.0).min(0.5))
    }

    /// Filter results by various criteria (streaming)
    pub fn filter_results(
        &self,
        results: Vec<SearchResult>,
        filters: &ResultFilters,
    ) -> AsyncStream<SearchResult> {
        let filters = filters.clone();
        
        AsyncStream::with_channel(move |sender| {
            for result in results {
                if Self::passes_filters(&result, &filters) {
                    let _ = sender.send(result);
                }
            }
        })
    }

    /// Check if a result passes all filters
    fn passes_filters(result: &SearchResult, filters: &ResultFilters) -> bool {
        // Score filter
        if let Some(min_score) = filters.min_score {
            if result.score < min_score {
                return false;
            }
        }
        
        // Date range filter
        if let Some(date_range) = &filters.date_range {
            if result.message.timestamp < date_range.start || result.message.timestamp > date_range.end {
                return false;
            }
        }
        
        // Role filter
        if let Some(role) = &filters.role_filter {
            if result.message.role != *role {
                return false;
            }
        }
        
        // Content length filter
        if let Some(min_length) = filters.min_content_length {
            if result.message.content.len() < min_length {
                return false;
            }
        }
        
        if let Some(max_length) = filters.max_content_length {
            if result.message.content.len() > max_length {
                return false;
            }
        }
        
        true
    }

    /// Get ranking statistics
    pub fn get_stats(&self) -> HashMap<String, f64> {
        self.stats.clone()
    }
}

/// Filters for search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultFilters {
    /// Minimum score threshold
    pub min_score: Option<f32>,
    /// Date range filter
    pub date_range: Option<DateRange>,
    /// Role filter
    pub role_filter: Option<crate::chat::message::MessageRole>,
    /// Minimum content length
    pub min_content_length: Option<usize>,
    /// Maximum content length
    pub max_content_length: Option<usize>,
    /// Custom filters
    pub custom_filters: HashMap<String, String>,
}

/// Date range for filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    /// Start date (inclusive)
    pub start: chrono::DateTime<chrono::Utc>,
    /// End date (inclusive)
    pub end: chrono::DateTime<chrono::Utc>,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            tf_weight: 1.0,
            idf_weight: 0.8,
            recency_weight: 0.3,
            exact_match_bonus: 0.5,
            phrase_match_bonus: 0.7,
            max_results: 100,
            min_score_threshold: 0.1,
        }
    }
}

impl Default for ResultRanker {
    fn default() -> Self {
        Self::new(RankingConfig::default())
    }
}

impl Default for ResultFilters {
    fn default() -> Self {
        Self {
            min_score: None,
            date_range: None,
            role_filter: None,
            min_content_length: None,
            max_content_length: None,
            custom_filters: HashMap::new(),
        }
    }
}