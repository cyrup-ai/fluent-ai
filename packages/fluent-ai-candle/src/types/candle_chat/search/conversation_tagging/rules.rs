//! Tagging rules and pattern matching
//!
//! This module implements the rule-based tagging system with pattern
//! matching and automatic classification capabilities.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Tagging rule for automatic classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaggingRule {
    /// Rule ID
    pub id: Uuid,
    /// Rule name
    pub name: String,
    /// Keywords to match
    pub keywords: Vec<Arc<str>>,
    /// Tag to apply
    pub tag_id: Uuid,
    /// Minimum confidence threshold
    pub min_confidence: f32,
    /// Rule weight
    pub weight: f32,
    /// Active flag
    pub active: bool,
}

impl TaggingRule {
    /// Create a new tagging rule
    pub fn new(name: String, keywords: Vec<Arc<str>>, tag_id: Uuid) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            keywords,
            tag_id,
            min_confidence: 0.5,
            weight: 1.0,
            active: true,
        }
    }

    /// Check if rule matches content
    pub fn matches(&self, content: &str) -> Option<f32> {
        if !self.active {
            return None;
        }

        let content_lower = content.to_lowercase();
        let mut matches = 0;
        let total_keywords = self.keywords.len();

        for keyword in &self.keywords {
            if content_lower.contains(keyword.to_lowercase().as_str()) {
                matches += 1;
            }
        }

        if matches > 0 {
            let confidence = (matches as f32 / total_keywords as f32) * self.weight;
            if confidence >= self.min_confidence {
                Some(confidence.min(1.0))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Set confidence threshold
    pub fn with_confidence(mut self, min_confidence: f32) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// Set rule weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Activate/deactivate rule
    pub fn set_active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }
}

/// Configuration for conversation tagger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaggerConfig {
    /// Enable automatic tagging
    pub auto_tagging: bool,
    /// Minimum confidence for auto tags
    pub min_auto_confidence: f32,
    /// Maximum tags per conversation
    pub max_tags_per_conversation: usize,
    /// Enable sentiment analysis
    pub enable_sentiment: bool,
    /// Enable topic detection
    pub enable_topic_detection: bool,
    /// Custom patterns
    pub custom_patterns: HashMap<String, String>,
}

impl Default for TaggerConfig {
    fn default() -> Self {
        Self {
            auto_tagging: true,
            min_auto_confidence: 0.6,
            max_tags_per_conversation: 10,
            enable_sentiment: true,
            enable_topic_detection: true,
            custom_patterns: HashMap::new(),
        }
    }
}

/// Statistics for tagging operations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TaggingStatistics {
    /// Total messages tagged
    pub total_messages_tagged: usize,
    /// Total tags applied
    pub total_tags_applied: usize,
    /// Auto-generated tags count
    pub auto_tags_count: usize,
    /// Manual tags count
    pub manual_tags_count: usize,
    /// Average confidence score
    pub average_confidence: f32,
    /// Tag category distribution
    pub category_distribution: HashMap<String, usize>,
    /// Most used tags
    pub most_used_tags: Vec<(Uuid, usize)>,
    /// Processing time statistics
    pub avg_processing_time_ms: f64,
}

impl TaggingStatistics {
    /// Update statistics with new tag application
    pub fn record_tag_application(&mut self, auto_generated: bool, confidence: f32, processing_time_ms: f64) {
        self.total_tags_applied += 1;
        
        if auto_generated {
            self.auto_tags_count += 1;
        } else {
            self.manual_tags_count += 1;
        }

        // Update average confidence
        let total_confidence = self.average_confidence * (self.total_tags_applied - 1) as f32;
        self.average_confidence = (total_confidence + confidence) / self.total_tags_applied as f32;

        // Update average processing time
        let total_time = self.avg_processing_time_ms * (self.total_tags_applied - 1) as f64;
        self.avg_processing_time_ms = (total_time + processing_time_ms) / self.total_tags_applied as f64;
    }

    /// Record message tagging
    pub fn record_message_tagged(&mut self) {
        self.total_messages_tagged += 1;
    }

    /// Update category distribution
    pub fn update_category_distribution(&mut self, category: String) {
        *self.category_distribution.entry(category).or_insert(0) += 1;
    }

    /// Update tag usage
    pub fn update_tag_usage(&mut self, tag_id: Uuid) {
        // Find and update existing tag usage or add new entry
        if let Some(pos) = self.most_used_tags.iter().position(|(id, _)| *id == tag_id) {
            self.most_used_tags[pos].1 += 1;
        } else {
            self.most_used_tags.push((tag_id, 1));
        }

        // Sort by usage count and keep top 10
        self.most_used_tags.sort_by(|a, b| b.1.cmp(&a.1));
        self.most_used_tags.truncate(10);
    }

    /// Get tagging efficiency
    pub fn get_efficiency(&self) -> f32 {
        if self.total_messages_tagged == 0 {
            0.0
        } else {
            self.total_tags_applied as f32 / self.total_messages_tagged as f32
        }
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}