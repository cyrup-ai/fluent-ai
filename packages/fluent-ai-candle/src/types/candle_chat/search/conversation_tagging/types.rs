//! Conversation tag types and category definitions
//!
//! This module defines all tag-related types and their operations
//! following zero-allocation, streaming-first architecture.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Conversation tag with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTag {
    /// Tag ID
    pub id: Uuid,
    /// Tag name
    pub name: Arc<str>,
    /// Tag description
    pub description: Option<String>,
    /// Tag color (hex)
    pub color: Option<String>,
    /// Tag category
    pub category: TagCategory,
    /// Confidence score
    pub confidence: f32,
    /// Auto-generated flag
    pub auto_generated: bool,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Usage count
    pub usage_count: usize,
    /// Associated keywords
    pub keywords: Vec<Arc<str>>,
    /// Tag metadata
    pub metadata: HashMap<String, String>,
}

/// Tag category classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TagCategory {
    /// Topic-based tag
    Topic,
    /// Sentiment-based tag
    Sentiment,
    /// Priority-based tag
    Priority,
    /// Status-based tag
    Status,
    /// User-defined tag
    Custom,
    /// System-generated tag
    System,
    /// Temporal tag
    Temporal,
    /// Context tag
    Context,
}

impl ConversationTag {
    /// Create a new conversation tag
    pub fn new(name: Arc<str>, category: TagCategory) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            description: None,
            color: None,
            category,
            confidence: 1.0,
            auto_generated: false,
            created_at: chrono::Utc::now(),
            usage_count: 0,
            keywords: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create with confidence score
    pub fn with_confidence(name: Arc<str>, category: TagCategory, confidence: f32) -> Self {
        let mut tag = Self::new(name, category);
        tag.confidence = confidence;
        tag.auto_generated = true;
        tag
    }

    /// Add keyword to tag
    pub fn add_keyword(&mut self, keyword: Arc<str>) {
        if !self.keywords.contains(&keyword) {
            self.keywords.push(keyword);
        }
    }

    /// Set description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Set color
    pub fn with_color(mut self, color: String) -> Self {
        self.color = Some(color);
        self
    }

    /// Increment usage count
    pub fn increment_usage(&mut self) {
        self.usage_count += 1;
    }

    /// Check if tag matches keyword
    pub fn matches_keyword(&self, keyword: &str) -> bool {
        self.keywords.iter().any(|k| k.as_ref() == keyword) ||
        self.name.to_lowercase().contains(&keyword.to_lowercase())
    }

    /// Get tag importance score
    pub fn importance_score(&self) -> f32 {
        self.confidence * (1.0 + self.usage_count as f32 / 100.0)
    }
}

/// Sentiment analysis score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentScore {
    /// Positive sentiment score (0.0 to 1.0)
    pub positive: f32,
    /// Negative sentiment score (0.0 to 1.0)
    pub negative: f32,
    /// Neutral sentiment score (0.0 to 1.0)
    pub neutral: f32,
    /// Overall sentiment (-1.0 to 1.0)
    pub compound: f32,
}

impl SentimentScore {
    /// Create new sentiment score
    pub fn new(positive: f32, negative: f32, neutral: f32) -> Self {
        let compound = positive - negative;
        Self {
            positive,
            negative,
            neutral,
            compound,
        }
    }

    /// Get dominant sentiment
    pub fn dominant_sentiment(&self) -> &'static str {
        if self.compound > 0.1 {
            "positive"
        } else if self.compound < -0.1 {
            "negative"
        } else {
            "neutral"
        }
    }

    /// Get confidence level
    pub fn confidence_level(&self) -> f32 {
        let max_score = self.positive.max(self.negative).max(self.neutral);
        max_score
    }
}

impl Default for SentimentScore {
    fn default() -> Self {
        Self {
            positive: 0.0,
            negative: 0.0,
            neutral: 1.0,
            compound: 0.0,
        }
    }
}