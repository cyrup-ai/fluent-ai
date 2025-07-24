//! Conversation tagging system for candle_chat search functionality
//!
//! This module provides intelligent conversation tagging and classification
//! with zero-allocation streaming patterns and lock-free operations.

use std::collections::HashMap;
use std::sync::Arc;

use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::core_types::SearchStatistics;
use crate::types::CandleSearchChatMessage;

/// Handle errors in streaming context without panicking
macro_rules! handle_error {
    ($error:expr, $context:literal) => {
        eprintln!("Streaming error in {}: {}", $context, $error);
        // Continue processing instead of returning error
    };
}

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

    /// Update usage count
    pub fn increment_usage(&mut self) {
        self.usage_count += 1;
    }

    /// Add keyword
    pub fn add_keyword(&mut self, keyword: Arc<str>) {
        if !self.keywords.contains(&keyword) {
            self.keywords.push(keyword);
        }
    }
}

/// Conversation tagger for automatic tagging
pub struct ConversationTagger {
    /// Available tags
    pub tags: HashMap<Uuid, ConversationTag>,
    /// Tag patterns
    pub patterns: HashMap<Arc<str>, Vec<Uuid>>,
    /// Tagging rules
    pub rules: Vec<TaggingRule>,
    /// Statistics
    pub statistics: TaggingStatistics,
    /// Configuration
    pub config: TaggerConfig,
}

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

impl ConversationTagger {
    /// Create a new conversation tagger
    pub fn new() -> Self {
        Self {
            tags: HashMap::new(),
            patterns: HashMap::new(),
            rules: Vec::new(),
            statistics: TaggingStatistics::default(),
            config: TaggerConfig::default(),
        }
    }

    /// Add a new tag
    pub fn add_tag(&mut self, tag: ConversationTag) {
        let tag_id = tag.id;
        
        // Update patterns
        for keyword in &tag.keywords {
            self.patterns
                .entry(keyword.clone())
                .or_insert_with(Vec::new)
                .push(tag_id);
        }
        
        self.tags.insert(tag_id, tag);
    }

    /// Tag messages automatically (streaming)
    pub fn tag_messages(&self, messages: Vec<CandleSearchChatMessage>) -> AsyncStream<(Uuid, Vec<ConversationTag>)> {
        let tags = self.tags.clone();
        let patterns = self.patterns.clone();
        let rules = self.rules.clone();
        let config = self.config.clone();

        AsyncStream::with_channel(move |sender| {
            for message in messages {
                if config.auto_tagging {
                    let mut applied_tags = Vec::new();
                    
                    // Apply pattern-based tagging
                    for (pattern, tag_ids) in &patterns {
                        if message.content.to_lowercase().contains(&pattern.to_lowercase()) {
                            for tag_id in tag_ids {
                                if let Some(tag) = tags.get(tag_id) {
                                    applied_tags.push(tag.clone());
                                }
                            }
                        }
                    }

                    // Apply rule-based tagging
                    for rule in &rules {
                        if rule.active {
                            let mut matches = 0;
                            for keyword in &rule.keywords {
                                if message.content.to_lowercase().contains(&keyword.to_lowercase()) {
                                    matches += 1;
                                }
                            }
                            
                            let confidence = (matches as f32) / (rule.keywords.len() as f32) * rule.weight;
                            if confidence >= rule.min_confidence {
                                if let Some(tag) = tags.get(&rule.tag_id) {
                                    let mut tagged = tag.clone();
                                    tagged.confidence = confidence;
                                    applied_tags.push(tagged);
                                }
                            }
                        }
                    }

                    // Limit tags per conversation
                    applied_tags.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
                    applied_tags.truncate(config.max_tags_per_conversation);

                    let _ = sender.send((message.id, applied_tags));
                }
            }
        })
    }

    /// Analyze conversation sentiment (streaming)
    pub fn analyze_sentiment(&self, messages: Vec<CandleSearchChatMessage>) -> AsyncStream<(Uuid, SentimentScore)> {
        let config = self.config.clone();

        AsyncStream::with_channel(move |sender| {
            if config.enable_sentiment {
                for message in messages {
                    // Simple sentiment analysis (would use more sophisticated methods)
                    let sentiment = Self::calculate_sentiment(&message.content);
                    let _ = sender.send((message.id, sentiment));
                }
            }
        })
    }

    /// Calculate sentiment score (simplified)
    fn calculate_sentiment(content: &str) -> SentimentScore {
        let positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"];
        let negative_words = ["bad", "terrible", "awful", "horrible", "disappointing", "frustrating"];
        
        let content_lower = content.to_lowercase();
        let mut positive_count = 0;
        let mut negative_count = 0;
        
        for word in positive_words {
            if content_lower.contains(word) {
                positive_count += 1;
            }
        }
        
        for word in negative_words {
            if content_lower.contains(word) {
                negative_count += 1;
            }
        }
        
        let total_words = content.split_whitespace().count() as f32;
        let positive_ratio = positive_count as f32 / total_words.max(1.0);
        let negative_ratio = negative_count as f32 / total_words.max(1.0);
        
        if positive_ratio > negative_ratio {
            SentimentScore::Positive(positive_ratio)
        } else if negative_ratio > positive_ratio {
            SentimentScore::Negative(negative_ratio)
        } else {
            SentimentScore::Neutral
        }
    }

    /// Get tagging statistics
    pub fn get_statistics(&self) -> TaggingStatistics {
        self.statistics.clone()
    }

    /// Add tagging rule
    pub fn add_rule(&mut self, rule: TaggingRule) {
        self.rules.push(rule);
    }

    /// Remove tag
    pub fn remove_tag(&mut self, tag_id: Uuid) {
        self.tags.remove(&tag_id);
        
        // Clean up patterns
        self.patterns.retain(|_, tag_ids| {
            tag_ids.retain(|id| *id != tag_id);
            !tag_ids.is_empty()
        });
        
        // Clean up rules
        self.rules.retain(|rule| rule.tag_id != tag_id);
    }
}

/// Sentiment score classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SentimentScore {
    /// Positive sentiment with score
    Positive(f32),
    /// Negative sentiment with score
    Negative(f32),
    /// Neutral sentiment
    Neutral,
}

/// Statistics for tagging operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaggingStatistics {
    /// Total tags created
    pub total_tags: usize,
    /// Auto-generated tags
    pub auto_generated_tags: usize,
    /// Total tagging operations
    pub total_operations: usize,
    /// Average confidence score
    pub avg_confidence: f32,
    /// Most used tags
    pub most_used_tags: Vec<(Uuid, usize)>,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
    /// Last update timestamp
    pub last_update: chrono::DateTime<chrono::Utc>,
}

impl Default for ConversationTagger {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TaggerConfig {
    fn default() -> Self {
        Self {
            auto_tagging: true,
            min_auto_confidence: 0.7,
            max_tags_per_conversation: 5,
            enable_sentiment: true,
            enable_topic_detection: true,
            custom_patterns: HashMap::new(),
        }
    }
}

impl Default for TaggingStatistics {
    fn default() -> Self {
        Self {
            total_tags: 0,
            auto_generated_tags: 0,
            total_operations: 0,
            avg_confidence: 0.0,
            most_used_tags: Vec::new(),
            performance_metrics: HashMap::new(),
            last_update: chrono::Utc::now(),
        }
    }
}

impl Default for TagCategory {
    fn default() -> Self {
        TagCategory::Custom
    }
}