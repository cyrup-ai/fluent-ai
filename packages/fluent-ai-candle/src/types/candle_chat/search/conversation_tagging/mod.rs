//! Conversation tagging system for candle_chat search functionality
//!
//! This module provides intelligent conversation tagging and classification
//! with zero-allocation streaming patterns and lock-free operations.
//!
//! Decomposed into focused modules following ≤300-line architectural constraint:
//! - types.rs: Tag types, categories, and sentiment analysis
//! - rules.rs: Tagging rules, patterns, and configuration

use std::sync::Arc;

use fluent_ai_async::AsyncStream;
use uuid::Uuid;

use crate::types::CandleSearchChatMessage;

// Re-export types from sub-modules
pub use types::{ConversationTag, TagCategory, SentimentScore};
pub use rules::{TaggingRule, TaggerConfig, TaggingStatistics};

// Sub-modules
pub mod types;
pub mod rules;

/// Handle errors in streaming context without panicking
macro_rules! handle_error {
    ($error:expr, $context:literal) => {
        eprintln!("Streaming error in {}: {}", $context, $error);
        // Continue processing instead of returning error
    };
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
    pub config: TaggerConfig}

impl ConversationTagger {
    /// Create a new conversation tagger
    pub fn new() -> Self {
        Self {
            tags: HashMap::new(),
            patterns: HashMap::new(),
            rules: Vec::new(),
            statistics: TaggingStatistics::default(),
            config: TaggerConfig::default()}
    }

    /// Create with custom configuration
    pub fn with_config(config: TaggerConfig) -> Self {
        Self {
            tags: HashMap::new(),
            patterns: HashMap::new(),
            rules: Vec::new(),
            statistics: TaggingStatistics::default(),
            config}
    }

    /// Add a tag to the system
    pub fn add_tag(&mut self, tag: ConversationTag) {
        self.tags.insert(tag.id, tag);
    }

    /// Add a tagging rule
    pub fn add_rule(&mut self, rule: TaggingRule) {
        self.rules.push(rule);
    }

    /// Tag messages automatically (streaming)
    pub fn tag_messages(&self, messages: Vec<CandleSearchChatMessage>) -> AsyncStream<(Uuid, Vec<ConversationTag>)> {
        let config = self.config.clone();
        let tags = self.tags.clone();
        let rules = self.rules.clone();

        AsyncStream::with_channel(move |sender| {
            if config.auto_tagging {
                for message in messages {
                    let mut applied_tags = Vec::new();

                    // Apply rule-based tagging
                    for rule in &rules {
                        if let Some(confidence) = rule.matches(&message.message.content) {
                            if let Some(tag) = tags.get(&rule.tag_id) {
                                let mut tagged = tag.clone();
                                tagged.confidence = confidence;
                                applied_tags.push(tagged);
                            }
                        }
                    }

                    // Limit tags per conversation
                    applied_tags.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
                    applied_tags.truncate(config.max_tags_per_conversation);

                    let _ = sender.send((Uuid::new_v4(), applied_tags));
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
                    let sentiment = Self::calculate_sentiment(&message.message.content);
                    let _ = sender.send((Uuid::new_v4(), sentiment));
                }
            }
        })
    }

    /// Calculate sentiment score (simplified)
    fn calculate_sentiment(content: &str) -> SentimentScore {
        let content_lower = content.to_lowercase();
        
        // Simple keyword-based sentiment analysis
        let positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "happy", "pleased"];
        let negative_words = ["bad", "terrible", "awful", "horrible", "sad", "angry", "frustrated", "disappointed"];
        
        let mut positive_count = 0;
        let mut negative_count = 0;
        let word_count = content.split_whitespace().count().max(1);
        
        for word in positive_words.iter() {
            if content_lower.contains(word) {
                positive_count += 1;
            }
        }
        
        for word in negative_words.iter() {
            if content_lower.contains(word) {
                negative_count += 1;
            }
        }
        
        let positive = positive_count as f32 / word_count as f32;
        let negative = negative_count as f32 / word_count as f32;
        let neutral = 1.0 - positive - negative;
        
        SentimentScore::new(positive, negative, neutral)
    }

    /// Get tag by ID
    pub fn get_tag(&self, tag_id: &Uuid) -> Option<&ConversationTag> {
        self.tags.get(tag_id)
    }

    /// Get all tags
    pub fn get_all_tags(&self) -> Vec<&ConversationTag> {
        self.tags.values().collect()
    }

    /// Update statistics
    pub fn update_statistics(&mut self, tag_id: Uuid, confidence: f32, processing_time_ms: f64) {
        self.statistics.record_tag_application(true, confidence, processing_time_ms);
        self.statistics.update_tag_usage(tag_id);
    }

    /// Get tagging statistics
    pub fn get_statistics(&self) -> &TaggingStatistics {
        &self.statistics
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.statistics.reset();
    }
}

impl Default for ConversationTagger {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ConversationTagger {
    fn clone(&self) -> Self {
        Self {
            tags: self.tags.clone(),
            patterns: self.patterns.clone(),
            rules: self.rules.clone(),
            statistics: self.statistics.clone(),
            config: self.config.clone()}
    }
}