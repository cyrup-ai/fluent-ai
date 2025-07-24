//! Conversation tagging system with lock-free operations
//!
//! This module provides comprehensive tagging functionality for conversations
//! with zero-allocation streaming patterns and atomic operations.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use atomic_counter::{AtomicCounter, ConsistentCounter};
use crossbeam_skiplist::SkipMap;
use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Represents a conversation tag with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTag {
    /// Unique tag identifier
    pub id: Arc<str>,
    /// Human-readable tag name
    pub name: Arc<str>,
    /// Tag description
    pub description: Arc<str>,
    /// Tag category for organization
    pub category: Arc<str>,
    /// Tag color for UI display
    pub color: Option<String>,
    /// Tag icon for UI display
    pub icon: Option<String>,
    /// Whether this tag is system-generated
    pub system_tag: bool,
    /// Parent tag ID for hierarchical tagging
    pub parent_id: Option<Arc<str>>,
    /// Child tag IDs
    pub children: Vec<Arc<str>>,
    /// Tag creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Tag last modified timestamp
    pub modified_at: chrono::DateTime<chrono::Utc>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl ConversationTag {
    /// Create a new conversation tag
    pub fn new(name: Arc<str>, description: Arc<str>, category: Arc<str>) -> Self {
        let now = chrono::Utc::now();
        Self {
            id: Arc::from(Uuid::new_v4().to_string()),
            name,
            description,
            category,
            color: None,
            icon: None,
            system_tag: false,
            parent_id: None,
            children: Vec::new(),
            created_at: now,
            modified_at: now,
            metadata: HashMap::new(),
        }
    }

    /// Check if this tag is a child of another tag
    pub fn is_child_of(&self, parent_id: &Arc<str>) -> bool {
        self.parent_id.as_ref() == Some(parent_id)
    }

    /// Add a child tag
    pub fn add_child(&mut self, child_id: Arc<str>) {
        if !self.children.contains(&child_id) {
            self.children.push(child_id);
            self.modified_at = chrono::Utc::now();
        }
    }

    /// Remove a child tag
    pub fn remove_child(&mut self, child_id: &Arc<str>) {
        self.children.retain(|id| id != child_id);
        self.modified_at = chrono::Utc::now();
    }
}

/// Lock-free conversation tagging system
pub struct ConversationTagger {
    /// Tag storage using lock-free skip map
    pub tags: Arc<SkipMap<Arc<str>, ConversationTag>>,
    /// Conversation to tags mapping
    pub conversation_tags: Arc<SkipMap<Uuid, Vec<Arc<str>>>>,
    /// Tag to conversations mapping (reverse index)
    pub tag_conversations: Arc<SkipMap<Arc<str>, Vec<Uuid>>>,
    /// Tag hierarchy (parent -> children)
    pub tag_hierarchy: Arc<SkipMap<Arc<str>, Vec<Arc<str>>>>,
    /// Statistics counters
    pub stats: Arc<TaggingStatistics>,
    /// Operation counters
    pub tag_counter: ConsistentCounter,
    pub operation_counter: ConsistentCounter,
}

impl ConversationTagger {
    /// Create a new conversation tagger
    pub fn new() -> Self {
        Self {
            tags: Arc::new(SkipMap::new()),
            conversation_tags: Arc::new(SkipMap::new()),
            tag_conversations: Arc::new(SkipMap::new()),
            tag_hierarchy: Arc::new(SkipMap::new()),
            stats: Arc::new(TaggingStatistics::default()),
            tag_counter: ConsistentCounter::new(0),
            operation_counter: ConsistentCounter::new(0),
        }
    }

    /// Create a new tag (streaming)
    pub fn create_tag(
        &self,
        name: Arc<str>,
        description: Arc<str>,
        category: Arc<str>,
    ) -> AsyncStream<Arc<str>> {
        let tags = Arc::clone(&self.tags);
        let tag_counter = self.tag_counter.clone();
        let operation_counter = self.operation_counter.clone();

        AsyncStream::with_channel(move |sender| {
            let tag = ConversationTag::new(name, description, category);
            let tag_id = tag.id.clone();
            
            tags.insert(tag_id.clone(), tag);
            tag_counter.inc();
            operation_counter.inc();
            
            let _ = sender.send(tag_id);
        })
    }

    /// Create a child tag (streaming)
    pub fn create_child_tag(
        &self,
        parent_id: Arc<str>,
        name: Arc<str>,
        description: Arc<str>,
        category: Arc<str>,
    ) -> AsyncStream<Arc<str>> {
        let tags = Arc::clone(&self.tags);
        let tag_hierarchy = Arc::clone(&self.tag_hierarchy);
        let tag_counter = self.tag_counter.clone();
        let operation_counter = self.operation_counter.clone();

        AsyncStream::with_channel(move |sender| {
            let mut tag = ConversationTag::new(name, description, category);
            tag.parent_id = Some(parent_id.clone());
            let tag_id = tag.id.clone();
            
            // Store the tag
            tags.insert(tag_id.clone(), tag);
            
            // Update hierarchy
            let mut children = tag_hierarchy.get(&parent_id)
                .map(|entry| entry.value().clone())
                .unwrap_or_default();
            children.push(tag_id.clone());
            tag_hierarchy.insert(parent_id, children);
            
            tag_counter.inc();
            operation_counter.inc();
            
            let _ = sender.send(tag_id);
        })
    }

    /// Tag a conversation (streaming)
    pub fn tag_conversation(&self, conversation_id: Uuid, tag_id: Arc<str>) -> AsyncStream<bool> {
        let conversation_tags = Arc::clone(&self.conversation_tags);
        let tag_conversations = Arc::clone(&self.tag_conversations);
        let tags = Arc::clone(&self.tags);
        let operation_counter = self.operation_counter.clone();

        AsyncStream::with_channel(move |sender| {
            // Check if tag exists
            let tag_exists = tags.contains_key(&tag_id);
            
            if tag_exists {
                // Add tag to conversation
                let mut conv_tags = conversation_tags.get(&conversation_id)
                    .map(|entry| entry.value().clone())
                    .unwrap_or_default();
                
                if !conv_tags.contains(&tag_id) {
                    conv_tags.push(tag_id.clone());
                    conversation_tags.insert(conversation_id, conv_tags);
                    
                    // Update reverse index
                    let mut tag_convs = tag_conversations.get(&tag_id)
                        .map(|entry| entry.value().clone())
                        .unwrap_or_default();
                    
                    if !tag_convs.contains(&conversation_id) {
                        tag_convs.push(conversation_id);
                        tag_conversations.insert(tag_id, tag_convs);
                    }
                    
                    operation_counter.inc();
                    let _ = sender.send(true);
                } else {
                    let _ = sender.send(false); // Already tagged
                }
            } else {
                let _ = sender.send(false); // Tag doesn't exist
            }
        })
    }

    /// Remove tag from conversation (streaming)
    pub fn untag_conversation(&self, conversation_id: Uuid, tag_id: Arc<str>) -> AsyncStream<bool> {
        let conversation_tags = Arc::clone(&self.conversation_tags);
        let tag_conversations = Arc::clone(&self.tag_conversations);
        let operation_counter = self.operation_counter.clone();

        AsyncStream::with_channel(move |sender| {
            let mut removed = false;
            
            // Remove from conversation tags
            if let Some(entry) = conversation_tags.get(&conversation_id) {
                let mut conv_tags = entry.value().clone();
                if let Some(pos) = conv_tags.iter().position(|id| id == &tag_id) {
                    conv_tags.remove(pos);
                    conversation_tags.insert(conversation_id, conv_tags);
                    removed = true;
                }
            }
            
            // Remove from reverse index
            if removed {
                if let Some(entry) = tag_conversations.get(&tag_id) {
                    let mut tag_convs = entry.value().clone();
                    if let Some(pos) = tag_convs.iter().position(|id| *id == conversation_id) {
                        tag_convs.remove(pos);
                        tag_conversations.insert(tag_id, tag_convs);
                    }
                }
                operation_counter.inc();
            }
            
            let _ = sender.send(removed);
        })
    }

    /// Get tags for a conversation (streaming)
    pub fn get_conversation_tags(&self, conversation_id: Uuid) -> AsyncStream<Vec<ConversationTag>> {
        let conversation_tags = Arc::clone(&self.conversation_tags);
        let tags = Arc::clone(&self.tags);

        AsyncStream::with_channel(move |sender| {
            let tag_ids = conversation_tags.get(&conversation_id)
                .map(|entry| entry.value().clone())
                .unwrap_or_default();
            
            let mut result_tags = Vec::new();
            for tag_id in tag_ids {
                if let Some(tag_entry) = tags.get(&tag_id) {
                    result_tags.push(tag_entry.value().clone());
                }
            }
            
            let _ = sender.send(result_tags);
        })
    }

    /// Get conversations with a specific tag (streaming)
    pub fn get_tagged_conversations(&self, tag_id: Arc<str>) -> AsyncStream<Vec<Uuid>> {
        let tag_conversations = Arc::clone(&self.tag_conversations);

        AsyncStream::with_channel(move |sender| {
            let conversations = tag_conversations.get(&tag_id)
                .map(|entry| entry.value().clone())
                .unwrap_or_default();
            
            let _ = sender.send(conversations);
        })
    }

    /// Delete a tag (streaming)
    pub fn delete_tag(&self, tag_id: Arc<str>) -> AsyncStream<bool> {
        let tags = Arc::clone(&self.tags);
        let conversation_tags = Arc::clone(&self.conversation_tags);
        let tag_conversations = Arc::clone(&self.tag_conversations);
        let tag_hierarchy = Arc::clone(&self.tag_hierarchy);
        let operation_counter = self.operation_counter.clone();

        AsyncStream::with_channel(move |sender| {
            let removed = tags.remove(&tag_id).is_some();
            
            if removed {
                // Remove from all conversations
                if let Some(conversations) = tag_conversations.remove(&tag_id) {
                    for conv_id in conversations.1 {
                        if let Some(entry) = conversation_tags.get(&conv_id) {
                            let mut conv_tags = entry.value().clone();
                            conv_tags.retain(|id| id != &tag_id);
                            conversation_tags.insert(conv_id, conv_tags);
                        }
                    }
                }
                
                // Remove from hierarchy
                tag_hierarchy.remove(&tag_id);
                
                operation_counter.inc();
            }
            
            let _ = sender.send(removed);
        })
    }

    /// Get all tags (streaming)
    pub fn get_all_tags(&self) -> AsyncStream<Vec<ConversationTag>> {
        let tags = Arc::clone(&self.tags);

        AsyncStream::with_channel(move |sender| {
            let all_tags: Vec<ConversationTag> = tags.iter()
                .map(|entry| entry.value().clone())
                .collect();
            
            let _ = sender.send(all_tags);
        })
    }

    /// Get tagging statistics (streaming)
    pub fn get_statistics(&self) -> AsyncStream<TaggingStatistics> {
        let stats = Arc::clone(&self.stats);
        let total_tags = self.tags.len();
        let total_tagged_conversations = self.conversation_tags.len();

        AsyncStream::with_channel(move |sender| {
            let mut current_stats = (*stats).clone();
            current_stats.total_tags = total_tags;
            current_stats.total_tagged_conversations = total_tagged_conversations;
            
            let _ = sender.send(current_stats);
        })
    }
}

impl Default for ConversationTagger {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the tagging system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaggingStatistics {
    /// Total number of tags
    pub total_tags: usize,
    /// Total number of tagged conversations
    pub total_tagged_conversations: usize,
    /// Average tags per conversation
    pub avg_tags_per_conversation: f64,
    /// Most used tags
    pub most_used_tags: Vec<(Arc<str>, usize)>,
    /// Tag usage distribution
    pub tag_usage_distribution: HashMap<Arc<str>, usize>,
    /// System vs user tags ratio
    pub system_tag_ratio: f64,
    /// Tag hierarchy depth statistics
    pub max_hierarchy_depth: usize,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
}

impl Default for TaggingStatistics {
    fn default() -> Self {
        Self {
            total_tags: 0,
            total_tagged_conversations: 0,
            avg_tags_per_conversation: 0.0,
            most_used_tags: Vec::new(),
            tag_usage_distribution: HashMap::new(),
            system_tag_ratio: 0.0,
            max_hierarchy_depth: 0,
            performance_metrics: HashMap::new(),
        }
    }
}