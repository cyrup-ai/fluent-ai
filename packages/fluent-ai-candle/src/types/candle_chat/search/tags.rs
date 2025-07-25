//! Tag management system for conversations
//!
//! This module provides hierarchical tag management with auto-tagging
//! capabilities and lock-free operations for conversation organization.

use std::collections::HashMap;
use std::sync::Arc;

use atomic_counter::{AtomicCounter, ConsistentCounter};
use crossbeam_skiplist::SkipMap;
use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::types::candle_chat::message::types::CandleMessage;
use super::types::StreamCollect;
use super::error::SearchError;

/// Tag with hierarchical structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTag {
    /// Tag ID
    pub id: Arc<str>,
    /// Tag name
    pub name: Arc<str>,
    /// Tag description
    pub description: Arc<str>,
    /// Tag color (hex code)
    pub color: Arc<str>,
    /// Parent tag ID for hierarchy
    pub parent_id: Option<Arc<str>>,
    /// Child tag IDs
    pub child_ids: Vec<Arc<str>>,
    /// Tag category
    pub category: Arc<str>,
    /// Tag metadata
    pub metadata: Option<Arc<str>>,
    /// Creation timestamp
    pub created_at: u64,
    /// Last updated timestamp
    pub updated_at: u64,
    /// Tag usage count
    pub usage_count: u64,
    /// Tag is active
    pub is_active: bool,
}

impl ConversationTag {
    /// Create a new tag
    pub fn new(name: Arc<str>, description: Arc<str>, category: Arc<str>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id: Arc::from(Uuid::new_v4().to_string()),
            name,
            description,
            color: Arc::from("#007bff"),
            parent_id: None,
            child_ids: Vec::new(),
            category,
            metadata: None,
            created_at: now,
            updated_at: now,
            usage_count: 0,
            is_active: true,
        }
    }

    /// Update tag usage statistics
    pub fn increment_usage(&mut self) {
        self.usage_count += 1;
        self.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    /// Add child tag
    pub fn add_child(&mut self, child_id: Arc<str>) {
        if !self.child_ids.contains(&child_id) {
            self.child_ids.push(child_id);
            self.updated_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
        }
    }

    /// Remove child tag
    pub fn remove_child(&mut self, child_id: &Arc<str>) {
        self.child_ids.retain(|id| id != child_id);
        self.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    /// Check if tag is a child of another tag
    pub fn is_child_of(&self, parent_id: &Arc<str>) -> bool {
        self.parent_id.as_ref() == Some(parent_id)
    }

    /// Get tag path (for hierarchical display)
    pub fn get_path(&self, tagger: &ConversationTagger) -> Vec<Arc<str>> {
        let mut path = Vec::new();
        let mut current_id = Some(self.id.clone());

        while let Some(id) = current_id {
            if let Some(tag) = tagger.get_tag(&id) {
                path.insert(0, tag.name.clone());
                current_id = tag.parent_id.clone();
            } else {
                break;
            }
        }

        path
    }
}

/// Conversation tagger with lock-free operations
pub struct ConversationTagger {
    /// Tags storage
    tags: SkipMap<Arc<str>, ConversationTag>,
    /// Message to tags mapping
    message_tags: SkipMap<Arc<str>, Vec<Arc<str>>>,
    /// Tag to messages mapping
    tag_messages: SkipMap<Arc<str>, Vec<Arc<str>>>,
    /// Tag hierarchy
    tag_hierarchy: SkipMap<Arc<str>, Vec<Arc<str>>>,
    /// Tag counter
    tag_counter: Arc<ConsistentCounter>,
    /// Tagging counter
    tagging_counter: Arc<ConsistentCounter>,
    /// Auto-tagging rules
    auto_tagging_rules: Arc<RwLock<HashMap<Arc<str>, Vec<Arc<str>>>>>,
}

impl ConversationTagger {
    /// Create a new conversation tagger
    pub fn new() -> Self {
        Self {
            tags: SkipMap::new(),
            message_tags: SkipMap::new(),
            tag_messages: SkipMap::new(),
            tag_hierarchy: SkipMap::new(),
            tag_counter: Arc::new(ConsistentCounter::new(0)),
            tagging_counter: Arc::new(ConsistentCounter::new(0)),
            auto_tagging_rules: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new tag (streaming)
    pub fn create_tag_stream(
        &self,
        name: Arc<str>,
        description: Arc<str>,
        category: Arc<str>,
    ) -> AsyncStream<Arc<str>> {
        AsyncStream::with_channel(move |sender| {
            let tag = ConversationTag::new(name, description, category);
            let tag_id = tag.id.clone();
            let _ = sender.send(tag_id);
        })
    }

    /// Create a child tag (streaming)
    pub fn create_child_tag_stream(
        &self,
        parent_id: Arc<str>,
        name: Arc<str>,
        description: Arc<str>,
        category: Arc<str>,
    ) -> AsyncStream<Arc<str>> {
        let parent_id_clone = parent_id.clone();
        let tags_clone = self.tags.clone();
        let tag_hierarchy_clone = self.tag_hierarchy.clone();
        let statistics_clone = self.statistics.clone();
        
        AsyncStream::with_channel(move |sender| {
            let mut tag = ConversationTag::new(name, description, category);
            tag.parent_id = Some(parent_id_clone.clone());
            let tag_id = tag.id.clone();

            // Update parent tag
            if let Some(parent_entry) = tags_clone.get(&parent_id_clone) {
                let mut parent_tag = parent_entry.value().clone();
                parent_tag.add_child(tag_id.clone());
                tags_clone.insert(parent_id_clone.clone(), parent_tag);
            }

            // Add to hierarchy
            let mut children = tag_hierarchy_clone
                .get(&parent_id_clone)
                .map(|c| c.value().clone())
                .unwrap_or_default();
            children.push(tag_id.clone());
            tag_hierarchy_clone.insert(parent_id_clone, children);

            tags_clone.insert(tag_id.clone(), tag);
            statistics_clone.total_tags.inc();

            let _ = sender.send(tag_id);
        })
    }

    /// Tag a message (streaming)
    pub fn tag_message_stream(
        &self,
        message_id: Arc<str>,
        tag_ids: Vec<Arc<str>>,
    ) -> AsyncStream<()> {
        let message_tags_clone = self.message_tags.clone();
        let tag_messages_clone = self.tag_messages.clone();
        let tags_clone = self.tags.clone();
        let statistics_clone = self.statistics.clone();
        
        AsyncStream::with_channel(move |sender| {
            // Update message tags mapping
            message_tags_clone.insert(message_id.clone(), tag_ids.clone());

            // Update tag messages mapping and usage counts
            for tag_id in &tag_ids {
                // Add message to tag's message list
                let mut messages = tag_messages_clone
                    .get(tag_id)
                    .map(|m| m.value().clone())
                    .unwrap_or_default();
                if !messages.contains(&message_id) {
                    messages.push(message_id.clone());
                    tag_messages_clone.insert(tag_id.clone(), messages);
                }

                // Update tag usage count
                if let Some(tag_entry) = tags_clone.get(tag_id) {
                    let mut tag = tag_entry.value().clone();
                    tag.increment_usage();
                    tags_clone.insert(tag_id.clone(), tag);
                }
            }

            statistics_clone.total_tagged_messages.inc();
            let _ = sender.send(());
        })
    }

    /// Auto-tag message based on content (streaming)
    pub fn auto_tag_message_stream(&self, message: CandleMessage) -> AsyncStream<Arc<str>> {
        AsyncStream::with_channel(move |sender| {
            let content = message.message.content.to_lowercase();

            if let Ok(rules) = self.auto_tagging_rules.try_read() {
                let mut suggested_tags = Vec::new();

                for (pattern, tag_ids) in rules.iter() {
                    if content.contains(pattern.as_ref()) {
                        suggested_tags.extend(tag_ids.clone());
                    }
                }

                // Remove duplicates
                suggested_tags.sort();
                suggested_tags.dedup();

                for tag in suggested_tags {
                    let _ = sender.send(tag);
                }
            }
        })
    }

    /// Add auto-tagging rule (streaming)
    pub fn add_auto_tagging_rule_stream(
        &self,
        pattern: Arc<str>,
        tag_ids: Vec<Arc<str>>,
    ) -> AsyncStream<()> {
        let auto_tagging_rules_clone = self.auto_tagging_rules.clone();
        
        AsyncStream::with_channel(move |sender| {
            if let Ok(mut rules) = auto_tagging_rules_clone.try_write() {
                rules.insert(pattern, tag_ids);
            }
            let _ = sender.send(());
        })
    }

    /// Remove auto-tagging rule (streaming)
    pub fn remove_auto_tagging_rule_stream(&self, pattern: Arc<str>) -> AsyncStream<()> {
        let auto_tagging_rules_clone = self.auto_tagging_rules.clone();
        
        AsyncStream::with_channel(move |sender| {
            if let Ok(mut rules) = auto_tagging_rules_clone.try_write() {
                rules.remove(&pattern);
            }
            let _ = sender.send(());
        })
    }

    /// Get tags for a message
    pub fn get_message_tags(&self, message_id: &Arc<str>) -> Vec<Arc<str>> {
        self.message_tags
            .get(message_id)
            .map(|tags| tags.value().clone())
            .unwrap_or_default()
    }

    /// Get messages for a tag
    pub fn get_tag_messages(&self, tag_id: &Arc<str>) -> Vec<Arc<str>> {
        self.tag_messages
            .get(tag_id)
            .map(|messages| messages.value().clone())
            .unwrap_or_default()
    }

    /// Get tag by ID
    pub fn get_tag(&self, tag_id: &Arc<str>) -> Option<ConversationTag> {
        self.tags.get(tag_id).map(|entry| entry.value().clone())
    }

    /// Get tag hierarchy
    pub fn get_tag_hierarchy(&self, tag_id: &Arc<str>) -> Vec<Arc<str>> {
        self.tag_hierarchy
            .get(tag_id)
            .map(|children| children.value().clone())
            .unwrap_or_default()
    }

    /// Get all tags
    pub fn get_all_tags(&self) -> Vec<ConversationTag> {
        self.tags
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get root tags (tags without parents)
    pub fn get_root_tags(&self) -> Vec<ConversationTag> {
        self.tags
            .iter()
            .filter(|entry| entry.value().parent_id.is_none())
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Search tags by name or description
    pub fn search_tags(&self, query: &str) -> Vec<ConversationTag> {
        let query_lower = query.to_lowercase();
        self.tags
            .iter()
            .filter(|entry| {
                let tag = entry.value();
                tag.name.to_lowercase().contains(&query_lower)
                    || tag.description.to_lowercase().contains(&query_lower)
                    || tag.category.to_lowercase().contains(&query_lower)
            })
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get tags by category
    pub fn get_tags_by_category(&self, category: &str) -> Vec<ConversationTag> {
        self.tags
            .iter()
            .filter(|entry| entry.value().category.as_ref() == category)
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get most used tags
    pub fn get_most_used_tags(&self, limit: usize) -> Vec<ConversationTag> {
        let mut tags: Vec<_> = self.tags
            .iter()
            .map(|entry| entry.value().clone())
            .collect();
        
        tags.sort_by(|a, b| b.usage_count.cmp(&a.usage_count));
        tags.truncate(limit);
        tags
    }

    /// Delete tag and update hierarchy
    pub fn delete_tag(&self, tag_id: &Arc<str>) -> bool {
        if let Some(tag_entry) = self.tags.get(tag_id) {
            let tag = tag_entry.value().clone();
            
            // Update parent to remove this child
            if let Some(parent_id) = &tag.parent_id {
                if let Some(parent_entry) = self.tags.get(parent_id) {
                    let mut parent_tag = parent_entry.value().clone();
                    parent_tag.remove_child(tag_id);
                    self.tags.insert(parent_id.clone(), parent_tag);
                }
                
                // Update hierarchy
                if let Some(siblings) = self.tag_hierarchy.get(parent_id) {
                    let mut siblings_vec = siblings.value().clone();
                    siblings_vec.retain(|id| id != tag_id);
                    self.tag_hierarchy.insert(parent_id.clone(), siblings_vec);
                }
            }
            
            // Reassign children to parent or make them root tags
            for child_id in &tag.child_ids {
                if let Some(child_entry) = self.tags.get(child_id) {
                    let mut child_tag = child_entry.value().clone();
                    child_tag.parent_id = tag.parent_id.clone();
                    self.tags.insert(child_id.clone(), child_tag);
                }
            }
            
            // Remove from all mappings
            self.tags.remove(tag_id);
            self.tag_hierarchy.remove(tag_id);
            
            // Clean up message mappings
            for message_id in self.tag_messages.get(tag_id).map(|m| m.value().clone()).unwrap_or_default() {
                if let Some(tags) = self.message_tags.get(&message_id) {
                    let mut tag_list = tags.value().clone();
                    tag_list.retain(|id| id != tag_id);
                    if tag_list.is_empty() {
                        self.message_tags.remove(&message_id);
                    } else {
                        self.message_tags.insert(message_id, tag_list);
                    }
                }
            }
            self.tag_messages.remove(tag_id);
            
            true
        } else {
            false
        }
    }

    /// Get tagging statistics
    pub fn get_statistics(&self) -> TaggingStatistics {
        TaggingStatistics {
            total_tags: self.tag_counter.get(),
            total_taggings: self.tagging_counter.get(),
            active_tags: self
                .tags
                .iter()
                .filter(|entry| entry.value().is_active)
                .count(),
            most_used_tag: self.get_most_used_tag(),
        }
    }

    /// Get most used tag
    fn get_most_used_tag(&self) -> Option<Arc<str>> {
        self.tags
            .iter()
            .max_by_key(|entry| entry.value().usage_count)
            .map(|entry| entry.value().name.clone())
    }

    /// Get auto-tagging rules
    pub fn get_auto_tagging_rules(&self) -> HashMap<Arc<str>, Vec<Arc<str>>> {
        self.auto_tagging_rules
            .try_read()
            .map(|rules| rules.clone())
            .unwrap_or_default()
    }

    /// Clear all auto-tagging rules
    pub fn clear_auto_tagging_rules(&self) {
        if let Ok(mut rules) = self.auto_tagging_rules.try_write() {
            rules.clear();
        }
    }
}

impl Default for ConversationTagger {
    fn default() -> Self {
        Self::new()
    }
}

/// Tagging statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TaggingStatistics {
    /// Total number of tags
    pub total_tags: usize,
    /// Total number of tagging operations
    pub total_taggings: usize,
    /// Number of active tags
    pub active_tags: usize,
    /// Most frequently used tag
    pub most_used_tag: Option<Arc<str>>,
}

impl TaggingStatistics {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Update statistics with new values
    pub fn update(&mut self, total_tags: usize, total_taggings: usize, active_tags: usize, most_used_tag: Option<Arc<str>>) {
        self.total_tags = total_tags;
        self.total_taggings = total_taggings;
        self.active_tags = active_tags;
        self.most_used_tag = most_used_tag;
    }

    /// Get tagging rate (taggings per tag)
    pub fn get_tagging_rate(&self) -> f64 {
        if self.total_tags > 0 {
            self.total_taggings as f64 / self.total_tags as f64
        } else {
            0.0
        }
    }

    /// Get active tag percentage
    pub fn get_active_percentage(&self) -> f64 {
        if self.total_tags > 0 {
            (self.active_tags as f64 / self.total_tags as f64) * 100.0
        } else {
            0.0
        }
    }
}