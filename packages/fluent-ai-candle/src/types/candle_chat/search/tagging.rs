//! Conversation tagging functionality with zero-allocation streaming
//!
//! This module provides hierarchical conversation tagging with lock-free operations,
//! auto-tagging rules, and streaming-first architecture using fluent-ai-async primitives.

use crate::types::CandleSearchChatMessage;
use crossbeam_skiplist::SkipMap;
use fluent_ai_async::{AsyncStream};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

/// Consistent counter for lock-free operations
#[derive(Debug)]
pub struct ConsistentCounter {
    value: std::sync::atomic::AtomicUsize}

impl ConsistentCounter {
    pub fn new(initial: usize) -> Self {
        Self {
            value: std::sync::atomic::AtomicUsize::new(initial)}
    }

    pub fn inc(&self) {
        self.value.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn dec(&self) {
        self.value.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn get(&self) -> usize {
        self.value.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn set(&self, value: usize) {
        self.value.store(value, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Clone for ConsistentCounter {
    fn clone(&self) -> Self {
        Self::new(self.get())
    }
}

impl Default for ConsistentCounter {
    fn default() -> Self {
        Self::new(0)
    }
}

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
    pub is_active: bool}

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
            is_active: true}
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
    /// Auto-tagging rules - lock-free concurrent access
    auto_tagging_rules: Arc<SkipMap<Arc<str>, Vec<Arc<str>>>>}

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
            auto_tagging_rules: Arc::new(SkipMap::new())}
    }

    /// Create a new tag (streaming)
    pub fn create_tag_stream(
        &self,
        name: Arc<str>,
        description: Arc<str>,
        category: Arc<str>,
    ) -> AsyncStream<Arc<str>> {
        // Create the tag immediately and get its ID
        let tag = ConversationTag::new(name, description, category);
        let tag_id = tag.id.clone();

        // Insert the tag
        self.tags.insert(tag_id.clone(), tag);
        self.tag_counter.inc();

        // Return a stream with the tag ID
        AsyncStream::with_channel(move |sender| {
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
        // Create the child tag immediately
        let mut tag = ConversationTag::new(name, description, category);
        tag.parent_id = Some(parent_id.clone());
        let tag_id = tag.id.clone();

        // Update parent tag
        if let Some(parent_entry) = self.tags.get(&parent_id) {
            let mut parent_tag = parent_entry.value().clone();
            parent_tag.child_ids.push(tag_id.clone());
            self.tags.insert(parent_id.clone(), parent_tag);
        }

        // Add to hierarchy
        if let Some(children) = self.tag_hierarchy.get(&parent_id) {
            let mut children_vec = children.value().clone();
            children_vec.push(tag_id.clone());
            self.tag_hierarchy.insert(parent_id.clone(), children_vec);
        } else {
            self.tag_hierarchy
                .insert(parent_id.clone(), vec![tag_id.clone()]);
        }

        self.tags.insert(tag_id.clone(), tag);
        self.tag_counter.inc();

        // Return a stream with the tag ID
        AsyncStream::with_channel(move |sender| {
            let _ = sender.send(tag_id);
        })
    }

    /// Create a child tag (legacy)
    pub fn create_child_tag(
        &self,
        parent_id: Arc<str>,
        name: Arc<str>,
        description: Arc<str>,
        category: Arc<str>,
    ) -> AsyncStream<Arc<str>> {
        self.create_child_tag_stream(parent_id, name, description, category)
    }

    /// Tag a message (streaming)
    pub fn tag_message_stream(
        &self,
        message_id: Arc<str>,
        tag_ids: Vec<Arc<str>>,
    ) -> AsyncStream<()> {
        // Perform tagging operations immediately
        self.message_tags
            .insert(message_id.clone(), tag_ids.clone());

        // Add to tag messages mapping
        for tag_id in &tag_ids {
            if let Some(messages) = self.tag_messages.get(tag_id) {
                let mut messages_vec = messages.value().clone();
                messages_vec.push(message_id.clone());
                self.tag_messages.insert(tag_id.clone(), messages_vec);
            } else {
                self.tag_messages
                    .insert(tag_id.clone(), vec![message_id.clone()]);
            }

            // Update tag usage count
            if let Some(tag_entry) = self.tags.get(tag_id) {
                let mut tag = tag_entry.value().clone();
                tag.usage_count += 1;
                tag.updated_at = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                self.tags.insert(tag_id.clone(), tag);
            }
        }

        self.tagging_counter.inc();

        // Return a stream with unit result
        AsyncStream::with_channel(move |sender| {
            let _ = sender.send(());
        })
    }

    /// Auto-tag message based on content (streaming)
    pub fn auto_tag_message_stream(
        &self,
        message: CandleSearchChatMessage,
    ) -> AsyncStream<Arc<str>> {
        // Perform auto-tagging analysis immediately
        let mut suggested_tags = Vec::new();
        let content = message.content.to_lowercase();

        // SkipMap is lock-free, access directly
        let rules = &self.auto_tagging_rules;

        for entry in rules.iter() {
            let pattern = entry.key();
            let tag_ids = entry.value();
            if content.contains(pattern.as_ref()) {
                suggested_tags.extend(tag_ids.clone());
            }
        }
        let _ = rules;

        // Remove duplicates
        suggested_tags.sort();
        suggested_tags.dedup();

        // Return a stream with the suggested tags
        AsyncStream::with_channel(move |sender| {
            for tag in suggested_tags {
                let _ = sender.send(tag);
            }
            // AsyncStream automatically closes when sender is dropped
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

    /// Search tags by name
    pub fn search_tags(&self, query: &str) -> Vec<ConversationTag> {
        let query_lower = query.to_lowercase();
        self.tags
            .iter()
            .filter(|entry| {
                let tag = entry.value();
                tag.name.to_lowercase().contains(&query_lower)
                    || tag.description.to_lowercase().contains(&query_lower)
            })
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Add auto-tagging rule (streaming)
    pub fn add_auto_tagging_rule_stream(
        &self,
        pattern: Arc<str>,
        tag_ids: Vec<Arc<str>>,
    ) -> AsyncStream<()> {
        // Perform the rule addition immediately
        // SkipMap is lock-free, access directly
        self.auto_tagging_rules.insert(pattern, tag_ids);

        // Return a stream with unit result
        AsyncStream::with_channel(move |sender| {
            let _ = sender.send(());
        })
    }

    /// Add auto-tagging rule with fluent-ai-async streaming architecture
    pub fn add_auto_tagging_rule(
        &self,
        pattern: Arc<str>,
        tag_ids: Vec<Arc<str>>,
    ) -> AsyncStream<()> {
        self.add_auto_tagging_rule_stream(pattern, tag_ids)
    }

    /// Remove auto-tagging rule (streaming)
    pub fn remove_auto_tagging_rule_stream(&self, pattern: Arc<str>) -> AsyncStream<()> {
        // Perform the rule removal immediately
        // SkipMap is lock-free, remove directly
        self.auto_tagging_rules.remove(&pattern);

        // Return a stream with unit result
        AsyncStream::with_channel(move |sender| {
            // Send unit result
            let _ = sender.send(());
        })
    }

    /// Remove auto-tagging rule with fluent-ai-async streaming architecture
    pub fn remove_auto_tagging_rule(&self, pattern: &Arc<str>) -> AsyncStream<()> {
        self.remove_auto_tagging_rule_stream(pattern.clone())
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
            most_used_tag: self.get_most_used_tag()}
    }

    /// Get most used tag
    fn get_most_used_tag(&self) -> Option<Arc<str>> {
        self.tags
            .iter()
            .max_by_key(|entry| entry.value().usage_count)
            .map(|entry| entry.value().name.clone())
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
    pub total_tags: usize,
    pub total_taggings: usize,
    pub active_tags: usize,
    pub most_used_tag: Option<Arc<str>>}
