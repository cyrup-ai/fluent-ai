//! Conversation tagger implementation

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use atomic_counter::{AtomicCounter, ConsistentCounter};
use crossbeam_skiplist::SkipMap;
use fluent_ai_async::AsyncStream;
use uuid::Uuid;

use super::types::{ConversationTag, TagFilter, TagCategory};
use super::statistics::TaggingStatistics;

/// Conversation tagger with lock-free operations
pub struct ConversationTagger {
    /// Tag storage using skip map for lock-free access
    tags: Arc<SkipMap<Arc<str>, ConversationTag>>,
    /// Conversation to tags mapping
    conversation_tags: Arc<SkipMap<Arc<str>, Vec<Arc<str>>>>,
    /// Tag usage counter
    usage_counter: ConsistentCounter,
    /// Operation statistics
    stats: Arc<std::sync::Mutex<TaggingStatistics>>,
    /// Tag creation counter
    tag_counter: AtomicUsize,
}

impl Default for ConversationTagger {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversationTagger {
    /// Create new conversation tagger
    pub fn new() -> Self {
        Self {
            tags: Arc::new(SkipMap::new()),
            conversation_tags: Arc::new(SkipMap::new()),
            usage_counter: ConsistentCounter::new(0),
            stats: Arc::new(std::sync::Mutex::new(TaggingStatistics::new())),
            tag_counter: AtomicUsize::new(0),
        }
    }

    /// Create a new tag
    pub fn create_tag(&self, name: impl Into<Arc<str>>, description: impl Into<Arc<str>>) -> AsyncStream<ConversationTag> {
        use fluent_ai_async::{emit, handle_error};

        let name: Arc<str> = name.into();
        let description: Arc<str> = description.into();

        AsyncStream::with_channel(move |sender| {
            let tag = ConversationTag::new(name.clone(), description);
            
            // Store the tag
            let tag_id = tag.id.clone();
            let tags = Arc::clone(&self.tags);
            tags.insert(tag_id, tag.clone());

            // Update statistics
            if let Ok(mut stats) = self.stats.lock() {
                stats.record_tag_created();
            }

            self.tag_counter.fetch_add(1, Ordering::Relaxed);

            emit!(sender, tag);
        })
    }

    /// Apply tag to conversation
    pub fn apply_tag(&self, conversation_id: impl Into<Arc<str>>, tag_id: impl Into<Arc<str>>) -> AsyncStream<bool> {
        use fluent_ai_async::{emit, handle_error};

        let conversation_id: Arc<str> = conversation_id.into();
        let tag_id: Arc<str> = tag_id.into();

        AsyncStream::with_channel(move |sender| {
            // Check if tag exists
            if !self.tags.contains_key(&tag_id) {
                handle_error!(format!("Tag {} not found", tag_id), "apply_tag");
                emit!(sender, false);
                return;
            }

            // Get or create conversation tag list
            let conversation_tags = Arc::clone(&self.conversation_tags);
            let existing_tags = conversation_tags
                .get(&conversation_id)
                .map(|entry| entry.value().clone())
                .unwrap_or_else(Vec::new);

            // Check if tag is already applied
            if existing_tags.contains(&tag_id) {
                emit!(sender, false);
                return;
            }

            // Add tag to conversation
            let mut updated_tags = existing_tags;
            updated_tags.push(tag_id.clone());
            conversation_tags.insert(conversation_id, updated_tags);

            // Update tag usage
            if let Some(entry) = self.tags.get(&tag_id) {
                let mut tag = entry.value().clone();
                tag.increment_usage();
                self.tags.insert(tag_id.clone(), tag);
            }

            // Update statistics
            if let Ok(mut stats) = self.stats.lock() {
                stats.record_tag_applied(&tag_id);
            }

            self.usage_counter.inc();

            emit!(sender, true);
        })
    }

    /// Remove tag from conversation
    pub fn remove_tag(&self, conversation_id: impl Into<Arc<str>>, tag_id: impl Into<Arc<str>>) -> AsyncStream<bool> {
        use fluent_ai_async::{emit, handle_error};

        let conversation_id: Arc<str> = conversation_id.into();
        let tag_id: Arc<str> = tag_id.into();

        AsyncStream::with_channel(move |sender| {
            let conversation_tags = Arc::clone(&self.conversation_tags);
            
            // Get conversation tags
            let existing_tags = match conversation_tags.get(&conversation_id) {
                Some(entry) => entry.value().clone(),
                None => {
                    emit!(sender, false);
                    return;
                }
            };

            // Remove tag from list
            let updated_tags: Vec<Arc<str>> = existing_tags
                .into_iter()
                .filter(|id| *id != tag_id)
                .collect();

            // Update conversation tags
            if updated_tags.is_empty() {
                conversation_tags.remove(&conversation_id);
            } else {
                conversation_tags.insert(conversation_id, updated_tags);
            }

            // Update statistics
            if let Ok(mut stats) = self.stats.lock() {
                stats.record_tag_removed();
            }

            emit!(sender, true);
        })
    }

    /// Get tags for conversation
    pub fn get_conversation_tags(&self, conversation_id: &str) -> Vec<ConversationTag> {
        let conversation_id: Arc<str> = Arc::from(conversation_id);
        
        let tag_ids = self.conversation_tags
            .get(&conversation_id)
            .map(|entry| entry.value().clone())
            .unwrap_or_else(Vec::new);

        let mut tags = Vec::new();
        for tag_id in tag_ids {
            if let Some(entry) = self.tags.get(&tag_id) {
                tags.push(entry.value().clone());
            }
        }

        tags
    }

    /// Search tags by filter
    pub fn search_tags(&self, filter: TagFilter) -> AsyncStream<ConversationTag> {
        use fluent_ai_async::{emit, handle_error};

        AsyncStream::with_channel(move |sender| {
            let tags = Arc::clone(&self.tags);
            
            for entry in tags.iter() {
                let tag = entry.value();
                
                if filter.matches(tag) {
                    emit!(sender, tag.clone());
                }
            }
        })
    }

    /// Get all tags
    pub fn get_all_tags(&self) -> Vec<ConversationTag> {
        self.tags
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get tag by ID
    pub fn get_tag(&self, tag_id: &str) -> Option<ConversationTag> {
        let tag_id: Arc<str> = Arc::from(tag_id);
        self.tags.get(&tag_id).map(|entry| entry.value().clone())
    }

    /// Delete tag
    pub fn delete_tag(&self, tag_id: &str) -> AsyncStream<bool> {
        use fluent_ai_async::{emit, handle_error};

        let tag_id: Arc<str> = Arc::from(tag_id);

        AsyncStream::with_channel(move |sender| {
            // Remove tag from storage
            let removed = self.tags.remove(&tag_id).is_some();

            if removed {
                // Remove from all conversations
                let conversation_tags = Arc::clone(&self.conversation_tags);
                let mut conversations_to_update = Vec::new();

                for entry in conversation_tags.iter() {
                    let conversation_id = entry.key().clone();
                    let mut tags = entry.value().clone();
                    
                    if let Some(pos) = tags.iter().position(|id| *id == tag_id) {
                        tags.remove(pos);
                        conversations_to_update.push((conversation_id, tags));
                    }
                }

                // Update conversations
                for (conversation_id, updated_tags) in conversations_to_update {
                    if updated_tags.is_empty() {
                        conversation_tags.remove(&conversation_id);
                    } else {
                        conversation_tags.insert(conversation_id, updated_tags);
                    }
                }
            }

            emit!(sender, removed);
        })
    }

    /// Get tagging statistics
    pub fn get_statistics(&self) -> TaggingStatistics {
        self.stats.lock().unwrap().clone()
    }

    /// Get tag count
    pub fn tag_count(&self) -> usize {
        self.tag_counter.load(Ordering::Relaxed)
    }

    /// Get usage count
    pub fn usage_count(&self) -> usize {
        self.usage_counter.get()
    }

    /// Clear all tags
    pub fn clear_all(&self) {
        self.tags.clear();
        self.conversation_tags.clear();
        self.usage_counter.reset();
        self.tag_counter.store(0, Ordering::Relaxed);

        if let Ok(mut stats) = self.stats.lock() {
            *stats = TaggingStatistics::new();
        }
    }
}