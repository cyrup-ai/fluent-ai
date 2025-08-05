use std::sync::Arc;
use serde::{Deserialize, Serialize};
use hashbrown::HashMap;

/// Tag with hierarchical structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTag {
    /// Unique identifier
    pub id: Arc<str>,
    /// Tag name
    pub name: Arc<str>,
    /// Optional description
    pub description: Arc<str>,
    /// Category for grouping
    pub category: Arc<str>,
    /// Parent tag ID, if any
    pub parent_id: Option<Arc<str>>,
    /// Child tags
    #[serde(skip_serializing, default)]
    pub children: Vec<Arc<str>>,
    /// Number of times this tag has been used
    pub usage_count: usize,
    /// When the tag was created (unix timestamp)
    pub created_at: u64,
    /// When the tag was last used (unix timestamp)
    pub last_used: u64,
}

/// Tagging statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TaggingStatistics {
    /// Total number of tags
    pub total_tags: usize,
    /// Number of tagged messages
    pub tagged_messages: usize,
    /// Most used tag
    pub most_used_tag: Option<Arc<str>>,
    /// Tags by category
    pub tags_by_category: HashMap<Arc<str>, usize>,
    /// Tags by depth in hierarchy
    pub tags_by_depth: HashMap<usize, usize>,
    /// Average tags per message
    pub avg_tags_per_message: f32,
}
