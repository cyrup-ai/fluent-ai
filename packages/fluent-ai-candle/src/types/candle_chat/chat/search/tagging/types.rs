//! Tagging types and data structures

use std::sync::Arc;
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
    /// Tag priority (higher = more important)
    pub priority: u8,
    /// Tag metadata
    pub metadata: std::collections::HashMap<String, String>,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last updated timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Tag usage count
    pub usage_count: u64,
    /// Tag author/creator
    pub author: Option<Arc<str>>}

impl Default for ConversationTag {
    fn default() -> Self {
        let now = chrono::Utc::now();
        Self {
            id: Arc::from(Uuid::new_v4().to_string()),
            name: Arc::from("Untitled Tag"),
            description: Arc::from(""),
            category: Arc::from("general"),
            color: None,
            icon: None,
            priority: 0,
            metadata: std::collections::HashMap::new(),
            created_at: now,
            updated_at: now,
            usage_count: 0,
            author: None}
    }
}

impl ConversationTag {
    /// Create a new conversation tag
    pub fn new(name: impl Into<Arc<str>>, description: impl Into<Arc<str>>) -> Self {
        let now = chrono::Utc::now();
        Self {
            id: Arc::from(Uuid::new_v4().to_string()),
            name: name.into(),
            description: description.into(),
            category: Arc::from("general"),
            color: None,
            icon: None,
            priority: 0,
            metadata: std::collections::HashMap::new(),
            created_at: now,
            updated_at: now,
            usage_count: 0,
            author: None}
    }

    /// Create a tag with category
    pub fn with_category(mut self, category: impl Into<Arc<str>>) -> Self {
        self.category = category.into();
        self
    }

    /// Set tag color
    pub fn with_color(mut self, color: String) -> Self {
        self.color = Some(color);
        self
    }

    /// Set tag icon
    pub fn with_icon(mut self, icon: String) -> Self {
        self.icon = Some(icon);
        self
    }

    /// Set tag priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Set tag author
    pub fn with_author(mut self, author: impl Into<Arc<str>>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Add metadata entry
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
        self.updated_at = chrono::Utc::now();
    }

    /// Increment usage count
    pub fn increment_usage(&mut self) {
        self.usage_count += 1;
        self.updated_at = chrono::Utc::now();
    }

    /// Check if tag matches search query
    pub fn matches_query(&self, query: &str) -> bool {
        let query_lower = query.to_lowercase();
        self.name.to_lowercase().contains(&query_lower)
            || self.description.to_lowercase().contains(&query_lower)
            || self.category.to_lowercase().contains(&query_lower)
    }
}

/// Tag category for organizing tags
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TagCategory {
    /// General purpose tags
    General,
    /// Project-related tags
    Project,
    /// Topic-based tags
    Topic,
    /// Priority tags
    Priority,
    /// Status tags
    Status,
    /// User-defined category
    Custom(Arc<str>)}

impl Default for TagCategory {
    fn default() -> Self {
        Self::General
    }
}

impl From<&str> for TagCategory {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "general" => Self::General,
            "project" => Self::Project,
            "topic" => Self::Topic,
            "priority" => Self::Priority,
            "status" => Self::Status,
            _ => Self::Custom(Arc::from(s))}
    }
}

impl ToString for TagCategory {
    fn to_string(&self) -> String {
        match self {
            Self::General => "general".to_string(),
            Self::Project => "project".to_string(),
            Self::Topic => "topic".to_string(),
            Self::Priority => "priority".to_string(),
            Self::Status => "status".to_string(),
            Self::Custom(name) => name.to_string()}
    }
}

/// Tag filter for searching tags
#[derive(Debug, Clone)]
pub struct TagFilter {
    /// Filter by category
    pub category: Option<TagCategory>,
    /// Filter by name pattern
    pub name_pattern: Option<String>,
    /// Filter by minimum priority
    pub min_priority: Option<u8>,
    /// Filter by maximum priority
    pub max_priority: Option<u8>,
    /// Filter by author
    pub author: Option<Arc<str>>,
    /// Filter by date range
    pub date_range: Option<crate::types::candle_chat::chat::search::export::types::DateRange>}

impl Default for TagFilter {
    fn default() -> Self {
        Self {
            category: None,
            name_pattern: None,
            min_priority: None,
            max_priority: None,
            author: None,
            date_range: None}
    }
}

impl TagFilter {
    /// Create empty filter
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by category
    pub fn with_category(mut self, category: TagCategory) -> Self {
        self.category = Some(category);
        self
    }

    /// Filter by name pattern
    pub fn with_name_pattern(mut self, pattern: String) -> Self {
        self.name_pattern = Some(pattern);
        self
    }

    /// Filter by priority range
    pub fn with_priority_range(mut self, min: u8, max: u8) -> Self {
        self.min_priority = Some(min);
        self.max_priority = Some(max);
        self
    }

    /// Check if tag matches this filter
    pub fn matches(&self, tag: &ConversationTag) -> bool {
        // Check category
        if let Some(ref category) = self.category {
            if tag.category.as_ref() != category.to_string() {
                return false;
            }
        }

        // Check name pattern
        if let Some(ref pattern) = self.name_pattern {
            if !tag.name.to_lowercase().contains(&pattern.to_lowercase()) {
                return false;
            }
        }

        // Check priority range
        if let Some(min_priority) = self.min_priority {
            if tag.priority < min_priority {
                return false;
            }
        }

        if let Some(max_priority) = self.max_priority {
            if tag.priority > max_priority {
                return false;
            }
        }

        // Check author
        if let Some(ref author) = self.author {
            match &tag.author {
                Some(tag_author) => {
                    if tag_author != author {
                        return false;
                    }
                }
                None => return false}
        }

        true
    }
}