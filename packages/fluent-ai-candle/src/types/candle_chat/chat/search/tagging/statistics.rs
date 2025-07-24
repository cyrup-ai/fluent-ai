//! Tagging statistics and metrics

use std::sync::atomic::{AtomicUsize, Ordering};
use serde::{Deserialize, Serialize};

/// Statistics for tagging operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaggingStatistics {
    /// Total number of tags created
    pub tags_created: usize,
    /// Total number of tags applied
    pub tags_applied: usize,
    /// Total number of tags removed
    pub tags_removed: usize,
    /// Total number of conversations tagged
    pub conversations_tagged: usize,
    /// Average tags per conversation
    pub avg_tags_per_conversation: f64,
    /// Most frequently used tags (tag_id, usage_count)
    pub popular_tags: Vec<(String, usize)>,
    /// Tag categories and their counts
    pub category_distribution: std::collections::HashMap<String, usize>,
    /// Tagging performance metrics
    pub performance_metrics: TaggingPerformanceMetrics,
}

impl Default for TaggingStatistics {
    fn default() -> Self {
        Self {
            tags_created: 0,
            tags_applied: 0,
            tags_removed: 0,
            conversations_tagged: 0,
            avg_tags_per_conversation: 0.0,
            popular_tags: Vec::new(),
            category_distribution: std::collections::HashMap::new(),
            performance_metrics: TaggingPerformanceMetrics::default(),
        }
    }
}

/// Performance metrics for tagging operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaggingPerformanceMetrics {
    /// Average time to create a tag (microseconds)
    pub avg_create_time_us: u64,
    /// Average time to apply a tag (microseconds)
    pub avg_apply_time_us: u64,
    /// Average time to search tags (microseconds)
    pub avg_search_time_us: u64,
    /// Cache hit rate for tag operations
    pub cache_hit_rate: f64,
    /// Total tagging operations performed
    pub total_operations: usize,
    /// Number of failed operations
    pub failed_operations: usize,
}

impl Default for TaggingPerformanceMetrics {
    fn default() -> Self {
        Self {
            avg_create_time_us: 0,
            avg_apply_time_us: 0,
            avg_search_time_us: 0,
            cache_hit_rate: 0.0,
            total_operations: 0,
            failed_operations: 0,
        }
    }
}

impl TaggingStatistics {
    /// Create new tagging statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record tag creation
    pub fn record_tag_created(&mut self) {
        self.tags_created += 1;
        self.performance_metrics.total_operations += 1;
    }

    /// Record tag application
    pub fn record_tag_applied(&mut self, tag_id: &str) {
        self.tags_applied += 1;
        self.performance_metrics.total_operations += 1;
        
        // Update popular tags
        if let Some(pos) = self.popular_tags.iter().position(|(id, _)| id == tag_id) {
            self.popular_tags[pos].1 += 1;
        } else {
            self.popular_tags.push((tag_id.to_string(), 1));
        }
        
        // Sort by usage count and keep top 10
        self.popular_tags.sort_by(|a, b| b.1.cmp(&a.1));
        self.popular_tags.truncate(10);
    }

    /// Record tag removal
    pub fn record_tag_removed(&mut self) {
        self.tags_removed += 1;
        self.performance_metrics.total_operations += 1;
    }

    /// Record conversation tagging
    pub fn record_conversation_tagged(&mut self, tag_count: usize) {
        self.conversations_tagged += 1;
        
        // Update average tags per conversation
        let total_tags = (self.avg_tags_per_conversation * (self.conversations_tagged - 1) as f64) + tag_count as f64;
        self.avg_tags_per_conversation = total_tags / self.conversations_tagged as f64;
    }

    /// Update category distribution
    pub fn update_category_distribution(&mut self, category: &str, delta: i32) {
        let count = self.category_distribution.get(category).copied().unwrap_or(0);
        if delta < 0 && (-delta) as usize > count {
            self.category_distribution.insert(category.to_string(), 0);
        } else {
            let new_count = (count as i32 + delta) as usize;
            self.category_distribution.insert(category.to_string(), new_count);
        }
    }

    /// Record operation timing
    pub fn record_create_timing(&mut self, duration_us: u64) {
        let total_ops = self.performance_metrics.total_operations;
        if total_ops > 0 {
            let current_avg = self.performance_metrics.avg_create_time_us;
            self.performance_metrics.avg_create_time_us = 
                (current_avg * (total_ops - 1) as u64 + duration_us) / total_ops as u64;
        } else {
            self.performance_metrics.avg_create_time_us = duration_us;
        }
    }

    /// Record apply timing
    pub fn record_apply_timing(&mut self, duration_us: u64) {
        let total_ops = self.performance_metrics.total_operations;
        if total_ops > 0 {
            let current_avg = self.performance_metrics.avg_apply_time_us;
            self.performance_metrics.avg_apply_time_us = 
                (current_avg * (total_ops - 1) as u64 + duration_us) / total_ops as u64;
        } else {
            self.performance_metrics.avg_apply_time_us = duration_us;
        }
    }

    /// Record search timing
    pub fn record_search_timing(&mut self, duration_us: u64) {
        let total_ops = self.performance_metrics.total_operations;
        if total_ops > 0 {
            let current_avg = self.performance_metrics.avg_search_time_us;
            self.performance_metrics.avg_search_time_us = 
                (current_avg * (total_ops - 1) as u64 + duration_us) / total_ops as u64;
        } else {
            self.performance_metrics.avg_search_time_us = duration_us;
        }
    }

    /// Record failed operation
    pub fn record_failed_operation(&mut self) {
        self.performance_metrics.failed_operations += 1;
        self.performance_metrics.total_operations += 1;
    }

    /// Update cache hit rate
    pub fn update_cache_hit_rate(&mut self, hit: bool) {
        let total_ops = self.performance_metrics.total_operations as f64;
        if total_ops > 0.0 {
            let current_rate = self.performance_metrics.cache_hit_rate;
            let hit_value = if hit { 1.0 } else { 0.0 };
            self.performance_metrics.cache_hit_rate = 
                (current_rate * (total_ops - 1.0) + hit_value) / total_ops;
        }
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.performance_metrics.total_operations;
        if total == 0 {
            return 1.0;
        }
        let successful = total - self.performance_metrics.failed_operations;
        successful as f64 / total as f64
    }

    /// Get most popular tag
    pub fn most_popular_tag(&self) -> Option<&(String, usize)> {
        self.popular_tags.first()
    }

    /// Get most popular category
    pub fn most_popular_category(&self) -> Option<(&String, &usize)> {
        self.category_distribution
            .iter()
            .max_by_key(|(_, count)| *count)
    }
}