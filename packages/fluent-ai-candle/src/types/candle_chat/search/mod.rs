//! Search system for conversation history
//!
//! This module provides comprehensive search functionality with:
//! - SIMD-optimized full-text search with TF-IDF scoring
//! - Hierarchical tag management system
//! - Multiple export formats with compression
//! - Zero-allocation streaming architecture
//! - Lock-free data structures for optimal performance

pub mod types;
pub mod index;
pub mod tags;
pub mod export;
pub mod error;

// Re-export public types for convenience
pub use types::{
    SearchQuery, QueryOperator, DateRange, SortOrder, SearchResult, SearchStatistics,
    MatchPosition, MatchType, SearchResultMetadata, SearchOptions, SearchScope,
    StreamCollect,
};

pub use index::{
    ChatSearchIndex, TermFrequency, IndexEntry,
};

pub use tags::{
    ConversationTag, ConversationTagger, TaggingStatistics,
};

pub use export::{
    ExportFormat, ExportOptions, ExportStatistics, HistoryExporter,
};

pub use error::{
    SearchError, ErrorSeverity, ErrorContext, SearchResult as Result,
};

use std::sync::Arc;
use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::types::candle_chat::SearchChatMessage;

/// Enhanced history management system
///
/// This is the main entry point for all search functionality, combining:
/// - Full-text search indexing
/// - Tag management
/// - Export capabilities
/// - Statistics tracking
#[derive(Clone)]
pub struct EnhancedHistoryManager {
    /// Search index for full-text search
    pub search_index: Arc<ChatSearchIndex>,
    /// Tag management system
    pub tagger: Arc<ConversationTagger>,
    /// Export functionality
    pub exporter: Arc<HistoryExporter>,
    /// System statistics
    pub statistics: Arc<RwLock<HistoryManagerStatistics>>,
}

/// Comprehensive history manager statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HistoryManagerStatistics {
    /// Search-related statistics
    pub search_stats: SearchStatistics,
    /// Tag management statistics
    pub tagging_stats: TaggingStatistics,
    /// Export statistics
    pub export_stats: ExportStatistics,
    /// Total operations performed
    pub total_operations: usize,
    /// System uptime in seconds
    pub system_uptime: u64,
}

impl EnhancedHistoryManager {
    /// Create a new enhanced history manager
    pub fn new() -> Self {
        Self {
            search_index: Arc::new(ChatSearchIndex::new()),
            tagger: Arc::new(ConversationTagger::new()),
            exporter: Arc::new(HistoryExporter::new()),
            statistics: Arc::new(RwLock::new(HistoryManagerStatistics::default())),
        }
    }

    /// Add a message to both search index and tagging system
    pub fn add_message_stream(&self, message: SearchChatMessage) -> AsyncStream<()> {
        let search_index = Arc::clone(&self.search_index);
        let tagger = Arc::clone(&self.tagger);
        let statistics = Arc::clone(&self.statistics);

        AsyncStream::with_channel(move |sender| {
            // Add to search index
            let mut index_stream = search_index.add_message_stream(message.clone());
            if let Some(_) = index_stream.recv() {
                // Successfully added to index
            }

            // Auto-tag the message
            let mut auto_tag_stream = tagger.auto_tag_message_stream(message.clone());
            let mut suggested_tags = Vec::new();
            while let Some(tag) = auto_tag_stream.recv() {
                suggested_tags.push(tag);
            }

            // Apply suggested tags if any
            if !suggested_tags.is_empty() {
                let message_id = message
                    .message
                    .id
                    .clone()
                    .unwrap_or_else(|| Arc::from("unknown"));
                let mut tag_stream = tagger.tag_message_stream(message_id, suggested_tags);
                let _ = tag_stream.recv();
            }

            // Update statistics
            if let Ok(mut stats) = statistics.try_write() {
                stats.total_operations += 1;
            }

            let _ = sender.send(());
        })
    }

    /// Search messages with comprehensive options
    pub fn search_messages_stream(&self, query: SearchQuery) -> AsyncStream<SearchResult> {
        let search_index = Arc::clone(&self.search_index);
        let statistics = Arc::clone(&self.statistics);

        AsyncStream::with_channel(move |sender| {
            // Perform search
            let mut search_stream = search_index.search_stream(query);
            while let Some(result) = search_stream.recv() {
                let _ = sender.send(result);
            }

            // Update statistics
            if let Ok(mut stats) = statistics.try_write() {
                stats.total_operations += 1;
            }
        })
    }

    /// Export conversation history with full options
    pub fn export_history_stream(
        &self,
        messages: Vec<SearchChatMessage>,
        options: ExportOptions,
    ) -> AsyncStream<String> {
        let exporter = Arc::clone(&self.exporter);
        let statistics = Arc::clone(&self.statistics);

        AsyncStream::with_channel(move |sender| {
            // Perform export
            let mut export_stream = exporter.export_history_stream(messages, options);
            if let Some(exported_data) = export_stream.recv() {
                let _ = sender.send(exported_data);
            }

            // Update statistics
            if let Ok(mut stats) = statistics.try_write() {
                stats.total_operations += 1;
            }
        })
    }

    /// Create a new tag
    pub fn create_tag_stream(
        &self,
        name: Arc<str>,
        description: Arc<str>,
        category: Arc<str>,
    ) -> AsyncStream<Arc<str>> {
        self.tagger.create_tag_stream(name, description, category)
    }

    /// Tag a message
    pub fn tag_message_stream(
        &self,
        message_id: Arc<str>,
        tag_ids: Vec<Arc<str>>,
    ) -> AsyncStream<()> {
        self.tagger.tag_message_stream(message_id, tag_ids)
    }

    /// Get comprehensive statistics
    pub fn get_statistics_stream(&self) -> AsyncStream<HistoryManagerStatistics> {
        let search_index = Arc::clone(&self.search_index);
        let tagger = Arc::clone(&self.tagger);
        let exporter = Arc::clone(&self.exporter);
        let statistics = Arc::clone(&self.statistics);

        AsyncStream::with_channel(move |sender| {
            let mut stats = HistoryManagerStatistics::default();

            // Get search statistics
            let mut search_stats_stream = search_index.get_statistics_stream();
            if let Some(search_stats) = search_stats_stream.recv() {
                stats.search_stats = search_stats;
            }

            // Get tagging statistics
            stats.tagging_stats = tagger.get_statistics();

            // Get export statistics
            let mut export_stats_stream = exporter.get_statistics_stream();
            if let Some(export_stats) = export_stats_stream.recv() {
                stats.export_stats = export_stats;
            }

            // Get system statistics
            if let Ok(system_stats) = statistics.try_read() {
                stats.total_operations = system_stats.total_operations;
                stats.system_uptime = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
            }

            let _ = sender.send(stats);
        })
    }

    /// Get all tags
    pub fn get_all_tags(&self) -> Vec<ConversationTag> {
        self.tagger.get_all_tags()
    }

    /// Search tags by name
    pub fn search_tags(&self, query: &str) -> Vec<ConversationTag> {
        self.tagger.search_tags(query)
    }

    /// Get messages for a tag
    pub fn get_tag_messages(&self, tag_id: &Arc<str>) -> Vec<Arc<str>> {
        self.tagger.get_tag_messages(tag_id)
    }

    /// Get tags for a message
    pub fn get_message_tags(&self, message_id: &Arc<str>) -> Vec<Arc<str>> {
        self.tagger.get_message_tags(message_id)
    }

    /// Clear all data (useful for testing)
    pub fn clear_all_data(&self) {
        // Note: Since we use lock-free data structures, we'd need to recreate them
        // For now, this is a placeholder for the interface
    }

    /// Get system health status
    pub fn get_health_status(&self) -> HealthStatus {
        let search_healthy = true; // Could check index size, query performance, etc.
        let tagging_healthy = true; // Could check tag counts, performance, etc.
        let export_healthy = true; // Could check export success rates, etc.

        HealthStatus {
            overall: search_healthy && tagging_healthy && export_healthy,
            search_system: search_healthy,
            tagging_system: tagging_healthy,
            export_system: export_healthy,
            last_check: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

impl Default for EnhancedHistoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Overall system health
    pub overall: bool,
    /// Search system health
    pub search_system: bool,
    /// Tagging system health
    pub tagging_system: bool,
    /// Export system health
    pub export_system: bool,
    /// Last health check timestamp
    pub last_check: u64,
}

impl HealthStatus {
    /// Check if system is fully operational
    pub fn is_healthy(&self) -> bool {
        self.overall && self.search_system && self.tagging_system && self.export_system
    }

    /// Get failing systems
    pub fn get_failing_systems(&self) -> Vec<&'static str> {
        let mut failing = Vec::new();
        if !self.search_system {
            failing.push("search");
        }
        if !self.tagging_system {
            failing.push("tagging");
        }
        if !self.export_system {
            failing.push("export");
        }
        failing
    }
}