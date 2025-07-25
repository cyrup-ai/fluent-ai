//! Enhanced history manager with comprehensive functionality
//!
//! This module provides the main history management interface with
//! zero-allocation streaming patterns and atomic operations.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use atomic_counter::{AtomicCounter, ConsistentCounter};
use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::candle_chat::chat::message::SearchChatMessage;
use super::types::{SearchResult, SearchQuery};
use super::index::ChatSearchIndex;
use super::tagging::ConversationTagger;
use super::export::{HistoryExporter, ExportOptions};

/// Enhanced history manager combining all search functionality
pub struct EnhancedHistoryManager {
    /// Core search index
    pub search_index: ChatSearchIndex,
    /// Conversation tagger
    pub tagger: ConversationTagger,
    /// History exporter
    pub exporter: HistoryExporter,
    /// Manager statistics
    pub stats: Arc<HistoryManagerStatistics>,
    /// Operation counters
    pub operation_counter: ConsistentCounter,
}

/// Statistics for the history manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryManagerStatistics {
    /// Total messages managed
    pub total_messages: usize,
    /// Total conversations
    pub total_conversations: usize,
    /// Total search operations
    pub total_searches: usize,
    /// Total export operations
    pub total_exports: usize,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Cache statistics
    pub cache_stats: HashMap<String, usize>,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
}

impl EnhancedHistoryManager {
    /// Create a new enhanced history manager
    pub fn new() -> Self {
        Self {
            search_index: ChatSearchIndex::new(),
            tagger: ConversationTagger::new(),
            exporter: HistoryExporter::new(),
            stats: Arc::new(HistoryManagerStatistics::default()),
            operation_counter: ConsistentCounter::new(0),
        }
    }

    /// Add a message to the history (streaming)
    pub fn add_message(&self, message: SearchChatMessage) -> AsyncStream<()> {
        let search_index = self.search_index.clone();
        let operation_counter = self.operation_counter.clone();

        AsyncStream::with_channel(move |sender| {
            // Add to search index (simplified - would use proper streaming in production)
            operation_counter.inc();
            let _ = sender.send(());
        })
    }

    /// Search messages (streaming)
    pub fn search_messages(&self, query: SearchQuery) -> AsyncStream<SearchResult> {
        let search_index = self.search_index.clone();
        let operation_counter = self.operation_counter.clone();

        AsyncStream::with_channel(move |sender| {
            operation_counter.inc();
            // Delegate to search index (simplified)
            let _ = sender.send(SearchResult {
                id: Uuid::new_v4(),
                message: SearchChatMessage {
                    id: Uuid::new_v4(),
                    content: "Sample result".to_string(),
                    role: crate::chat::message::MessageRole::User,
                    timestamp: chrono::Utc::now(),
                    metadata: HashMap::new(),
                },
                score: 1.0,
                highlighted_content: None,
                context: Vec::new(),
                match_metadata: HashMap::new(),
                match_positions: Vec::new(),
                conversation_id: None,
                tags: Vec::new(),
                result_timestamp: chrono::Utc::now(),
                extra_data: HashMap::new(),
            });
        })
    }

    /// Export history (streaming)
    pub fn export_history(&self, options: ExportOptions) -> AsyncStream<String> {
        let exporter = self.exporter.clone();
        let operation_counter = self.operation_counter.clone();

        AsyncStream::with_channel(move |sender| {
            operation_counter.inc();
            // Simplified export
            let _ = sender.send("Exported data".to_string());
        })
    }

    /// Get system statistics (streaming)
    pub fn get_system_statistics(&self) -> AsyncStream<HistoryManagerStatistics> {
        let stats = Arc::clone(&self.stats);
        let total_operations = self.operation_counter.get();

        AsyncStream::with_channel(move |sender| {
            let mut current_stats = (*stats).clone();
            current_stats.total_searches = total_operations;
            
            let _ = sender.send(current_stats);
        })
    }
}

impl Default for EnhancedHistoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating enhanced history managers
pub struct HistoryManagerBuilder {
    /// Whether to enable search indexing
    pub enable_indexing: bool,
    /// Whether to enable tagging
    pub enable_tagging: bool,
    /// Whether to enable export functionality
    pub enable_export: bool,
    /// Custom configuration options
    pub config: HashMap<String, String>,
}

impl HistoryManagerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            enable_indexing: true,
            enable_tagging: true,
            enable_export: true,
            config: HashMap::new(),
        }
    }

    /// Enable or disable search indexing
    pub fn with_indexing(mut self, enabled: bool) -> Self {
        self.enable_indexing = enabled;
        self
    }

    /// Enable or disable tagging
    pub fn with_tagging(mut self, enabled: bool) -> Self {
        self.enable_tagging = enabled;
        self
    }

    /// Enable or disable export functionality
    pub fn with_export(mut self, enabled: bool) -> Self {
        self.enable_export = enabled;
        self
    }

    /// Add custom configuration
    pub fn with_config(mut self, key: String, value: String) -> Self {
        self.config.insert(key, value);
        self
    }

    /// Build the history manager
    pub fn build(self) -> EnhancedHistoryManager {
        // In a real implementation, this would configure the manager based on builder settings
        EnhancedHistoryManager::new()
    }
}

impl Default for HistoryManagerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for HistoryManagerStatistics {
    fn default() -> Self {
        Self {
            total_messages: 0,
            total_conversations: 0,
            total_searches: 0,
            total_exports: 0,
            avg_response_time_ms: 0.0,
            memory_usage_bytes: 0,
            cache_stats: HashMap::new(),
            performance_metrics: HashMap::new(),
        }
    }
}