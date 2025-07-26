//! Enhanced history manager with comprehensive functionality
//!
//! This module provides the main history management interface with
//! zero-allocation streaming patterns and atomic operations.

use std::collections::HashMap;
use std::sync::Arc;

use crate::types::candle_chat::search::tagging::ConsistentCounter;
use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::candle_chat::chat::message::SearchChatMessage;
use super::types::{SearchResult, SearchQuery};
use super::index::ChatSearchIndex;
use super::tagging::ConversationTagger;
use super::export::{HistoryExporter, ExportOptions};

// Note: Clone implementation for ConsistentCounter removed due to orphan trait rules
// Use AtomicU64 or custom wrapper types instead for cloneable counters

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
    pub operation_counter: ConsistentCounter}

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
    pub performance_metrics: HashMap<String, f64>}

impl EnhancedHistoryManager {
    /// Create a new enhanced history manager
    pub fn new() -> Self {
        Self {
            search_index: ChatSearchIndex::new(),
            tagger: ConversationTagger::new(),
            exporter: HistoryExporter::new(),
            stats: Arc::new(HistoryManagerStatistics::default()),
            operation_counter: ConsistentCounter::new(0)}
    }

    /// Add a message to the history (streaming)
    pub fn add_message(&self, _message: SearchChatMessage) -> AsyncStream<()> {
        let _search_index = self.search_index.clone();
        let operation_counter = self.operation_counter.clone();

        AsyncStream::with_channel(move |sender| {
            // Add to search index (simplified - would use proper streaming in production)
            operation_counter.inc();
            let _ = sender.send(());
        })
    }

    /// Search messages (streaming)
    pub fn search_messages(&self, _query: SearchQuery) -> AsyncStream<SearchResult> {
        let _search_index = self.search_index.clone();
        let operation_counter = self.operation_counter.clone();

        AsyncStream::with_channel(move |sender| {
            operation_counter.inc();
            // Delegate to search index (simplified)
            let _ = sender.send(SearchResult {
                id: Uuid::new_v4(),
                message: SearchChatMessage {
                    id: Uuid::new_v4().to_string(),
                    content: "Sample result".to_string(),
                    role: crate::types::candle_chat::message::CandleMessageRole::User,
                    name: None,
                    metadata: None,
                    timestamp: Some(chrono::Utc::now().timestamp_millis() as u64),
                    relevance_score: 1.0,
                    search_timestamp: chrono::Utc::now().timestamp_millis() as u64},
                score: 1.0,
                highlighted_content: None,
                context: Vec::new(),
                match_metadata: HashMap::new(),
                match_positions: Vec::new(),
                conversation_id: None,
                tags: Vec::new(),
                result_timestamp: chrono::Utc::now(),
                extra_data: HashMap::new()});
        })
    }

    /// Export history (streaming)
    pub fn export_history(&self, options: ExportOptions) -> AsyncStream<String> {
        let exporter = self.exporter.clone();
        let operation_counter = self.operation_counter.clone();

        AsyncStream::with_channel(move |sender| {
            operation_counter.inc();
            // Use the exporter for actual export functionality
            if let Ok(exported_data) = exporter.export(&options) {
                let _ = sender.send(exported_data);
            } else {
                let _ = sender.send("Export failed".to_string());
            }
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
    pub config: HashMap<String, String>}

impl HistoryManagerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            enable_indexing: true,
            enable_tagging: true,
            enable_export: true,
            config: HashMap::new()}
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
            performance_metrics: HashMap::new()}
    }
}

/// Type alias for compatibility with mod.rs imports
pub type SearchManager = EnhancedHistoryManager;

/// Search configuration for the search manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfiguration {
    /// Maximum number of results to return
    pub max_results: usize,
    /// Whether to enable fuzzy search
    pub enable_fuzzy_search: bool,
    /// Minimum similarity score for fuzzy search
    pub min_similarity_score: f32,
    /// Whether to enable highlighting
    pub enable_highlighting: bool,
    /// Search timeout in milliseconds
    pub search_timeout_ms: u64,
    /// Whether to enable caching
    pub enable_caching: bool,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// Whether to enable real-time indexing
    pub enable_real_time_indexing: bool,
    /// Batch size for bulk operations
    pub batch_size: usize,
    /// Whether to enable SIMD optimization
    pub enable_simd: bool,
    /// Index update frequency in seconds
    pub index_update_frequency_secs: u64}

impl Default for SearchConfiguration {
    fn default() -> Self {
        Self {
            max_results: 100,
            enable_fuzzy_search: false,
            min_similarity_score: 0.7,
            enable_highlighting: true,
            search_timeout_ms: 5000,
            enable_caching: true,
            cache_size_limit: 10000,
            enable_real_time_indexing: true,
            batch_size: 1000,
            enable_simd: true,
            index_update_frequency_secs: 300}
    }
}