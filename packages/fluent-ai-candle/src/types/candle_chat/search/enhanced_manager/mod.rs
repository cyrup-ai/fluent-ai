//! Enhanced history manager for candle_chat search system
//!
//! This module provides advanced history management capabilities with
//! caching, statistics, and performance optimization features.
//!
//! Decomposed into focused modules following ≤300-line architectural constraint:
//! - config.rs: Configuration structures and builder pattern
//! - statistics.rs: Statistics tracking and performance monitoring  
//! - operations.rs: Core operational functionality

use std::collections::HashMap;

use fluent_ai_async::AsyncStream;
use super::core_types::{SearchQuery, SearchResult};
use super::search_index::ChatSearchIndex;
use super::conversation_tagging::{ConversationTagger, ConversationTag};
use super::history_export::{HistoryExporter, ExportOptions, ExportChunk};
use crate::types::CandleSearchChatMessage;

// Re-export sub-module types
pub use config::{ManagerConfig, BackupConfig, CleanupResult, OptimizationResult, HistoryManagerBuilder};
pub use statistics::HistoryManagerStatistics;
pub use operations::ManagerOperations;

// Sub-modules
pub mod config;
pub mod statistics;
pub mod operations;

/// Enhanced history manager with advanced features
pub struct EnhancedHistoryManager {
    /// Core operations handler
    operations: ManagerOperations,
    /// Manager statistics
    pub statistics: HistoryManagerStatistics,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
}

impl EnhancedHistoryManager {
    /// Create a new enhanced history manager
    pub fn new(config: ManagerConfig) -> Self {
        let search_index = ChatSearchIndex::new();
        let tagger = ConversationTagger::new();
        let exporter = HistoryExporter::new(ExportOptions::default());
        
        let operations = ManagerOperations::new(
            search_index,
            tagger,
            exporter,
            config,
        );

        Self {
            operations,
            statistics: HistoryManagerStatistics::default(),
            performance_metrics: HashMap::new(),
        }
    }

    /// Create with builder pattern
    pub fn builder() -> HistoryManagerBuilder {
        HistoryManagerBuilder::new()
    }

    /// Add messages to the manager (streaming)
    pub fn add_messages(&mut self, messages: Vec<CandleSearchChatMessage>) -> AsyncStream<()> {
        self.statistics.update_message_count(messages.len());
        self.operations.add_messages(messages, &mut self.statistics)
    }

    /// Search messages with caching (streaming)
    pub fn search_messages(&mut self, query: &SearchQuery) -> AsyncStream<SearchResult> {
        self.operations.search_messages(query, &mut self.statistics)
    }

    /// Tag conversations automatically (streaming)
    pub fn tag_conversations(&mut self, messages: Vec<CandleSearchChatMessage>) -> AsyncStream<(uuid::Uuid, Vec<ConversationTag>)> {
        self.operations.tag_conversations(messages)
    }

    /// Export history with options (streaming)
    pub fn export_history(&mut self, messages: Vec<CandleSearchChatMessage>) -> AsyncStream<ExportChunk> {
        self.operations.export_history(messages)
    }

    /// Perform cleanup operations (streaming)
    pub fn cleanup(&mut self) -> AsyncStream<CleanupResult> {
        self.operations.cleanup(&mut self.statistics)
    }

    /// Optimize performance (streaming)
    pub fn optimize(&mut self) -> AsyncStream<OptimizationResult> {
        self.operations.optimize(&mut self.statistics)
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> &HistoryManagerStatistics {
        &self.statistics
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &HashMap<String, f64> {
        &self.performance_metrics
    }

    /// Update configuration
    pub fn update_config(&mut self, config: ManagerConfig) {
        self.operations.config = config;
    }

    /// Reset all statistics
    pub fn reset_statistics(&mut self) {
        self.statistics.reset();
        self.performance_metrics.clear();
    }
}

impl Clone for EnhancedHistoryManager {
    fn clone(&self) -> Self {
        // Create new instance with same config
        let mut new_manager = Self::new(self.operations.config.clone());
        new_manager.statistics = self.statistics.clone();
        new_manager.performance_metrics = self.performance_metrics.clone();
        new_manager
    }
}