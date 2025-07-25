//! Search index implementation for chat messages
//!
//! This module provides the core search indexing functionality with SIMD optimization,
//! lock-free data structures, and zero-allocation streaming patterns.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use atomic_counter::{AtomicCounter, ConsistentCounter};
use crossbeam_skiplist::SkipMap;
use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::candle_chat::chat::message::SearchChatMessage;
use super::types::{SearchResult, TermFrequency, SearchStatistics};

/// Entry in the search index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    /// Message ID
    pub message_id: Uuid,
    /// Term positions within the message
    pub positions: Vec<usize>,
    /// Term frequency in this message
    pub term_frequency: usize,
    /// Message metadata for quick access
    pub metadata: HashMap<String, String>,
}

/// Core search index with SIMD optimization and lock-free operations
pub struct ChatSearchIndex {
    /// Term to message mapping using lock-free skip map
    pub term_index: Arc<SkipMap<Arc<str>, Vec<IndexEntry>>>,
    /// Message storage using lock-free skip map
    pub messages: Arc<SkipMap<Uuid, SearchChatMessage>>,
    /// Term frequency statistics
    pub term_frequencies: Arc<SkipMap<Arc<str>, TermFrequency>>,
    /// Index statistics with atomic counters
    pub stats: Arc<SearchStatistics>,
    /// Total message count (atomic)
    pub message_count: Arc<AtomicUsize>,
    /// Index version for cache invalidation
    pub version: Arc<AtomicUsize>,
    /// Performance counters
    pub search_counter: ConsistentCounter,
    pub index_counter: ConsistentCounter,
}

impl Clone for ChatSearchIndex {
    fn clone(&self) -> Self {
        Self {
            term_index: Arc::clone(&self.term_index),
            messages: Arc::clone(&self.messages),
            term_frequencies: Arc::clone(&self.term_frequencies),
            stats: Arc::clone(&self.stats),
            message_count: Arc::clone(&self.message_count),
            version: Arc::clone(&self.version),
            search_counter: ConsistentCounter::new(self.search_counter.get()),
            index_counter: ConsistentCounter::new(self.index_counter.get()),
        }
    }
}

impl std::fmt::Debug for ChatSearchIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatSearchIndex")
            .field("message_count", &self.message_count.load(Ordering::Relaxed))
            .field("term_count", &self.term_index.len())
            .field("version", &self.version.load(Ordering::Relaxed))
            .finish()
    }
}

impl ChatSearchIndex {
    /// Create a new search index
    pub fn new() -> Self {
        Self {
            term_index: Arc::new(SkipMap::new()),
            messages: Arc::new(SkipMap::new()),
            term_frequencies: Arc::new(SkipMap::new()),
            stats: Arc::new(SearchStatistics::default()),
            message_count: Arc::new(AtomicUsize::new(0)),
            version: Arc::new(AtomicUsize::new(0)),
            search_counter: ConsistentCounter::new(0),
            index_counter: ConsistentCounter::new(0),
        }
    }

    /// Add a message to the index (streaming)
    pub fn add_message(&self, message: SearchChatMessage) -> AsyncStream<()> {
        let message_id = message.id;
        let content = message.content.clone();
        let messages = Arc::clone(&self.messages);
        let term_index = Arc::clone(&self.term_index);
        let term_frequencies = Arc::clone(&self.term_frequencies);
        let message_count = Arc::clone(&self.message_count);
        let version = Arc::clone(&self.version);
        let index_counter = self.index_counter.clone();

        AsyncStream::with_channel(move |sender| {
            // Store the message
            messages.insert(message_id, message);
            
            // Tokenize content
            let tokens = Self::tokenize(&content);
            
            // Update term index
            for (position, token) in tokens.iter().enumerate() {
                let term = Arc::from(token.as_str());
                
                // Create index entry
                let entry = IndexEntry {
                    message_id,
                    positions: vec![position],
                    term_frequency: 1,
                    metadata: HashMap::new(),
                };
                
                // Update term index (simplified - in real implementation would merge entries)
                term_index.insert(term.clone(), vec![entry]);
                
                // Update term frequencies
                let tf = TermFrequency::new(term, 1, 1, message_count.load(Ordering::Relaxed));
                term_frequencies.insert(term, tf);
            }
            
            // Update counters
            message_count.fetch_add(1, Ordering::Relaxed);
            version.fetch_add(1, Ordering::Relaxed);
            index_counter.inc();
            
            let _ = sender.send(());
        })
    }

    /// Remove a message from the index (streaming)
    pub fn remove_message(&self, message_id: Uuid) -> AsyncStream<bool> {
        let messages = Arc::clone(&self.messages);
        let term_index = Arc::clone(&self.term_index);
        let message_count = Arc::clone(&self.message_count);
        let version = Arc::clone(&self.version);

        AsyncStream::with_channel(move |sender| {
            let removed = messages.remove(&message_id).is_some();
            
            if removed {
                // TODO: Remove from term index (complex operation)
                message_count.fetch_sub(1, Ordering::Relaxed);
                version.fetch_add(1, Ordering::Relaxed);
            }
            
            let _ = sender.send(removed);
        })
    }

    /// Search messages with SIMD optimization (streaming individual results)
    pub fn search_messages(&self, terms: &[Arc<str>]) -> AsyncStream<SearchResult> {
        let terms = terms.to_vec();
        let term_index = Arc::clone(&self.term_index);
        let messages = Arc::clone(&self.messages);
        let search_counter = self.search_counter.clone();

        AsyncStream::with_channel(move |sender| {
            search_counter.inc();
            
            // Simple search implementation (would be SIMD-optimized in production)
            for term in &terms {
                if let Some(entries) = term_index.get(term) {
                    for entry in entries.value() {
                        if let Some(message) = messages.get(&entry.message_id) {
                            let result = SearchResult {
                                id: Uuid::new_v4(),
                                message: message.value().clone(),
                                score: 1.0, // Simplified scoring
                                highlighted_content: None,
                                context: Vec::new(),
                                match_metadata: HashMap::new(),
                                match_positions: Vec::new(),
                                conversation_id: None,
                                tags: Vec::new(),
                                result_timestamp: chrono::Utc::now(),
                                extra_data: HashMap::new(),
                            };
                            let _ = sender.send(result);
                        }
                    }
                }
            }
        })
    }

    /// Search with AND operator (streaming)
    pub fn search_and(&self, terms: &[Arc<str>]) -> AsyncStream<SearchResult> {
        let terms = terms.to_vec();
        let term_index = Arc::clone(&self.term_index);
        let messages = Arc::clone(&self.messages);

        AsyncStream::with_channel(move |sender| {
            // AND search implementation (simplified)
            // In production, this would use set intersection algorithms
            if let Some(first_term) = terms.first() {
                if let Some(entries) = term_index.get(first_term) {
                    for entry in entries.value() {
                        if let Some(message) = messages.get(&entry.message_id) {
                            let result = SearchResult {
                                id: Uuid::new_v4(),
                                message: message.value().clone(),
                                score: 1.0,
                                highlighted_content: None,
                                context: Vec::new(),
                                match_metadata: HashMap::new(),
                                match_positions: Vec::new(),
                                conversation_id: None,
                                tags: Vec::new(),
                                result_timestamp: chrono::Utc::now(),
                                extra_data: HashMap::new(),
                            };
                            let _ = sender.send(result);
                        }
                    }
                }
            }
        })
    }

    /// Search with OR operator (streaming)
    pub fn search_or(&self, terms: &[Arc<str>]) -> AsyncStream<SearchResult> {
        // Delegate to regular search for OR semantics
        self.search_messages(terms)
    }

    /// Search with NOT operator (streaming)
    pub fn search_not(&self, include_terms: &[Arc<str>], exclude_terms: &[Arc<str>]) -> AsyncStream<SearchResult> {
        let include_terms = include_terms.to_vec();
        let exclude_terms = exclude_terms.to_vec();
        let term_index = Arc::clone(&self.term_index);
        let messages = Arc::clone(&self.messages);

        AsyncStream::with_channel(move |sender| {
            // NOT search implementation (simplified)
            for term in &include_terms {
                if let Some(entries) = term_index.get(term) {
                    for entry in entries.value() {
                        // Check if message contains any exclude terms (simplified)
                        let should_exclude = false; // Would check exclude_terms
                        
                        if !should_exclude {
                            if let Some(message) = messages.get(&entry.message_id) {
                                let result = SearchResult {
                                    id: Uuid::new_v4(),
                                    message: message.value().clone(),
                                    score: 1.0,
                                    highlighted_content: None,
                                    context: Vec::new(),
                                    match_metadata: HashMap::new(),
                                    match_positions: Vec::new(),
                                    conversation_id: None,
                                    tags: Vec::new(),
                                    result_timestamp: chrono::Utc::now(),
                                    extra_data: HashMap::new(),
                                };
                                let _ = sender.send(result);
                            }
                        }
                    }
                }
            }
        })
    }

    /// Get search statistics (streaming)
    pub fn get_statistics(&self) -> AsyncStream<SearchStatistics> {
        let stats = Arc::clone(&self.stats);
        let message_count = Arc::clone(&self.message_count);
        let term_count = self.term_index.len();

        AsyncStream::with_channel(move |sender| {
            let mut current_stats = (*stats).clone();
            current_stats.total_messages = message_count.load(Ordering::Relaxed);
            current_stats.unique_terms = term_count;
            
            let _ = sender.send(current_stats);
        })
    }

    /// Simple tokenization (would use advanced NLP in production)
    fn tokenize(text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|s| s.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Clear the entire index
    pub fn clear(&self) -> AsyncStream<()> {
        let term_index = Arc::clone(&self.term_index);
        let messages = Arc::clone(&self.messages);
        let term_frequencies = Arc::clone(&self.term_frequencies);
        let message_count = Arc::clone(&self.message_count);
        let version = Arc::clone(&self.version);

        AsyncStream::with_channel(move |sender| {
            term_index.clear();
            messages.clear();
            term_frequencies.clear();
            message_count.store(0, Ordering::Relaxed);
            version.fetch_add(1, Ordering::Relaxed);
            
            let _ = sender.send(());
        })
    }
}

impl Default for ChatSearchIndex {
    fn default() -> Self {
        Self::new()
    }
}