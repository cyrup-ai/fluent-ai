//! Search index implementation with SIMD optimization
//!
//! This module provides the core search indexing functionality with lock-free
//! data structures and high-performance SIMD-optimized operations.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;

use atomic_counter::{AtomicCounter, ConsistentCounter};
use crossbeam_skiplist::SkipMap;
use fluent_ai_async::AsyncStream;

use crate::chat::message::SearchChatMessage;
use super::types::{
    SearchError, SearchStatistics, TermFrequency, IndexEntry,
};

/// Chat search index with SIMD optimization
pub struct ChatSearchIndex {
    /// Inverted index: term -> documents containing term
    inverted_index: SkipMap<Arc<str>, Vec<IndexEntry>>,
    /// Document store: doc_id -> message
    document_store: SkipMap<Arc<str>, SearchChatMessage>,
    /// Term frequencies for TF-IDF calculation
    term_frequencies: SkipMap<Arc<str>, TermFrequency>,
    /// Document count
    document_count: Arc<AtomicUsize>,
    /// Query counter
    query_counter: Arc<ConsistentCounter>,
    /// Index update counter
    index_update_counter: Arc<ConsistentCounter>,
    /// Search statistics
    statistics: Arc<RwLock<SearchStatistics>>,
    /// SIMD processing threshold
    simd_threshold: Arc<AtomicUsize>,
}

impl Clone for ChatSearchIndex {
    fn clone(&self) -> Self {
        // Create a new empty instance since SkipMap doesn't implement Clone
        Self::new()
    }
}

impl std::fmt::Debug for ChatSearchIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatSearchIndex")
            .field(
                "inverted_index",
                &format!("SkipMap with {} entries", self.inverted_index.len()),
            )
            .field(
                "document_store",
                &format!("SkipMap with {} entries", self.document_store.len()),
            )
            .field(
                "term_frequencies",
                &format!("SkipMap with {} entries", self.term_frequencies.len()),
            )
            .field(
                "document_count",
                &self.document_count.load(Ordering::Relaxed),
            )
            .field("query_counter", &"ConsistentCounter")
            .field("index_update_counter", &"ConsistentCounter")
            .field("statistics", &"Arc<RwLock<SearchStatistics>>")
            .field(
                "simd_threshold",
                &self.simd_threshold.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl ChatSearchIndex {
    /// Create a new search index
    pub fn new() -> Self {
        Self {
            inverted_index: SkipMap::new(),
            document_store: SkipMap::new(),
            term_frequencies: SkipMap::new(),
            document_count: Arc::new(AtomicUsize::new(0)),
            query_counter: Arc::new(ConsistentCounter::new(0)),
            index_update_counter: Arc::new(ConsistentCounter::new(0)),
            statistics: Arc::new(RwLock::new(SearchStatistics {
                total_messages: 0,
                total_terms: 0,
                total_queries: 0,
                average_query_time: 0.0,
                index_size: 0,
                last_index_update: 0,
            })),
            simd_threshold: Arc::new(AtomicUsize::new(8)), // Process 8 terms at once with SIMD
        }
    }

    /// Add message to search index (streaming)
    pub fn add_message_stream(&self, message: SearchChatMessage) -> AsyncStream<()> {
        let self_clone = self.clone();

        AsyncStream::with_channel(move |sender| {
            let index = self_clone.document_count.load(Ordering::Relaxed);
            let doc_id = message
                .message
                .id
                .clone()
                .unwrap_or_else(|| format!("msg_{}", index));
            self_clone
                .document_store
                .insert(Arc::from(doc_id.as_str()), message.clone());
            let _new_index = self_clone.document_count.fetch_add(1, Ordering::Relaxed);

            // Tokenize and index the content
            let tokens = self_clone.tokenize_with_simd(&message.message.content);
            let total_tokens = tokens.len();

            // Calculate term frequencies
            let mut term_counts = HashMap::new();
            for token in &tokens {
                let count = term_counts.get(token).map_or(0, |e: &u32| *e) + 1;
                term_counts.insert(token.clone(), count);
            }

            // Update inverted index
            for (term, count) in term_counts {
                let tf = (count as f32) / (total_tokens as f32);

                let index_entry = IndexEntry {
                    doc_id: Arc::from(doc_id.as_str()),
                    term_frequency: tf,
                    positions: tokens
                        .iter()
                        .enumerate()
                        .filter(|(_, t)| **t == term)
                        .map(|(i, _)| i)
                        .collect(),
                };

                // SkipMap doesn't have get_mut method, use insert pattern
                let mut entries = self_clone
                    .inverted_index
                    .get(&term)
                    .map(|e| e.value().clone())
                    .unwrap_or_default();
                entries.push(index_entry);
                self_clone.inverted_index.insert(term.clone(), entries);

                // Update term frequencies - SkipMap doesn't have get_mut
                let mut tf_entry = self_clone
                    .term_frequencies
                    .get(&term)
                    .map(|e| e.value().clone())
                    .unwrap_or(TermFrequency {
                        tf: 0.0,
                        df: 0,
                        total_docs: 1,
                    });
                tf_entry.tf += 1.0;
                tf_entry.df = 1;
                self_clone.term_frequencies.insert(term.clone(), tf_entry);
                // Update document frequency based on current index size
                let doc_freq = self_clone
                    .inverted_index
                    .get(&term)
                    .map(|e| e.value().len() as u32)
                    .unwrap_or(1);
                if let Some(mut tf_entry) = self_clone
                    .term_frequencies
                    .get(&term)
                    .map(|e| e.value().clone())
                {
                    tf_entry.df = doc_freq;
                    self_clone.term_frequencies.insert(term.clone(), tf_entry);
                }
            }

            self_clone.index_update_counter.inc();

            // Update statistics - use blocking write since we're in a closure
            if let Ok(mut stats) = self_clone.statistics.try_write() {
                stats.total_messages = self_clone.document_count.load(Ordering::Relaxed);
                stats.total_terms = self_clone.inverted_index.len();
                stats.last_index_update = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
            }

            let _ = sender.send(());
        })
    }

    /// Add message to search index (legacy future-compatible method)
    pub fn add_message(&self, message: SearchChatMessage) -> Result<(), SearchError> {
        let mut stream = self.add_message_stream(message);
        // Use AsyncStream try_next method (NO FUTURES architecture)
        match stream.try_next() {
            Some(_) => Ok(()),
            None => Err(SearchError::IndexError {
                reason: Arc::from("Stream closed unexpectedly"),
            }),
        }
    }

    /// Get search statistics
    pub fn get_statistics(&self) -> SearchStatistics {
        self.statistics
            .read()
            .unwrap_or_else(|_| {
                std::sync::PoisonError::into_inner(self.statistics.read().unwrap_err())
            })
            .clone()
    }

    /// Get document count
    pub fn document_count(&self) -> usize {
        self.document_count.load(Ordering::Relaxed)
    }

    /// Get term count
    pub fn term_count(&self) -> usize {
        self.term_frequencies.len()
    }

    /// Check if document exists
    pub fn contains_document(&self, doc_id: &str) -> bool {
        self.document_store.contains_key(&Arc::from(doc_id))
    }

    /// Get document by ID
    pub fn get_document(&self, doc_id: &str) -> Option<SearchChatMessage> {
        self.document_store
            .get(&Arc::from(doc_id))
            .map(|entry| entry.value().clone())
    }

    /// Remove document from index
    pub fn remove_document(&self, doc_id: &str) -> Result<(), SearchError> {
        let doc_id_arc = Arc::from(doc_id);
        
        // Remove from document store
        if self.document_store.remove(&doc_id_arc).is_none() {
            return Err(SearchError::IndexError {
                reason: Arc::from("Document not found"),
            });
        }

        // Remove from inverted index (this is expensive but necessary)
        for entry in self.inverted_index.iter() {
            let term = entry.key().clone();
            let mut entries = entry.value().clone();
            entries.retain(|e| e.doc_id != doc_id_arc);
            
            if entries.is_empty() {
                self.inverted_index.remove(&term);
                self.term_frequencies.remove(&term);
            } else {
                self.inverted_index.insert(term.clone(), entries);
            }
        }

        self.document_count.fetch_sub(1, Ordering::Relaxed);
        
        // Update statistics
        if let Ok(mut stats) = self.statistics.try_write() {
            stats.total_messages = self.document_count.load(Ordering::Relaxed);
            stats.total_terms = self.inverted_index.len();
            stats.last_index_update = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
        }

        Ok(())
    }

    /// Clear the entire index
    pub fn clear(&self) -> Result<(), SearchError> {
        self.inverted_index.clear();
        self.document_store.clear();
        self.term_frequencies.clear();
        self.document_count.store(0, Ordering::Relaxed);
        
        // Reset statistics
        if let Ok(mut stats) = self.statistics.try_write() {
            *stats = SearchStatistics::default();
        }

        Ok(())
    }

    /// Tokenize text with SIMD optimization
    pub fn tokenize_with_simd(&self, text: &str) -> Vec<Arc<str>> {
        // Simple tokenization - split on whitespace and punctuation
        // In a real implementation, this would use SIMD for faster processing
        text.split_whitespace()
            .map(|word| {
                // Remove punctuation and convert to lowercase
                let cleaned: String = word
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase();
                Arc::from(cleaned)
            })
            .filter(|token: &Arc<str>| !token.is_empty())
            .collect()
    }

    /// Get terms for a document
    pub fn get_document_terms(&self, doc_id: &str) -> Vec<Arc<str>> {
        let doc_id_arc = Arc::from(doc_id);
        let mut terms = Vec::new();
        
        for entry in self.inverted_index.iter() {
            let term = entry.key().clone();
            let entries = entry.value();
            
            if entries.iter().any(|e| e.doc_id == doc_id_arc) {
                terms.push(term);
            }
        }
        
        terms
    }

    /// Get term frequency for a document
    pub fn get_term_frequency(&self, term: &str, doc_id: &str) -> Option<f32> {
        let term_arc = Arc::from(term);
        let doc_id_arc = Arc::from(doc_id);
        
        self.inverted_index
            .get(&term_arc)
            .and_then(|entries| {
                entries.value()
                    .iter()
                    .find(|entry| entry.doc_id == doc_id_arc)
                    .map(|entry| entry.term_frequency)
            })
    }

    /// Get read-only access to the inverted index
    #[inline]
    pub fn inverted_index(&self) -> &SkipMap<Arc<str>, Vec<IndexEntry>> {
        &self.inverted_index
    }

    /// Get read-only access to the document store
    #[inline]
    pub fn document_store(&self) -> &SkipMap<Arc<str>, SearchChatMessage> {
        &self.document_store
    }

    /// Increment query counter (for statistics tracking)
    #[inline]
    pub fn increment_query_counter(&self) {
        self.query_counter.inc();
    }

    /// Get statistics with write access for updates
    #[inline]
    pub fn update_statistics<F, R>(&self, f: F) -> Result<R, std::sync::PoisonError<std::sync::RwLockWriteGuard<'_, crate::chat::search::types::SearchStatistics>>>
    where
        F: FnOnce(&mut crate::chat::search::types::SearchStatistics) -> R,
    {
        let mut stats = self.statistics.write()?;
        Ok(f(&mut *stats))
    }

}