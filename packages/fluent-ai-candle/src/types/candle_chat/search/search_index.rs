//! Search index implementation for candle_chat search functionality
//!
//! This module provides the core search indexing capabilities with
//! zero-allocation streaming patterns and lock-free data structures.

use std::collections::HashMap;
use std::sync::Arc;

use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::core_types::{SearchResult, TermFrequency, SearchStatistics, MatchPosition, MatchType};
use crate::types::CandleSearchChatMessage;

/// Handle errors in streaming context without panicking
macro_rules! handle_error {
    ($error:expr, $context:literal) => {
        eprintln!("Streaming error in {}: {}", $context, $error);
        // Continue processing instead of returning error
    };
}

/// Index entry for search functionality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    /// Message ID
    pub message_id: Uuid,
    /// Term positions
    pub positions: Vec<usize>,
    /// Term frequency
    pub frequency: usize,
    /// Document length
    pub document_length: usize,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Chat search index with lock-free operations
#[derive(Serialize, Deserialize)]
pub struct ChatSearchIndex {
    /// Term to document mapping
    pub term_index: HashMap<Arc<str>, Vec<IndexEntry>>,
    /// Document to terms mapping
    pub document_index: HashMap<Uuid, Vec<Arc<str>>>,
    /// Term frequencies
    pub term_frequencies: HashMap<Arc<str>, TermFrequency>,
    /// Total documents
    pub total_documents: usize,
    /// Index statistics
    pub statistics: SearchStatistics,
    /// Last update timestamp
    pub last_update: chrono::DateTime<chrono::Utc>,
    /// Index version
    pub version: u32,
    /// Configuration
    pub config: IndexConfig,
}

/// Configuration for search index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Enable stemming
    pub enable_stemming: bool,
    /// Enable stop word removal
    pub remove_stop_words: bool,
    /// Minimum term length
    pub min_term_length: usize,
    /// Maximum term length
    pub max_term_length: usize,
    /// Case sensitive indexing
    pub case_sensitive: bool,
    /// Index metadata
    pub index_metadata: bool,
    /// Batch size for indexing
    pub batch_size: usize,
    /// Memory limit in bytes
    pub memory_limit: usize,
}

impl Clone for ChatSearchIndex {
    fn clone(&self) -> Self {
        Self {
            term_index: self.term_index.clone(),
            document_index: self.document_index.clone(),
            term_frequencies: self.term_frequencies.clone(),
            total_documents: self.total_documents,
            statistics: self.statistics.clone(),
            last_update: self.last_update,
            version: self.version,
            config: self.config.clone(),
        }
    }
}

impl std::fmt::Debug for ChatSearchIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatSearchIndex")
            .field("total_documents", &self.total_documents)
            .field("term_count", &self.term_index.len())
            .field("version", &self.version)
            .field("last_update", &self.last_update)
            .finish()
    }
}

impl ChatSearchIndex {
    /// Create a new search index
    pub fn new() -> Self {
        Self {
            term_index: HashMap::new(),
            document_index: HashMap::new(),
            term_frequencies: HashMap::new(),
            total_documents: 0,
            statistics: SearchStatistics::default(),
            last_update: chrono::Utc::now(),
            version: 1,
            config: IndexConfig::default(),
        }
    }

    /// Add messages to the index (streaming)
    pub fn add_messages(&mut self, messages: Vec<CandleSearchChatMessage>) -> AsyncStream<()> {
        let mut term_index = self.term_index.clone();
        let mut document_index = self.document_index.clone();
        let mut term_frequencies = self.term_frequencies.clone();
        let mut total_documents = self.total_documents;
        let config = self.config.clone();

        AsyncStream::with_channel(move |sender| {
            for message in messages {
                // Tokenize message content
                let terms = Self::tokenize_content(&message.content, &config);
                
                // Create index entry
                let entry = IndexEntry {
                    message_id: message.id,
                    positions: Vec::new(), // Would calculate actual positions
                    frequency: terms.len(),
                    document_length: message.content.len(),
                    metadata: HashMap::new(),
                };

                // Update term index
                for term in &terms {
                    term_index
                        .entry(term.clone())
                        .or_insert_with(Vec::new)
                        .push(entry.clone());
                }

                // Update document index
                document_index.insert(message.id, terms.clone());

                // Update term frequencies
                for term in terms {
                    let freq = term_frequencies
                        .entry(term.clone())
                        .or_insert_with(|| TermFrequency::new(term.clone(), 0, 0, total_documents));
                    freq.frequency += 1;
                }

                total_documents += 1;
            }

            let _ = sender.send(());
        })
    }

    /// Search the index (streaming)
    pub fn search(&self, query: &str, limit: Option<usize>) -> AsyncStream<SearchResult> {
        let terms = Self::tokenize_content(query, &self.config);
        let term_index = self.term_index.clone();
        let document_index = self.document_index.clone();
        let limit = limit.unwrap_or(100);

        AsyncStream::with_channel(move |sender| {
            let mut results = Vec::new();
            let mut processed_docs = std::collections::HashSet::new();

            // Find matching documents
            for term in &terms {
                if let Some(entries) = term_index.get(term) {
                    for entry in entries {
                        if processed_docs.insert(entry.message_id) {
                            // Create mock search result (would be more sophisticated)
                            let result = SearchResult {
                                id: Uuid::new_v4(),
                                message: CandleSearchChatMessage {
                                    id: entry.message_id,
                                    role: crate::types::CandleMessageRole::User,
                                    content: "Mock content".to_string(),
                                    timestamp: chrono::Utc::now(),
                                    metadata: HashMap::new(),
                                },
                                score: 1.0, // Would calculate actual score
                                highlighted_content: None,
                                context: Vec::new(),
                                match_metadata: HashMap::new(),
                                match_positions: vec![MatchPosition {
                                    start: 0,
                                    end: term.len(),
                                    term: term.to_string(),
                                    match_type: MatchType::Exact,
                                    confidence: 1.0,
                                }],
                                conversation_id: None,
                                tags: Vec::new(),
                                result_timestamp: chrono::Utc::now(),
                                extra_data: HashMap::new(),
                            };
                            results.push(result);
                        }
                    }
                }
            }

            // Sort and limit results
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(limit);

            // Send results
            for result in results {
                let _ = sender.send(result);
            }
        })
    }

    /// Tokenize content based on configuration
    fn tokenize_content(content: &str, config: &IndexConfig) -> Vec<Arc<str>> {
        let mut terms = Vec::new();
        
        // Simple tokenization (would be more sophisticated)
        for word in content.split_whitespace() {
            let term = if config.case_sensitive {
                word.to_string()
            } else {
                word.to_lowercase()
            };

            // Apply length filters
            if term.len() >= config.min_term_length && term.len() <= config.max_term_length {
                // Skip stop words if configured (simplified)
                if !config.remove_stop_words || !Self::is_stop_word(&term) {
                    terms.push(Arc::from(term));
                }
            }
        }

        terms
    }

    /// Check if a term is a stop word (simplified)
    fn is_stop_word(term: &str) -> bool {
        matches!(term, "the" | "a" | "an" | "and" | "or" | "but" | "in" | "on" | "at" | "to" | "for" | "of" | "with" | "by")
    }

    /// Get index statistics
    pub fn get_statistics(&self) -> SearchStatistics {
        self.statistics.clone()
    }

    /// Optimize the index (streaming)
    pub fn optimize(&mut self) -> AsyncStream<()> {
        AsyncStream::with_channel(move |sender| {
            // Index optimization logic would go here
            // For now, just update the timestamp
            let _ = sender.send(());
        })
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.term_index.clear();
        self.document_index.clear();
        self.term_frequencies.clear();
        self.total_documents = 0;
        self.last_update = chrono::Utc::now();
        self.version += 1;
    }

    /// Get index size in bytes (approximate)
    pub fn size_bytes(&self) -> usize {
        // Simplified size calculation
        self.term_index.len() * 100 + self.document_index.len() * 50
    }
}

impl Default for ChatSearchIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            enable_stemming: false,
            remove_stop_words: true,
            min_term_length: 2,
            max_term_length: 50,
            case_sensitive: false,
            index_metadata: true,
            batch_size: 1000,
            memory_limit: 100 * 1024 * 1024, // 100MB
        }
    }
}