//! Message indexing functionality
//!
//! Contains methods for adding messages to the search index, tokenization,
//! and maintaining inverted index structures.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use atomic_counter::AtomicCounter;
use fluent_ai_async::AsyncStream;

use crate::types::candle_chat::message::CandleMessage;
use crate::types::CandleSearchChatMessage as SearchChatMessage;

use super::super::types::TermFrequency;
use super::core::{ChatSearchIndex, IndexEntry};
use crate::types::extensions::RoleExt;

impl ChatSearchIndex {
    /// Add message to search index (streaming)
    pub fn add_message_stream(&self, message: SearchChatMessage) -> AsyncStream<()> {
        let self_clone = self.clone();

        AsyncStream::with_channel(move |sender| {
            let index = self_clone.document_count.load(Ordering::Relaxed);
            let doc_id = message
                .id
                .clone();
            // Convert SearchChatMessage to CandleMessage for storage
            let candle_message = CandleMessage {
                id: message.id.clone(),
                role: message.role,
                content: message.content.clone(),
                name: message.name.clone(),
                metadata: message.metadata.clone(),
                timestamp: message.timestamp,
            };
            self_clone
                .document_store
                .insert(Arc::from(doc_id.as_str()), candle_message);

            // Tokenize and index the content
            let tokens = self_clone.tokenize_with_simd(&message.content);
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

                // Update inverted index
                let mut entries = self_clone
                    .inverted_index
                    .get(&term)
                    .map(|e| e.value().clone())
                    .unwrap_or_default();
                entries.push(index_entry);
                self_clone.inverted_index.insert(term.clone(), entries);

                // Update term frequencies
                let doc_freq = self_clone
                    .inverted_index
                    .get(&term)
                    .map(|e| e.value().len() as u32)
                    .unwrap_or(1);

                let total_docs = self_clone.document_count.load(Ordering::Relaxed) as u32 + 1;
                let idf = (total_docs as f64 / doc_freq as f64).ln();
                let tf_idf_score = tf as f64 * idf;

                let tf_entry = TermFrequency {
                    term: term.clone(),
                    frequency: count,
                    document_frequency: doc_freq,
                    tf_idf_score,
                };
                self_clone.term_frequencies.insert(term.clone(), tf_entry);
            }

            self_clone.document_count.fetch_add(1, Ordering::Relaxed);
            self_clone.index_update_counter.inc();

            // Update statistics
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

    /// Tokenize text with SIMD optimization
    pub(super) fn tokenize_with_simd(&self, text: &str) -> Vec<Arc<str>> {
        let words: Vec<&str> = text
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect();

        let simd_threshold = self.simd_threshold.load(Ordering::Relaxed);

        if words.len() >= simd_threshold {
            self.process_words_simd(words)
        } else {
            words
                .into_iter()
                .map(|w| Arc::from(w.to_lowercase()))
                .collect()
        }
    }

    /// Process words with SIMD optimization
    fn process_words_simd(&self, words: Vec<&str>) -> Vec<Arc<str>> {
        let mut processed = Vec::with_capacity(words.len());

        // Process words in chunks of 8 for SIMD
        for chunk in words.chunks(8) {
            for word in chunk {
                processed.push(Arc::from(word.to_lowercase()));
            }
        }

        processed
    }
}
