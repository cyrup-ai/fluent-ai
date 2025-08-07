use crossbeam_skiplist::SkipMap;
use std::sync::Arc;
use fluent_ai_async::AsyncStream;

use crate::domain::chat::message::CandleSearchChatMessage;
use super::{SearchQuery as CandleSearchQuery, SearchResult as CandleSearchResult};

/// Enhanced history management system with domain-level integration
#[derive(Debug)]
pub struct CandleEnhancedHistoryManager {
    /// Search index
    search_index: Arc<super::index::ChatSearchIndex>,
    /// Conversation tagger
    #[allow(dead_code)] // TODO: Integrate tagging with history management
    tagger: Arc<super::tagger::CandleConversationTagger>,
    /// History exporter
    #[allow(dead_code)] // TODO: Implement history export functionality
    exporter: Arc<super::export::HistoryExporter>,
    /// Message store
    messages: Arc<SkipMap<String, CandleSearchChatMessage>>,
    /// Message index by timestamp
    message_timestamps: Arc<SkipMap<i64, String>>,
}

impl CandleEnhancedHistoryManager {
    /// Create a new enhanced history manager
    pub fn new() -> Self {
        Self {
            search_index: Arc::new(super::index::ChatSearchIndex::new()),
            tagger: Arc::new(super::tagger::CandleConversationTagger::new()),
            exporter: Arc::new(super::export::HistoryExporter::new()),
            messages: Arc::new(SkipMap::new()),
            message_timestamps: Arc::new(SkipMap::new()),
        }
    }

    /// Add message to history manager (streaming)
    pub fn add_message_stream(&self, message: CandleSearchChatMessage) -> AsyncStream<()> {
        let message_id = message.message.id.clone();
        let timestamp = message.message.timestamp;
        let message_clone = message.clone();
        let messages = Arc::clone(&self.messages);
        let message_timestamps = Arc::clone(&self.message_timestamps);
        let search_index: Arc<super::index::ChatSearchIndex> = Arc::clone(&self.search_index);
        
        AsyncStream::with_channel(move |sender| {
            // Add to message store - only if message_id is present
            if let Some(id) = message_id.clone() {
                messages.insert(id.clone(), message_clone.clone());
                
                // Index by timestamp - only if timestamp is present
                if let Some(ts_u64) = timestamp {
                    let ts_i64 = ts_u64 as i64; // Convert u64 to i64 for timestamp indexing
                    message_timestamps.insert(ts_i64, id);
                }
            }
            
            // Index for search
            let _ = search_index.add_message_stream(message_clone);
            
            // Emit completion
            let _ = sender.try_send(());
        })
    }

    /// Search messages (streaming)
    pub fn search_messages_stream(&self, query: CandleSearchQuery) -> AsyncStream<CandleSearchResult> {
        // TODO: Implement full search functionality in Phase 2.3
        // For now, return empty stream until domain ChatSearchIndex has search methods
        let _query = query; // Avoid unused parameter warning
        AsyncStream::with_channel(move |_sender| {
            // Placeholder - will be implemented with enhanced search features
        })
    }
}