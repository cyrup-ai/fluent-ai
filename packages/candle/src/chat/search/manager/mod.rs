use std::sync::Arc;

use crossbeam_skiplist::SkipMap;
use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};

use crate::chat::message::SearchChatMessage;
use super::{SearchError, SearchQuery, SearchResult};

/// Enhanced history management system
#[derive(Debug)]
pub struct EnhancedHistoryManager {
    /// Search index
    search_index: super::index::ChatSearchIndex,
    /// Conversation tagger
    tagger: super::tagger::ConversationTagger,
    /// History exporter
    exporter: super::export::HistoryExporter,
    /// Message store
    messages: SkipMap<String, SearchChatMessage>,
    /// Message index by timestamp
    message_timestamps: SkipMap<i64, String>,
}

impl EnhancedHistoryManager {
    /// Create a new enhanced history manager
    pub fn new() -> Self {
        Self {
            search_index: super::index::ChatSearchIndex::new(),
            tagger: super::tagger::ConversationTagger::new(),
            exporter: super::export::HistoryExporter::new(),
            messages: SkipMap::new(),
            message_timestamps: SkipMap::new(),
        }
    }

    /// Add message to history manager (streaming)
    pub fn add_message_stream(&self, message: SearchChatMessage) -> AsyncStream<()> {
        let message_id = message.id.clone();
        let timestamp = message.timestamp;
        
        AsyncStream::with_channel(move |sender| {
            // Add to message store
            self.messages.insert(message_id.clone(), message);
            
            // Index by timestamp
            self.message_timestamps.insert(timestamp, message_id);
            
            // Index for search
            let _ = self.search_index.add_message_stream(message);
            
            // Emit completion
            let _ = sender.try_send(());
        })
    }

    /// Search messages (streaming)
    pub fn search_messages_stream(&self, query: SearchQuery) -> AsyncStream<SearchResult> {
        self.search_index.search_stream(query)
    }
}
