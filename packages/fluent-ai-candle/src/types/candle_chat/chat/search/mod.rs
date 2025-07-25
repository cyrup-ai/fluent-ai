//! Chat search module with decomposed submodules
//!
//! This module provides comprehensive chat search functionality with clear
//! separation of concerns across focused submodules.

pub mod types;
pub mod index;
pub mod tagging;
pub mod export;
pub mod manager;
pub mod searcher;
pub mod query;
pub mod ranking;

// Re-export core types with explicit, non-conflicting imports
// Types module - available types only
pub use types::{DateRange, SearchResult, SearchQuery, SearchStatistics, TermFrequency, MatchPosition, MatchType, SearchError, QueryError, SearchFilter, SearchMetadata};
pub use index::{ChatSearchIndex, SearchIndex, IndexBuilder, IndexStatistics};
// Tagging module - qualified statistics  
pub use tagging::{ConversationTagger, TaggingStatistics as TagStats};
// Export module - qualified statistics
pub use export::{HistoryExporter, ExportStatistics as ExportStats};
pub use export::{ExportFormat, ExportOptions}; 
// Export format handlers directly from their module
pub use export::formats::{JsonFormatHandler, CsvFormatHandler, MarkdownFormatHandler, FormatHandler, get_format_handler};
// Manager module 
pub use manager::{SearchManager, SearchConfiguration};
// Searcher module - qualified SearchOptions
pub use searcher::{ChatSearcher, SearchOptions as SearcherOptions};
// Query module - qualified SearchOptions
pub use query::{QueryProcessor, SearchOptions as QueryOptions};
// Ranking module - qualified DateRange to avoid conflict
pub use ranking::{ResultRanker, RankingConfig, DateRange as RankingDateRange};

use fluent_ai_async::AsyncStream;

/// Stream collection trait to provide .collect() method for future-like behavior
pub trait StreamCollect<T> {
    fn collect_sync(self) -> AsyncStream<Vec<T>>;
}

impl<T> StreamCollect<T> for AsyncStream<T>
where
    T: Send + 'static,
{
    fn collect_sync(self) -> AsyncStream<Vec<T>> {
        AsyncStream::with_channel(move |sender| {
            // AsyncStream doesn't implement Iterator - use proper streaming pattern
            let results = Vec::new();
            // For now, send empty results - this would need proper stream collection logic
            let _ = sender.send(results);
        })
    }
}

// Removed unused handle_error macro

