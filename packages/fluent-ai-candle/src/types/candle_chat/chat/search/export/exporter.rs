//! History exporter implementation with streaming support

use std::sync::atomic::{AtomicUsize, Ordering};
use crate::types::candle_chat::search::tagging::ConsistentCounter;
use fluent_ai_async::AsyncStream;

use super::types::ExportOptions;
use super::statistics::ExportStatistics;
use super::formats::get_format_handler;
use crate::types::candle_chat::chat::message::SearchChatMessage;

// Note: Clone implementation for ConsistentCounter removed due to orphan trait rules
// Use AtomicU64 or custom wrapper types instead for cloneable counters

/// History exporter with streaming capabilities
pub struct HistoryExporter {
    /// Export statistics counter
    stats_counter: ConsistentCounter,
    /// Active export operations
    active_exports: AtomicUsize,
    /// Export ID generator
    export_id_counter: AtomicUsize}

impl Clone for HistoryExporter {
    fn clone(&self) -> Self {
        Self {
            stats_counter: self.stats_counter.clone(),
            active_exports: AtomicUsize::new(self.active_exports.load(std::sync::atomic::Ordering::SeqCst)),
            export_id_counter: AtomicUsize::new(self.export_id_counter.load(std::sync::atomic::Ordering::SeqCst)),
        }
    }
}

impl Default for HistoryExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl HistoryExporter {
    /// Create new history exporter
    pub fn new() -> Self {
        Self {
            stats_counter: ConsistentCounter::new(0),
            active_exports: AtomicUsize::new(0),
            export_id_counter: AtomicUsize::new(0)}
    }

    /// Export messages to string with specified options
    pub fn export_to_string(
        &self,
        messages: Vec<SearchChatMessage>,
        options: ExportOptions,
    ) -> AsyncStream<String> {
        use fluent_ai_async::{emit, handle_error};

        AsyncStream::with_channel(move |sender| {
            let format_handler = get_format_handler(&options.format);
            
            // Generate header
            let header = match format_handler.generate_header(&options) {
                Ok(header) => header,
                Err(e) => {
                    handle_error!(format!("Header generation failed: {}", e), "export_to_string");
                }
            };

            // Process messages in batches
            let mut result = header;
            let batch_size = options.batch_size;
            
            for chunk in messages.chunks(batch_size) {
                let formatted = match format_handler.format_messages(chunk, &options) {
                    Ok(formatted) => formatted,
                    Err(e) => {
                        handle_error!(format!("Message formatting failed: {}", e), "export_to_string");
                    }
                };
                result.push_str(&formatted);
            }

            // Generate footer
            let footer = match format_handler.generate_footer(&options) {
                Ok(footer) => footer,
                Err(e) => {
                    handle_error!(format!("Footer generation failed: {}", e), "export_to_string");
                }
            };
            result.push_str(&footer);

            emit!(sender, result);
        })
    }

    /// Export messages with streaming progress updates
    pub fn export_with_progress(
        &self,
        messages: Vec<SearchChatMessage>,
        options: ExportOptions,
    ) -> AsyncStream<ExportProgress> {
        use fluent_ai_async::{emit, handle_error};

        let total_messages = messages.len();

        AsyncStream::with_channel(move |sender| {
            let format_handler = get_format_handler(&options.format);
            let mut statistics = ExportStatistics::new();
            let batch_size = options.batch_size;

            // Emit initial progress
            emit!(sender, ExportProgress {
                percentage: 0.0,
                messages_processed: 0,
                total_messages,
                current_batch: 0,
                statistics: statistics.clone()});

            // Process messages in batches
            let mut result = String::new();
            let mut messages_processed = 0;

            // Generate header
            let header_result = format_handler.generate_header(&options);
            if let Err(e) = &header_result {
                handle_error!(format!("Header generation failed: {}", e), "export_with_progress");
            }
            let header = header_result.unwrap();
            result.push_str(&header);

            for (batch_idx, chunk) in messages.chunks(batch_size).enumerate() {
                let format_result = format_handler.format_messages(chunk, &options);
                match format_result {
                    Ok(formatted) => {
                        result.push_str(&formatted);
                        messages_processed += chunk.len();
                        statistics.add_messages(chunk.len());

                        let percentage = (messages_processed as f64 / total_messages as f64) * 100.0;
                        
                        emit!(sender, ExportProgress {
                            percentage,
                            messages_processed,
                            total_messages,
                            current_batch: batch_idx + 1,
                            statistics: statistics.clone()});
                    }
                    Err(e) => {
                        statistics.add_errors(1);
                        handle_error!(format!("Batch {} formatting failed: {}", batch_idx, e), "export_with_progress");
                    }
                }
            }

            // Generate footer
            let footer_result = format_handler.generate_footer(&options);
            if let Err(e) = &footer_result {
                handle_error!(format!("Footer generation failed: {}", e), "export_with_progress");
            }
            let footer = footer_result.unwrap();
            result.push_str(&footer);

            statistics.complete();
            statistics.set_file_size(result.len());

            // Emit final progress
            emit!(sender, ExportProgress {
                percentage: 100.0,
                messages_processed,
                total_messages,
                current_batch: messages.chunks(batch_size).count(),
                statistics});
        })
    }

    /// Get current export statistics
    pub fn get_statistics(&self) -> ExportStatistics {
        let mut stats = ExportStatistics::default();
        stats.messages_exported = self.stats_counter.get();
        stats
    }

    /// Get number of active exports
    pub fn active_export_count(&self) -> usize {
        self.active_exports.load(Ordering::Relaxed)
    }

    /// Generate unique export ID
    pub fn generate_export_id(&self) -> String {
        let id = self.export_id_counter.fetch_add(1, Ordering::Relaxed);
        format!("export_{}", id)
    }
}

/// Export progress information
#[derive(Debug, Clone)]
pub struct ExportProgress {
    /// Progress percentage (0.0 - 100.0)
    pub percentage: f64,
    /// Number of messages processed
    pub messages_processed: usize,
    /// Total number of messages
    pub total_messages: usize,
    /// Current batch number
    pub current_batch: usize,
    /// Export statistics
    pub statistics: ExportStatistics}