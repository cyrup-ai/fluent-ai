//! History exporter implementation with streaming support

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use atomic_counter::{AtomicCounter, ConsistentCounter};
use fluent_ai_async::AsyncStream;
use uuid::Uuid;

use super::types::{ExportOptions, ExportFormat};
use super::statistics::ExportStatistics;
use super::formats::{get_format_handler, FormatHandler};
use crate::chat::message::SearchChatMessage;

/// History exporter with streaming capabilities
pub struct HistoryExporter {
    /// Export statistics counter
    stats_counter: ConsistentCounter,
    /// Active export operations
    active_exports: AtomicUsize,
    /// Export ID generator
    export_id_counter: AtomicUsize,
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
            export_id_counter: AtomicUsize::new(0),
        }
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
                Ok(h) => h,
                Err(e) => {
                    handle_error!(format!("Header generation failed: {}", e), "export_to_string");
                    return;
                }
            };

            // Process messages in batches
            let mut result = header;
            let batch_size = options.batch_size;
            
            for chunk in messages.chunks(batch_size) {
                match format_handler.format_messages(chunk, &options) {
                    Ok(formatted) => {
                        result.push_str(&formatted);
                    }
                    Err(e) => {
                        handle_error!(format!("Message formatting failed: {}", e), "export_to_string");
                        continue;
                    }
                }
            }

            // Generate footer
            match format_handler.generate_footer(&options) {
                Ok(footer) => {
                    result.push_str(&footer);
                }
                Err(e) => {
                    handle_error!(format!("Footer generation failed: {}", e), "export_to_string");
                }
            }

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
                statistics: statistics.clone(),
            });

            // Process messages in batches
            let mut result = String::new();
            let mut messages_processed = 0;

            // Generate header
            match format_handler.generate_header(&options) {
                Ok(header) => result.push_str(&header),
                Err(e) => {
                    handle_error!(format!("Header generation failed: {}", e), "export_with_progress");
                    return;
                }
            }

            for (batch_idx, chunk) in messages.chunks(batch_size).enumerate() {
                match format_handler.format_messages(chunk, &options) {
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
                            statistics: statistics.clone(),
                        });
                    }
                    Err(e) => {
                        handle_error!(format!("Batch {} formatting failed: {}", batch_idx, e), "export_with_progress");
                        statistics.add_errors(1);
                        continue;
                    }
                }
            }

            // Generate footer
            match format_handler.generate_footer(&options) {
                Ok(footer) => result.push_str(&footer),
                Err(e) => {
                    handle_error!(format!("Footer generation failed: {}", e), "export_with_progress");
                }
            }

            statistics.complete();
            statistics.set_file_size(result.len());

            // Emit final progress
            emit!(sender, ExportProgress {
                percentage: 100.0,
                messages_processed,
                total_messages,
                current_batch: messages.chunks(batch_size).count(),
                statistics,
            });
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
    pub statistics: ExportStatistics,
}