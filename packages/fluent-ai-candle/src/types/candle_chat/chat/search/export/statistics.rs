//! Export statistics and progress tracking

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use atomic_counter::{AtomicCounter, ConsistentCounter};
use serde::{Deserialize, Serialize};

/// Statistics for export operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportStatistics {
    /// Total messages exported
    pub messages_exported: usize,
    /// Total conversations exported
    pub conversations_exported: usize,
    /// Export duration in milliseconds
    pub duration_ms: u64,
    /// Export file size in bytes
    pub file_size_bytes: usize,
    /// Compression ratio (if applicable)
    pub compression_ratio: Option<f64>,
    /// Number of errors encountered
    pub error_count: usize,
    /// Export start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// Export end time
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
}

impl Default for ExportStatistics {
    fn default() -> Self {
        Self {
            messages_exported: 0,
            conversations_exported: 0,
            duration_ms: 0,
            file_size_bytes: 0,
            compression_ratio: None,
            error_count: 0,
            start_time: chrono::Utc::now(),
            end_time: None,
        }
    }
}

impl ExportStatistics {
    /// Create new export statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark export as completed
    pub fn complete(&mut self) {
        self.end_time = Some(chrono::Utc::now());
        if let Some(end_time) = self.end_time {
            self.duration_ms = (end_time - self.start_time).num_milliseconds() as u64;
        }
    }

    /// Add exported message count
    pub fn add_messages(&mut self, count: usize) {
        self.messages_exported += count;
    }

    /// Add exported conversation count
    pub fn add_conversations(&mut self, count: usize) {
        self.conversations_exported += count;
    }

    /// Add error count
    pub fn add_errors(&mut self, count: usize) {
        self.error_count += count;
    }

    /// Set file size
    pub fn set_file_size(&mut self, size: usize) {
        self.file_size_bytes = size;
    }

    /// Set compression ratio
    pub fn set_compression_ratio(&mut self, ratio: f64) {
        self.compression_ratio = Some(ratio);
    }

    /// Get export progress as percentage
    pub fn progress_percentage(&self, total_messages: usize) -> f64 {
        if total_messages == 0 {
            return 100.0;
        }
        (self.messages_exported as f64 / total_messages as f64) * 100.0
    }

    /// Get export rate (messages per second)
    pub fn export_rate(&self) -> f64 {
        if self.duration_ms == 0 {
            return 0.0;
        }
        (self.messages_exported as f64) / (self.duration_ms as f64 / 1000.0)
    }
}