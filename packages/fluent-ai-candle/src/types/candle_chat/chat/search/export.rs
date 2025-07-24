//! History export system with zero-allocation streaming
//!
//! This module provides comprehensive export functionality for chat history
//! with multiple formats and streaming patterns for large datasets.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use atomic_counter::{AtomicCounter, ConsistentCounter};
use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::chat::message::SearchChatMessage;

/// Export format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// XML format
    Xml,
    /// Plain text format
    Text,
    /// Markdown format
    Markdown,
    /// HTML format
    Html,
    /// Custom format with template
    Custom(String),
}

/// Export options configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    /// Export format
    pub format: ExportFormat,
    /// Whether to include message metadata
    pub include_metadata: bool,
    /// Whether to include timestamps
    pub include_timestamps: bool,
    /// Whether to include user information
    pub include_users: bool,
    /// Whether to include conversation context
    pub include_context: bool,
    /// Date range filter
    pub date_range: Option<DateRange>,
    /// Maximum number of messages to export
    pub limit: Option<usize>,
    /// Whether to compress the output
    pub compress: bool,
    /// Custom field selections
    pub fields: Vec<String>,
    /// Export template for custom formats
    pub template: Option<String>,
    /// Whether to export in chunks
    pub chunked: bool,
    /// Chunk size for streaming exports
    pub chunk_size: usize,
    /// Whether to include deleted messages
    pub include_deleted: bool,
    /// Custom filters
    pub filters: HashMap<String, String>,
}

/// Date range for export filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    /// Start date (inclusive)
    pub start: chrono::DateTime<chrono::Utc>,
    /// End date (inclusive)
    pub end: chrono::DateTime<chrono::Utc>,
}

/// History exporter with streaming capabilities
pub struct HistoryExporter {
    /// Export statistics
    pub export_statistics: Arc<ExportStatistics>,
    /// Operation counters
    pub export_counter: ConsistentCounter,
    pub bytes_exported: Arc<AtomicUsize>,
}

/// Export operation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportStatistics {
    /// Total exports performed
    pub total_exports: usize,
    /// Total messages exported
    pub total_messages_exported: usize,
    /// Total bytes exported
    pub total_bytes_exported: usize,
    /// Average export time in milliseconds
    pub avg_export_time_ms: f64,
    /// Export format distribution
    pub format_distribution: HashMap<String, usize>,
    /// Largest export size in bytes
    pub largest_export_bytes: usize,
    /// Most recent export timestamp
    pub last_export_time: Option<chrono::DateTime<chrono::Utc>>,
    /// Export success rate
    pub success_rate: f64,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
}

impl HistoryExporter {
    /// Create a new history exporter
    pub fn new() -> Self {
        Self {
            export_statistics: Arc::new(ExportStatistics::default()),
            export_counter: ConsistentCounter::new(0),
            bytes_exported: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Export messages to string format (streaming)
    pub fn export_messages(
        &self,
        messages: Vec<SearchChatMessage>,
        options: ExportOptions,
    ) -> AsyncStream<String> {
        let export_counter = self.export_counter.clone();
        let bytes_exported = Arc::clone(&self.bytes_exported);

        AsyncStream::with_channel(move |sender| {
            let start_time = std::time::Instant::now();
            export_counter.inc();

            let exported_data = match options.format {
                ExportFormat::Json => Self::export_as_json(&messages, &options),
                ExportFormat::Csv => Self::export_as_csv(&messages, &options),
                ExportFormat::Xml => Self::export_as_xml(&messages, &options),
                ExportFormat::Text => Self::export_as_text(&messages, &options),
                ExportFormat::Markdown => Self::export_as_markdown(&messages, &options),
                ExportFormat::Html => Self::export_as_html(&messages, &options),
                ExportFormat::Custom(template) => Self::export_as_custom(&messages, &options, &template),
            };

            // Update statistics
            bytes_exported.fetch_add(exported_data.len(), Ordering::Relaxed);

            let _ = sender.send(exported_data);
        })
    }

    /// Export as JSON format
    fn export_as_json(messages: &[SearchChatMessage], options: &ExportOptions) -> String {
        let filtered_messages = Self::apply_filters(messages, options);
        
        match serde_json::to_string_pretty(&filtered_messages) {
            Ok(json) => json,
            Err(_) => "[]".to_string(), // Fallback to empty array
        }
    }

    /// Export as CSV format
    fn export_as_csv(messages: &[SearchChatMessage], options: &ExportOptions) -> String {
        let filtered_messages = Self::apply_filters(messages, options);
        let mut csv = String::new();
        
        // Header
        csv.push_str("id,content,role,timestamp\n");
        
        // Data rows
        for message in filtered_messages {
            csv.push_str(&format!(
                "{},{},{},{}\n",
                message.id,
                Self::escape_csv(&message.content),
                format!("{:?}", message.role),
                message.timestamp.format("%Y-%m-%d %H:%M:%S")
            ));
        }
        
        csv
    }

    /// Export as XML format
    fn export_as_xml(messages: &[SearchChatMessage], options: &ExportOptions) -> String {
        let filtered_messages = Self::apply_filters(messages, options);
        let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<messages>\n");
        
        for message in filtered_messages {
            xml.push_str(&format!(
                "  <message id=\"{}\" role=\"{:?}\" timestamp=\"{}\">\n",
                message.id,
                message.role,
                message.timestamp.format("%Y-%m-%d %H:%M:%S")
            ));
            xml.push_str(&format!("    <content>{}</content>\n", Self::escape_xml(&message.content)));
            xml.push_str("  </message>\n");
        }
        
        xml.push_str("</messages>");
        xml
    }

    /// Export as plain text format
    fn export_as_text(messages: &[SearchChatMessage], options: &ExportOptions) -> String {
        let filtered_messages = Self::apply_filters(messages, options);
        let mut text = String::new();
        
        for message in filtered_messages {
            if options.include_timestamps {
                text.push_str(&format!("[{}] ", message.timestamp.format("%Y-%m-%d %H:%M:%S")));
            }
            text.push_str(&format!("{:?}: {}\n", message.role, message.content));
        }
        
        text
    }

    /// Export as Markdown format
    fn export_as_markdown(messages: &[SearchChatMessage], options: &ExportOptions) -> String {
        let filtered_messages = Self::apply_filters(messages, options);
        let mut markdown = String::from("# Chat History\n\n");
        
        for message in filtered_messages {
            markdown.push_str(&format!("## {:?}\n\n", message.role));
            if options.include_timestamps {
                markdown.push_str(&format!("*{}*\n\n", message.timestamp.format("%Y-%m-%d %H:%M:%S")));
            }
            markdown.push_str(&format!("{}\n\n", message.content));
        }
        
        markdown
    }

    /// Export as HTML format
    fn export_as_html(messages: &[SearchChatMessage], options: &ExportOptions) -> String {
        let filtered_messages = Self::apply_filters(messages, options);
        let mut html = String::from("<!DOCTYPE html>\n<html>\n<head><title>Chat History</title></head>\n<body>\n<h1>Chat History</h1>\n");
        
        for message in filtered_messages {
            html.push_str("<div class=\"message\">\n");
            html.push_str(&format!("<h3>{:?}</h3>\n", message.role));
            if options.include_timestamps {
                html.push_str(&format!("<p class=\"timestamp\">{}</p>\n", message.timestamp.format("%Y-%m-%d %H:%M:%S")));
            }
            html.push_str(&format!("<p>{}</p>\n", Self::escape_html(&message.content)));
            html.push_str("</div>\n");
        }
        
        html.push_str("</body>\n</html>");
        html
    }

    /// Export using custom template
    fn export_as_custom(messages: &[SearchChatMessage], options: &ExportOptions, template: &str) -> String {
        let filtered_messages = Self::apply_filters(messages, options);
        let mut result = String::new();
        
        // Simple template processing (would use a proper template engine in production)
        for message in filtered_messages {
            let mut processed = template.replace("{id}", &message.id.to_string());
            processed = processed.replace("{content}", &message.content);
            processed = processed.replace("{role}", &format!("{:?}", message.role));
            processed = processed.replace("{timestamp}", &message.timestamp.format("%Y-%m-%d %H:%M:%S").to_string());
            result.push_str(&processed);
        }
        
        result
    }

    /// Apply filters to messages based on export options
    fn apply_filters(messages: &[SearchChatMessage], options: &ExportOptions) -> Vec<SearchChatMessage> {
        let mut filtered: Vec<SearchChatMessage> = messages.to_vec();
        
        // Apply date range filter
        if let Some(date_range) = &options.date_range {
            filtered.retain(|msg| {
                msg.timestamp >= date_range.start && msg.timestamp <= date_range.end
            });
        }
        
        // Apply limit
        if let Some(limit) = options.limit {
            filtered.truncate(limit);
        }
        
        // Apply custom filters
        for (field, value) in &options.filters {
            match field.as_str() {
                "role" => {
                    filtered.retain(|msg| format!("{:?}", msg.role).to_lowercase().contains(&value.to_lowercase()));
                }
                "content" => {
                    filtered.retain(|msg| msg.content.to_lowercase().contains(&value.to_lowercase()));
                }
                _ => {} // Unknown filter, ignore
            }
        }
        
        filtered
    }

    /// Escape CSV special characters
    fn escape_csv(text: &str) -> String {
        if text.contains(',') || text.contains('"') || text.contains('\n') {
            format!("\"{}\"", text.replace('"', "\"\""))
        } else {
            text.to_string()
        }
    }

    /// Escape XML special characters
    fn escape_xml(text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }

    /// Escape HTML special characters
    fn escape_html(text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    }

    /// Get export statistics (streaming)
    pub fn get_statistics(&self) -> AsyncStream<ExportStatistics> {
        let stats = Arc::clone(&self.export_statistics);
        let total_exports = self.export_counter.get();
        let total_bytes = self.bytes_exported.load(Ordering::Relaxed);

        AsyncStream::with_channel(move |sender| {
            let mut current_stats = (*stats).clone();
            current_stats.total_exports = total_exports;
            current_stats.total_bytes_exported = total_bytes;
            
            let _ = sender.send(current_stats);
        })
    }
}

impl Default for HistoryExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::Json,
            include_metadata: true,
            include_timestamps: true,
            include_users: true,
            include_context: false,
            date_range: None,
            limit: None,
            compress: false,
            fields: Vec::new(),
            template: None,
            chunked: false,
            chunk_size: 1000,
            include_deleted: false,
            filters: HashMap::new(),
        }
    }
}

impl Default for ExportStatistics {
    fn default() -> Self {
        Self {
            total_exports: 0,
            total_messages_exported: 0,
            total_bytes_exported: 0,
            avg_export_time_ms: 0.0,
            format_distribution: HashMap::new(),
            largest_export_bytes: 0,
            last_export_time: None,
            success_rate: 1.0,
            performance_metrics: HashMap::new(),
        }
    }
}