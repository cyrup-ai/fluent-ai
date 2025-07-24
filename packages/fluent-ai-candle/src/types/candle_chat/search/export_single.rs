//! Export functionality for conversation history
//!
//! This module provides comprehensive export capabilities with multiple formats,
//! compression options, and streaming-first architecture for optimal performance.

use std::sync::Arc;
use std::time::Instant;

use atomic_counter::{AtomicCounter, ConsistentCounter};
use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::chat::message::{MessageRole, SearchChatMessage};
use super::types::{DateRange, StreamCollect, SearchError};

/// Export format enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Markdown format
    Markdown,
    /// HTML format
    Html,
    /// XML format
    Xml,
    /// Plain text format
    PlainText,
}

/// Export options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    /// Export format
    pub format: ExportFormat,
    /// Include metadata
    pub include_metadata: bool,
    /// Include tags
    pub include_tags: bool,
    /// Include timestamps
    pub include_timestamps: bool,
    /// Compress output
    pub compress: bool,
    /// Date range filter
    pub date_range: Option<DateRange>,
    /// User filter
    pub user_filter: Option<Arc<str>>,
    /// Tag filter
    pub tag_filter: Option<Vec<Arc<str>>>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::Json,
            include_metadata: true,
            include_tags: true,
            include_timestamps: true,
            compress: false,
            date_range: None,
            user_filter: None,
            tag_filter: None,
        }
    }
}

/// Export statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExportStatistics {
    /// Total number of exports performed
    pub total_exports: usize,
    /// Total number of messages exported
    pub total_messages_exported: usize,
    /// Most popular export format
    pub most_popular_format: Option<ExportFormat>,
    /// Average export time in milliseconds
    pub average_export_time: f64,
    /// Last export timestamp
    pub last_export_time: u64,
}

impl ExportStatistics {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Update statistics with new export
    pub fn update_export(&mut self, messages_count: usize, export_time: f64, format: ExportFormat) {
        self.total_exports += 1;
        self.total_messages_exported += messages_count;
        self.average_export_time = 
            (self.average_export_time * (self.total_exports - 1) as f64 + export_time) 
            / self.total_exports as f64;
        self.most_popular_format = Some(format);
        self.last_export_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    /// Get export rate (exports per message)
    pub fn get_export_rate(&self) -> f64 {
        if self.total_messages_exported > 0 {
            self.total_exports as f64 / self.total_messages_exported as f64
        } else {
            0.0
        }
    }
}

/// History exporter with zero-allocation streaming
#[derive(Clone)]
pub struct HistoryExporter {
    /// Export counter
    export_counter: Arc<ConsistentCounter>,
    /// Export statistics
    export_statistics: Arc<RwLock<ExportStatistics>>,
}

impl HistoryExporter {
    /// Create a new history exporter
    pub fn new() -> Self {
        Self {
            export_counter: Arc::new(ConsistentCounter::new(0)),
            export_statistics: Arc::new(RwLock::new(ExportStatistics::default())),
        }
    }

    /// Export conversation history (streaming)
    pub fn export_history_stream(
        &self,
        messages: Vec<SearchChatMessage>,
        options: ExportOptions,
    ) -> AsyncStream<String> {
        let self_clone = self.clone();

        AsyncStream::with_channel(move |sender| {
            let start_time = Instant::now();
            self_clone.export_counter.inc();

            // Apply filters
            let filtered_messages = self_clone.apply_filters(messages, &options);

            // Generate export data based on format
            let exported_data = match options.format {
                ExportFormat::Json => self_clone.export_json(&filtered_messages, &options),
                ExportFormat::Csv => self_clone.export_csv(&filtered_messages, &options),
                ExportFormat::Markdown => self_clone.export_markdown(&filtered_messages, &options),
                ExportFormat::Html => self_clone.export_html(&filtered_messages, &options),
                ExportFormat::Xml => self_clone.export_xml(&filtered_messages, &options),
                ExportFormat::PlainText => self_clone.export_plain_text(&filtered_messages, &options),
            };

            // Apply compression if requested
            let final_data = if options.compress {
                self_clone.compress_data(&exported_data)
            } else {
                exported_data
            };

            // Update statistics
            let export_time = start_time.elapsed().as_millis() as f64;
            if let Ok(mut stats) = self_clone.export_statistics.try_write() {
                stats.update_export(filtered_messages.len(), export_time, options.format);
            }

            let _ = sender.send(final_data);
        })
    }

    /// Apply export filters to messages
    fn apply_filters(&self, messages: Vec<SearchChatMessage>, options: &ExportOptions) -> Vec<SearchChatMessage> {
        messages
            .into_iter()
            .filter(|message| {
                // Date range filter
                if let Some(date_range) = &options.date_range {
                    if let Some(timestamp) = message.message.timestamp {
                        if timestamp < date_range.start || timestamp > date_range.end {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }

                // User filter
                if let Some(user_filter) = &options.user_filter {
                    let role_filter = match user_filter.as_ref() {
                        "system" => MessageRole::System,
                        "user" => MessageRole::User,
                        "assistant" => MessageRole::Assistant,
                        "tool" => MessageRole::Tool,
                        _ => MessageRole::User,
                    };
                    if message.message.role != role_filter {
                        return false;
                    }
                }

                true
            })
            .collect()
    }

    /// Export to JSON format
    fn export_json(&self, messages: &[SearchChatMessage], options: &ExportOptions) -> String {
        let mut json_messages = Vec::new();

        for message in messages {
            let mut json_obj = serde_json::json!({
                "role": format!("{:?}", message.message.role),
                "content": message.message.content,
                "relevance_score": message.relevance_score,
            });

            if options.include_timestamps {
                if let Some(timestamp) = message.message.timestamp {
                    json_obj["timestamp"] = serde_json::Value::Number(timestamp.into());
                }
            }

            if options.include_metadata {
                json_obj["relevance_score"] = serde_json::Value::from(message.relevance_score);
            }

            json_messages.push(json_obj);
        }

        let export_obj = serde_json::json!({
            "messages": json_messages,
            "export_info": {
                "format": "json",
                "exported_at": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                "total_messages": messages.len(),
            }
        });

        serde_json::to_string_pretty(&export_obj).unwrap_or_else(|_| "{}".to_string())
    }

    /// Export to CSV format
    fn export_csv(&self, messages: &[SearchChatMessage], options: &ExportOptions) -> String {
        let mut csv_output = String::new();

        // Header
        let mut headers = vec!["role", "content"];
        if options.include_timestamps {
            headers.push("timestamp");
        }
        if options.include_metadata {
            headers.push("relevance_score");
        }
        csv_output.push_str(&headers.join(","));
        csv_output.push('\n');

        // Data rows
        for message in messages {
            let escaped_content = message
                .message
                .content
                .replace(',', "\\,")
                .replace('\n', "\\n")
                .replace('"', "\\\"");
            let role_str = format!("{:?}", message.message.role);

            let mut row = vec![
                format!("\"{}\"", role_str),
                format!("\"{}\"", escaped_content),
            ];

            if options.include_timestamps {
                let timestamp_str = message
                    .message
                    .timestamp
                    .map_or_else(|| "0".to_string(), |t| t.to_string());
                row.push(timestamp_str);
            }

            if options.include_metadata {
                row.push(message.relevance_score.to_string());
            }

            csv_output.push_str(&row.join(","));
            csv_output.push('\n');
        }

        csv_output
    }

    /// Export to Markdown format
    fn export_markdown(&self, messages: &[SearchChatMessage], options: &ExportOptions) -> String {
        let mut markdown_output = String::new();

        markdown_output.push_str("# Conversation History\n\n");

        if options.include_timestamps {
            markdown_output.push_str(&format!(
                "Exported at: {}\n\n",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            ));
        }

        for message in messages {
            markdown_output.push_str(&format!("## {:?}\n\n", message.message.role));
            markdown_output.push_str(&format!("{}\n\n", message.message.content));

            if options.include_timestamps {
                let timestamp_str = message
                    .message
                    .timestamp
                    .map_or_else(|| "Unknown".to_string(), |t| t.to_string());
                markdown_output.push_str(&format!("*Timestamp: {}*\n\n", timestamp_str));
            }

            if options.include_metadata {
                markdown_output.push_str(&format!(
                    "*Relevance Score: {:.2}*\n\n",
                    message.relevance_score
                ));
            }

            markdown_output.push_str("---\n\n");
        }

        markdown_output
    }

    /// Export to HTML format
    fn export_html(&self, messages: &[SearchChatMessage], options: &ExportOptions) -> String {
        let mut html_output = String::new();

        html_output.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html_output.push_str("<title>Conversation History</title>\n");
        html_output.push_str("<style>\n");
        html_output.push_str("body { font-family: Arial, sans-serif; margin: 20px; }\n");
        html_output.push_str(".message { border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px; }\n");
        html_output.push_str(".role { font-weight: bold; color: #333; }\n");
        html_output.push_str(".timestamp { color: #666; font-size: 0.9em; }\n");
        html_output.push_str(".metadata { color: #999; font-size: 0.8em; }\n");
        html_output.push_str("</style>\n");
        html_output.push_str("</head>\n<body>\n");

        html_output.push_str("<h1>Conversation History</h1>\n");

        for message in messages {
            html_output.push_str("<div class=\"message\">\n");
            html_output.push_str(&format!(
                "<div class=\"role\">{:?}</div>\n",
                message.message.role
            ));
            html_output.push_str(&format!(
                "<div class=\"content\">{}</div>\n",
                message
                    .message
                    .content
                    .replace('<', "&lt;")
                    .replace('>', "&gt;")
                    .replace('&', "&amp;")
            ));

            if options.include_timestamps {
                let timestamp_str = message
                    .message
                    .timestamp
                    .map_or_else(|| "Unknown".to_string(), |t| t.to_string());
                html_output.push_str(&format!(
                    "<div class=\"timestamp\">Timestamp: {}</div>\n",
                    timestamp_str
                ));
            }

            if options.include_metadata {
                html_output.push_str(&format!(
                    "<div class=\"metadata\">Relevance Score: {:.2}</div>\n",
                    message.relevance_score
                ));
            }

            html_output.push_str("</div>\n");
        }

        html_output.push_str("</body>\n</html>");
        html_output
    }

    /// Export to XML format
    fn export_xml(&self, messages: &[SearchChatMessage], options: &ExportOptions) -> String {
        let mut xml_output = String::new();

        xml_output.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml_output.push_str("<conversation>\n");

        if options.include_timestamps {
            xml_output.push_str(&format!(
                "  <export_info exported_at=\"{}\"/>\n",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            ));
        }

        for message in messages {
            xml_output.push_str("  <message>\n");
            xml_output.push_str(&format!("    <role>{:?}</role>\n", message.message.role));
            xml_output.push_str(&format!(
                "    <content>{}</content>\n",
                message
                    .message
                    .content
                    .replace('<', "&lt;")
                    .replace('>', "&gt;")
                    .replace('&', "&amp;")
            ));

            if options.include_timestamps {
                let timestamp_str = message
                    .message
                    .timestamp
                    .map_or_else(|| "Unknown".to_string(), |t| t.to_string());
                xml_output.push_str(&format!("    <timestamp>{}</timestamp>\n", timestamp_str));
            }

            if options.include_metadata {
                xml_output.push_str(&format!(
                    "    <relevance_score>{:.2}</relevance_score>\n",
                    message.relevance_score
                ));
            }

            xml_output.push_str("  </message>\n");
        }

        xml_output.push_str("</conversation>");
        xml_output
    }

    /// Export to plain text format
    fn export_plain_text(&self, messages: &[SearchChatMessage], options: &ExportOptions) -> String {
        let mut text_output = String::new();

        text_output.push_str("CONVERSATION HISTORY\n");
        text_output.push_str("===================\n\n");

        if options.include_timestamps {
            text_output.push_str(&format!(
                "Exported at: {}\n\n",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            ));
        }

        for message in messages {
            text_output.push_str(&format!(
                "{:?}: {}\n",
                message.message.role, message.message.content
            ));

            if options.include_timestamps {
                let timestamp_str = message
                    .message
                    .timestamp
                    .map_or_else(|| "Unknown".to_string(), |t| t.to_string());
                text_output.push_str(&format!("Timestamp: {}\n", timestamp_str));
            }

            if options.include_metadata {
                text_output.push_str(&format!(
                    "Relevance Score: {:.2}\n",
                    message.relevance_score
                ));
            }

            text_output.push_str("\n");
        }

        text_output
    }

    /// Compress data using LZ4
    fn compress_data(&self, data: &str) -> String {
        match lz4::block::compress(data.as_bytes(), None, true) {
            Ok(compressed) => {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD.encode(compressed)
            }
            Err(_) => data.to_string(), // Fallback to uncompressed on error
        }
    }

    /// Decompress LZ4 data
    pub fn decompress_data(&self, compressed_data: &str) -> Result<String, SearchError> {
        use base64::Engine;
        
        let compressed = base64::engine::general_purpose::STANDARD
            .decode(compressed_data)
            .map_err(|e| SearchError::ExportError {
                reason: Arc::from(format!("Base64 decode failed: {}", e)),
            })?;

        let decompressed = lz4::block::decompress(&compressed, None)
            .map_err(|e| SearchError::ExportError {
                reason: Arc::from(format!("LZ4 decompress failed: {}", e)),
            })?;

        String::from_utf8(decompressed).map_err(|e| SearchError::ExportError {
            reason: Arc::from(format!("UTF-8 decode failed: {}", e)),
        })
    }

    /// Get export statistics (streaming)
    pub fn get_statistics_stream(&self) -> AsyncStream<ExportStatistics> {
        let self_clone = self.clone();

        AsyncStream::with_channel(move |sender| {
            if let Ok(stats) = self_clone.export_statistics.try_read() {
                let _ = sender.send(stats.clone());
            } else {
                let _ = sender.send(ExportStatistics::default());
            }
        })
    }

    /// Get export count
    pub fn get_export_count(&self) -> usize {
        self.export_counter.get()
    }

    /// Reset statistics
    pub fn reset_statistics(&self) {
        if let Ok(mut stats) = self.export_statistics.try_write() {
            *stats = ExportStatistics::default();
        }
        self.export_counter.reset();
    }
}

impl Default for HistoryExporter {
    fn default() -> Self {
        Self::new()
    }
}