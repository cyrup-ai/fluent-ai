//! History export functionality for candle_chat search system
//!
//! This module provides comprehensive history export capabilities with
//! multiple formats, filtering options, and streaming export operations.

use std::collections::HashMap;
use std::sync::Arc;

use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::types::{SearchQuery, DateRange};
use crate::types::CandleSearchChatMessage;

/// Handle errors in streaming context without panicking
macro_rules! handle_error {
    ($error:expr, $context:literal) => {
        eprintln!("Streaming error in {}: {}", $context, $error);
        // Continue processing instead of returning error
    };
}

/// Export options for history export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    /// Export format
    pub format: ExportFormat,
    /// Date range filter
    pub date_range: Option<DateRange>,
    /// Role filter
    pub role_filter: Option<crate::types::CandleMessageRole>,
    /// Include metadata
    pub include_metadata: bool,
    /// Include system messages
    pub include_system: bool,
    /// Include deleted messages
    pub include_deleted: bool,
    /// Maximum messages to export
    pub max_messages: Option<usize>,
    /// Compression options
    pub compression: CompressionOptions,
    /// Privacy options
    pub privacy: PrivacyOptions,
    /// Custom fields to include
    pub custom_fields: Vec<String>,
    /// Export filename
    pub filename: Option<String>,
    /// Batch size for streaming
    pub batch_size: usize,
    /// Include attachments
    pub include_attachments: bool,
    /// Language filter
    pub language_filter: Option<String>,
    /// Minimum message length
    pub min_length: Option<usize>,
    /// Maximum message length
    pub max_length: Option<usize>,
}

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
    /// PDF format (future)
    Pdf,
    /// Custom format
    Custom(String),
}

/// Compression options for export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionOptions {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level (1-9)
    pub level: u8,
    /// Archive format
    pub archive_format: Option<ArchiveFormat>,
}

/// Compression algorithm options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// Gzip compression
    Gzip,
    /// Zlib compression
    Zlib,
    /// Brotli compression
    Brotli,
    /// LZ4 compression
    Lz4,
    /// None (no compression)
    None,
}

/// Archive format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchiveFormat {
    /// ZIP archive
    Zip,
    /// TAR archive
    Tar,
    /// 7Z archive
    SevenZ,
}

/// Privacy options for export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyOptions {
    /// Anonymize user data
    pub anonymize_users: bool,
    /// Redact sensitive information
    pub redact_sensitive: bool,
    /// Hash user IDs
    pub hash_user_ids: bool,
    /// Remove timestamps
    pub remove_timestamps: bool,
    /// Custom redaction patterns
    pub redaction_patterns: Vec<String>,
    /// Encryption options
    pub encryption: Option<EncryptionOptions>,
}

/// Encryption options for export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionOptions {
    /// Encryption algorithm
    pub algorithm: String,
    /// Key derivation method
    pub key_derivation: String,
    /// Salt for key derivation
    pub salt: Option<String>,
    /// Additional authenticated data
    pub aad: Option<String>,
}

/// History exporter for chat data
pub struct HistoryExporter {
    /// Export configuration
    pub config: ExportOptions,
    /// Export statistics
    pub export_statistics: ExportStatistics,
    /// Active exports
    pub active_exports: HashMap<Uuid, ExportJob>,
    /// Export history
    pub export_history: Vec<ExportRecord>,
}

/// Export job tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportJob {
    /// Job ID
    pub id: Uuid,
    /// Job status
    pub status: ExportStatus,
    /// Progress percentage
    pub progress: f32,
    /// Start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// End time
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    /// Messages processed
    pub messages_processed: usize,
    /// Total messages
    pub total_messages: usize,
    /// Export options used
    pub options: ExportOptions,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Export status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportStatus {
    /// Export is pending
    Pending,
    /// Export is running
    Running,
    /// Export completed successfully
    Completed,
    /// Export failed
    Failed,
    /// Export was cancelled
    Cancelled,
}

/// Export record for history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRecord {
    /// Record ID
    pub id: Uuid,
    /// Export timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Export format used
    pub format: ExportFormat,
    /// Messages exported
    pub message_count: usize,
    /// File size in bytes
    pub file_size: usize,
    /// Export duration
    pub duration_ms: u64,
    /// Success flag
    pub success: bool,
    /// Export metadata
    pub metadata: HashMap<String, String>,
}

/// Export statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportStatistics {
    /// Total exports performed
    pub total_exports: usize,
    /// Successful exports
    pub successful_exports: usize,
    /// Failed exports
    pub failed_exports: usize,
    /// Total messages exported
    pub total_messages_exported: usize,
    /// Total export time
    pub total_export_time_ms: u64,
    /// Average export time
    pub avg_export_time_ms: f64,
    /// Most popular format
    pub most_popular_format: Option<ExportFormat>,
    /// Export size statistics
    pub size_statistics: HashMap<String, usize>,
    /// Last export timestamp
    pub last_export: Option<chrono::DateTime<chrono::Utc>>,
}

impl HistoryExporter {
    /// Create a new history exporter
    pub fn new(config: ExportOptions) -> Self {
        Self {
            config,
            export_statistics: ExportStatistics::default(),
            active_exports: HashMap::new(),
            export_history: Vec::new(),
        }
    }

    /// Export messages with streaming (main export function)
    pub fn export_messages(&mut self, messages: Vec<CandleSearchChatMessage>) -> AsyncStream<ExportChunk> {
        let config = self.config.clone();
        let job_id = Uuid::new_v4();
        let start_time = chrono::Utc::now();

        // Create export job
        let job = ExportJob {
            id: job_id,
            status: ExportStatus::Running,
            progress: 0.0,
            start_time,
            end_time: None,
            messages_processed: 0,
            total_messages: messages.len(),
            options: config.clone(),
            error_message: None,
        };
        self.active_exports.insert(job_id, job);

        AsyncStream::with_channel(move |sender| {
            let mut filtered_messages = messages;
            
            // Apply filters
            if let Some(role_filter) = &config.role_filter {
                filtered_messages.retain(|msg| msg.role == *role_filter);
            }
            
            if let Some(date_range) = &config.date_range {
                if let Some(start) = date_range.start {
                    filtered_messages.retain(|msg| msg.timestamp >= start);
                }
                if let Some(end) = date_range.end {
                    filtered_messages.retain(|msg| msg.timestamp <= end);
                }
            }

            // Apply length filters
            if let Some(min_length) = config.min_length {
                filtered_messages.retain(|msg| msg.content.len() >= min_length);
            }
            if let Some(max_length) = config.max_length {
                filtered_messages.retain(|msg| msg.content.len() <= max_length);
            }

            // Limit messages if specified
            if let Some(max_messages) = config.max_messages {
                filtered_messages.truncate(max_messages);
            }

            // Process messages in batches
            for chunk in filtered_messages.chunks(config.batch_size) {
                let export_chunk = match config.format {
                    ExportFormat::Json => Self::export_as_json(chunk, &config),
                    ExportFormat::Csv => Self::export_as_csv(chunk, &config),
                    ExportFormat::Text => Self::export_as_text(chunk, &config),
                    ExportFormat::Markdown => Self::export_as_markdown(chunk, &config),
                    _ => Self::export_as_json(chunk, &config), // Default to JSON
                };

                let _ = sender.send(export_chunk);
            }
        })
    }

    /// Export as JSON format
    fn export_as_json(messages: &[CandleSearchChatMessage], config: &ExportOptions) -> ExportChunk {
        let mut json_data = Vec::new();
        
        for message in messages {
            let mut msg_data = serde_json::json!({
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "timestamp": message.timestamp,
            });

            if config.include_metadata {
                msg_data["metadata"] = serde_json::to_value(&message.metadata).unwrap_or_default();
            }

            json_data.push(msg_data);
        }

        let content = serde_json::to_string_pretty(&json_data).unwrap_or_default();
        ExportChunk {
            id: Uuid::new_v4(),
            content: content.into_bytes(),
            format: ExportFormat::Json,
            chunk_index: 0,
            total_chunks: 1,
            metadata: HashMap::new(),
        }
    }

    /// Export as CSV format
    fn export_as_csv(messages: &[CandleSearchChatMessage], _config: &ExportOptions) -> ExportChunk {
        let mut csv_content = String::from("id,role,content,timestamp\n");
        
        for message in messages {
            csv_content.push_str(&format!(
                "{},{},{},{}\n",
                message.id,
                format!("{:?}", message.role),
                message.content.replace(',', ";").replace('\n', " "),
                message.timestamp.format("%Y-%m-%d %H:%M:%S")
            ));
        }

        ExportChunk {
            id: Uuid::new_v4(),
            content: csv_content.into_bytes(),
            format: ExportFormat::Csv,
            chunk_index: 0,
            total_chunks: 1,
            metadata: HashMap::new(),
        }
    }

    /// Export as plain text format
    fn export_as_text(messages: &[CandleSearchChatMessage], _config: &ExportOptions) -> ExportChunk {
        let mut text_content = String::new();
        
        for message in messages {
            text_content.push_str(&format!(
                "[{}] {:?}: {}\n\n",
                message.timestamp.format("%Y-%m-%d %H:%M:%S"),
                message.role,
                message.content
            ));
        }

        ExportChunk {
            id: Uuid::new_v4(),
            content: text_content.into_bytes(),
            format: ExportFormat::Text,
            chunk_index: 0,
            total_chunks: 1,
            metadata: HashMap::new(),
        }
    }

    /// Export as Markdown format
    fn export_as_markdown(messages: &[CandleSearchChatMessage], _config: &ExportOptions) -> ExportChunk {
        let mut md_content = String::from("# Chat History Export\n\n");
        
        for message in messages {
            md_content.push_str(&format!(
                "## {:?} - {}\n\n{}\n\n---\n\n",
                message.role,
                message.timestamp.format("%Y-%m-%d %H:%M:%S"),
                message.content
            ));
        }

        ExportChunk {
            id: Uuid::new_v4(),
            content: md_content.into_bytes(),
            format: ExportFormat::Markdown,
            chunk_index: 0,
            total_chunks: 1,
            metadata: HashMap::new(),
        }
    }

    /// Get export statistics
    pub fn get_statistics(&self) -> ExportStatistics {
        self.export_statistics.clone()
    }

    /// Get active export jobs
    pub fn get_active_jobs(&self) -> Vec<ExportJob> {
        self.active_exports.values().cloned().collect()
    }

    /// Cancel export job
    pub fn cancel_export(&mut self, job_id: Uuid) -> bool {
        if let Some(job) = self.active_exports.get_mut(&job_id) {
            job.status = ExportStatus::Cancelled;
            job.end_time = Some(chrono::Utc::now());
            true
        } else {
            false
        }
    }
}

/// Export chunk for streaming export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportChunk {
    /// Chunk ID
    pub id: Uuid,
    /// Chunk content
    pub content: Vec<u8>,
    /// Export format
    pub format: ExportFormat,
    /// Chunk index
    pub chunk_index: usize,
    /// Total chunks
    pub total_chunks: usize,
    /// Chunk metadata
    pub metadata: HashMap<String, String>,
}

impl Default for HistoryExporter {
    fn default() -> Self {
        Self::new(ExportOptions::default())
    }
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::Json,
            date_range: None,
            role_filter: None,
            include_metadata: true,
            include_system: false,
            include_deleted: false,
            max_messages: None,
            compression: CompressionOptions::default(),
            privacy: PrivacyOptions::default(),
            custom_fields: Vec::new(),
            filename: None,
            batch_size: 1000,
            include_attachments: false,
            language_filter: None,
            min_length: None,
            max_length: None,
        }
    }
}

impl Default for CompressionOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: CompressionAlgorithm::None,
            level: 6,
            archive_format: None,
        }
    }
}

impl Default for PrivacyOptions {
    fn default() -> Self {
        Self {
            anonymize_users: false,
            redact_sensitive: false,
            hash_user_ids: false,
            remove_timestamps: false,
            redaction_patterns: Vec::new(),
            encryption: None,
        }
    }
}

impl Default for ExportStatistics {
    fn default() -> Self {
        Self {
            total_exports: 0,
            successful_exports: 0,
            failed_exports: 0,
            total_messages_exported: 0,
            total_export_time_ms: 0,
            avg_export_time_ms: 0.0,
            most_popular_format: None,
            size_statistics: HashMap::new(),
            last_export: None,
        }
    }
}