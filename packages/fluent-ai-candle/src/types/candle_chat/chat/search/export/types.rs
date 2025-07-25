//! Export types and configuration structures

use serde::{Deserialize, Serialize};

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
    /// YAML format
    Yaml,
    /// Custom format with user-defined structure
    Custom {
        /// Format name
        name: String,
        /// Template for formatting
        template: String}}

impl Default for ExportFormat {
    fn default() -> Self {
        Self::Json
    }
}

/// Date range for filtering exports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    /// Start date (inclusive)
    pub start: Option<chrono::DateTime<chrono::Utc>>,
    /// End date (inclusive)
    pub end: Option<chrono::DateTime<chrono::Utc>>}

impl Default for DateRange {
    fn default() -> Self {
        Self {
            start: None,
            end: None}
    }
}

/// Export configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    /// Export format
    pub format: ExportFormat,
    /// Include message metadata
    pub include_metadata: bool,
    /// Include system messages
    pub include_system_messages: bool,
    /// Include deleted messages
    pub include_deleted: bool,
    /// Date range filter
    pub date_range: DateRange,
    /// Maximum number of messages to export
    pub max_messages: Option<usize>,
    /// Batch size for streaming export
    pub batch_size: usize,
    /// Pretty print output (for JSON/XML)
    pub pretty_print: bool,
    /// Include conversation tags
    pub include_tags: bool,
    /// Include search metadata
    pub include_search_metadata: bool,
    /// Compression level (0-9)
    pub compression_level: Option<u8>,
    /// Custom export template (for Custom format)
    pub custom_template: Option<String>,
    /// Filter by conversation IDs
    pub conversation_ids: Vec<String>,
    /// Filter by message roles
    pub message_roles: Vec<String>,
    /// Include attachments
    pub include_attachments: bool}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::Json,
            include_metadata: true,
            include_system_messages: false,
            include_deleted: false,
            date_range: DateRange::default(),
            max_messages: None,
            batch_size: 1000,
            pretty_print: true,
            include_tags: true,
            include_search_metadata: false,
            compression_level: None,
            custom_template: None,
            conversation_ids: Vec::new(),
            message_roles: Vec::new(),
            include_attachments: true}
    }
}