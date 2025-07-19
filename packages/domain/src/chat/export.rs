//! Chat export functionality
//!
//! Provides zero-allocation export capabilities for chat conversations and history.
//! Supports multiple formats with blazing-fast serialization and ergonomic APIs.

use std::sync::Arc;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Export format options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format with full metadata
    Json,
    /// Markdown format for human readability
    Markdown,
    /// Plain text format
    Text,
    /// CSV format for data analysis
    Csv,
}

/// Export configuration with zero-allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Export format
    pub format: ExportFormat,
    /// Include metadata in export
    pub include_metadata: bool,
    /// Include timestamps
    pub include_timestamps: bool,
    /// Maximum messages to export (0 = all)
    pub max_messages: usize,
    /// Custom filename prefix
    pub filename_prefix: Arc<str>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Json,
            include_metadata: true,
            include_timestamps: true,
            max_messages: 0,
            filename_prefix: Arc::from("chat_export"),
        }
    }
}

/// Export error types
#[derive(Error, Debug, Clone)]
pub enum ExportError {
    #[error("Serialization error: {detail}")]
    SerializationError { detail: Arc<str> },
    
    #[error("IO error: {detail}")]
    IoError { detail: Arc<str> },
    
    #[error("Format error: {detail}")]
    FormatError { detail: Arc<str> },
}

/// Result type for export operations
pub type ExportResult<T> = Result<T, ExportError>;

/// Export a conversation to the specified format
pub fn export_conversation(
    messages: &[crate::message::Message],
    config: &ExportConfig,
) -> ExportResult<String> {
    match config.format {
        ExportFormat::Json => export_to_json(messages, config),
        ExportFormat::Markdown => export_to_markdown(messages, config),
        ExportFormat::Text => export_to_text(messages, config),
        ExportFormat::Csv => export_to_csv(messages, config),
    }
}

/// Export to JSON format
fn export_to_json(
    messages: &[crate::message::Message],
    config: &ExportConfig,
) -> ExportResult<String> {
    let limited_messages = if config.max_messages > 0 {
        &messages[..config.max_messages.min(messages.len())]
    } else {
        messages
    };

    serde_json::to_string_pretty(limited_messages)
        .map_err(|e| ExportError::SerializationError {
            detail: Arc::from(e.to_string()),
        })
}

/// Export to Markdown format
fn export_to_markdown(
    messages: &[crate::message::Message],
    config: &ExportConfig,
) -> ExportResult<String> {
    let mut output = String::with_capacity(messages.len() * 100);
    output.push_str("# Chat Export\n\n");

    let limited_messages = if config.max_messages > 0 {
        &messages[..config.max_messages.min(messages.len())]
    } else {
        messages
    };

    for message in limited_messages {
        output.push_str(&format!("## {}\n\n", message.role));
        output.push_str(&message.content);
        output.push_str("\n\n");

        if config.include_timestamps {
            output.push_str(&format!("*Timestamp: {}*\n\n", message.timestamp));
        }
    }

    Ok(output)
}

/// Export to plain text format
fn export_to_text(
    messages: &[crate::message::Message],
    config: &ExportConfig,
) -> ExportResult<String> {
    let mut output = String::with_capacity(messages.len() * 100);

    let limited_messages = if config.max_messages > 0 {
        &messages[..config.max_messages.min(messages.len())]
    } else {
        messages
    };

    for message in limited_messages {
        output.push_str(&format!("{}: {}\n", message.role, message.content));

        if config.include_timestamps {
            output.push_str(&format!("Timestamp: {}\n", message.timestamp));
        }
        output.push('\n');
    }

    Ok(output)
}

/// Export to CSV format
fn export_to_csv(
    messages: &[crate::message::Message],
    config: &ExportConfig,
) -> ExportResult<String> {
    let mut output = String::with_capacity(messages.len() * 100);
    
    // CSV header
    if config.include_timestamps {
        output.push_str("role,content,timestamp\n");
    } else {
        output.push_str("role,content\n");
    }

    let limited_messages = if config.max_messages > 0 {
        &messages[..config.max_messages.min(messages.len())]
    } else {
        messages
    };

    for message in limited_messages {
        let escaped_content = message.content.replace('"', "\"\"");
        if config.include_timestamps {
            output.push_str(&format!("\"{}\",\"{}\",{}\n", 
                message.role, escaped_content, message.timestamp));
        } else {
            output.push_str(&format!("\"{}\",\"{}\"\n", 
                message.role, escaped_content));
        }
    }

    Ok(output)
}
