//! Export format definitions and configuration
//!
//! This module defines all export formats and their configuration options
//! following the zero-allocation, streaming-first architecture.

use serde::{Deserialize, Serialize};

use super::super::types::SearchResult;

/// Export format enumeration
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
    Html}

/// Export options for customizing output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    /// Export format
    pub format: ExportFormat,
    /// Include metadata in export
    pub include_metadata: bool,
    /// Include statistics in export
    pub include_statistics: bool,
    /// Maximum results to export
    pub max_results: Option<usize>,
    /// Pretty print output (for JSON/XML)
    pub pretty_print: bool,
    /// Include timestamps
    pub include_timestamps: bool,
    /// Include relevance scores
    pub include_scores: bool}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::Json,
            include_metadata: true,
            include_statistics: true,
            max_results: None,
            pretty_print: true,
            include_timestamps: true,
            include_scores: true}
    }
}

/// Export metadata structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMetadata {
    pub export_time: u64,
    pub format: ExportFormat,
    pub total_results: usize,
    pub options: ExportOptions}

/// Export data structure for JSON serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportData {
    pub results: Vec<SearchResult>,
    pub metadata: Option<ExportMetadata>}

/// Export error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum ExportError {
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Unsupported format: {0:?}")]
    UnsupportedFormat(ExportFormat),
    
    #[error("Export failed: {0}")]
    ExportFailed(String)}