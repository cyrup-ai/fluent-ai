//! Search export functionality for various formats
//!
//! This module provides comprehensive export capabilities with streaming-first architecture,
//! zero-allocation patterns, and support for multiple output formats.
//!
//! Decomposed into focused modules following ≤300-line architectural constraint:
//! - formats.rs: Format definitions and configuration
//! - json.rs: JSON export implementation  
//! - csv.rs: CSV export implementation
//! - text_formats.rs: XML, Text, Markdown, HTML exports

use std::time::Instant;
use fluent_ai_async::{AsyncStream, emit, handle_error};
use serde::{Deserialize, Serialize};

use super::types::{SearchResult, SearchStatistics};

// Re-export all format-related types
pub use formats::{ExportFormat, ExportOptions, ExportMetadata, ExportData, ExportError};

// Sub-modules
pub mod formats;
pub mod json;
pub mod csv;
pub mod text_formats;

/// Export statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExportStatistics {
    /// Total exports performed
    pub total_exports: usize,
    /// Total messages exported
    pub total_messages_exported: usize,
    /// Average export time in milliseconds
    pub average_export_time: f64,
    /// Last export timestamp
    pub last_export_time: u64,
}

/// Search exporter for generating output in various formats
#[derive(Debug, Clone)]
pub struct SearchExporter {
    /// Export statistics
    stats: ExportStatistics,
}

impl SearchExporter {
    /// Create a new search exporter
    pub fn new() -> Self {
        Self {
            stats: ExportStatistics::default(),
        }
    }

    /// Export search results using fluent-ai-async streaming architecture
    pub fn export_results(
        &self,
        results: Vec<SearchResult>,
        options: ExportOptions,
    ) -> AsyncStream<String> {
        AsyncStream::with_channel(move |sender| {
            let _start_time = Instant::now();
            
            // Apply result limit if specified
            let limited_results = if let Some(max) = options.max_results {
                results.into_iter().take(max).collect()
            } else {
                results
            };

            // Generate export based on format
            let exported_data = match options.format {
                ExportFormat::Json => json::JsonExporter::export_results(&limited_results, &options),
                ExportFormat::Csv => csv::CsvExporter::export_results(&limited_results, &options),
                ExportFormat::Xml => text_formats::TextFormatsExporter::export_as_xml(&limited_results, &options),
                ExportFormat::Text => text_formats::TextFormatsExporter::export_as_text(&limited_results, &options),
                ExportFormat::Markdown => text_formats::TextFormatsExporter::export_as_markdown(&limited_results, &options),
                ExportFormat::Html => text_formats::TextFormatsExporter::export_as_html(&limited_results, &options),
            };

            match exported_data {
                Ok(data) => emit!(sender, data),
                Err(error) => handle_error!(error, "Export failed"),
            }
        })
    }

    /// Export search statistics using fluent-ai-async streaming architecture
    pub fn export_statistics(
        &self,
        stats: SearchStatistics,
        format: ExportFormat,
    ) -> AsyncStream<String> {
        AsyncStream::with_channel(move |sender| {
            let exported_stats = match format {
                ExportFormat::Json => json::JsonExporter::export_statistics(&stats, true),
                ExportFormat::Csv => csv::CsvExporter::export_statistics(&stats),
                ExportFormat::Text => text_formats::TextFormatsExporter::export_statistics_as_text(&stats),
                ExportFormat::Markdown => text_formats::TextFormatsExporter::export_statistics_as_markdown(&stats),
                _ => format!("Statistics export not supported for format: {:?}", format),
            };

            emit!(sender, exported_stats);
        })
    }

    /// Get export statistics
    pub fn get_statistics(&self) -> &ExportStatistics {
        &self.stats
    }
}

impl Default for SearchExporter {
    fn default() -> Self {
        Self::new()
    }
}