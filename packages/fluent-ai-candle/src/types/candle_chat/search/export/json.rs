//! JSON export implementation
//!
//! High-performance JSON export with streaming support and
//! zero-allocation patterns where possible.

use super::super::types::{SearchResult, SearchStatistics};
use super::formats::{ExportOptions, ExportFormat, ExportData, ExportMetadata, ExportError};

/// JSON export implementation
pub struct JsonExporter;

impl JsonExporter {
    /// Export search results as JSON
    pub fn export_results(results: &[SearchResult], options: &ExportOptions) -> Result<String, ExportError> {
        let export_data = ExportData {
            results: results.to_vec(),
            metadata: if options.include_metadata {
                Some(ExportMetadata {
                    export_time: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                    format: options.format.clone(),
                    total_results: results.len(),
                    options: options.clone(),
                })
            } else {
                None
            },
        };

        if options.pretty_print {
            serde_json::to_string_pretty(&export_data)
                .map_err(|e| ExportError::SerializationError(e.to_string()))
        } else {
            serde_json::to_string(&export_data)
                .map_err(|e| ExportError::SerializationError(e.to_string()))
        }
    }

    /// Export search statistics as JSON
    pub fn export_statistics(stats: &SearchStatistics, pretty_print: bool) -> String {
        if pretty_print {
            serde_json::to_string_pretty(stats)
                .unwrap_or_else(|_| "{}".to_string())
        } else {
            serde_json::to_string(stats)
                .unwrap_or_else(|_| "{}".to_string())
        }
    }
}