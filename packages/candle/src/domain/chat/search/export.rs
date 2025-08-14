//! Search result export functionality

use std::sync::Arc;

use fluent_ai_async::AsyncStream;
use serde_json;

use super::types::{ExportFormat, ExportOptions, SearchError, SearchResult};

/// Search result exporter with streaming capabilities
pub struct SearchExporter {
    /// Default export options
    default_options: ExportOptions,
}

/// History exporter for chat conversation history (domain version)
#[derive(Debug)]
pub struct HistoryExporter {
    /// Default export options
    default_options: ExportOptions,
}

impl SearchExporter {
    /// Create a new search exporter
    pub fn new() -> Self {
        Self {
            default_options: ExportOptions::default(),
        }
    }

    /// Export search results as a stream
    pub fn export_stream(
        &self,
        results: Vec<SearchResult>,
        options: Option<ExportOptions>,
    ) -> AsyncStream<String> {
        let export_options = options.unwrap_or_else(|| self.default_options.clone());
        let limited_results = if let Some(max) = export_options.max_results {
            results.into_iter().take(max).collect()
        } else {
            results
        };

        // Clone self to avoid borrowing issues
        let self_clone = self.clone();

        AsyncStream::with_channel(move |sender| {
            match export_options.format {
                ExportFormat::Json => {
                    if let Ok(json) = self_clone.export_json_sync(&limited_results, &export_options)
                    {
                        let _ = sender.send(json);
                    }
                }
                _ => {
                    // Other formats not implemented in simplified version
                    let _ = sender.send(String::from("Export format not supported"));
                }
            }
        })
    }

    /// Export to JSON format
    fn export_json_sync(
        &self,
        results: &[SearchResult],
        _options: &ExportOptions,
    ) -> Result<String, SearchError> {
        serde_json::to_string_pretty(&results).map_err(|e| SearchError::ExportError {
            reason: Arc::from(format!("JSON serialization failed: {}", e)),
        })
    }
}

impl Default for SearchExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SearchExporter {
    fn clone(&self) -> Self {
        Self {
            default_options: self.default_options.clone(),
        }
    }
}

impl HistoryExporter {
    /// Create a new history exporter
    pub fn new() -> Self {
        Self {
            default_options: ExportOptions::default(),
        }
    }

    /// Create exporter with custom default options
    pub fn with_options(options: ExportOptions) -> Self {
        Self {
            default_options: options,
        }
    }
}

impl Default for HistoryExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for HistoryExporter {
    fn clone(&self) -> Self {
        Self {
            default_options: self.default_options.clone(),
        }
    }
}
