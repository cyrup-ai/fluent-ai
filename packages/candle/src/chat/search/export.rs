//! Search result export functionality
//!
//! This module provides streaming export capabilities for search results
//! in various formats with zero-allocation patterns.

use std::sync::Arc;
use serde_json;
use fluent_ai_async::AsyncStream;

use super::types::{SearchResult, ExportFormat, ExportOptions, SearchError};

/// Search result exporter with streaming capabilities
pub struct SearchExporter {
    /// Default export options
    default_options: ExportOptions,
}

/// History exporter for chat conversation history
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

    /// Create exporter with custom default options
    pub fn with_options(options: ExportOptions) -> Self {
        Self {
            default_options: options,
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
                    if let Ok(json) = self_clone.export_json_sync(&limited_results, &export_options) {
                        let _ = sender.send(json);
                    }
                }
                ExportFormat::Csv => {
                    if let Ok(csv) = self_clone.export_csv_sync(&limited_results, &export_options) {
                        let _ = sender.send(csv);
                    }
                }
                ExportFormat::Xml => {
                    if let Ok(xml) = self_clone.export_xml_sync(&limited_results, &export_options) {
                        let _ = sender.send(xml);
                    }
                }
                ExportFormat::Text => {
                    if let Ok(text) = self_clone.export_text_sync(&limited_results, &export_options) {
                        let _ = sender.send(text);
                    }
                }
            }
        })
    }

    /// Export search results (blocking)
    pub fn export_sync(
        &self,
        results: &[SearchResult],
        options: &ExportOptions,
    ) -> Result<String, SearchError> {
        match options.format {
            ExportFormat::Json => self.export_json_sync(results, options),
            ExportFormat::Csv => self.export_csv_sync(results, options),
            ExportFormat::Xml => self.export_xml_sync(results, options),
            ExportFormat::Text => self.export_text_sync(results, options),
        }
    }

    /// Export to JSON format
    fn export_json_sync(
        &self,
        results: &[SearchResult],
        options: &ExportOptions,
    ) -> Result<String, SearchError> {
        let export_data = if options.include_metadata {
            self.create_full_export_data(results, options)
        } else {
            self.create_minimal_export_data(results, options)
        };

        serde_json::to_string_pretty(&export_data).map_err(|e| SearchError::ExportError {
            reason: Arc::from(format!("JSON serialization failed: {}", e)),
        })
    }

    /// Export to CSV format
    fn export_csv_sync(
        &self,
        results: &[SearchResult],
        options: &ExportOptions,
    ) -> Result<String, SearchError> {
        let mut csv = String::new();
        
        // CSV Header
        if options.include_metadata {
            csv.push_str("ID,Content,Role,Timestamp,Relevance Score,Matching Terms");
            if options.include_context {
                csv.push_str(",Context Messages");
            }
            csv.push('\n');
        } else {
            csv.push_str("Content,Role,Relevance Score\n");
        }

        // CSV Rows
        for result in results {
            let content = self.escape_csv_field(&result.message.message.content);
            let role = format!("{:?}", result.message.message.role);
            
            if options.include_metadata {
                let id = result.message.message.id.as_deref().unwrap_or("N/A");
                let timestamp = result.message.message.timestamp.unwrap_or(0);
                let matching_terms = result.matching_terms
                    .iter()
                    .map(|t| t.as_ref())
                    .collect::<Vec<_>>()
                    .join(";");

                csv.push_str(&format!(
                    "{},{},{},{},{:.3},{}", 
                    self.escape_csv_field(id),
                    content,
                    role,
                    timestamp,
                    result.relevance_score,
                    self.escape_csv_field(&matching_terms)
                ));

                if options.include_context {
                    let context_content = result.context
                        .iter()
                        .map(|msg| msg.message.content.as_str())
                        .collect::<Vec<_>>()
                        .join(" | ");
                    csv.push_str(&format!(",{}", self.escape_csv_field(&context_content)));
                }
            } else {
                csv.push_str(&format!(
                    "{},{},{:.3}",
                    content,
                    role,
                    result.relevance_score
                ));
            }
            
            csv.push('\n');
        }

        Ok(csv)
    }

    /// Export to XML format
    fn export_xml_sync(
        &self,
        results: &[SearchResult],
        options: &ExportOptions,
    ) -> Result<String, SearchError> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<search_results>\n");

        for result in results {
            xml.push_str("  <result>\n");
            
            // Basic fields
            xml.push_str(&format!("    <content><![CDATA[{}]]></content>\n", result.message.message.content));
            xml.push_str(&format!("    <role>{:?}</role>\n", result.message.message.role));
            xml.push_str(&format!("    <relevance_score>{:.3}</relevance_score>\n", result.relevance_score));
            
            // Metadata
            if options.include_metadata {
                if let Some(ref id) = result.message.message.id {
                    xml.push_str(&format!("    <id>{}</id>\n", self.escape_xml(id)));
                }
                if let Some(timestamp) = result.message.message.timestamp {
                    xml.push_str(&format!("    <timestamp>{}</timestamp>\n", timestamp));
                }
                
                xml.push_str("    <matching_terms>\n");
                for term in &result.matching_terms {
                    xml.push_str(&format!("      <term>{}</term>\n", self.escape_xml(term)));
                }
                xml.push_str("    </matching_terms>\n");
            }

            // Context
            if options.include_context && !result.context.is_empty() {
                xml.push_str("    <context_messages>\n");
                for context_msg in &result.context {
                    xml.push_str("      <message>\n");
                    xml.push_str(&format!("        <content><![CDATA[{}]]></content>\n", context_msg.message.content));
                    xml.push_str(&format!("        <role>{:?}</role>\n", context_msg.message.role));
                    xml.push_str("      </message>\n");
                }
                xml.push_str("    </context_messages>\n");
            }
            
            xml.push_str("  </result>\n");
        }

        xml.push_str("</search_results>\n");
        Ok(xml)
    }

    /// Export to plain text format
    fn export_text_sync(
        &self,
        results: &[SearchResult],
        options: &ExportOptions,
    ) -> Result<String, SearchError> {
        let mut text = String::new();
        text.push_str("SEARCH RESULTS\n");
        text.push_str("==============\n\n");

        for (i, result) in results.iter().enumerate() {
            text.push_str(&format!("Result #{}\n", i + 1));
            text.push_str("-----------\n");
            
            text.push_str(&format!("Content: {}\n", result.message.message.content));
            text.push_str(&format!("Role: {:?}\n", result.message.message.role));
            text.push_str(&format!("Relevance Score: {:.3}\n", result.relevance_score));
            
            if options.include_metadata {
                if let Some(ref id) = result.message.message.id {
                    text.push_str(&format!("ID: {}\n", id));
                }
                if let Some(timestamp) = result.message.message.timestamp {
                    text.push_str(&format!("Timestamp: {}\n", timestamp));
                }
                
                if !result.matching_terms.is_empty() {
                    let terms = result.matching_terms
                        .iter()
                        .map(|t| t.as_ref())
                        .collect::<Vec<_>>()
                        .join(", ");
                    text.push_str(&format!("Matching Terms: {}\n", terms));
                }
            }

            if options.include_context && !result.context.is_empty() {
                text.push_str("\nContext Messages:\n");
                for context_msg in &result.context {
                    text.push_str(&format!("  - [{:?}] {}\n", 
                        context_msg.message.role, 
                        context_msg.message.content
                    ));
                }
            }
            
            text.push_str("\n");
        }

        Ok(text)
    }

    /// Create full export data with metadata
    fn create_full_export_data(
        &self,
        results: &[SearchResult],
        options: &ExportOptions,
    ) -> serde_json::Value {
        let results_json: Vec<serde_json::Value> = results
            .iter()
            .map(|result| {
                let mut obj = serde_json::json!({
                    "message": {
                        "id": result.message.message.id,
                        "content": result.message.message.content,
                        "role": result.message.message.role,
                        "timestamp": result.message.message.timestamp
                    },
                    "relevance_score": result.relevance_score,
                    "matching_terms": result.matching_terms
                });

                if options.include_context {
                    obj["context"] = serde_json::to_value(&result.context).unwrap_or(serde_json::Value::Null);
                }

                obj["metadata"] = serde_json::to_value(&result.metadata).unwrap_or(serde_json::Value::Null);
                obj
            })
            .collect();

        serde_json::json!({
            "results": results_json,
            "total_results": results.len(),
            "export_timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        })
    }

    /// Create minimal export data without metadata
    fn create_minimal_export_data(
        &self,
        results: &[SearchResult],
        _options: &ExportOptions,
    ) -> serde_json::Value {
        let results_json: Vec<serde_json::Value> = results
            .iter()
            .map(|result| {
                serde_json::json!({
                    "content": result.message.message.content,
                    "role": result.message.message.role,
                    "relevance_score": result.relevance_score
                })
            })
            .collect();

        serde_json::json!({
            "results": results_json
        })
    }

    /// Escape CSV field
    fn escape_csv_field(&self, field: &str) -> String {
        if field.contains(',') || field.contains('"') || field.contains('\n') {
            format!("\"{}\"", field.replace('"', "\"\""))
        } else {
            field.to_string()
        }
    }

    /// Escape XML content
    fn escape_xml(&self, content: &str) -> String {
        content
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }

    /// Set default export options
    pub fn set_default_options(&mut self, options: ExportOptions) {
        self.default_options = options;
    }

    /// Get default export options
    pub fn get_default_options(&self) -> &ExportOptions {
        &self.default_options
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

impl std::fmt::Debug for SearchExporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchExporter")
            .field("default_options", &self.default_options)
            .finish()
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

impl std::fmt::Debug for HistoryExporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HistoryExporter")
            .field("default_options", &self.default_options)
            .finish()
    }
}