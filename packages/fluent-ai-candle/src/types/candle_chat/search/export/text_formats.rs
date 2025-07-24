//! Text-based export formats (XML, Text, Markdown, HTML)
//!
//! High-performance text format exports with proper escaping and
//! streaming-compatible architecture.

use super::super::types::{SearchResult, SearchStatistics};
use super::formats::{ExportOptions, ExportError};

/// Text-based export implementations
pub struct TextFormatsExporter;

impl TextFormatsExporter {
    /// Export search results as XML
    pub fn export_as_xml(results: &[SearchResult], options: &ExportOptions) -> Result<String, ExportError> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<search_results>\n");

        if options.include_metadata {
            xml.push_str(&format!(
                "  <metadata>\n    <export_time>{}</export_time>\n    <total_results>{}</total_results>\n  </metadata>\n",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                results.len()
            ));
        }

        xml.push_str("  <results>\n");
        for result in results {
            xml.push_str("    <result>\n");
            xml.push_str(&format!("      <id>{}</id>\n", Self::escape_xml(&result.message.message.id)));
            xml.push_str(&format!("      <user>{}</user>\n", Self::escape_xml(&result.message.message.user.as_deref().unwrap_or(""))));
            xml.push_str(&format!("      <content>{}</content>\n", Self::escape_xml(&result.message.message.content)));
            xml.push_str(&format!("      <timestamp>{}</timestamp>\n", result.message.message.timestamp.map(|t| t.to_string()).unwrap_or_default()));
            
            if options.include_scores {
                xml.push_str(&format!("      <relevance_score>{}</relevance_score>\n", result.relevance_score));
            }
            
            xml.push_str("    </result>\n");
        }
        xml.push_str("  </results>\n");
        xml.push_str("</search_results>\n");

        Ok(xml)
    }

    /// Export search results as plain text
    pub fn export_as_text(results: &[SearchResult], options: &ExportOptions) -> Result<String, ExportError> {
        let mut text = String::new();
        
        if options.include_metadata {
            text.push_str("Search Results Export\n");
            text.push_str(&format!("Total Results: {}\n", results.len()));
            text.push_str(&format!("Export Time: {}\n", 
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            ));
            text.push_str(&"=".repeat(50));
            text.push('\n');
        }

        for (i, result) in results.iter().enumerate() {
            text.push_str(&format!("Result {}\n", i + 1));
            text.push_str(&format!("User: {}\n", result.message.message.user.as_deref().unwrap_or("")));
            text.push_str(&format!("Content: {}\n", result.message.message.content));
            text.push_str(&format!("Timestamp: {}\n", result.message.message.timestamp.map(|t| t.to_string()).unwrap_or_default()));
            
            if options.include_scores {
                text.push_str(&format!("Relevance Score: {:.3}\n", result.relevance_score));
            }
            
            text.push_str(&"-".repeat(30));
            text.push('\n');
        }

        Ok(text)
    }

    /// Export search results as Markdown
    pub fn export_as_markdown(results: &[SearchResult], options: &ExportOptions) -> Result<String, ExportError> {
        let mut md = String::new();
        
        md.push_str("# Search Results\n\n");
        
        if options.include_metadata {
            md.push_str(&format!("**Total Results:** {}\n", results.len()));
            md.push_str(&format!("**Export Time:** {}\n\n", 
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            ));
        }

        for (i, result) in results.iter().enumerate() {
            md.push_str(&format!("## Result {}\n\n", i + 1));
            md.push_str(&format!("**User:** {}\n\n", result.message.message.user.as_deref().unwrap_or("")));
            md.push_str(&format!("**Content:** {}\n\n", result.message.message.content));
            md.push_str(&format!("**Timestamp:** {}\n\n", result.message.message.timestamp.map(|t| t.to_string()).unwrap_or_default()));
            
            if options.include_scores {
                md.push_str(&format!("**Relevance Score:** {:.3}\n\n", result.relevance_score));
            }
            
            md.push_str("---\n\n");
        }

        Ok(md)
    }

    /// Export search results as HTML
    pub fn export_as_html(results: &[SearchResult], options: &ExportOptions) -> Result<String, ExportError> {
        let mut html = String::new();
        
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>Search Results</title>\n");
        html.push_str("<style>body{font-family:Arial,sans-serif;margin:20px;}");
        html.push_str(".result{border:1px solid #ccc;margin:10px 0;padding:15px;border-radius:5px;}");
        html.push_str(".user{font-weight:bold;color:#0066cc;}.score{color:#666;font-size:0.9em;}</style>\n");
        html.push_str("</head>\n<body>\n");
        
        html.push_str("<h1>Search Results</h1>\n");
        
        if options.include_metadata {
            html.push_str(&format!("<p><strong>Total Results:</strong> {}</p>\n", results.len()));
            html.push_str(&format!("<p><strong>Export Time:</strong> {}</p>\n", 
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            ));
        }

        for (i, result) in results.iter().enumerate() {
            html.push_str("<div class=\"result\">\n");
            html.push_str(&format!("<h3>Result {}</h3>\n", i + 1));
            html.push_str(&format!("<p class=\"user\">User: {}</p>\n", Self::escape_html(&result.message.message.user.as_deref().unwrap_or(""))));
            html.push_str(&format!("<p>Content: {}</p>\n", Self::escape_html(&result.message.message.content)));
            html.push_str(&format!("<p>Timestamp: {}</p>\n", result.message.message.timestamp.map(|t| t.to_string()).unwrap_or_default()));
            
            if options.include_scores {
                html.push_str(&format!("<p class=\"score\">Relevance Score: {:.3}</p>\n", result.relevance_score));
            }
            
            html.push_str("</div>\n");
        }
        
        html.push_str("</body>\n</html>\n");

        Ok(html)
    }

    /// Export search statistics as text
    pub fn export_statistics_as_text(stats: &SearchStatistics) -> String {
        format!(
            "Search Statistics\n=================\nTotal Queries: {}\nTotal Results: {}\nAverage Query Time: {:.2}ms\nCache Hit Rate: {:.1}%\n",
            stats.total_queries,
            stats.total_results,
            stats.average_query_time,
            stats.cache_hit_rate * 100.0
        )
    }

    /// Export search statistics as Markdown
    pub fn export_statistics_as_markdown(stats: &SearchStatistics) -> String {
        format!(
            "# Search Statistics\n\n- **Total Queries:** {}\n- **Total Results:** {}\n- **Average Query Time:** {:.2}ms\n- **Cache Hit Rate:** {:.1}%\n",
            stats.total_queries,
            stats.total_results,
            stats.average_query_time,
            stats.cache_hit_rate * 100.0
        )
    }

    /// Escape XML content
    fn escape_xml(content: &str) -> String {
        content
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }

    /// Escape HTML content
    fn escape_html(content: &str) -> String {
        content
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    }
}