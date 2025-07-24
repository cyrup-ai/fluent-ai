//! CSV export implementation  
//!
//! High-performance CSV export with proper field escaping and
//! streaming-compatible architecture.

use super::super::types::{SearchResult, SearchStatistics};
use super::formats::{ExportOptions, ExportError};

/// CSV export implementation
pub struct CsvExporter;

impl CsvExporter {
    /// Export search results as CSV
    pub fn export_results(results: &[SearchResult], options: &ExportOptions) -> Result<String, ExportError> {
        let mut csv = String::new();
        
        // CSV header
        csv.push_str("id,user,content,timestamp");
        if options.include_scores {
            csv.push_str(",relevance_score");
        }
        if options.include_timestamps {
            csv.push_str(",created_at");
        }
        csv.push('\n');

        // CSV rows
        for result in results {
            csv.push_str(&format!(
                "{},{},{},{}",
                Self::escape_csv_field(&result.message.message.id),
                Self::escape_csv_field(&result.message.message.user.as_deref().unwrap_or("")),
                Self::escape_csv_field(&result.message.message.content),
                result.message.message.timestamp.map(|t| t.to_string()).unwrap_or_default()
            ));
            
            if options.include_scores {
                csv.push_str(&format!(",{}", result.relevance_score));
            }
            
            if options.include_timestamps {
                if let Some(timestamp) = result.message.message.timestamp {
                    csv.push_str(&format!(",{}", timestamp));
                } else {
                    csv.push_str(",");
                }
            }
            
            csv.push('\n');
        }

        Ok(csv)
    }

    /// Export search statistics as CSV
    pub fn export_statistics(stats: &SearchStatistics) -> String {
        let mut csv = String::new();
        csv.push_str("metric,value\n");
        csv.push_str(&format!("total_queries,{}\n", stats.total_queries));
        csv.push_str(&format!("total_results,{}\n", stats.total_results));
        csv.push_str(&format!("average_query_time,{}\n", stats.average_query_time));
        csv.push_str(&format!("cache_hit_rate,{}\n", stats.cache_hit_rate));
        csv
    }

    /// Escape CSV field for proper formatting
    fn escape_csv_field(field: &str) -> String {
        if field.contains(',') || field.contains('"') || field.contains('\n') {
            format!("\"{}\"", field.replace('"', "\"\""))
        } else {
            field.to_string()
        }
    }
}