//! Export format implementations and serialization

use super::types::{ExportFormat, ExportOptions};
use crate::types::candle_chat::chat::message::SearchChatMessage;
use std::collections::HashMap;

/// Format handler trait for different export formats
pub trait FormatHandler {
    /// Format a batch of messages
    fn format_messages(&self, messages: &[SearchChatMessage], options: &ExportOptions) -> Result<String, String>;
    
    /// Generate header for the export format
    fn generate_header(&self, options: &ExportOptions) -> Result<String, String>;
    
    /// Generate footer for the export format
    fn generate_footer(&self, options: &ExportOptions) -> Result<String, String>;
    
    /// Get file extension for this format
    fn file_extension(&self) -> &'static str;
    
    /// Get MIME type for this format
    fn mime_type(&self) -> &'static str;
}

/// JSON format handler
pub struct JsonFormatHandler;

impl FormatHandler for JsonFormatHandler {
    fn format_messages(&self, messages: &[SearchChatMessage], options: &ExportOptions) -> Result<String, String> {
        let mut json_messages = Vec::new();
        
        for message in messages {
            let mut json_obj = serde_json::Map::new();
            json_obj.insert("content".to_string(), serde_json::Value::String(message.content.to_string()));
            json_obj.insert("role".to_string(), serde_json::Value::String(message.role.to_string()));
            
            if options.include_metadata {
                json_obj.insert("id".to_string(), serde_json::Value::String(message.id.to_string()));
                json_obj.insert("timestamp".to_string(), serde_json::Value::String(message.timestamp.to_rfc3339()));
            }
            
            json_messages.push(serde_json::Value::Object(json_obj));
        }
        
        let json_array = serde_json::Value::Array(json_messages);
        if options.pretty_print {
            serde_json::to_string_pretty(&json_array).map_err(|e| e.to_string())
        } else {
            serde_json::to_string(&json_array).map_err(|e| e.to_string())
        }
    }
    
    fn generate_header(&self, _options: &ExportOptions) -> Result<String, String> {
        Ok("[\n".to_string())
    }
    
    fn generate_footer(&self, _options: &ExportOptions) -> Result<String, String> {
        Ok("]\n".to_string())
    }
    
    fn file_extension(&self) -> &'static str {
        "json"
    }
    
    fn mime_type(&self) -> &'static str {
        "application/json"
    }
}

/// CSV format handler
pub struct CsvFormatHandler;

impl FormatHandler for CsvFormatHandler {
    fn format_messages(&self, messages: &[SearchChatMessage], options: &ExportOptions) -> Result<String, String> {
        let mut csv_content = String::new();
        
        for message in messages {
            let content_escaped = message.content.replace("\"", "\"\"");
            let role_str = message.role.to_string();
            
            if options.include_metadata {
                csv_content.push_str(&format!(
                    "\"{}\",\"{}\",\"{}\",\"{}\"\n",
                    message.id,
                    role_str,
                    content_escaped,
                    message.timestamp.to_rfc3339()
                ));
            } else {
                csv_content.push_str(&format!(
                    "\"{}\",\"{}\"\n",
                    role_str,
                    content_escaped
                ));
            }
        }
        
        Ok(csv_content)
    }
    
    fn generate_header(&self, options: &ExportOptions) -> Result<String, String> {
        if options.include_metadata {
            Ok("ID,Role,Content,Timestamp\n".to_string())
        } else {
            Ok("Role,Content\n".to_string())
        }
    }
    
    fn generate_footer(&self, _options: &ExportOptions) -> Result<String, String> {
        Ok(String::new())
    }
    
    fn file_extension(&self) -> &'static str {
        "csv"
    }
    
    fn mime_type(&self) -> &'static str {
        "text/csv"
    }
}

/// Markdown format handler
pub struct MarkdownFormatHandler;

impl FormatHandler for MarkdownFormatHandler {
    fn format_messages(&self, messages: &[SearchChatMessage], options: &ExportOptions) -> Result<String, String> {
        let mut md_content = String::new();
        
        for message in messages {
            let role_str = message.role.to_string();
            md_content.push_str(&format!("## {}\n\n", role_str));
            md_content.push_str(&message.content);
            md_content.push_str("\n\n");
            
            if options.include_metadata {
                md_content.push_str(&format!(
                    "*ID: {} | Timestamp: {}*\n\n",
                    message.id,
                    message.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
                ));
            }
            
            md_content.push_str("---\n\n");
        }
        
        Ok(md_content)
    }
    
    fn generate_header(&self, _options: &ExportOptions) -> Result<String, String> {
        Ok("# Chat Export\n\n".to_string())
    }
    
    fn generate_footer(&self, _options: &ExportOptions) -> Result<String, String> {
        Ok("\n---\n*Export completed*\n".to_string())
    }
    
    fn file_extension(&self) -> &'static str {
        "md"
    }
    
    fn mime_type(&self) -> &'static str {
        "text/markdown"
    }
}

/// Get format handler for export format
pub fn get_format_handler(format: &ExportFormat) -> Box<dyn FormatHandler> {
    match format {
        ExportFormat::Json => Box::new(JsonFormatHandler),
        ExportFormat::Csv => Box::new(CsvFormatHandler),
        ExportFormat::Markdown => Box::new(MarkdownFormatHandler),
        // Add other format handlers as needed
        _ => Box::new(JsonFormatHandler), // Default to JSON
    }
}