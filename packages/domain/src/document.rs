use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::async_task::error_handlers::BadTraitImpl;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub data: String,
    pub format: Option<ContentFormat>,
    pub media_type: Option<DocumentMediaType>,
    #[serde(flatten)]
    pub additional_props: HashMap<String, Value>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ContentFormat {
    Base64,
    Text,
    Html,
    Markdown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DocumentMediaType {
    PDF,
    DOCX,
    TXT,
    RTF,
    ODT,
}

// Builder implementations moved to fluent_ai/src/builders/document.rs

impl Document {
    /// Create a new document from file path (simplified version for domain use)
    /// Full builder functionality is in fluent-ai/src/builders/document.rs
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> DocumentLoader {
        DocumentLoader {
            path: path.as_ref().display().to_string(),
        }
    }

    /// Extract the text content from the document
    pub fn content(&self) -> String {
        match self.format {
            Some(ContentFormat::Base64) => "[Base64 Document]".to_string(),
            _ => self.data.clone(),
        }
    }
}

/// Simple document loader for domain use
pub struct DocumentLoader {
    path: String,
}

impl DocumentLoader {
    /// Load the document (simplified version)
    pub fn load(self) -> Document {
        // For domain use, return a simple text document
        // Full loading logic is in fluent-ai builders
        Document {
            data: format!("Document from: {}", self.path),
            format: Some(ContentFormat::Text),
            media_type: Some(DocumentMediaType::TXT),
            additional_props: HashMap::new(),
        }
    }
}

// All builder implementations moved to fluent_ai/src/builders/document.rs

/// BadTraitImpl for Document - returns empty document as the bad implementation
impl BadTraitImpl for Document {
    fn bad_impl(error: &str) -> Self {
        eprintln!("Document BadTraitImpl: {}", error);
        Document {
            data: format!("Error loading document: {}", error),
            format: Some(ContentFormat::Text),
            media_type: Some(DocumentMediaType::TXT),
            additional_props: HashMap::new(),
        }
    }
}
