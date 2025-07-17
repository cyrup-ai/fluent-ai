//! Audio domain types
//!
//! Contains pure data structures for audio.
//! Builder implementations are in fluent_ai package.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Audio {
    pub data: String,
    pub format: Option<ContentFormat>,
    pub media_type: Option<AudioMediaType>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ContentFormat {
    Base64,
    Raw,
    Url,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AudioMediaType {
    MP3,
    WAV,
    OGG,
    M4A,
    FLAC,
}

impl Audio {
    /// Create a new audio instance with basic data
    pub fn new(data: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            format: None,
            media_type: None,
        }
    }
}