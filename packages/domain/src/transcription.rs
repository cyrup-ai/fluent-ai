//! Types for audio transcription functionality

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Request for transcribing audio content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionRequest {
    /// Binary audio data to transcribe
    pub data: Vec<u8>,
    /// Original filename of the audio file
    pub filename: String,
    /// Language of the audio content (ISO 639-1)
    pub language: String,
    /// Optional prompt to guide the transcription
    pub prompt: Option<String>,
    /// Optional temperature for sampling
    pub temperature: Option<f64>,
    /// Additional provider-specific parameters
    pub additional_params: Option<Value>,
}

/// Response from a transcription operation
/// 
/// Wraps the actual transcription text with the original provider response
#[derive(Debug, Clone)]
pub struct TranscriptionResponse<T> {
    /// The transcribed text
    pub text: String,
    /// The original provider response
    pub response: T,
}

impl<T> TranscriptionResponse<T> {
    /// Create a new transcription response
    pub fn new(text: String, response: T) -> Self {
        Self { text, response }
    }
    
    /// Get the transcribed text
    pub fn text(&self) -> &str {
        &self.text
    }
    
    /// Get the original response
    pub fn into_inner(self) -> T {
        self.response
    }
}
