//! Voice processing module
//!
//! This module provides functionality for handling voice-related operations,
//! including audio processing and transcription services.

pub mod audio;
pub mod transcription;

// Re-export types for public API
pub use audio::{Audio, AudioMediaType, ContentFormat as AudioContentFormat};
use serde::{Deserialize, Serialize};
pub use transcription::{Transcription, TranscriptionRequest, TranscriptionResponse};

/// Voice processing error type
#[derive(Debug, thiserror::Error)]
pub enum VoiceError {
    /// Audio format error
    #[error("Invalid audio format: {0}")]
    FormatError(String),

    /// Transcription error
    #[error("Transcription failed: {0}")]
    TranscriptionError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Other errors
    #[error("Voice processing error: {0}")]
    Other(String)}

/// Type alias for voice processing results
pub type Result<T> = std::result::Result<T, VoiceError>;

/// Voice processing service trait
pub trait VoiceService: Send + Sync + 'static {
    /// Transcribe audio data to text
    fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> crate::AsyncTask<Result<TranscriptionResponse<()>>>;

    /// Convert text to speech
    fn synthesize(&self, text: &str, voice_id: &str) -> crate::AsyncTask<Result<Vec<u8>>>;

    /// List available voices
    fn list_voices(&self) -> crate::AsyncTask<Result<Vec<VoiceInfo>>>;
}

/// Information about an available voice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceInfo {
    /// Unique identifier for the voice
    pub id: String,
    /// Display name of the voice
    pub name: String,
    /// Language code (BCP-47)
    pub language: String,
    /// Gender of the voice (if applicable)
    pub gender: Option<String>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_creation() {
        let audio = Audio::new("test".to_string());
        assert_eq!(audio.data, "test");
        assert!(audio.format.is_none());
        assert!(audio.media_type.is_none());
    }

    #[test]
    fn test_transcription_response() {
        let response = TranscriptionResponse::new("test".to_string(), ());
        assert_eq!(response.text(), "test");
    }
}
