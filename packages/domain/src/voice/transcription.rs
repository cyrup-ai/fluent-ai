//! Types and functionality for audio transcription
//!
//! This module provides types for handling audio transcription requests and responses.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Simple transcription result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcription {
    /// The transcribed text
    pub text: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: Option<f64>,
    /// Language detected (ISO 639-1)
    pub language: Option<String>}

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

    /// Optional temperature for sampling (0.0 to 1.0)
    pub temperature: Option<f64>,

    /// Additional provider-specific parameters
    pub additional_params: Option<Value>}

impl TranscriptionRequest {
    /// Create a new transcription request with required fields
    pub fn new(data: Vec<u8>, filename: impl Into<String>, language: impl Into<String>) -> Self {
        Self {
            data,
            filename: filename.into(),
            language: language.into(),
            prompt: None,
            temperature: None,
            additional_params: None}
    }

    /// Set an optional prompt to guide the transcription
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Set the temperature for sampling
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Add additional provider-specific parameters
    pub fn with_additional_params(mut self, params: Value) -> Self {
        self.additional_params = Some(params);
        self
    }
}

/// Response from a transcription operation
///
/// Wraps the actual transcription text with the original provider response
#[derive(Debug, Clone)]
pub struct TranscriptionResponse<T> {
    /// The transcribed text
    pub text: String,

    /// The original provider response
    pub response: T}

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

    /// Map the inner response to a different type
    pub fn map<U, F>(self, f: F) -> TranscriptionResponse<U>
    where
        F: FnOnce(T) -> U,
    {
        TranscriptionResponse {
            text: self.text,
            response: f(self.response)}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcription_request_creation() {
        let data = vec![1, 2, 3];
        let request = TranscriptionRequest::new(data.clone(), "test.mp3", "en");

        assert_eq!(request.data, data);
        assert_eq!(request.filename, "test.mp3");
        assert_eq!(request.language, "en");
        assert!(request.prompt.is_none());
        assert!(request.temperature.is_none());
        assert!(request.additional_params.is_none());
    }

    #[test]
    fn test_transcription_request_builder() {
        let data = vec![1, 2, 3];
        let params = serde_json::json!({ "model": "whisper-1" });

        let request = TranscriptionRequest::new(data, "test.mp3", "en")
            .with_prompt("Transcribe this audio")
            .with_temperature(0.7)
            .with_additional_params(params.clone());

        assert_eq!(request.prompt, Some("Transcribe this audio".to_string()));
        assert_eq!(request.temperature, Some(0.7));
        assert_eq!(request.additional_params, Some(params));
    }

    #[test]
    fn test_transcription_response() {
        let response = TranscriptionResponse::new("test".to_string(), ());

        assert_eq!(response.text(), "test");
        assert_eq!(response.into_inner(), ());
    }

    #[test]
    fn test_transcription_response_map() {
        let response = TranscriptionResponse::new("test".to_string(), 42);
        let mapped = response.map(|x| x.to_string());

        assert_eq!(mapped.text, "test");
        assert_eq!(mapped.response, "42");
    }
}
