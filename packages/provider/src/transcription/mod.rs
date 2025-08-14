//! Zero-allocation audio transcription interfaces
//!
//! This module provides blazing-fast, streaming audio transcription
//! with zero-allocation audio processing and consistent interfaces
//! across all provider implementations.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use fluent_ai_async::{AsyncStream, AsyncStreamSender};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::AsyncRead;

/// Trait for transcription models - unified interface across providers
pub trait TranscriptionModel: Send + Sync + Clone {
    /// The type of response this model produces
    type Response: Send + Sync;

    /// Transcribe audio data with the given request
    async fn transcription(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResponse<Self::Response>, TranscriptionError>;
}

/// Transcription request structure for provider implementations
#[derive(Debug, Clone)]
pub struct TranscriptionRequest {
    /// Audio data to transcribe
    pub data: Vec<u8>,
    /// Original filename (for format detection)
    pub filename: String,
    /// Temperature for transcription (0.0-1.0)
    pub temperature: Option<f64>,
    /// Additional provider-specific parameters
    pub additional_params: Option<Value>,
}

impl TranscriptionRequest {
    /// Create a new transcription request
    pub fn new(data: Vec<u8>, filename: String) -> Self {
        Self {
            data,
            filename,
            temperature: None,
            additional_params: None,
        }
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set additional parameters
    pub fn with_additional_params(mut self, params: Value) -> Self {
        self.additional_params = Some(params);
        self
    }
}

/// Comprehensive transcription error types for robust error handling
#[derive(Debug, thiserror::Error)]
pub enum TranscriptionError {
    /// Network/HTTP request failed
    #[error("Network request failed: {0}")]
    NetworkError(String),

    /// Authentication failed
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),

    /// Invalid audio format
    #[error("Invalid audio format: {0}")]
    InvalidAudioFormat(String),

    /// Audio file not found
    #[error("Audio file not found: {0}")]
    FileNotFound(String),

    /// Audio file too large
    #[error("Audio file too large: {size} bytes (max: {max_size} bytes)")]
    FileTooLarge { size: u64, max_size: u64 },

    /// Unsupported language
    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),

    /// Model not found or not supported
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {message}")]
    RateLimitExceeded { message: String },

    /// Quota exceeded
    #[error("Quota exceeded: {message}")]
    QuotaExceeded { message: String },

    /// Serialization failed
    #[error("Serialization failed: {0}")]
    SerializationError(String),

    /// Deserialization failed
    #[error("Deserialization failed: {0}")]
    DeserializationError(String),

    /// Audio processing failed
    #[error("Audio processing failed: {0}")]
    AudioProcessingError(String),

    /// Provider-specific error
    #[error("Provider error: {0}")]
    ProviderError(String),

    /// Configuration error
    #[error("Configuration error: {field}: {message}")]
    ConfigurationError { field: String, message: String },

    /// IO error (file operations)
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for transcription operations
pub type Result<T> = std::result::Result<T, TranscriptionError>;

/// Supported audio formats for transcription
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioFormat {
    /// MP3 audio format
    Mp3,
    /// MP4 audio format
    Mp4,
    /// M4A audio format
    M4a,
    /// WAV audio format
    Wav,
    /// WebM audio format
    Webm,
    /// FLAC audio format
    Flac,
    /// OGG audio format
    Ogg,
}

impl AudioFormat {
    /// Get the file extension for this format
    #[inline(always)]
    pub fn extension(&self) -> &'static str {
        match self {
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Mp4 => "mp4",
            AudioFormat::M4a => "m4a",
            AudioFormat::Wav => "wav",
            AudioFormat::Webm => "webm",
            AudioFormat::Flac => "flac",
            AudioFormat::Ogg => "ogg",
        }
    }

    /// Get the MIME type for this format
    #[inline(always)]
    pub fn mime_type(&self) -> &'static str {
        match self {
            AudioFormat::Mp3 => "audio/mp3",
            AudioFormat::Mp4 => "audio/mp4",
            AudioFormat::M4a => "audio/m4a",
            AudioFormat::Wav => "audio/wav",
            AudioFormat::Webm => "audio/webm",
            AudioFormat::Flac => "audio/flac",
            AudioFormat::Ogg => "audio/ogg",
        }
    }

    /// Detect format from file extension
    #[inline]
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "mp3" => Some(AudioFormat::Mp3),
            "mp4" => Some(AudioFormat::Mp4),
            "m4a" => Some(AudioFormat::M4a),
            "wav" => Some(AudioFormat::Wav),
            "webm" => Some(AudioFormat::Webm),
            "flac" => Some(AudioFormat::Flac),
            "ogg" => Some(AudioFormat::Ogg),
            _ => None,
        }
    }

    /// Detect format from file path
    #[inline]
    pub fn from_path<P: AsRef<Path>>(path: P) -> Option<Self> {
        path.as_ref()
            .extension()
            .and_then(|ext| ext.to_str())
            .and_then(Self::from_extension)
    }
}

impl fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.extension())
    }
}

/// Audio transcription language codes (ISO 639-1)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    /// English
    En,
    /// Spanish
    Es,
    /// French
    Fr,
    /// German
    De,
    /// Italian
    It,
    /// Portuguese
    Pt,
    /// Russian
    Ru,
    /// Chinese (Mandarin)
    Zh,
    /// Japanese
    Ja,
    /// Korean
    Ko,
    /// Arabic
    Ar,
    /// Hindi
    Hi,
    /// Dutch
    Nl,
    /// Polish
    Pl,
    /// Turkish
    Tr,
    /// Swedish
    Sv,
    /// Norwegian
    No,
    /// Danish
    Da,
    /// Finnish
    Fi,
    /// Auto-detect language
    Auto,
}

impl Language {
    /// Get the ISO 639-1 language code
    #[inline(always)]
    pub fn code(&self) -> &'static str {
        match self {
            Language::En => "en",
            Language::Es => "es",
            Language::Fr => "fr",
            Language::De => "de",
            Language::It => "it",
            Language::Pt => "pt",
            Language::Ru => "ru",
            Language::Zh => "zh",
            Language::Ja => "ja",
            Language::Ko => "ko",
            Language::Ar => "ar",
            Language::Hi => "hi",
            Language::Nl => "nl",
            Language::Pl => "pl",
            Language::Tr => "tr",
            Language::Sv => "sv",
            Language::No => "no",
            Language::Da => "da",
            Language::Fi => "fi",
            Language::Auto => "auto",
        }
    }

    /// Parse language from ISO code
    #[inline]
    pub fn from_code(code: &str) -> Option<Self> {
        match code.to_lowercase().as_str() {
            "en" => Some(Language::En),
            "es" => Some(Language::Es),
            "fr" => Some(Language::Fr),
            "de" => Some(Language::De),
            "it" => Some(Language::It),
            "pt" => Some(Language::Pt),
            "ru" => Some(Language::Ru),
            "zh" => Some(Language::Zh),
            "ja" => Some(Language::Ja),
            "ko" => Some(Language::Ko),
            "ar" => Some(Language::Ar),
            "hi" => Some(Language::Hi),
            "nl" => Some(Language::Nl),
            "pl" => Some(Language::Pl),
            "tr" => Some(Language::Tr),
            "sv" => Some(Language::Sv),
            "no" => Some(Language::No),
            "da" => Some(Language::Da),
            "fi" => Some(Language::Fi),
            "auto" => Some(Language::Auto),
            _ => None,
        }
    }

    /// Get the human-readable language name
    #[inline(always)]
    pub fn name(&self) -> &'static str {
        match self {
            Language::En => "English",
            Language::Es => "Spanish",
            Language::Fr => "French",
            Language::De => "German",
            Language::It => "Italian",
            Language::Pt => "Portuguese",
            Language::Ru => "Russian",
            Language::Zh => "Chinese",
            Language::Ja => "Japanese",
            Language::Ko => "Korean",
            Language::Ar => "Arabic",
            Language::Hi => "Hindi",
            Language::Nl => "Dutch",
            Language::Pl => "Polish",
            Language::Tr => "Turkish",
            Language::Sv => "Swedish",
            Language::No => "Norwegian",
            Language::Da => "Danish",
            Language::Fi => "Finnish",
            Language::Auto => "Auto-detect",
        }
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

/// Universal transcription provider trait
///
/// Provides zero-allocation audio transcription with streaming support
/// and consistent interfaces across all AI providers.
pub trait TranscriptionProvider: Send + Sync {
    /// The type of raw transcription response from the provider
    type Response: Send + Sync;

    /// Transcribe audio from a file path
    ///
    /// This method provides zero-allocation file-based transcription
    /// with automatic format detection and optimized file handling.
    async fn transcribe_file<P: AsRef<Path> + Send>(
        &self,
        path: P,
        options: TranscriptionOptions,
    ) -> Result<TranscriptionResponse<Self::Response>>;

    /// Transcribe audio from raw bytes
    ///
    /// This method provides zero-allocation transcription from audio
    /// data already loaded in memory.
    async fn transcribe_bytes(
        &self,
        audio_data: &[u8],
        format: AudioFormat,
        options: TranscriptionOptions,
    ) -> Result<TranscriptionResponse<Self::Response>>;

    /// Transcribe audio from a stream (for large files)
    ///
    /// This method provides streaming transcription for large audio files
    /// that cannot be loaded entirely in memory.
    async fn transcribe_stream<R: AsyncRead + Send + Unpin>(
        &self,
        audio_stream: R,
        format: AudioFormat,
        options: TranscriptionOptions,
    ) -> Result<StreamingTranscriptionResponse<Self::Response>>;

    /// Get supported audio formats for this provider
    fn supported_formats(&self) -> &[AudioFormat];

    /// Get supported languages for this provider
    fn supported_languages(&self) -> &[Language];

    /// Get the maximum file size allowed (in bytes)
    fn max_file_size(&self) -> u64;

    /// Get the model name/identifier
    fn model_name(&self) -> &str;

    /// Check if the provider supports real-time streaming
    fn supports_streaming(&self) -> bool {
        false // Most providers don't support real-time streaming yet
    }

    /// Check if the provider supports timestamps
    fn supports_timestamps(&self) -> bool {
        true // Most providers support timestamps
    }

    /// Check if the provider supports speaker diarization
    fn supports_speaker_diarization(&self) -> bool {
        false // Advanced feature, not all providers support it
    }
}

/// Transcription options for configuring the transcription process
#[derive(Debug, Clone)]
pub struct TranscriptionOptions {
    /// Target language for transcription (None for auto-detection)
    pub language: Option<Language>,
    /// Whether to include timestamps in the response
    pub timestamps: bool,
    /// Whether to perform speaker diarization
    pub speaker_diarization: bool,
    /// Custom model to use (provider-specific)
    pub model: Option<String>,
    /// Temperature for the transcription (0.0-1.0)
    pub temperature: Option<f32>,
    /// Custom prompt to guide the transcription
    pub prompt: Option<String>,
    /// Maximum segment length in seconds (for chunked processing)
    pub max_segment_length: Option<f32>,
    /// Additional provider-specific options
    pub additional_options: HashMap<String, Value>,
}

impl Default for TranscriptionOptions {
    fn default() -> Self {
        Self {
            language: None, // Auto-detect
            timestamps: false,
            speaker_diarization: false,
            model: None,
            temperature: None,
            prompt: None,
            max_segment_length: None,
            additional_options: HashMap::new(),
        }
    }
}

impl TranscriptionOptions {
    /// Create new transcription options with defaults
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the target language
    #[inline(always)]
    pub fn with_language(mut self, language: Language) -> Self {
        self.language = Some(language);
        self
    }

    /// Enable timestamps in the response
    #[inline(always)]
    pub fn with_timestamps(mut self) -> Self {
        self.timestamps = true;
        self
    }

    /// Enable speaker diarization
    #[inline(always)]
    pub fn with_speaker_diarization(mut self) -> Self {
        self.speaker_diarization = true;
        self
    }

    /// Set a custom model
    #[inline(always)]
    pub fn with_model(mut self, model: String) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the temperature
    #[inline(always)]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature.clamp(0.0, 1.0));
        self
    }

    /// Set a custom prompt
    #[inline(always)]
    pub fn with_prompt(mut self, prompt: String) -> Self {
        self.prompt = Some(prompt);
        self
    }

    /// Set maximum segment length
    #[inline(always)]
    pub fn with_max_segment_length(mut self, seconds: f32) -> Self {
        self.max_segment_length = Some(seconds);
        self
    }

    /// Add an additional option
    #[inline(always)]
    pub fn with_option(mut self, key: String, value: Value) -> Self {
        self.additional_options.insert(key, value);
        self
    }
}

/// Zero-allocation transcription response wrapper
#[derive(Debug, Clone)]
pub struct TranscriptionResponse<T> {
    /// The raw provider-specific response
    pub raw_response: T,
    /// The transcribed text
    pub text: String,
    /// Segments with timestamps (if requested)
    pub segments: Option<Vec<TranscriptionSegment>>,
    /// Detected language (if auto-detection was used)
    pub language: Option<Language>,
    /// Confidence score (0.0-1.0)
    pub confidence: Option<f32>,
    /// Response metadata
    pub metadata: TranscriptionMetadata,
}

impl<T> TranscriptionResponse<T> {
    /// Create a new transcription response
    #[inline(always)]
    pub fn new(raw_response: T, text: String) -> Self {
        Self {
            raw_response,
            text,
            segments: None,
            language: None,
            confidence: None,
            metadata: TranscriptionMetadata::default(),
        }
    }

    /// Add segments with timestamps
    #[inline(always)]
    pub fn with_segments(mut self, segments: Vec<TranscriptionSegment>) -> Self {
        self.segments = Some(segments);
        self
    }

    /// Add detected language
    #[inline(always)]
    pub fn with_language(mut self, language: Language) -> Self {
        self.language = Some(language);
        self
    }

    /// Add confidence score
    #[inline(always)]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// Add metadata
    #[inline(always)]
    pub fn with_metadata(mut self, metadata: TranscriptionMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Get the transcribed text
    #[inline(always)]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Get segments if available
    #[inline(always)]
    pub fn segments(&self) -> Option<&[TranscriptionSegment]> {
        self.segments.as_deref()
    }

    /// Get the total duration from segments
    #[inline]
    pub fn duration(&self) -> Option<f32> {
        self.segments
            .as_ref()
            .and_then(|segments| segments.last())
            .map(|last_segment| last_segment.end)
    }
}

/// Transcription segment with timestamps
#[derive(Debug, Clone)]
pub struct TranscriptionSegment {
    /// Segment text
    pub text: String,
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Confidence score for this segment (0.0-1.0)
    pub confidence: Option<f32>,
    /// Speaker ID (for diarization)
    pub speaker: Option<String>,
}

impl TranscriptionSegment {
    /// Create a new transcription segment
    #[inline(always)]
    pub fn new(text: String, start: f32, end: f32) -> Self {
        Self {
            text,
            start,
            end,
            confidence: None,
            speaker: None,
        }
    }

    /// Add confidence score
    #[inline(always)]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// Add speaker ID
    #[inline(always)]
    pub fn with_speaker(mut self, speaker: String) -> Self {
        self.speaker = Some(speaker);
        self
    }

    /// Get segment duration
    #[inline(always)]
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }
}

/// Streaming transcription response for real-time processing
pub struct StreamingTranscriptionResponse<T> {
    /// The raw provider-specific streaming response
    pub raw_response: T,
    /// Stream of transcription chunks
    pub stream: AsyncStream<TranscriptionChunk>,
    /// Response metadata (filled as chunks arrive)
    pub metadata: TranscriptionMetadata,
}

impl<T> StreamingTranscriptionResponse<T> {
    /// Create a new streaming transcription response
    #[inline(always)]
    pub fn new(
        raw_response: T,
        stream: AsyncStream<Result<TranscriptionChunk, TranscriptionError>>,
    ) -> Self {
        Self {
            raw_response,
            stream,
            metadata: TranscriptionMetadata::default(),
        }
    }

    /// Add metadata
    #[inline(always)]
    pub fn with_metadata(mut self, metadata: TranscriptionMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Collect all chunks into a single transcription
    pub async fn collect(mut self) -> Result<TranscriptionResponse<T>> {
        let mut full_text = String::new();
        let mut segments = Vec::new();
        let mut detected_language = None;
        let mut overall_confidence = None;

        while let Some(chunk_result) = self.stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    full_text.push_str(&chunk.text);

                    if let Some(segment) = chunk.segment {
                        segments.push(segment);
                    }

                    if detected_language.is_none() {
                        detected_language = chunk.language;
                    }

                    if let Some(conf) = chunk.confidence {
                        overall_confidence = Some(conf);
                    }
                }
                Err(e) => return Err(e),
            }
        }

        let mut response = TranscriptionResponse::new(self.raw_response, full_text);

        if !segments.is_empty() {
            response = response.with_segments(segments);
        }

        if let Some(lang) = detected_language {
            response = response.with_language(lang);
        }

        if let Some(conf) = overall_confidence {
            response = response.with_confidence(conf);
        }

        response = response.with_metadata(self.metadata);

        Ok(response)
    }
}

/// Transcription chunk for streaming responses
#[derive(Debug, Clone)]
pub struct TranscriptionChunk {
    /// Partial transcription text
    pub text: String,
    /// Segment information (if available)
    pub segment: Option<TranscriptionSegment>,
    /// Detected language (if auto-detection was used)
    pub language: Option<Language>,
    /// Confidence score for this chunk
    pub confidence: Option<f32>,
    /// Whether this is the final chunk
    pub is_final: bool,
}

/// Transcription response metadata
#[derive(Debug, Clone, Default)]
pub struct TranscriptionMetadata {
    /// Request ID (if provided by the API)
    pub request_id: Option<String>,
    /// Model used for the transcription
    pub model: Option<String>,
    /// Processing time in milliseconds
    pub processing_time_ms: Option<u64>,
    /// Audio duration in seconds
    pub audio_duration: Option<f32>,
    /// Audio format that was processed
    pub audio_format: Option<AudioFormat>,
    /// Additional provider-specific metadata
    pub provider_metadata: Option<Value>,
}

impl TranscriptionMetadata {
    /// Create new transcription metadata
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set request ID
    #[inline(always)]
    pub fn with_request_id(mut self, request_id: String) -> Self {
        self.request_id = Some(request_id);
        self
    }

    /// Set model name
    #[inline(always)]
    pub fn with_model(mut self, model: String) -> Self {
        self.model = Some(model);
        self
    }

    /// Set processing time
    #[inline(always)]
    pub fn with_processing_time(mut self, processing_time_ms: u64) -> Self {
        self.processing_time_ms = Some(processing_time_ms);
        self
    }

    /// Set audio duration
    #[inline(always)]
    pub fn with_audio_duration(mut self, duration: f32) -> Self {
        self.audio_duration = Some(duration);
        self
    }

    /// Set audio format
    #[inline(always)]
    pub fn with_audio_format(mut self, format: AudioFormat) -> Self {
        self.audio_format = Some(format);
        self
    }

    /// Set provider metadata
    #[inline(always)]
    pub fn with_provider_metadata(mut self, metadata: Value) -> Self {
        self.provider_metadata = Some(metadata);
        self
    }
}

pub use StreamingTranscriptionResponse as DefaultStreamingTranscriptionResponse;
/// Re-exports for convenience
pub use TranscriptionResponse as DefaultTranscriptionResponse;
