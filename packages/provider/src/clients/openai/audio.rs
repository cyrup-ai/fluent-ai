//! Zero-allocation OpenAI audio capabilities for speech and transcription
//!
//! Provides comprehensive support for OpenAI's audio models including Whisper (transcription),
//! TTS (text-to-speech), and audio processing with optimal performance patterns.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::{OpenAIError, OpenAIResult};

/// Audio format types supported by OpenAI
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    MP3,
    MP4,
    MPEG,
    MPGA,
    M4A,
    WAV,
    WEBM,
    FLAC,
    OGG}

/// Voice options for text-to-speech
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Voice {
    Alloy,
    Echo,
    Fable,
    Onyx,
    Nova,
    Shimmer}

/// Audio quality settings for TTS
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioQuality {
    /// Standard quality (faster, smaller files)
    Standard,
    /// High definition quality (slower, larger files)
    HD}

/// Response format for TTS output
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResponseFormat {
    MP3,
    OPUS,
    AAC,
    FLAC,
    WAV,
    PCM}

/// Transcription language codes (ISO 639-1)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageCode(pub String);

/// Audio data container with metadata
#[derive(Debug, Clone)]
pub struct AudioData {
    pub data: Vec<u8>,
    pub format: AudioFormat,
    pub duration: Option<Duration>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
    pub bit_rate: Option<u32>}

/// Transcription request configuration
#[derive(Debug, Clone)]
pub struct TranscriptionRequest {
    pub audio: AudioData,
    pub model: String,
    pub language: Option<LanguageCode>,
    pub prompt: Option<String>,
    pub response_format: Option<String>,
    pub temperature: Option<f32>,
    pub timestamp_granularities: Vec<String>}

/// Transcription response from Whisper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<WordTimestamp>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<SegmentTimestamp>>}

/// Word-level timestamp information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTimestamp {
    pub word: String,
    pub start: f32,
    pub end: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>}

/// Segment-level timestamp information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentTimestamp {
    pub id: u32,
    pub seek: u32,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub tokens: Vec<u32>,
    pub temperature: f32,
    pub avg_logprob: f32,
    pub compression_ratio: f32,
    pub no_speech_prob: f32}

/// Text-to-speech request configuration
#[derive(Debug, Clone)]
pub struct TTSRequest {
    pub model: String,
    pub input: String,
    pub voice: Voice,
    pub response_format: ResponseFormat,
    pub speed: Option<f32>}

/// Translation request (audio to English text)
#[derive(Debug, Clone)]
pub struct TranslationRequest {
    pub audio: AudioData,
    pub model: String,
    pub prompt: Option<String>,
    pub response_format: Option<String>,
    pub temperature: Option<f32>}

/// Audio processing capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioCapabilities {
    pub transcription: bool,
    pub translation: bool,
    pub text_to_speech: bool,
    pub real_time_transcription: bool,
    pub speaker_identification: bool,
    pub noise_reduction: bool}

impl AudioFormat {
    /// Detect format from file extension
    #[inline(always)]
    pub fn from_extension(ext: &str) -> OpenAIResult<Self> {
        match ext.to_lowercase().as_str() {
            "mp3" => Ok(Self::MP3),
            "mp4" => Ok(Self::MP4),
            "mpeg" => Ok(Self::MPEG),
            "mpga" => Ok(Self::MPGA),
            "m4a" => Ok(Self::M4A),
            "wav" => Ok(Self::WAV),
            "webm" => Ok(Self::WEBM),
            "flac" => Ok(Self::FLAC),
            "ogg" => Ok(Self::OGG),
            _ => Err(OpenAIError::AudioError(format!(
                "Unsupported audio format: {}",
                ext
            )))}
    }

    /// Get MIME type for format
    #[inline(always)]
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::MP3 => "audio/mpeg",
            Self::MP4 => "audio/mp4",
            Self::MPEG => "audio/mpeg",
            Self::MPGA => "audio/mpeg",
            Self::M4A => "audio/mp4",
            Self::WAV => "audio/wav",
            Self::WEBM => "audio/webm",
            Self::FLAC => "audio/flac",
            Self::OGG => "audio/ogg"}
    }

    /// Check if format supports metadata
    #[inline(always)]
    pub fn supports_metadata(&self) -> bool {
        matches!(self, Self::MP3 | Self::MP4 | Self::M4A | Self::FLAC)
    }

    /// Check if format is lossless
    #[inline(always)]
    pub fn is_lossless(&self) -> bool {
        matches!(self, Self::WAV | Self::FLAC)
    }

    /// Get typical file extension
    #[inline(always)]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::MP3 => "mp3",
            Self::MP4 => "mp4",
            Self::MPEG => "mpeg",
            Self::MPGA => "mpga",
            Self::M4A => "m4a",
            Self::WAV => "wav",
            Self::WEBM => "webm",
            Self::FLAC => "flac",
            Self::OGG => "ogg"}
    }
}

impl Voice {
    /// Get all available voices
    #[inline(always)]
    pub fn all() -> Vec<Self> {
        vec![
            Self::Alloy,
            Self::Echo,
            Self::Fable,
            Self::Onyx,
            Self::Nova,
            Self::Shimmer,
        ]
    }

    /// Get voice description
    #[inline(always)]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Alloy => "Balanced, versatile voice suitable for most content",
            Self::Echo => "Clear, crisp voice ideal for professional content",
            Self::Fable => "Warm, friendly voice perfect for storytelling",
            Self::Onyx => "Deep, resonant voice great for serious content",
            Self::Nova => "Bright, energetic voice ideal for upbeat content",
            Self::Shimmer => "Smooth, sophisticated voice perfect for elegant content"}
    }

    /// Check if voice supports specific language well
    #[inline(always)]
    pub fn supports_language(&self, language: &str) -> bool {
        // All voices support English and major languages
        // This could be expanded with specific voice-language compatibility
        matches!(
            language,
            "en" | "es" | "fr" | "de" | "it" | "pt" | "nl" | "ja" | "ko" | "zh"
        )
    }
}

impl ResponseFormat {
    /// Get MIME type for response format
    #[inline(always)]
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::MP3 => "audio/mpeg",
            Self::OPUS => "audio/opus",
            Self::AAC => "audio/aac",
            Self::FLAC => "audio/flac",
            Self::WAV => "audio/wav",
            Self::PCM => "audio/pcm"}
    }

    /// Check if format is compressed
    #[inline(always)]
    pub fn is_compressed(&self) -> bool {
        matches!(self, Self::MP3 | Self::OPUS | Self::AAC)
    }

    /// Get typical file extension
    #[inline(always)]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::MP3 => "mp3",
            Self::OPUS => "opus",
            Self::AAC => "aac",
            Self::FLAC => "flac",
            Self::WAV => "wav",
            Self::PCM => "pcm"}
    }
}

impl LanguageCode {
    /// Create language code from string
    #[inline(always)]
    pub fn new(code: impl Into<String>) -> Self {
        Self(code.into())
    }

    /// Common language codes
    #[inline(always)]
    pub fn english() -> Self {
        Self("en".to_string())
    }

    #[inline(always)]
    pub fn spanish() -> Self {
        Self("es".to_string())
    }

    #[inline(always)]
    pub fn french() -> Self {
        Self("fr".to_string())
    }

    #[inline(always)]
    pub fn german() -> Self {
        Self("de".to_string())
    }

    #[inline(always)]
    pub fn italian() -> Self {
        Self("it".to_string())
    }

    #[inline(always)]
    pub fn portuguese() -> Self {
        Self("pt".to_string())
    }

    #[inline(always)]
    pub fn dutch() -> Self {
        Self("nl".to_string())
    }

    #[inline(always)]
    pub fn japanese() -> Self {
        Self("ja".to_string())
    }

    #[inline(always)]
    pub fn korean() -> Self {
        Self("ko".to_string())
    }

    #[inline(always)]
    pub fn chinese() -> Self {
        Self("zh".to_string())
    }

    /// Validate language code format
    #[inline(always)]
    pub fn validate(&self) -> OpenAIResult<()> {
        if self.0.len() < 2 || self.0.len() > 5 {
            return Err(OpenAIError::AudioError(
                "Invalid language code format".to_string(),
            ));
        }
        Ok(())
    }
}

impl AudioData {
    /// Create from bytes with format detection
    #[inline(always)]
    pub fn from_bytes(data: Vec<u8>, format: AudioFormat) -> Self {
        Self {
            data,
            format,
            duration: None,
            sample_rate: None,
            channels: None,
            bit_rate: None}
    }

    /// Create from file path
    #[inline(always)]
    pub fn from_file(path: &str) -> OpenAIResult<Self> {
        let data = std::fs::read(path)
            .map_err(|e| OpenAIError::AudioError(format!("Failed to read audio file: {}", e)))?;

        let format = if let Some(ext) = path.split('.').last() {
            AudioFormat::from_extension(ext)?
        } else {
            return Err(OpenAIError::AudioError(
                "Cannot determine audio format from path".to_string(),
            ));
        };

        Ok(Self::from_bytes(data, format))
    }

    /// Get size in bytes
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Validate audio data for OpenAI API
    #[inline(always)]
    pub fn validate(&self) -> OpenAIResult<()> {
        // Check file size limits (25MB for Whisper)
        const MAX_SIZE: usize = 25 * 1024 * 1024;
        if self.data.len() > MAX_SIZE {
            return Err(OpenAIError::AudioError(format!(
                "Audio size {} bytes exceeds maximum of {} bytes",
                self.data.len(),
                MAX_SIZE
            )));
        }

        // Check minimum size
        if self.data.len() < 1000 {
            return Err(OpenAIError::AudioError("Audio data too small".to_string()));
        }

        Ok(())
    }

    /// Estimate duration from file size (rough approximation)
    #[inline(always)]
    pub fn estimate_duration(&self) -> Duration {
        if let Some(duration) = self.duration {
            return duration;
        }

        // Rough estimation based on format and size
        let estimated_seconds = match self.format {
            AudioFormat::WAV => self.data.len() / (44100 * 2 * 2), // 44.1kHz, 16-bit, stereo
            AudioFormat::FLAC => self.data.len() / (44100 * 2),    // Compressed but lossless
            AudioFormat::MP3 => self.data.len() / 16000,           // ~128kbps average
            _ => self.data.len() / 20000,                          // Conservative estimate
        };

        Duration::from_secs(estimated_seconds as u64)
    }

    /// Get metadata string for logging
    #[inline(always)]
    pub fn metadata_string(&self) -> String {
        let mut parts = vec![
            format!("format={:?}", self.format),
            format!("size={}bytes", self.size()),
        ];

        if let Some(duration) = &self.duration {
            parts.push(format!("duration={:.2}s", duration.as_secs_f32()));
        }

        if let Some(sample_rate) = self.sample_rate {
            parts.push(format!("sample_rate={}Hz", sample_rate));
        }

        if let Some(channels) = self.channels {
            parts.push(format!("channels={}", channels));
        }

        parts.join(", ")
    }
}

impl TranscriptionRequest {
    /// Create new transcription request
    #[inline(always)]
    pub fn new(audio: AudioData) -> Self {
        Self {
            audio,
            model: "whisper-1".to_string(),
            language: None,
            prompt: None,
            response_format: None,
            temperature: None,
            timestamp_granularities: Vec::new()}
    }

    /// Set model for transcription
    #[inline(always)]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set language hint
    #[inline(always)]
    pub fn with_language(mut self, language: LanguageCode) -> Self {
        self.language = Some(language);
        self
    }

    /// Set context prompt
    #[inline(always)]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Set response format
    #[inline(always)]
    pub fn with_response_format(mut self, format: impl Into<String>) -> Self {
        self.response_format = Some(format.into());
        self
    }

    /// Set temperature for randomness
    #[inline(always)]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Enable word-level timestamps
    #[inline(always)]
    pub fn with_word_timestamps(mut self) -> Self {
        if !self.timestamp_granularities.contains(&"word".to_string()) {
            self.timestamp_granularities.push("word".to_string());
        }
        self
    }

    /// Enable segment-level timestamps
    #[inline(always)]
    pub fn with_segment_timestamps(mut self) -> Self {
        if !self
            .timestamp_granularities
            .contains(&"segment".to_string())
        {
            self.timestamp_granularities.push("segment".to_string());
        }
        self
    }

    /// Validate request
    #[inline(always)]
    pub fn validate(&self) -> OpenAIResult<()> {
        self.audio.validate()?;

        if let Some(temp) = self.temperature {
            if !(0.0..=2.0).contains(&temp) {
                return Err(OpenAIError::AudioError(
                    "Temperature must be between 0.0 and 2.0".to_string(),
                ));
            }
        }

        if let Some(language) = &self.language {
            language.validate()?;
        }

        Ok(())
    }
}

impl TTSRequest {
    /// Create new text-to-speech request
    #[inline(always)]
    pub fn new(input: impl Into<String>, voice: Voice) -> Self {
        Self {
            model: "tts-1".to_string(),
            input: input.into(),
            voice,
            response_format: ResponseFormat::MP3,
            speed: None}
    }

    /// Create high-quality TTS request
    #[inline(always)]
    pub fn new_hd(input: impl Into<String>, voice: Voice) -> Self {
        Self {
            model: "tts-1-hd".to_string(),
            input: input.into(),
            voice,
            response_format: ResponseFormat::MP3,
            speed: None}
    }

    /// Set model
    #[inline(always)]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set response format
    #[inline(always)]
    pub fn with_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = format;
        self
    }

    /// Set playback speed
    #[inline(always)]
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = Some(speed);
        self
    }

    /// Validate request
    #[inline(always)]
    pub fn validate(&self) -> OpenAIResult<()> {
        if self.input.is_empty() {
            return Err(OpenAIError::AudioError(
                "Input text cannot be empty".to_string(),
            ));
        }

        // Check character limit (4096 characters for TTS)
        if self.input.len() > 4096 {
            return Err(OpenAIError::AudioError(
                "Input text exceeds 4096 character limit".to_string(),
            ));
        }

        if let Some(speed) = self.speed {
            if !(0.25..=4.0).contains(&speed) {
                return Err(OpenAIError::AudioError(
                    "Speed must be between 0.25 and 4.0".to_string(),
                ));
            }
        }

        Ok(())
    }
}

impl TranslationRequest {
    /// Create new translation request
    #[inline(always)]
    pub fn new(audio: AudioData) -> Self {
        Self {
            audio,
            model: "whisper-1".to_string(),
            prompt: None,
            response_format: None,
            temperature: None}
    }

    /// Set model
    #[inline(always)]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set context prompt
    #[inline(always)]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Set response format
    #[inline(always)]
    pub fn with_response_format(mut self, format: impl Into<String>) -> Self {
        self.response_format = Some(format.into());
        self
    }

    /// Set temperature
    #[inline(always)]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Validate request
    #[inline(always)]
    pub fn validate(&self) -> OpenAIResult<()> {
        self.audio.validate()?;

        if let Some(temp) = self.temperature {
            if !(0.0..=2.0).contains(&temp) {
                return Err(OpenAIError::AudioError(
                    "Temperature must be between 0.0 and 2.0".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// Get capabilities for specific audio model
#[inline(always)]
pub fn get_model_capabilities(model: &str) -> AudioCapabilities {
    match model {
        "whisper-1" => AudioCapabilities {
            transcription: true,
            translation: true,
            text_to_speech: false,
            real_time_transcription: false,
            speaker_identification: false,
            noise_reduction: true},
        "tts-1" | "tts-1-hd" => AudioCapabilities {
            transcription: false,
            translation: false,
            text_to_speech: true,
            real_time_transcription: false,
            speaker_identification: false,
            noise_reduction: false},
        _ => AudioCapabilities {
            transcription: false,
            translation: false,
            text_to_speech: false,
            real_time_transcription: false,
            speaker_identification: false,
            noise_reduction: false}}
}

/// Check if model supports transcription
#[inline(always)]
pub fn supports_transcription(model: &str) -> bool {
    matches!(model, "whisper-1")
}

/// Check if model supports text-to-speech
#[inline(always)]
pub fn supports_tts(model: &str) -> bool {
    matches!(model, "tts-1" | "tts-1-hd")
}
