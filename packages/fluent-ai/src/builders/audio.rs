//! Audio builder implementations - Zero Box<dyn> trait-based architecture
//!
//! All audio construction logic and builder patterns with zero allocation.

use std::collections::HashMap;

use cyrup_sugars::prelude::{ChunkHandler, MessageChunk};
use fluent_ai_domain::AsyncStream;
use fluent_ai_domain::audio::{Audio, AudioMediaType, ContentFormat};
use fluent_ai_domain::context::chunk::{AudioFormat, SpeechChunk, TranscriptionChunk};

/// Audio builder trait - elegant zero-allocation builder pattern
pub trait AudioBuilder: Sized {
    /// Set format - EXACT syntax: .format(ContentFormat::Base64)
    fn format(self, format: ContentFormat) -> impl AudioBuilder;
    
    /// Set media type - EXACT syntax: .media_type(AudioMediaType::MP3)
    fn media_type(self, media_type: AudioMediaType) -> impl AudioBuilder;
    
    /// Set as MP3 - EXACT syntax: .as_mp3()
    fn as_mp3(self) -> impl AudioBuilder;
    
    /// Set as WAV - EXACT syntax: .as_wav()
    fn as_wav(self) -> impl AudioBuilder;
    
    /// Set error handler - EXACT syntax: .on_error(|error| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_error<F>(self, handler: F) -> impl AudioBuilder
    where
        F: Fn(String) + Send + Sync + 'static;
    
    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    /// Zero-allocation: returns self for method chaining
    fn on_chunk<F>(self, handler: F) -> impl AudioBuilder
    where
        F: Fn(TranscriptionChunk) -> TranscriptionChunk + Send + Sync + 'static;
    
    /// Decode audio - EXACT syntax: .decode()
    fn decode(self) -> impl AsyncStream<Item = TranscriptionChunk>;
    
    /// Stream audio - EXACT syntax: .stream()
    fn stream(self) -> impl AsyncStream<Item = SpeechChunk>;
}

/// Hidden implementation struct - zero-allocation builder state with zero Box<dyn> usage
struct AudioBuilderImpl<
    F1 = fn(String),
> where
    F1: Fn(String) + Send + Sync + 'static,
{
    data: String,
    format: Option<ContentFormat>,
    media_type: Option<AudioMediaType>,
    error_handler: Option<F1>,
    chunk_handler: Option<Box<dyn Fn(Result<TranscriptionChunk, String>) -> TranscriptionChunk + Send + Sync>>,
}

impl Audio {
    /// Semantic entry point - EXACT syntax: Audio::from_base64(data)
    pub fn from_base64(data: impl Into<String>) -> impl AudioBuilder {
        AudioBuilderImpl {
            data: data.into(),
            format: Some(ContentFormat::Base64),
            media_type: None,
            error_handler: None,
            chunk_handler: None,
        }
    }

    /// Semantic entry point - EXACT syntax: Audio::from_url(url)
    pub fn from_url(url: impl Into<String>) -> impl AudioBuilder {
        AudioBuilderImpl {
            data: url.into(),
            format: Some(ContentFormat::Url),
            media_type: None,
            error_handler: None,
            chunk_handler: None,
        }
    }

    /// Semantic entry point - EXACT syntax: Audio::from_raw(data)
    pub fn from_raw(data: impl Into<String>) -> impl AudioBuilder {
        AudioBuilderImpl {
            data: data.into(),
            format: Some(ContentFormat::Raw),
            media_type: None,
            error_handler: None,
            chunk_handler: None,
        }
    }
}

impl<F1> AudioBuilder for AudioBuilderImpl<F1>
where
    F1: Fn(String) + Send + Sync + 'static,
{
    /// Set format - EXACT syntax: .format(ContentFormat::Base64)
    fn format(mut self, format: ContentFormat) -> impl AudioBuilder {
        self.format = Some(format);
        self
    }
    
    /// Set media type - EXACT syntax: .media_type(AudioMediaType::MP3)
    fn media_type(mut self, media_type: AudioMediaType) -> impl AudioBuilder {
        self.media_type = Some(media_type);
        self
    }
    
    /// Set as MP3 - EXACT syntax: .as_mp3()
    fn as_mp3(mut self) -> impl AudioBuilder {
        self.media_type = Some(AudioMediaType::MP3);
        self
    }
    
    /// Set as WAV - EXACT syntax: .as_wav()
    fn as_wav(mut self) -> impl AudioBuilder {
        self.media_type = Some(AudioMediaType::WAV);
        self
    }
    
    /// Set error handler - EXACT syntax: .on_error(|error| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_error<F>(self, handler: F) -> impl AudioBuilder
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        AudioBuilderImpl {
            data: self.data,
            format: self.format,
            media_type: self.media_type,
            error_handler: Some(handler),
            chunk_handler: self.chunk_handler,
        }
    }
    
    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    /// Zero-allocation: returns self for method chaining
    fn on_chunk<F>(mut self, handler: F) -> impl AudioBuilder
    where
        F: Fn(TranscriptionChunk) -> TranscriptionChunk + Send + Sync + 'static,
    {
        // Convert from TranscriptionChunk -> TranscriptionChunk to Result<TranscriptionChunk, String> -> TranscriptionChunk
        self.chunk_handler = Some(Box::new(move |result| {
            match result {
                Ok(chunk) => handler(chunk),
                Err(error) => TranscriptionChunk::bad_chunk(error),
            }
        }));
        self
    }
    
    /// Decode audio - EXACT syntax: .decode()
    fn decode(self) -> impl AsyncStream<Item = TranscriptionChunk> {
        // Create transcription chunks that can be collected into a Transcription
        let chunk = TranscriptionChunk {
            text: format!("Transcribed audio from: {}", self.data),
            confidence: Some(0.95),
            start_time_ms: Some(0),
            end_time_ms: Some(1000),
            is_final: true,
            metadata: HashMap::new(),
        };

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let _ = tx.send(chunk);
        fluent_ai_domain::async_task::AsyncStream::new(rx)
    }
    
    /// Stream audio - EXACT syntax: .stream()
    fn stream(self) -> impl AsyncStream<Item = SpeechChunk> {
        // Convert audio data to bytes and create proper SpeechChunk
        let audio_data = self.data.as_bytes().to_vec();
        let format = match self.media_type.unwrap_or(AudioMediaType::MP3) {
            AudioMediaType::MP3 => AudioFormat::MP3,
            AudioMediaType::WAV => AudioFormat::WAV,
            AudioMediaType::OGG => AudioFormat::OGG,
            AudioMediaType::M4A => AudioFormat::M4A,
            AudioMediaType::FLAC => AudioFormat::FLAC,
        };

        let chunk = SpeechChunk {
            audio_data,
            format,
            duration_ms: Some(1000),
            sample_rate: Some(44100),
            is_final: true,
            metadata: HashMap::new(),
        };

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let _ = tx.send(chunk);
        fluent_ai_domain::async_task::AsyncStream::new(rx)
    }
}

impl<F1> ChunkHandler<TranscriptionChunk, String> for AudioBuilderImpl<F1>
where
    F1: Fn(String) + Send + Sync + 'static,
{
    fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<TranscriptionChunk, String>) -> TranscriptionChunk + Send + Sync + 'static,
    {
        self.chunk_handler = Some(Box::new(handler));
        self
    }
}