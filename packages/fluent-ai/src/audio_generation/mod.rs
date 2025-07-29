// ============================================================================
// File: src/audio_generation.rs
// ----------------------------------------------------------------------------
// Text-to-Audio core primitives – RIG “zero-alloc” edition
//
// • Public APIs surface only `AsyncTask` / `AsyncStream` handles
// • No blocking, no heap traffic on the happy path
// • Typestate builder guarantees all mandatory fields are set
// ============================================================================

#![allow(clippy::type_complexity)]

use std::sync::Arc;

// Removed futures_util::FutureExt - using AsyncStream patterns instead
use serde_json::Value;
use thiserror::Error;

use crate::runtime::{AsyncStream, AsyncTask};

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------
#[derive(Debug, Error)]
pub enum AudioGenerationError {
    #[error("HTTP error: {0}")]
    Http(#[from] fluent_ai_http3::HttpError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("request build error: {0}")]
    Build(String),
    #[error("provider error: {0}")]
    Provider(String)}

// ---------------------------------------------------------------------------
// Stream item + final response
// ---------------------------------------------------------------------------
pub type AudioGenerationChunk = Vec<u8>;

#[derive(Debug)]
pub struct AudioGenerationResponse<T> {
    pub audio: Vec<u8>,
    pub provider_response: T}

// ---------------------------------------------------------------------------
// Streaming flow returned by every model
// ---------------------------------------------------------------------------
pub const STREAM_CAP: usize = 256;

pub struct AudioGenerationFlow<R> {
    pub chunks: AsyncStream<AudioGenerationChunk, STREAM_CAP>,
    pub done: AsyncTask<Result<AudioGenerationResponse<R>, AudioGenerationError>>}

// ---------------------------------------------------------------------------
// Request payload
// ---------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct AudioGenerationRequest {
    pub text: String,
    pub voice: String,
    pub speed: f32,
    pub additional_params: Option<Value>}

// ---------------------------------------------------------------------------
// Typestate builder  (ensures `text` *and* `voice` are set)
// ---------------------------------------------------------------------------
pub struct RequestBuilder<M, const HAS_TEXT: bool, const HAS_VOICE: bool> {
    model: M,
    text: Option<String>,
    voice: Option<String>,
    speed: f32,
    additional_params: Option<Value>}

impl<M: Clone> RequestBuilder<M, false, false> {
    #[inline(always)]
    pub fn new(model: M) -> Self {
        Self {
            model,
            text: None,
            voice: None,
            speed: 1.0,
            additional_params: None}
    }
}

impl<M, const T: bool, const V: bool> RequestBuilder<M, T, V> {
    #[inline(always)]
    pub fn text(self, txt: impl Into<String>) -> RequestBuilder<M, true, V> {
        RequestBuilder {
            text: Some(txt.into()),
            ..self
        }
    }

    #[inline(always)]
    pub fn voice(self, voice: impl Into<String>) -> RequestBuilder<M, T, true> {
        RequestBuilder {
            voice: Some(voice.into()),
            ..self
        }
    }

    #[inline(always)]
    pub fn speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    #[inline(always)]
    pub fn additional_params(mut self, params: Value) -> Self {
        self.additional_params = Some(params);
        self
    }
}

// Builder becomes “executable” only when both mandatory fields are set
impl<M: AudioGenerationModel> RequestBuilder<M, true, true> {
    #[inline(always)]
    fn build(self) -> AudioGenerationRequest {
        AudioGenerationRequest {
            text: self.text.unwrap_or_default(),
            voice: self.voice.unwrap_or_default(),
            speed: self.speed,
            additional_params: self.additional_params}
    }

    /// One-shot API – resolves to the full audio buffer.
    #[inline(always)]
    pub fn send(
        self,
    ) -> AsyncTask<Result<AudioGenerationResponse<M::Response>, AudioGenerationError>>
    where
        M::Response: Send + 'static,
    {
        self.model.audio_generation(self.build()).done
    }

    /// Streaming API – immediately yields a chunk stream + terminal task.
    #[inline(always)]
    pub fn stream(self) -> AudioGenerationFlow<M::Response>
    where
        M::Response: Send + 'static,
    {
        self.model.audio_generation(self.build())
    }
}

// ---------------------------------------------------------------------------
// Provider-side model trait
// ---------------------------------------------------------------------------
pub trait AudioGenerationModel: Clone + Send + Sync + 'static {
    type Response: Send + Sync + 'static;

    /// Must spawn work and return immediately, never blocking.
    fn audio_generation(
        &self,
        request: AudioGenerationRequest,
    ) -> AudioGenerationFlow<Self::Response>;

    /// Entry-point for end-users.
    #[inline(always)]
    fn audio_generation_request(&self) -> RequestBuilder<Self, false, false>
    where
        Self: Sized,
    {
        RequestBuilder::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Dyn-erase concrete models so heterogeneous providers share one API
// ---------------------------------------------------------------------------
pub trait AudioGenerationModelDyn: Send + Sync {
    fn audio_generation(&self, request: AudioGenerationRequest) -> AudioGenerationFlow<()>;

    fn audio_generation_request(&self) -> RequestBuilder<AudioGenerationModelHandle, false, false>;
}

impl<T> AudioGenerationModelDyn for T
where
    T: AudioGenerationModel + 'static,
{
    fn audio_generation(&self, request: AudioGenerationRequest) -> AudioGenerationFlow<()> {
        let flow = AudioGenerationModel::audio_generation(self, request);
        AudioGenerationFlow {
            chunks: flow.chunks,
            done: flow.done.map(|r| {
                r.map(|resp| AudioGenerationResponse {
                    audio: resp.audio,
                    provider_response: ()})
            })}
    }

    fn audio_generation_request(&self) -> RequestBuilder<AudioGenerationModelHandle, false, false> {
        RequestBuilder::new(AudioGenerationModelHandle {
            inner: Arc::new(self.clone())})
    }
}

/// Cheap handle that hides the concrete model type.
#[derive(Clone)]
pub struct AudioGenerationModelHandle {
    inner: Arc<dyn AudioGenerationModelDyn + 'static>}

impl AudioGenerationModel for AudioGenerationModelHandle {
    type Response = ();

    #[inline(always)]
    fn audio_generation(
        &self,
        request: AudioGenerationRequest,
    ) -> AudioGenerationFlow<Self::Response> {
        self.inner.audio_generation(request)
    }
}
