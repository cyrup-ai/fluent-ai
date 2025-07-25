// ============================================================================
// File: src/client/audio_generation.rs
// ----------------------------------------------------------------------------
// Zero-alloc text-to-audio abstraction wired into the new runtime primitives.
// ============================================================================

#![cfg(feature = "audio")]

use crate::{
    audio_generation::{AudioGenerationChunk, AudioGenerationError, AudioGenerationRequest},
    client::{AsAudioGeneration, ProviderClient},
    runtime::{AsyncStream, AsyncTask}};

// ---------------------------------------------------------------------------
// Stream ring capacity – amend at build-time if benchmarking dictates.
// ---------------------------------------------------------------------------
const STREAM_CAPACITY: usize = 256;

// ---------------------------------------------------------------------------
// Flow: every request yields (chunk-stream, terminal-task).
// ---------------------------------------------------------------------------
pub struct AudioGenerationFlow {
    pub chunks: AsyncStream<AudioGenerationChunk, STREAM_CAPACITY>,
    pub done: AsyncTask<Result<(), AudioGenerationError>>}

// ---------------------------------------------------------------------------
// Provider-side model trait (implemented by concrete SDK wrappers).
// ---------------------------------------------------------------------------
pub trait AudioGenerationModel: Send + Sync + Clone + 'static {
    fn audio_generation(&self, req: AudioGenerationRequest) -> AudioGenerationFlow;
}

// ---------------------------------------------------------------------------
// High-level provider trait.
// ---------------------------------------------------------------------------
pub trait AudioGenerationClient: ProviderClient + Clone + Send + Sync + 'static {
    type Model: AudioGenerationModel;

    fn audio_generation_model(&self, name: &str) -> Self::Model;
}

// ---------------------------------------------------------------------------
// Dynamic façade – mirror pattern used by Completion / Embeddings.
// ---------------------------------------------------------------------------
pub trait AudioGenerationClientDyn: ProviderClient + Send + Sync {
    fn audio_generation_model<'a>(&self, name: &str) -> Box<dyn AudioGenerationModelDyn + 'a>;
}

pub trait AudioGenerationModelDyn: Send + Sync {
    fn audio_generation(&self, req: AudioGenerationRequest) -> AudioGenerationFlow;
}

// blanket conversions
impl<T> AudioGenerationModelDyn for T
where
    T: AudioGenerationModel + 'static,
{
    #[inline(always)]
    fn audio_generation(&self, req: AudioGenerationRequest) -> AudioGenerationFlow {
        AudioGenerationModel::audio_generation(self, req)
    }
}

impl<C> AudioGenerationClientDyn for C
where
    C: AudioGenerationClient,
    C::Model: AudioGenerationModel + 'static,
{
    #[inline(always)]
    fn audio_generation_model<'a>(&self, name: &str) -> Box<dyn AudioGenerationModelDyn + 'a> {
        Box::new(self.audio_generation_model(name))
    }
}

// ---------------------------------------------------------------------------
// Blanket AsAudioGeneration – any compliant client gets it for free
// ---------------------------------------------------------------------------

/// Trait for converting types to audio generation client
pub trait AsAudioGeneration {
    /// Convert to audio generation client
    fn as_audio_generation(&self) -> Option<Box<dyn AudioGenerationClientDyn>>;
}

impl<T> AsAudioGeneration for T
where
    T: AudioGenerationClient + Clone + Send + Sync + 'static,
{
    #[inline(always)]
    fn as_audio_generation(&self) -> Option<Box<dyn AudioGenerationClientDyn>> {
        Some(Box::new(self.clone()))
    }
}

// Handy dyn-erased handle
#[derive(Clone)]
pub struct AudioGenerationModelHandle<'a> {
    inner: &'a dyn AudioGenerationModelDyn}

impl<'a> AudioGenerationModel for AudioGenerationModelHandle<'a> {
    #[inline(always)]
    fn audio_generation(&self, req: AudioGenerationRequest) -> AudioGenerationFlow {
        self.inner.audio_generation(req)
    }
}
