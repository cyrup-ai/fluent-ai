// ============================================================================
// File: client/transcription.rs
// ----------------------------------------------------------------------------
// Transcription-capable provider abstraction.
//
// • Zero-blocking; every operation yields an `AsyncTask`.
// • Mirrors completion / embeddings wrappers for a uniform public surface.
// ============================================================================

#![allow(clippy::type_complexity)]

use std::sync::Arc;

use crate::{
    client::ProviderClient,
    runtime::AsyncTask,
    transcription::{
        TranscriptionError, TranscriptionModel, TranscriptionModelDyn, TranscriptionRequest,
        TranscriptionResponse}};

// -----------------------------------------------------------------------------
// Provider-side trait to be implemented by concrete SDK wrappers
// -----------------------------------------------------------------------------
pub trait TranscriptionClient: ProviderClient + Clone + Send + Sync + 'static {
    type Model: TranscriptionModel;

    /// Instantiate a transcription model by name.
    fn transcription_model(&self, name: &str) -> Self::Model;
}

// -----------------------------------------------------------------------------
// Dynamic-dispatch façade – enables runtime provider selection
// -----------------------------------------------------------------------------
pub trait TranscriptionClientDyn: ProviderClient + Send + Sync {
    fn transcription_model<'a>(&self, name: &str) -> Box<dyn TranscriptionModelDyn + 'a>;
}

impl<C, M> TranscriptionClientDyn for C
where
    C: TranscriptionClient<Model = M>,
    M: TranscriptionModel + 'static,
{
    #[inline]
    fn transcription_model<'a>(&self, name: &str) -> Box<dyn TranscriptionModelDyn + 'a> {
        Box::new(self.transcription_model(name))
    }
}

// -----------------------------------------------------------------------------
// Blanket AsTranscription – any compliant client gets it for free
// -----------------------------------------------------------------------------

/// Trait for converting types to transcription client
pub trait AsTranscription {
    /// Convert to transcription client
    fn as_transcription(&self) -> Option<Box<dyn TranscriptionClientDyn>>;
}

impl<T> AsTranscription for T
where
    T: TranscriptionClient + Clone + Send + Sync + 'static,
{
    #[inline(always)]
    fn as_transcription(&self) -> Option<Box<dyn TranscriptionClientDyn>> {
        Some(Box::new(self.clone()))
    }
}

// -----------------------------------------------------------------------------
// Dyn-erased handle so builders can stay generic
// -----------------------------------------------------------------------------
#[derive(Clone)]
pub struct TranscriptionModelHandle<'a> {
    inner: Arc<dyn TranscriptionModelDyn + 'a>}

impl TranscriptionModel for TranscriptionModelHandle<'_> {
    type Response = ();

    #[inline]
    fn transcription(
        &self,
        req: TranscriptionRequest,
    ) -> AsyncTask<Result<TranscriptionResponse<Self::Response>, TranscriptionError>> {
        self.inner.transcription(req)
    }
}
