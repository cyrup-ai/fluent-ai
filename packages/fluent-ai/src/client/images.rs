// ============================================================================
// File: client/images.rs
// ----------------------------------------------------------------------------
// Text-to-Image provider abstraction (zero-alloc, lock-free hot path).
//
// • Streaming API ⇒ every request yields a `Receiver<Result<Chunk, Err>>`.
// • No Tokio / no blocking – pure cross-beam MPMC queue.
// • Dyn-erasure identical to completion / embeddings.
// • No heap churn once the receiver is created.
// ============================================================================

#![cfg(feature = "image")]

use std::sync::Arc;

use crossbeam_channel::{Receiver, unbounded};

use crate::{
    client::{AsImageGeneration, ProviderClient},
    image_generation::{ImageGenerationChunk, ImageGenerationError, ImageGenerationRequest}};

// -----------------------------------------------------------------------------
// Stream wrapper – ergonomic alias
// -----------------------------------------------------------------------------
pub type ImageStream = Receiver<Result<ImageGenerationChunk, ImageGenerationError>>;

// -----------------------------------------------------------------------------
// Provider-specific model abstraction
// -----------------------------------------------------------------------------
pub trait ImageGenerationModel: Send + Sync + Clone + 'static {
    /// Start generation – *returns immediately* with a channel.
    fn image_generation(&self, req: ImageGenerationRequest) -> ImageStream;
}

// -----------------------------------------------------------------------------
// High-level provider trait implemented by concrete SDK clients
// -----------------------------------------------------------------------------
pub trait ImageGenerationClient: ProviderClient + Clone + Send + Sync + 'static {
    type Model: ImageGenerationModel;

    /// Instantiate a model by name.
    fn image_generation_model(&self, name: &str) -> Self::Model;
}

// -----------------------------------------------------------------------------
// Dynamic-dispatch façade
// -----------------------------------------------------------------------------
pub trait ImageGenerationClientDyn: ProviderClient + Send + Sync {
    fn image_generation_model<'a>(&self, name: &str) -> Box<dyn ImageGenerationModelDyn + 'a>;
}

// Helper trait object so the dyn client can hide concrete model types.
pub trait ImageGenerationModelDyn: Send + Sync {
    fn image_generation(&self, req: ImageGenerationRequest) -> ImageStream;
}

// -------- impls ----------------------------------------------------------------

impl<T, M> ImageGenerationModelDyn for T
where
    T: ImageGenerationModel + 'static,
{
    #[inline]
    fn image_generation(&self, req: ImageGenerationRequest) -> ImageStream {
        self.image_generation(req)
    }
}

impl<C, M> ImageGenerationClientDyn for C
where
    C: ImageGenerationClient<Model = M>,
    M: ImageGenerationModel + 'static,
{
    #[inline]
    fn image_generation_model<'a>(&self, name: &str) -> Box<dyn ImageGenerationModelDyn + 'a> {
        Box::new(self.image_generation_model(name))
    }
}

// -----------------------------------------------------------------------------
// Blanket conversion – any compliant client gains `as_image_generation()`.
// -----------------------------------------------------------------------------

/// Trait for converting types to image generation client
pub trait AsImageGeneration {
    /// Convert to image generation client
    fn as_image_generation(&self) -> Option<Box<dyn ImageGenerationClientDyn>>;
}

impl<T> AsImageGeneration for T
where
    T: ImageGenerationClient + Clone + Send + Sync + 'static,
{
    #[inline(always)]
    fn as_image_generation(&self) -> Option<Box<dyn ImageGenerationClientDyn>> {
        Some(Box::new(self.clone()))
    }
}

// -----------------------------------------------------------------------------
// Dyn-erased handle – for use inside generic builders
// -----------------------------------------------------------------------------
#[derive(Clone)]
pub struct ImageGenerationModelHandle<'a> {
    inner: Arc<dyn ImageGenerationModelDyn + 'a>}

impl ImageGenerationModel for ImageGenerationModelHandle<'_> {
    #[inline]
    fn image_generation(&self, req: ImageGenerationRequest) -> ImageStream {
        self.inner.image_generation(req)
    }
}

// -----------------------------------------------------------------------------
// End of file
// -----------------------------------------------------------------------------
