// ============================================================================
// File: client/embeddings.rs
// ----------------------------------------------------------------------------
// Embedding-capable provider abstraction.
//
// • Pure async, zero-blocking on the hot-path.
// • Mirrors `client/completion.rs` for API symmetry.
// • All outward-facing methods return `AsyncTask` (blueprint rule #1).
// ============================================================================

#![allow(clippy::type_complexity)]

use std::sync::Arc;

use crate::{
    client::ProviderClient,
    embedding::{
        Embed, Embedding, EmbeddingError,
        builder::EmbeddingsBuilder,
        embedding::{EmbeddingModel, EmbeddingModelDyn}},
    runtime::AsyncTask};

// -----------------------------------------------------------------------------
// Provider trait implemented by concrete SDK wrappers
// -----------------------------------------------------------------------------
pub trait EmbeddingsClient: ProviderClient + Clone + Send + Sync + 'static {
    type Model: EmbeddingModel;

    /// Instantiate a model by name.
    /// If the dimensionality is unknown use
    /// [`embedding_model_with_ndims`](Self::embedding_model_with_ndims).
    fn embedding_model(&self, name: &str) -> Self::Model;

    fn embedding_model_with_ndims(&self, name: &str, ndims: usize) -> Self::Model;

    // -------- convenience fluent builders -----------------------------------

    #[inline(always)]
    fn embeddings<D: Embed>(&self, name: &str) -> EmbeddingsBuilder<Self::Model, D> {
        EmbeddingsBuilder::new(self.embedding_model(name))
    }

    #[inline(always)]
    fn embeddings_with_ndims<D: Embed>(
        &self,
        name: &str,
        ndims: usize,
    ) -> EmbeddingsBuilder<Self::Model, D> {
        EmbeddingsBuilder::new(self.embedding_model_with_ndims(name, ndims))
    }
}

// -----------------------------------------------------------------------------
// Dynamic-dispatch façade — enables runtime provider selection
// -----------------------------------------------------------------------------
pub trait EmbeddingsClientDyn: ProviderClient + Send + Sync {
    fn embedding_model<'a>(&self, name: &str) -> Box<dyn EmbeddingModelDyn + 'a>;
    fn embedding_model_with_ndims<'a>(
        &self,
        name: &str,
        ndims: usize,
    ) -> Box<dyn EmbeddingModelDyn + 'a>;
}

impl<C, M> EmbeddingsClientDyn for C
where
    C: EmbeddingsClient<Model = M>,
    M: EmbeddingModel + 'static,
{
    #[inline]
    fn embedding_model<'a>(&self, name: &str) -> Box<dyn EmbeddingModelDyn + 'a> {
        Box::new(self.embedding_model(name))
    }

    #[inline]
    fn embedding_model_with_ndims<'a>(
        &self,
        name: &str,
        ndims: usize,
    ) -> Box<dyn EmbeddingModelDyn + 'a> {
        Box::new(self.embedding_model_with_ndims(name, ndims))
    }
}

// -----------------------------------------------------------------------------
// Blanket AsEmbeddings — opt-in via trait bounds only
// -----------------------------------------------------------------------------

/// Trait for converting types to embeddings client
pub trait AsEmbeddings {
    /// Convert to embeddings client
    fn as_embeddings(&self) -> Option<Box<dyn EmbeddingsClientDyn>>;
}

impl<T> AsEmbeddings for T
where
    T: EmbeddingsClient + Clone + Send + Sync + 'static,
{
    #[inline(always)]
    fn as_embeddings(&self) -> Option<Box<dyn EmbeddingsClientDyn>> {
        Some(Box::new(self.clone()))
    }
}

// -----------------------------------------------------------------------------
// Dyn-erased handle (mirrors CompletionModelHandle)
// -----------------------------------------------------------------------------
#[derive(Clone)]
pub struct EmbeddingModelHandle<'a> {
    inner: Arc<dyn EmbeddingModelDyn + 'a>}

impl EmbeddingModel for EmbeddingModelHandle<'_> {
    const MAX_DOCUMENTS: usize = usize::MAX;

    #[inline(always)]
    fn ndims(&self) -> usize {
        self.inner.ndims()
    }

    #[inline]
    fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send + 'static,
    ) -> AsyncTask<Result<Vec<Embedding>, EmbeddingError>> {
        self.inner.embed_texts(texts)
    }
}
