// ============================================================================
// File: src/embeddings/embedding.rs
// ----------------------------------------------------------------------------
// Embedding-model abstractions (text + image) plus error & data carriers.
// Designed for absolute zero overhead on the critical path.
// ============================================================================

use fluent_ai_http3::async_task::AsyncStream;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// 0. Error enumeration – single source of truth
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    /// Transport or timeout failure
    #[error("HttpError: {0}")]
    HttpError(#[from] fluent_ai_http3::HttpError),

    /// (De)serialisation failure
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    /// User-supplied document processing failure
    #[error("DocumentError: {0}")]
    DocumentError(Box<dyn std::error::Error + Send + Sync + 'static>),

    /// Provider response malformed or unexpected
    #[error("ResponseError: {0}")]
    ResponseError(String),

    /// Remote provider surfaced an explicit error
    #[error("ProviderError: {0}")]
    ProviderError(String),
}

// ---------------------------------------------------------------------------
// 1. Core model trait (text)
// ---------------------------------------------------------------------------

pub trait EmbeddingModel: Clone + Send + Sync + 'static {
    /// Max batch size accepted by the backend.
    const MAX_DOCUMENTS: usize;

    /// Dimensionality of each embedding vector.
    fn ndims(&self) -> usize;

    /// Batch embed – MUST be implemented by concrete providers.
    fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> AsyncStream<Result<Vec<Embedding>, EmbeddingError>>;

    /// Convenience: single-document embed.
    #[inline(always)]
    fn embed_text(&self, text: &str) -> AsyncStream<Result<Embedding, EmbeddingError>> {
        let (tx, stream) = AsyncStream::channel();
        let text_owned = text.to_string();
        let mut texts_stream = self.embed_texts(std::iter::once(text_owned));
        
        tokio::spawn(async move {
            let result = match texts_stream.next() {
                Some(Ok(embeddings)) => {
                    match embeddings.into_iter().next() {
                        Some(embedding) => Ok(embedding),
                        None => Err(EmbeddingError::Provider(
                            "No embeddings returned from provider".to_string(),
                        )),
                    }
                }
                Some(Err(e)) => Err(e),
                None => Err(EmbeddingError::Provider(
                    "No response from provider".to_string(),
                )),
            };
            let _ = tx.send(result);
        });
        
        stream
    }
}

// ---------------------------------------------------------------------------
// 2. Dyn-erased adaptor so heterogeneous models can co-exist
// ---------------------------------------------------------------------------

pub trait EmbeddingModelDyn: Send + Sync {
    fn max_documents(&self) -> usize;
    fn ndims(&self) -> usize;

    fn embed_text(&self, text: &str) -> AsyncStream<Result<Embedding, EmbeddingError>>;

    fn embed_texts(
        &self,
        texts: Vec<String>,
    ) -> AsyncStream<Result<Vec<Embedding>, EmbeddingError>>;
}

impl<T: EmbeddingModel> EmbeddingModelDyn for T {
    #[inline(always)]
    fn max_documents(&self) -> usize {
        T::MAX_DOCUMENTS
    }

    #[inline(always)]
    fn ndims(&self) -> usize {
        self.ndims()
    }

    #[inline(always)]
    fn embed_text(&self, text: &str) -> AsyncStream<Result<Embedding, EmbeddingError>> {
        EmbeddingModel::embed_text(self, text)
    }

    #[inline(always)]
    fn embed_texts(
        &self,
        texts: Vec<String>,
    ) -> AsyncStream<Result<Vec<Embedding>, EmbeddingError>> {
        self.embed_texts(texts)
    }
}

// ---------------------------------------------------------------------------
// 3. Image model trait – mirrors the text variant
// ---------------------------------------------------------------------------

pub trait ImageEmbeddingModel: Clone + Send + Sync + 'static {
    const MAX_DOCUMENTS: usize;

    fn ndims(&self) -> usize;

    fn embed_images(
        &self,
        images: impl IntoIterator<Item = Vec<u8>> + Send,
    ) -> AsyncStream<Result<Vec<Embedding>, EmbeddingError>>;

    #[inline(always)]
    fn embed_image(
        &self,
        bytes: &[u8],
    ) -> AsyncStream<Result<Embedding, EmbeddingError>> {
        let bytes_owned = bytes.to_owned();
        let images_stream = self.embed_images(std::iter::once(bytes_owned));
        
        let (tx, stream) = AsyncStream::channel();
        tokio::spawn(async move {
            let result = match images_stream.collect().into_iter().next() {
                Some(Ok(embeddings)) => match embeddings.into_iter().next() {
                    Some(embedding) => Ok(embedding),
                    None => Err(EmbeddingError::Provider(
                        "No embeddings returned from provider".to_string(),
                    )),
                },
                Some(Err(e)) => Err(e),
                None => Err(EmbeddingError::Provider(
                    "No embeddings returned from provider".to_string(),
                )),
            };
            let _ = tx.send(result);
        });
        stream
    }
}

// ---------------------------------------------------------------------------
// 4. Data carrier – one document + its vector
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Embedding {
    /// Original document (handy for debugging / tracing)
    pub document: String,
    /// Dense numeric vector
    pub vec: Vec<f64>,
}

impl PartialEq for Embedding {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.document == other.document
    }
}
impl Eq for Embedding {}

// ---------------------------------------------------------------------------
// 5. Tiny helper so we can inline `pipe` ergonomically
// ---------------------------------------------------------------------------

trait Pipe: Sized {
    #[inline(always)]
    fn pipe<U, F: FnOnce(Self) -> U>(self, f: F) -> U {
        f(self)
    }
}
impl<T> Pipe for T {}
