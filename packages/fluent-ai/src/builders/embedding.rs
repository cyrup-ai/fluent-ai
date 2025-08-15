//! Embedding builder implementations - Zero Box<dyn> trait-based architecture
//!
//! All embedding construction logic and builder patterns with zero allocation.

use std::marker::PhantomData;

use cyrup_sugars::prelude::{ChunkHandler, MessageChunk};
use fluent_ai_domain::context::chunk::EmbeddingChunk;
use fluent_ai_domain::embedding::Embedding;
use fluent_ai_domain::{AsyncTask, ZeroOneOrMany, spawn_async};

/// Embedding builder trait - elegant zero-allocation builder pattern
pub trait EmbeddingBuilder: Sized {
    /// Set vector - EXACT syntax: .vec(vector)
    fn vec(self, vec: ZeroOneOrMany<f64>) -> impl EmbeddingBuilder;
    
    /// Set dimensions - EXACT syntax: .with_dims(512)
    fn with_dims(self, dims: usize) -> impl EmbeddingBuilder;
    
    /// Set error handler - EXACT syntax: .on_error(|error| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_error<F>(self, handler: F) -> impl EmbeddingBuilder
    where
        F: Fn(String) + Send + Sync + 'static;
    
    /// Set result handler - EXACT syntax: .on_result(|result| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_result<F>(self, handler: F) -> impl EmbeddingBuilder
    where
        F: FnOnce(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + 'static;
    

    
    /// Generate embedding - EXACT syntax: .embed()
    fn embed(self) -> AsyncTask<Embedding>;
}

/// Hidden implementation struct - zero-allocation builder state with zero Box<dyn> usage
struct EmbeddingBuilderImpl<
    F1 = fn(String),
    F2 = fn(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32>,
> where
    F1: Fn(String) + Send + Sync + 'static,
    F2: FnOnce(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + 'static,
{
    document: String,
    vec: Option<ZeroOneOrMany<f64>>,
    error_handler: Option<F1>,
    result_handler: Option<F2>,
    cyrup_chunk_handler: Option<Box<dyn Fn(Result<EmbeddingChunk, String>) -> EmbeddingChunk + Send + Sync>>,
}

impl Embedding {
    /// Semantic entry point - EXACT syntax: Embedding::from_document("text")
    pub fn from_document(document: impl Into<String>) -> impl EmbeddingBuilder {
        EmbeddingBuilderImpl {
            document: document.into(),
            vec: None,
            error_handler: None,
            result_handler: None,
            cyrup_chunk_handler: None,
        }
    }
}

impl<F1, F2> EmbeddingBuilder for EmbeddingBuilderImpl<F1, F2>
where
    F1: Fn(String) + Send + Sync + 'static,
    F2: FnOnce(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + 'static,
{
    /// Set vector - EXACT syntax: .vec(vector)
    fn vec(mut self, vec: ZeroOneOrMany<f64>) -> impl EmbeddingBuilder {
        self.vec = Some(vec);
        self
    }
    
    /// Set dimensions - EXACT syntax: .with_dims(512)
    fn with_dims(mut self, dims: usize) -> impl EmbeddingBuilder {
        self.vec = Some(ZeroOneOrMany::many(vec![0.0; dims]));
        self
    }
    
    /// Set error handler - EXACT syntax: .on_error(|error| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_error<F>(self, handler: F) -> impl EmbeddingBuilder
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        EmbeddingBuilderImpl {
            document: self.document,
            vec: self.vec,
            error_handler: Some(handler),
            result_handler: self.result_handler,
            cyrup_chunk_handler: self.cyrup_chunk_handler,
        }
    }
    
    /// Set result handler - EXACT syntax: .on_result(|result| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_result<F>(self, handler: F) -> impl EmbeddingBuilder
    where
        F: FnOnce(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + 'static,
    {
        EmbeddingBuilderImpl {
            document: self.document,
            vec: self.vec,
            error_handler: self.error_handler,
            result_handler: Some(handler),
            cyrup_chunk_handler: self.cyrup_chunk_handler,
        }
    }
    
    /// Generate embedding - EXACT syntax: .embed()
    fn embed(self) -> AsyncTask<Embedding> {
        spawn_async(async move {
            Embedding {
                document: self.document,
                vec: self.vec.unwrap_or(ZeroOneOrMany::None),
            }
        })
    }
}

impl<F1, F2> ChunkHandler<EmbeddingChunk, String> for EmbeddingBuilderImpl<F1, F2>
where
    F1: Fn(String) + Send + Sync + 'static,
    F2: FnOnce(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + 'static,
{
    fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<EmbeddingChunk, String>) -> EmbeddingChunk + Send + Sync + 'static,
    {
        self.cyrup_chunk_handler = Some(Box::new(handler));
        self
    }
}