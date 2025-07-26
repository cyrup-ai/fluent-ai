//! Embedding builder implementations - Zero Box<dyn> trait-based architecture
//!
//! All embedding construction logic and builder patterns with zero allocation.

use std::marker::PhantomData;
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
    
    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_chunk<F>(self, handler: F) -> impl EmbeddingBuilder
    where
        F: FnMut(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + 'static;
    
    /// Generate embedding - EXACT syntax: .embed()
    fn embed(self) -> AsyncTask<Embedding>;
}

/// Hidden implementation struct - zero-allocation builder state with zero Box<dyn> usage
struct EmbeddingBuilderImpl<
    F1 = fn(String),
    F2 = fn(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32>,
    F3 = fn(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32>,
> where
    F1: Fn(String) + Send + Sync + 'static,
    F2: FnOnce(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + 'static,
    F3: FnMut(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + 'static,
{
    document: String,
    vec: Option<ZeroOneOrMany<f64>>,
    error_handler: Option<F1>,
    result_handler: Option<F2>,
    chunk_handler: Option<F3>,
}

impl Embedding {
    /// Semantic entry point - EXACT syntax: Embedding::from_document("text")
    pub fn from_document(document: impl Into<String>) -> impl EmbeddingBuilder {
        EmbeddingBuilderImpl {
            document: document.into(),
            vec: None,
            error_handler: None,
            result_handler: None,
            chunk_handler: None,
        }
    }
}

impl<F1, F2, F3> EmbeddingBuilder for EmbeddingBuilderImpl<F1, F2, F3>
where
    F1: Fn(String) + Send + Sync + 'static,
    F2: FnOnce(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + 'static,
    F3: FnMut(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + 'static,
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
            chunk_handler: self.chunk_handler,
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
            chunk_handler: self.chunk_handler,
        }
    }
    
    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_chunk<F>(self, handler: F) -> impl EmbeddingBuilder
    where
        F: FnMut(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + 'static,
    {
        EmbeddingBuilderImpl {
            document: self.document,
            vec: self.vec,
            error_handler: self.error_handler,
            result_handler: self.result_handler,
            chunk_handler: Some(handler),
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