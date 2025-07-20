//! Embedding builder implementations
//!
//! All embedding construction logic and builder patterns.

use fluent_ai_domain::embedding::Embedding;
use fluent_ai_domain::{AsyncTask, ZeroOneOrMany, spawn_async};

pub struct EmbeddingBuilder {
    document: String,
    vec: Option<ZeroOneOrMany<f64>>,
}

pub struct EmbeddingBuilderWithHandler {
    #[allow(dead_code)] // TODO: Use for source document text content for embedding generation
    document: String,
    #[allow(dead_code)] // TODO: Use for pre-computed embedding vector values
    vec: Option<ZeroOneOrMany<f64>>,
    #[allow(dead_code)] // TODO: Use for polymorphic error handling during embedding operations
    error_handler: Box<dyn Fn(String) + Send + Sync>,
    #[allow(dead_code)] // TODO: Use for embedding result processing and transformation
    result_handler:
        Option<Box<dyn FnOnce(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + 'static>>,
    #[allow(dead_code)] // TODO: Use for embedding streaming chunk processing
    chunk_handler:
        Option<Box<dyn FnMut(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + 'static>>,
}

impl Embedding {
    // Semantic entry point
    pub fn from_document(document: impl Into<String>) -> EmbeddingBuilder {
        EmbeddingBuilder {
            document: document.into(),
            vec: None,
        }
    }
}

impl EmbeddingBuilder {
    pub fn vec(mut self, vec: ZeroOneOrMany<f64>) -> Self {
        self.vec = Some(vec);
        self
    }

    pub fn with_dims(mut self, dims: usize) -> Self {
        self.vec = Some(ZeroOneOrMany::many(vec![0.0; dims]));
        self
    }

    // Error handling - required before terminal methods
    pub fn on_error<F>(self, handler: F) -> EmbeddingBuilderWithHandler
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        EmbeddingBuilderWithHandler {
            document: self.document,
            vec: self.vec,
            error_handler: Box::new(handler),
            result_handler: None,
            chunk_handler: None,
        }
    }

    pub fn on_result<F>(self, handler: F) -> EmbeddingBuilderWithHandler
    where
        F: FnOnce(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + 'static,
    {
        EmbeddingBuilderWithHandler {
            document: self.document,
            vec: self.vec,
            error_handler: Box::new(|e| eprintln!("Embedding error: {}", e)),
            result_handler: Some(Box::new(handler)),
            chunk_handler: None,
        }
    }

    pub fn on_chunk<F>(self, handler: F) -> EmbeddingBuilderWithHandler
    where
        F: FnMut(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + 'static,
    {
        EmbeddingBuilderWithHandler {
            document: self.document,
            vec: self.vec,
            error_handler: Box::new(|e| eprintln!("Embedding chunk error: {}", e)),
            result_handler: None,
            chunk_handler: Some(Box::new(handler)),
        }
    }
}

impl EmbeddingBuilderWithHandler {
    // Terminal method - returns AsyncTask<Embedding>
    pub fn embed(self) -> AsyncTask<Embedding> {
        spawn_async(async move {
            Embedding {
                document: self.document,
                vec: self.vec.unwrap_or(ZeroOneOrMany::None),
            }
        })
    }
}
