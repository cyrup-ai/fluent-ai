use crate::async_task::{AsyncStream, AsyncTask, BadTraitImpl};
use crate::domain::chunk::EmbeddingChunk;
use crate::ZeroOneOrMany;
use serde::{Deserialize, Serialize};

/// Core trait for embedding models
pub trait EmbeddingModel: Send + Sync + Clone {
    /// Create embeddings for a single text
    fn embed(&self, text: &str) -> AsyncTask<ZeroOneOrMany<f32>>;

    /// Create embeddings for multiple texts with streaming
    fn embed_batch(&self, texts: Vec<String>) -> AsyncStream<EmbeddingChunk>;

    /// Simple embedding with handler
    fn on_embedding<F>(&self, text: &str, handler: F) -> AsyncTask<ZeroOneOrMany<f32>>
    where
        F: FnOnce(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + 'static,
    {
        let embed_task = self.embed(text);
        AsyncTask::from_future(async move {
            let embedding = embed_task.await.unwrap_or(ZeroOneOrMany::None);
            handler(embedding)
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    pub document: String,
    pub vec: Vec<f64>,
}

pub struct EmbeddingBuilder {
    document: String,
    vec: Option<Vec<f64>>,
}

pub struct EmbeddingBuilderWithHandler {
    document: String,
    vec: Option<Vec<f64>>,
    error_handler: Box<dyn Fn(String) + Send + Sync>,
    result_handler: Option<Box<dyn FnOnce(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + 'static>>,
    chunk_handler: Option<Box<dyn FnMut(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + 'static>>,
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
    pub fn vec(mut self, vec: Vec<f64>) -> Self {
        self.vec = Some(vec);
        self
    }

    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.vec = Some(vec![0.0; dims]);
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
        AsyncTask::from_value(Embedding {
            document: self.document,
            vec: self.vec.unwrap_or_default(),
        })
    }
}
