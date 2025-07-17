use crate::{AsyncTask, spawn_async};
use crate::chunk::EmbeddingChunk;
use crate::ZeroOneOrMany;
use serde::{Deserialize, Serialize};

/// Core trait for embedding models
pub trait EmbeddingModel: Send + Sync + Clone {
    /// Create embeddings for a single text
    fn embed(&self, text: &str) -> AsyncTask<ZeroOneOrMany<f32>>;

    /// Create embeddings for multiple texts with streaming
    fn embed_batch(&self, texts: ZeroOneOrMany<String>) -> crate::async_task::AsyncStream<EmbeddingChunk>;

    /// Simple embedding with handler
    /// Performance: Zero allocation, direct await without Result unwrapping
    fn on_embedding<F>(&self, text: &str, handler: F) -> AsyncTask<ZeroOneOrMany<f32>>
    where
        F: FnOnce(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + 'static,
    {
        let embed_task = self.embed(text);
        crate::async_task::spawn_async(async move {
            let embedding = match embed_task.await {
                Ok(embedding) => embedding,
                Err(_) => ZeroOneOrMany::None, // Handle JoinError properly
            };
            handler(embedding)
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    pub document: String,
    pub vec: ZeroOneOrMany<f64>,
}

// Builder implementations moved to fluent_ai/src/builders/embedding.rs
