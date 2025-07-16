// ============================================================================
// File: src/embeddings/builder.rs
// ----------------------------------------------------------------------------
// Batch-embedding builder that turns an arbitrary collection of documents
// (any type implementing `Embed`) into vectors via an `EmbeddingModel`.
//
// Public surface complies with blueprint rule #1: the only async output is an
// `AsyncTask`, guaranteeing a single `.await` for callers.
//
// Hot-path after `build()`:
//   • no allocations
//   • no locking
//   • bounded concurrency (CPU-adaptive)
// ============================================================================

#![allow(clippy::type_complexity)]

use core::cmp::max;
use std::collections::HashMap;

use crate::{
    embeddings::{
        embed::{EmbedError, TextEmbedder},
        Embed, Embedding, EmbeddingError, EmbeddingModel,
    },
    one_or_many::OneOrMany,
    runtime::{self, AsyncTask},
};

/// Fluent builder that accumulates documents then generates embeddings in
/// batched, parallel requests.
pub struct EmbeddingsBuilder<M: EmbeddingModel, T: Embed> {
    model: M,
    documents: Vec<(T, Vec<String>)>,
}

impl<M: EmbeddingModel, T: Embed> EmbeddingsBuilder<M, T> {
    /// Start a new builder for `model`.
    #[inline(always)]
    pub fn new(model: M) -> Self {
        Self {
            model,
            documents: Vec::new(),
        }
    }

    /// Push a single document.
    #[inline]
    pub fn document(mut self, doc: T) -> Result<Self, EmbedError> {
        let mut extractor = TextEmbedder::default();
        doc.embed(&mut extractor)?;
        self.documents.push((doc, extractor.texts));
        Ok(self)
    }

    /// Push many documents at once.
    #[inline]
    pub fn documents<I>(self, iter: I) -> Result<Self, EmbedError>
    where
        I: IntoIterator<Item = T>,
    {
        iter.into_iter().try_fold(self, |acc, d| acc.document(d))
    }
}

impl<M, T> EmbeddingsBuilder<M, T>
where
    M: EmbeddingModel + Clone + Send + Sync + 'static,
    T: Embed + Send + 'static,
{
    /// Stream embeddings as they complete - closures receive clean unwrapped (doc, embedding) pairs
    #[inline]
    pub fn stream<F>(self, handler: F) -> AsyncTask<Result<(), EmbeddingError>>
    where
        F: Fn(T, OneOrMany<Embedding>) + Send + Sync + 'static,
    {
        runtime::spawn_async(async move {
            // Process documents in batches and stream results as they complete
            let mut docs: HashMap<usize, T> = HashMap::with_capacity(self.documents.len());
            let mut per_doc_texts: Vec<(usize, Vec<String>)> =
                Vec::with_capacity(self.documents.len());

            for (idx, (doc, texts)) in self.documents.into_iter().enumerate() {
                docs.insert(idx, doc);
                per_doc_texts.push((idx, texts));
            }

            // Flatten texts into (doc_id, text) for batching
            let mut flat: Vec<(usize, String)> = Vec::new();
            per_doc_texts.into_iter().for_each(|(id, txts)| {
                flat.extend(txts.into_iter().map(|t| (id, t)));
            });

            // Process in batches and stream results immediately
            let batch_cap = M::MAX_DOCUMENTS;
            assert!(batch_cap > 0, "EmbeddingModel::MAX_DOCUMENTS must be > 0");

            let mut embeddings_by_doc: HashMap<usize, OneOrMany<Embedding>> = HashMap::new();
            let mut start = 0;
            
            while start < flat.len() {
                let end = (start + batch_cap).min(flat.len());
                let slice = &flat[start..end];
                let ids: Vec<usize> = slice.iter().map(|(i, _)| *i).collect();
                let docs_batch: Vec<String> = slice.iter().map(|(_, t)| t.clone()).collect();
                
                // Process this batch
                let embeds = self.model.embed_texts(docs_batch).await?;
                
                // Accumulate embeddings by document
                for (doc_id, embedding) in ids.into_iter().zip(embeds) {
                    embeddings_by_doc
                        .entry(doc_id)
                        .and_modify(|slot| slot.push(embedding.clone()))
                        .or_insert_with(|| OneOrMany::one(embedding));
                }
                
                start = end;
            }

            // Stream completed document embeddings
            for (doc_id, doc) in docs {
                if let Some(embeddings) = embeddings_by_doc.remove(&doc_id) {
                    handler(doc, embeddings); // Pre-unwrapped! No Result handling needed
                }
            }

            Ok(())
        })
    }

    /// Collect all embeddings (convenience method for when you want the full result)
    #[inline]
    pub fn collect_all(self) -> AsyncTask<Result<Vec<(T, OneOrMany<Embedding>)>, EmbeddingError>> {
        runtime::spawn_async(async move {
            let mut results = Vec::new();
            
            // Use streaming internally but collect results
            let stream_task = self.stream(|doc, embeddings| {
                results.push((doc, embeddings));
            });
            
            stream_task.await?;
            Ok(results)
        })
    }
}

// ============================================================================
// End of file
// ============================================================================
