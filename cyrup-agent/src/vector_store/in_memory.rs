// ============================================================================
// File: src/vector_store/in_memory.rs
// ----------------------------------------------------------------------------
// Zero-dependency, zero-alloc in-memory vector-store with cosine ranking.
// Conforms to the updated VectorStoreIndex trait (AsyncTask-only surface).
// ============================================================================

#![allow(clippy::type_complexity)]

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
};

use ordered_float::OrderedFloat;
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    embeddings::{Embedding, EmbeddingModel},
    runtime::{self, AsyncTask},
    vector_store::{VectorStoreError, VectorStoreIndex},
    OneOrMany,
};

// ---------------------------------------------------------------------------
// Small helper kept identical to the earlier revision
// ---------------------------------------------------------------------------
#[derive(Eq, PartialEq)]
struct Ranked<'a, D: Serialize>(
    OrderedFloat<f64>, // cosine similarity
    &'a str,           // doc-id
    &'a D,             // user payload (JSON-serialised)
    &'a str,           // best embedding’s `document` field
);

impl<D: Serialize> Ord for Ranked<'_, D> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}
impl<D: Serialize> PartialOrd for Ranked<'_, D> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// In-memory store (doc id → (payload , embeddings))
// ---------------------------------------------------------------------------
#[derive(Clone, Default)]
pub struct InMemoryVectorStore<D: Serialize> {
    inner: HashMap<String, (D, OneOrMany<Embedding>)>,
}

impl<D: Serialize> InMemoryVectorStore<D> {
    // ---------- constructors & mutators (unchanged) ------------------------
    pub fn from_documents<I>(docs: I) -> Self
    where
        I: IntoIterator<Item = (D, OneOrMany<Embedding>)>,
    {
        Self::with_doc_iter(docs, |idx, _| format!("doc{idx}"))
    }

    pub fn from_documents_with_ids<I, S>(docs: I) -> Self
    where
        I: IntoIterator<Item = (S, D, OneOrMany<Embedding>)>,
        S: ToString,
    {
        let mut map = HashMap::new();
        for (id, doc, emb) in docs {
            map.insert(id.to_string(), (doc, emb));
        }
        Self { inner: map }
    }

    pub fn from_documents_with<F, I>(docs: I, make_id: F) -> Self
    where
        I: IntoIterator<Item = (D, OneOrMany<Embedding>)>,
        F: Fn(usize, &D) -> String,
    {
        Self::with_doc_iter(docs, make_id)
    }

    pub fn add_documents<I>(&mut self, docs: I)
    where
        I: IntoIterator<Item = (D, OneOrMany<Embedding>)>,
    {
        let start = self.inner.len();
        for (idx, (doc, emb)) in docs.into_iter().enumerate() {
            self.inner.insert(format!("doc{}", start + idx), (doc, emb));
        }
    }

    pub fn add_documents_with_ids<I, S>(&mut self, docs: I)
    where
        I: IntoIterator<Item = (S, D, OneOrMany<Embedding>)>,
        S: ToString,
    {
        for (id, doc, emb) in docs {
            self.inner.insert(id.to_string(), (doc, emb));
        }
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = (&str, &(D, OneOrMany<Embedding>))> {
        self.inner.iter().map(|(id, pair)| (id.as_str(), pair))
    }

    #[inline(always)]
    fn with_doc_iter<F, I>(docs: I, make_id: F) -> Self
    where
        I: IntoIterator<Item = (D, OneOrMany<Embedding>)>,
        F: Fn(usize, &D) -> String,
    {
        let mut map = HashMap::new();
        for (idx, (doc, emb)) in docs.into_iter().enumerate() {
            map.insert(make_id(idx, &doc), (doc, emb));
        }
        Self { inner: map }
    }
}

// ---------------------------------------------------------------------------
// Concrete index tying store + embedding model
// ---------------------------------------------------------------------------
pub struct InMemoryVectorIndex<M: EmbeddingModel, D: Serialize> {
    model: M,
    store: InMemoryVectorStore<D>,
}

impl<M: EmbeddingModel, D: Serialize> InMemoryVectorIndex<M, D> {
    #[inline(always)]
    pub fn new(model: M, store: InMemoryVectorStore<D>) -> Self {
        Self { model, store }
    }

    // Re-expose for diagnostics
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.store.inner.len()
    }
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = (&str, &(D, OneOrMany<Embedding>))> {
        self.store.iter()
    }

    // ------------ internal ranking helper (cosine, descending) ------------
    fn rank(&self, query: &Embedding, k: usize) -> BinaryHeap<Reverse<Ranked<'_, D>>> {
        let mut heap: BinaryHeap<Reverse<Ranked<'_, D>>> = BinaryHeap::with_capacity(k + 1);

        for (id, (payload, embeds)) in &self.store.inner {
            if let Some(best) = embeds
                .iter()
                .map(|e| (OrderedFloat(e.cosine_similarity(query, false)), &e.document))
                .max_by(|a, b| a.0.cmp(&b.0))
            {
                heap.push(Reverse(Ranked(best.0, id, payload, best.1)));
                if heap.len() > k {
                    heap.pop();
                }
            }
        }
        heap
    }
}

// ---------------------------------------------------------------------------
// VectorStoreIndex implementation – NOW returns `AsyncTask`
// ---------------------------------------------------------------------------
impl<M, D> VectorStoreIndex for InMemoryVectorIndex<M, D>
where
    M: EmbeddingModel + Clone + Send + Sync + 'static,
    D: Serialize + Clone + Send + Sync + 'static,
{
    fn top_n<T>(
        &self,
        query: &str,
        n: usize,
    ) -> AsyncTask<Result<Vec<(f64, String, T)>, VectorStoreError>>
    where
        T: DeserializeOwned + Send + 'static,
    {
        let model = self.model.clone();
        let store = self.store.clone();
        let prompt = query.to_owned();

        runtime::spawn_async(async move {
            let q_embedding = model.embed_text(prompt).await?;
            let idx = InMemoryVectorIndex { model, store };
            let heap = idx.rank(&q_embedding, n);

            heap.into_iter()
                .map(|Reverse(Ranked(score, id, doc, _))| {
                    serde_json::from_str::<T>(&serde_json::to_string(doc)?)
                        .map(|d| (score.0, id.to_owned(), d))
                        .map_err(VectorStoreError::from)
                })
                .collect()
        })
    }

    fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> AsyncTask<Result<Vec<(f64, String)>, VectorStoreError>> {
        let model = self.model.clone();
        let store = self.store.clone();
        let prompt = query.to_owned();

        runtime::spawn_async(async move {
            let q_embedding = model.embed_text(prompt).await?;
            let idx = InMemoryVectorIndex { model, store };
            let heap = idx.rank(&q_embedding, n);

            Ok(heap
                .into_iter()
                .map(|Reverse(Ranked(score, id, ..))| (score.0, id.to_owned()))
                .collect())
        })
    }
}
