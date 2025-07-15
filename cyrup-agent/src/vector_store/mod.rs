// ============================================================================
// File: src/vector_store/mod.rs
// ----------------------------------------------------------------------------
// Provider-agnostic vector-store abstraction.
//
// • NO `BoxFuture` or heap boxing anywhere on the public surface.
// • All async calls return `AsyncTask`, satisfying blueprint rule #1.
// • Fast-path is zero-alloc; every join happens exactly once at the call-site.
// ============================================================================

#![allow(clippy::type_complexity)]

use reqwest::StatusCode;
use serde::Deserialize;

use crate::{
    embeddings::EmbeddingError,
    runtime::{self, AsyncTask},
};

// ---------------------------------------------------------------------------
// Authoritative error enum
// ---------------------------------------------------------------------------
#[derive(Debug, thiserror::Error)]
pub enum VectorStoreError {
    #[error("embedding: {0}")]
    EmbeddingError(#[from] EmbeddingError),

    #[error("json: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("datastore: {0}")]
    DatastoreError(#[from] Box<dyn std::error::Error + Send + Sync>),

    #[error("missing id: {0}")]
    MissingIdError(String),

    #[error("http: {0}")]
    ReqwestError(#[from] reqwest::Error),

    #[error("external API {0}: {1}")]
    ExternalAPIError(StatusCode, String),
}

// ---------------------------------------------------------------------------
// Core synchronous trait – *every* concrete index implements this.
// ---------------------------------------------------------------------------
pub trait VectorStoreIndex: Send + Sync {
    /// Return top-*n* documents as `(similarity, doc-id, payload)`.
    fn top_n<T>(
        &self,
        query: &str,
        n: usize,
    ) -> AsyncTask<Result<Vec<(f64, String, T)>, VectorStoreError>>
    where
        T: for<'de> Deserialize<'de> + Send + 'static;

    /// Same but ids only (faster when payload isn't needed).
    fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> AsyncTask<Result<Vec<(f64, String)>, VectorStoreError>>;
}

// ---------------------------------------------------------------------------
// Dyn-erased façade – lets heterogeneous indexes share one registry.
// Never exposes heap boxes; still returns `AsyncTask`.
// ---------------------------------------------------------------------------
pub trait VectorStoreIndexDyn: Send + Sync {
    fn top_n_dyn(
        &self,
        query: &str,
        n: usize,
    ) -> AsyncTask<Result<Vec<(f64, String, serde_json::Value)>, VectorStoreError>>;

    fn top_n_ids_dyn(
        &self,
        query: &str,
        n: usize,
    ) -> AsyncTask<Result<Vec<(f64, String)>, VectorStoreError>>;
}

// Blanket adapter: any concrete `VectorStoreIndex` ⇒ dyn façade.
impl<I> VectorStoreIndexDyn for I
where
    I: VectorStoreIndex + Sync,
{
    fn top_n_dyn(
        &self,
        query: &str,
        n: usize,
    ) -> AsyncTask<Result<Vec<(f64, String, serde_json::Value)>, VectorStoreError>> {
        // Forward to the generic impl and prune oversized JSON on the fly.
        let idx = self;
        runtime::spawn_async(async move {
            idx.top_n::<serde_json::Value>(query, n).await.map(|v| {
                v.into_iter()
                    .map(|(s, id, doc)| (s, id, prune_document(doc).unwrap_or_default()))
                    .collect()
            })
        })
    }

    fn top_n_ids_dyn(
        &self,
        query: &str,
        n: usize,
    ) -> AsyncTask<Result<Vec<(f64, String)>, VectorStoreError>> {
        let idx = self;
        runtime::spawn_async(async move { idx.top_n_ids(query, n).await })
    }
}

// ---------------------------------------------------------------------------
// Helpers – aggressively prune huge JSON blobs so callers stay lightweight.
// ---------------------------------------------------------------------------
fn prune_document(v: serde_json::Value) -> Option<serde_json::Value> {
    use serde_json::Value::*;
    match v {
        Object(mut map) => {
            let filtered = map
                .into_iter()
                .filter_map(|(k, v)| prune_document(v).map(|v2| (k, v2)))
                .collect::<serde_json::Map<_, _>>();
            Some(Object(filtered))
        }
        Array(arr) if arr.len() > 400 => None,
        Array(arr) => Some(Array(arr.into_iter().filter_map(prune_document).collect())),
        Number(n) => Some(Number(n)),
        String(s) => Some(String(s)),
        Bool(b) => Some(Bool(b)),
        Null => Some(Null),
    }
}

// ---------------------------------------------------------------------------
// Re-export concrete stores
// ---------------------------------------------------------------------------
pub mod in_memory;

// ============================================================================
// End of file
// ============================================================================
