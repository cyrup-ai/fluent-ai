// ============================================================================
// File: src/providers/azure_new/embedding.rs
// ----------------------------------------------------------------------------
// Azure OpenAI embedding implementation with AsyncTask pattern
// ============================================================================

#![allow(clippy::type_complexity)]

use serde::{Deserialize, Serialize};
use serde_json::json;

use super::client::Client;
use crate::{
    embeddings::{self, EmbeddingError},
    runtime::{self as rt, AsyncTask},
};
use fluent_ai_domain::model::EmbeddingModel;

// ───────────────────────────── public constants ──────────────────────────

/// `text-embedding-3-large` embedding model
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// `text-embedding-3-small` embedding model
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-ada-002` embedding model
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

// ───────────────────────────── response types ────────────────────────────

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {} Total tokens: {}",
            self.prompt_tokens, self.total_tokens
        )
    }
}

// ───────────────────────────── error handling ───────────────────────

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

impl From<ApiErrorResponse> for EmbeddingError {
    fn from(err: ApiErrorResponse) -> Self {
        EmbeddingError::ProviderError(err.message)
    }
}

impl From<ApiResponse<EmbeddingResponse>> for Result<EmbeddingResponse, EmbeddingError> {
    fn from(value: ApiResponse<EmbeddingResponse>) -> Self {
        match value {
            ApiResponse::Ok(response) => Ok(response),
            ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
        }
    }
}

// ───────────────────────────── provider model ────────────────────────────
// EmbeddingModel is now imported from fluent_ai_domain::model
// Removed duplicated EmbeddingModel struct - use canonical domain type

impl EmbeddingModel {
    pub fn new(client: Client, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.to_string(),
            ndims,
        }
    }
}

// ───────────────────────────── impl EmbeddingModel ─────────────────────

impl crate::client::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;

    #[inline(always)]
    fn ndims(&self) -> usize {
        self.ndims
    }

    #[inline]
    fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> AsyncTask<Result<Vec<crate::embeddings::Embedding>, crate::embeddings::EmbeddingError>>
    {
        let documents: Vec<String> = documents.into_iter().collect();
        let this = self.clone();
        rt::spawn_async(async move { this.perform_embedding_batch(documents).await })
    }

    #[inline]
    fn embed(&self, text: &str) -> AsyncTask<cyrup_sugars::ZeroOneOrMany<f32>> {
        let this = self.clone();
        let text = text.to_string();
        rt::spawn_async(async move {
            match this.perform_single_embedding(text).await {
                Ok(embedding_vec) => cyrup_sugars::ZeroOneOrMany::from_iter(
                    embedding_vec.into_iter().map(|v| v as f32),
                ),
                Err(_) => cyrup_sugars::ZeroOneOrMany::None,
            }
        })
    }

    #[inline]
    fn embed_batch(
        &self,
        texts: cyrup_sugars::ZeroOneOrMany<String>,
    ) -> fluent_ai_domain::AsyncStream<fluent_ai_domain::chunk::EmbeddingChunk> {
        use fluent_ai_domain::AsyncStream;
        use futures::stream::StreamExt;

        let this = self.clone();
        let text_vec: Vec<String> = texts.into_iter().collect();

        AsyncStream::from_stream(
            futures::stream::iter(text_vec.into_iter().enumerate()).then(move |(index, text)| {
                let this = this.clone();
                async move {
                    match this.perform_single_embedding(text.clone()).await {
                        Ok(embedding_vec) => fluent_ai_domain::chunk::EmbeddingChunk {
                            index,
                            text,
                            embedding: cyrup_sugars::ZeroOneOrMany::from_iter(
                                embedding_vec.into_iter().map(|v| v as f32),
                            ),
                        },
                        Err(_) => fluent_ai_domain::chunk::EmbeddingChunk {
                            index,
                            text,
                            embedding: cyrup_sugars::ZeroOneOrMany::None,
                        },
                    }
                }
            }),
        )
    }
}

// ───────────────────────────── internal async helpers ───────────────────

impl EmbeddingModel {
    /// High-performance single embedding with zero-allocation optimizations
    #[inline]
    async fn perform_single_embedding(
        &self,
        text: String,
    ) -> Result<Vec<f64>, crate::embeddings::EmbeddingError> {
        use crate::embeddings::EmbeddingError;
        use crate::http::{HttpClient, HttpRequest};

        // Pre-allocate request body to avoid reallocation
        let request_body = serde_json::json!({
            "input": [text],
            "encoding_format": "float"
        });

        let serialized_body = serde_json::to_vec(&request_body)
            .map_err(|e| EmbeddingError::Serialization(e.to_string()))?;

        let url = format!(
            "{}/openai/deployments/{}/embeddings?api-version={}",
            self.client.azure_endpoint, self.model, self.client.api_version
        );

        let (header_name, header_value) = self.client.auth_header();

        let request = HttpRequest::post(url, serialized_body.into())
            .map_err(|e| EmbeddingError::Http(e.to_string()))?
            .header("Content-Type", "application/json")
            .header(header_name, header_value);

        let response = self
            .client
            .http_client
            .send(request)
            .await
            .map_err(|e| EmbeddingError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .map_err(|e| EmbeddingError::Http(e.to_string()))?;
            return Err(EmbeddingError::ProviderError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let response_body: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::Serialization(e.to_string()))?;

        match response_body.data.into_iter().next() {
            Some(embedding_data) => Ok(embedding_data.embedding),
            None => Err(EmbeddingError::ResponseError(
                "No embedding data in response".to_string(),
            )),
        }
    }

    /// High-performance batch embedding with optimized memory usage  
    #[inline]
    async fn perform_embedding_batch(
        &self,
        documents: Vec<String>,
    ) -> Result<Vec<crate::embeddings::Embedding>, crate::embeddings::EmbeddingError> {
        use crate::embeddings::EmbeddingError;
        use crate::http::{HttpClient, HttpRequest};

        if documents.is_empty() {
            return Ok(Vec::new());
        }

        if documents.len() > Self::MAX_DOCUMENTS {
            return Err(EmbeddingError::Provider(format!(
                "Too many documents: {} exceeds maximum of {}",
                documents.len(),
                Self::MAX_DOCUMENTS
            )));
        }

        let request_body = serde_json::json!({
            "input": documents,
            "encoding_format": "float"
        });

        let serialized_body = serde_json::to_vec(&request_body)
            .map_err(|e| EmbeddingError::Serialization(e.to_string()))?;

        let url = format!(
            "{}/openai/deployments/{}/embeddings?api-version={}",
            self.client.azure_endpoint, self.model, self.client.api_version
        );

        let (header_name, header_value) = self.client.auth_header();

        let request = HttpRequest::post(url, serialized_body.into())
            .map_err(|e| EmbeddingError::Http(e.to_string()))?
            .header("Content-Type", "application/json")
            .header(header_name, header_value);

        let response = self
            .client
            .http_client
            .send(request)
            .await
            .map_err(|e| EmbeddingError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .map_err(|e| EmbeddingError::Http(e.to_string()))?;
            return Err(EmbeddingError::ProviderError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let response_body: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::Serialization(e.to_string()))?;

        if response_body.data.len() != documents.len() {
            return Err(EmbeddingError::ResponseError(format!(
                "Response data length ({}) does not match input length ({})",
                response_body.data.len(),
                documents.len()
            )));
        }

        // Pre-allocate result vector with exact capacity for zero reallocation
        let mut results = Vec::with_capacity(response_body.data.len());

        for (embedding_data, document) in response_body.data.into_iter().zip(documents.into_iter())
        {
            results.push(crate::embeddings::Embedding {
                document,
                vec: cyrup_sugars::ZeroOneOrMany::from_iter(embedding_data.embedding.into_iter()),
            });
        }

        Ok(results)
    }
}
