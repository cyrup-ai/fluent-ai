use serde::Deserialize;
use serde_json::json;

use super::client::{ApiResponse, Client, Usage};
use crate::embeddings::{self, EmbeddingError};

// ================================================================
// Mistral Embedding API
// ================================================================
/// `mistral-embed` embedding model
pub const MISTRAL_EMBED: &str = "mistral-embed";
pub const MAX_DOCUMENTS: usize = 1024;

#[derive(Clone)]
pub struct EmbeddingModel {
    client: Client,
    pub model: String,
    ndims: usize,
}

impl EmbeddingModel {
    pub fn new(client: Client, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.to_string(),
            ndims,
        }
    }
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = MAX_DOCUMENTS;
    fn ndims(&self) -> usize {
        self.ndims
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents = documents.into_iter().collect::<Vec<_>>();

        let request_body = serde_json::to_vec(&json!({
            "model": self.model,
            "input": documents,
        }))
        .map_err(|e| {
            EmbeddingError::ProviderError(format!("Failed to serialize request: {}", e))
        })?;

        let http_request = self
            .client
            .post("v1/embeddings", request_body)
            .map_err(|e| {
                EmbeddingError::ProviderError(format!("Failed to create request: {}", e))
            })?;

        let response = self
            .client
            .http_client
            .send(http_request)
            .await
            .map_err(|e| EmbeddingError::ProviderError(format!("Request failed: {}", e)))?;

        if response.status().is_success() {
            match serde_json::from_slice::<ApiResponse<EmbeddingResponse>>(response.body())? {
                ApiResponse::Ok(response) => {
                    tracing::debug!(target: "rig",
                        "Mistral embedding token usage: {}",
                        response.usage
                    );

                    if response.data.len() != documents.len() {
                        return Err(EmbeddingError::ResponseError(
                            "Response data length does not match input length".into(),
                        ));
                    }

                    Ok(response
                        .data
                        .into_iter()
                        .zip(documents.into_iter())
                        .map(|(embedding, document)| embeddings::Embedding {
                            document,
                            vec: embedding.embedding,
                        })
                        .collect())
                }
                ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
            }
        } else {
            let error_body = String::from_utf8_lossy(response.body());
            Err(EmbeddingError::ProviderError(error_body.to_string()))
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub usage: Usage,
    pub data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: usize,
}
