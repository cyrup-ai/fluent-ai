// ============================================================================
// File: src/providers/azure_new/image_generation.rs
// ----------------------------------------------------------------------------
// Azure OpenAI image generation implementation with AsyncTask pattern
// ============================================================================

#![allow(clippy::type_complexity)]

use fluent_ai_domain::model::ImageGenerationModel;
use serde_json::json;

use super::client::Client;
use crate::{
    client::ImageGenerationClient,
    image_generation::{self, ImageGenerationError, ImageGenerationRequest},
    providers::openai::ImageGenerationResponse,
    runtime::{self, AsyncTask}};

// ───────────────────────────── error handling ───────────────────────

#[derive(Debug, serde::Deserialize)]
struct ApiErrorResponse {
    message: String}

#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse)}

// ───────────────────────────── provider model ────────────────────────────
// ImageGenerationModel is now imported from fluent_ai_domain::model
// Removed duplicated ImageGenerationModel struct - use canonical domain type

impl ImageGenerationModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string()}
    }
}

// ───────────────────────────── impl ImageGenerationModel ─────────────────────

impl image_generation::ImageGenerationModel for ImageGenerationModel {
    type Response = ImageGenerationResponse;

    fn image_generation(
        &self,
        generation_request: ImageGenerationRequest,
    ) -> AsyncTask<
        Result<image_generation::ImageGenerationResponse<Self::Response>, ImageGenerationError>,
    > {
        let this = self.clone();
        runtime::spawn_async(async move { this.perform_image_generation(generation_request).await })
    }
}

// ───────────────────────────── internal async helpers ───────────────────

impl ImageGenerationModel {
    async fn perform_image_generation(
        self,
        generation_request: ImageGenerationRequest,
    ) -> Result<
        image_generation::ImageGenerationResponse<ImageGenerationResponse>,
        ImageGenerationError,
    > {
        let request = json!({
            "model": self.model,
            "prompt": generation_request.prompt,
            "size": format!("{}x{}", generation_request.width, generation_request.height),
            "response_format": "b64_json"
        });

        let response = self
            .client
            .post_image_generation(&self.model)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(ImageGenerationError::ProviderError(format!(
                "{}: {}",
                response.status(),
                response.text().await?
            )));
        }

        let t = response.text().await?;

        match serde_json::from_str::<ApiResponse<ImageGenerationResponse>>(&t)? {
            ApiResponse::Ok(response) => response.try_into(),
            ApiResponse::Err(err) => Err(ImageGenerationError::ProviderError(err.message))}
    }
}

// ───────────────────────────── client implementation ───────────────────

impl ImageGenerationClient for Client {
    type ImageGenerationModel = ImageGenerationModel;

    fn image_generation_model(&self, model: &str) -> Self::ImageGenerationModel {
        ImageGenerationModel::new(self.clone(), model)
    }
}
