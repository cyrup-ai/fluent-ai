use serde::{Deserialize, Serialize};

/// OpenAI-compatible /v1/models response structure used by OpenAI, Mistral, XAI
#[derive(serde::Deserialize, serde::Serialize, Default, Debug)]
pub struct OpenAiModelsListResponse {
    pub data: Vec<OpenAiModel>,
    pub object: String,
}

/// Standard model object from OpenAI-compatible /v1/models endpoints
#[derive(serde::Deserialize, serde::Serialize, Default, Debug)]
pub struct OpenAiModel {
    pub id: String,
    pub object: String,
    pub created: Option<u64>,
    pub owned_by: Option<String>,
}

/// HuggingFace API response format (different structure)
#[derive(serde::Deserialize, serde::Serialize, Default, Debug)]
pub struct HuggingFaceModelsListResponse {
    pub models: Vec<HuggingFaceModel>,
}

#[derive(serde::Deserialize, serde::Serialize, Default, Debug)]
pub struct HuggingFaceModel {
    pub id: String,
    pub pipeline_tag: Option<String>,
    pub tags: Vec<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
}

/// Together.ai /v1/models API response type - Future batch API
/// Returns a direct array of models (not wrapped in "data" field)
#[allow(dead_code)] // Future API - for batch processing instead of streaming
pub type TogetherModelsListResponse = Vec<TogetherModel>;

/// Individual model in Together.ai response
#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct TogetherModel {
    pub id: String,
    pub object: String,
    #[serde(default)]
    pub created: u64,
    #[serde(rename = "type", default)]
    pub model_type: String,
    #[serde(default)]
    pub running: bool,
    #[serde(default)]
    pub display_name: String,
    #[serde(default)]
    pub organization: String,
    pub link: Option<String>,
    pub license: Option<String>,
    #[serde(default)]
    pub context_length: u64,
    #[serde(default)]
    pub config: TogetherConfig,
    #[serde(default)]
    pub pricing: Option<TogetherPricing>,
}

/// Together.ai model configuration
#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct TogetherConfig {
    pub chat_template: Option<String>,
    #[serde(default)]
    pub stop: Vec<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
}

/// Together.ai pricing information
#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct TogetherPricing {
    #[serde(default)]
    pub hourly: f64,
    #[serde(default)]
    pub input: f64,
    #[serde(default)]
    pub output: f64,
    #[serde(default)]
    pub base: f64,
    #[serde(default)]
    pub finetune: f64,
}

/// Huggingface models API response type - Future batch API
/// Returns a direct array of models (not wrapped)
#[allow(dead_code)] // Future API - for batch processing instead of streaming
pub type HFModelsListResponse = Vec<HFModel>;

/// Individual model in Huggingface response
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct HFModel {
    #[serde(rename = "_id")]
    pub internal_id: String,
    pub id: String,
    pub likes: u64,
    #[serde(rename = "private")]
    pub is_private: bool,
    pub downloads: u64,
    pub tags: Vec<String>,
    pub pipeline_tag: String,
    pub library_name: String,
    #[serde(rename = "createdAt")]
    pub created_at: String,
    #[serde(rename = "modelId")]
    pub model_id: String,
}
