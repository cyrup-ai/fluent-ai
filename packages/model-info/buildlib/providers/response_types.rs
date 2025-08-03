use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Together.ai /v1/models API response type
/// Returns a direct array of models (not wrapped in "data" field)
pub type TogetherModelsListResponse = Vec<TogetherModel>;

/// Individual model in Together.ai response
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct TogetherModel {
    pub id: String,
    pub object: String,
    pub created: u64,
    #[serde(rename = "type")]
    pub model_type: String,
    pub running: bool,
    pub display_name: String,
    pub organization: String,
    pub link: Option<String>,
    pub license: Option<String>,
    pub context_length: u64,
    pub config: TogetherConfig,
    pub pricing: TogetherPricing,
}

/// Together.ai model configuration
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct TogetherConfig {
    pub chat_template: Option<String>,
    pub stop: Vec<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
}

/// Together.ai pricing information
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct TogetherPricing {
    pub hourly: f64,
    pub input: f64,
    pub output: f64,
    pub base: f64,
    pub finetune: f64,
}

/// Huggingface models API response type
/// Returns a direct array of models (not wrapped)
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