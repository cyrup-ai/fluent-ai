/// OpenAI-compatible /v1/models request (empty for GET requests)
#[derive(serde::Deserialize, serde::Serialize, Default, Debug)]
pub struct OpenAiModelsListRequest;

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

/// Together.ai direct array response format
#[derive(serde::Deserialize, serde::Serialize, Default, Debug)]
pub struct TogetherModelsListResponse(pub Vec<TogetherModel>);

/// Together.ai model object with extended pricing and config data
#[derive(serde::Deserialize, serde::Serialize, Default, Debug)]
pub struct TogetherModel {
    pub id: String,
    pub object: String,
    pub created: Option<u64>,
    #[serde(rename = "type")]
    pub model_type: Option<String>,
    pub running: Option<bool>,
    pub display_name: Option<String>,
    pub organization: Option<String>,
    pub link: Option<String>,
    pub context_length: Option<u64>,
    pub config: Option<TogetherModelConfig>,
    pub pricing: Option<TogetherModelPricing>,
}

#[derive(serde::Deserialize, serde::Serialize, Default, Debug)]
pub struct TogetherModelConfig {
    pub chat_template: Option<String>,
    pub stop: Vec<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
}

#[derive(serde::Deserialize, serde::Serialize, Default, Debug)]
pub struct TogetherModelPricing {
    pub hourly: Option<f64>,
    pub input: Option<f64>,
    pub output: Option<f64>,
    pub base: Option<f64>,
    pub finetune: Option<f64>,
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

/// Legacy aliases for backward compatibility
pub type StandardModelsResponse = OpenAiModelsListResponse;
pub type StandardModel = OpenAiModel;
