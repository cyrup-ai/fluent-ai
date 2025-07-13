//! Model information data structures and implementations

use serde::{Deserialize, Serialize};

/// Information about a specific AI model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelInfoData {
    pub provider_name: String,
    pub name: String,
    pub max_input_tokens: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub input_price: Option<f64>,
    pub output_price: Option<f64>,
    pub supports_vision: Option<bool>,
    pub supports_function_calling: Option<bool>,
    pub require_max_tokens: Option<bool>,
}

// AUTO-GENERATED START
/// Get model info for gpt-4.1
pub fn get_gpt41_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openai".to_string(),
name: "gpt-4.1".to_string(),
max_input_tokens: Some(1047576u64),
max_output_tokens: Some(32768u64),
input_price: Some(2f64),
output_price: Some(8f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for gpt-4.1-mini
pub fn get_gpt41mini_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openai".to_string(),
name: "gpt-4.1-mini".to_string(),
max_input_tokens: Some(1047576u64),
max_output_tokens: Some(32768u64),
input_price: Some(0.4f64),
output_price: Some(1.6f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for gpt-4.1-nano
pub fn get_gpt41nano_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openai".to_string(),
name: "gpt-4.1-nano".to_string(),
max_input_tokens: Some(1047576u64),
max_output_tokens: Some(32768u64),
input_price: Some(0.1f64),
output_price: Some(0.4f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for gpt-4o
pub fn get_gpt4o_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openai".to_string(),
name: "gpt-4o".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: Some(16384u64),
input_price: Some(2.5f64),
output_price: Some(10f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for gpt-4o-search-preview
pub fn get_gpt4osearchpreview_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openai".to_string(),
name: "gpt-4o-search-preview".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: Some(16384u64),
input_price: Some(2.5f64),
output_price: Some(10f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for gpt-4o-mini
pub fn get_gpt4omini_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openai".to_string(),
name: "gpt-4o-mini".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: Some(16384u64),
input_price: Some(0.15f64),
output_price: Some(0.6f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for gpt-4o-mini-search-preview
pub fn get_gpt4ominisearchpreview_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openai".to_string(),
name: "gpt-4o-mini-search-preview".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: Some(16384u64),
input_price: Some(0.15f64),
output_price: Some(0.6f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for chatgpt-4o-latest
pub fn get_chatgpt4olatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openai".to_string(),
name: "chatgpt-4o-latest".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: Some(16384u64),
input_price: Some(5f64),
output_price: Some(15f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for o4-mini
pub fn get_o4mini_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openai".to_string(),
name: "o4-mini".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: None,
input_price: Some(1.1f64),
output_price: Some(4.4f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for o4-mini-high
pub fn get_o4minihigh_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openai".to_string(),
name: "o4-mini-high".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: None,
input_price: Some(1.1f64),
output_price: Some(4.4f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for o3
pub fn get_o3_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openai".to_string(),
name: "o3".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: None,
input_price: Some(10f64),
output_price: Some(40f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for o3-mini
pub fn get_o3mini_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openai".to_string(),
name: "o3-mini".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: None,
input_price: Some(1.1f64),
output_price: Some(4.4f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for o3-mini-high
pub fn get_o3minihigh_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openai".to_string(),
name: "o3-mini-high".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: None,
input_price: Some(1.1f64),
output_price: Some(4.4f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for gpt-4-turbo
pub fn get_gpt4turbo_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openai".to_string(),
name: "gpt-4-turbo".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: Some(4096u64),
input_price: Some(10f64),
output_price: Some(30f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for gpt-3.5-turbo
pub fn get_gpt35turbo_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openai".to_string(),
name: "gpt-3.5-turbo".to_string(),
max_input_tokens: Some(16385u64),
max_output_tokens: Some(4096u64),
input_price: Some(0.5f64),
output_price: Some(1.5f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for text-embedding-3-large
pub fn get_textembedding3large_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openai".to_string(),
name: "text-embedding-3-large".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0.13f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for text-embedding-3-small
pub fn get_textembedding3small_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openai".to_string(),
name: "text-embedding-3-small".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0.02f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for gemini-2.5-flash
pub fn get_gemini25flash_info() -> ModelInfoData {
ModelInfoData {
provider_name: "gemini".to_string(),
name: "gemini-2.5-flash".to_string(),
max_input_tokens: Some(1048576u64),
max_output_tokens: Some(65536u64),
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for gemini-2.5-pro
pub fn get_gemini25pro_info() -> ModelInfoData {
ModelInfoData {
provider_name: "gemini".to_string(),
name: "gemini-2.5-pro".to_string(),
max_input_tokens: Some(1048576u64),
max_output_tokens: Some(65536u64),
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for gemini-2.5-flash-lite-preview-06-17
pub fn get_gemini25flashlitepreview0617_info() -> ModelInfoData {
ModelInfoData {
provider_name: "gemini".to_string(),
name: "gemini-2.5-flash-lite-preview-06-17".to_string(),
max_input_tokens: Some(1000000u64),
max_output_tokens: Some(64000u64),
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for gemini-2.0-flash
pub fn get_gemini20flash_info() -> ModelInfoData {
ModelInfoData {
provider_name: "gemini".to_string(),
name: "gemini-2.0-flash".to_string(),
max_input_tokens: Some(1048576u64),
max_output_tokens: Some(8192u64),
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for gemini-2.0-flash-lite
pub fn get_gemini20flashlite_info() -> ModelInfoData {
ModelInfoData {
provider_name: "gemini".to_string(),
name: "gemini-2.0-flash-lite".to_string(),
max_input_tokens: Some(1048576u64),
max_output_tokens: Some(8192u64),
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for gemma-3-27b-it
pub fn get_gemma327bit_info() -> ModelInfoData {
ModelInfoData {
provider_name: "gemini".to_string(),
name: "gemma-3-27b-it".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: Some(8192u64),
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for text-embedding-004
pub fn get_textembedding004_info() -> ModelInfoData {
ModelInfoData {
provider_name: "gemini".to_string(),
name: "text-embedding-004".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for claude-opus-4-20250514
pub fn get_claudeopus420250514_info() -> ModelInfoData {
ModelInfoData {
provider_name: "claude".to_string(),
name: "claude-opus-4-20250514".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(8192u64),
input_price: Some(15f64),
output_price: Some(75f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for claude-opus-4-20250514:thinking
pub fn get_claudeopus420250514thinking_info() -> ModelInfoData {
ModelInfoData {
provider_name: "claude".to_string(),
name: "claude-opus-4-20250514:thinking".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(24000u64),
input_price: Some(15f64),
output_price: Some(75f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for claude-sonnet-4-20250514
pub fn get_claudesonnet420250514_info() -> ModelInfoData {
ModelInfoData {
provider_name: "claude".to_string(),
name: "claude-sonnet-4-20250514".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(8192u64),
input_price: Some(3f64),
output_price: Some(15f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for claude-sonnet-4-20250514:thinking
pub fn get_claudesonnet420250514thinking_info() -> ModelInfoData {
ModelInfoData {
provider_name: "claude".to_string(),
name: "claude-sonnet-4-20250514:thinking".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(24000u64),
input_price: Some(3f64),
output_price: Some(15f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for claude-3-7-sonnet-20250219
pub fn get_claude37sonnet20250219_info() -> ModelInfoData {
ModelInfoData {
provider_name: "claude".to_string(),
name: "claude-3-7-sonnet-20250219".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(8192u64),
input_price: Some(3f64),
output_price: Some(15f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for claude-3-7-sonnet-20250219:thinking
pub fn get_claude37sonnet20250219thinking_info() -> ModelInfoData {
ModelInfoData {
provider_name: "claude".to_string(),
name: "claude-3-7-sonnet-20250219:thinking".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(24000u64),
input_price: Some(3f64),
output_price: Some(15f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: Some(true),
}
}

/// Get model info for claude-3-5-sonnet-20241022
pub fn get_claude35sonnet20241022_info() -> ModelInfoData {
ModelInfoData {
provider_name: "claude".to_string(),
name: "claude-3-5-sonnet-20241022".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(8192u64),
input_price: Some(3f64),
output_price: Some(15f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for claude-3-5-haiku-20241022
pub fn get_claude35haiku20241022_info() -> ModelInfoData {
ModelInfoData {
provider_name: "claude".to_string(),
name: "claude-3-5-haiku-20241022".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.8f64),
output_price: Some(4f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for mistral-medium-latest
pub fn get_mistralmediumlatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "mistral".to_string(),
name: "mistral-medium-latest".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0.4f64),
output_price: Some(2f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for mistral-small-latest
pub fn get_mistralsmalllatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "mistral".to_string(),
name: "mistral-small-latest".to_string(),
max_input_tokens: Some(32768u64),
max_output_tokens: None,
input_price: Some(0.1f64),
output_price: Some(0.3f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for magistral-medium-latest
pub fn get_magistralmediumlatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "mistral".to_string(),
name: "magistral-medium-latest".to_string(),
max_input_tokens: Some(40960u64),
max_output_tokens: None,
input_price: Some(2f64),
output_price: Some(5f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for magistral-small-latest
pub fn get_magistralsmalllatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "mistral".to_string(),
name: "magistral-small-latest".to_string(),
max_input_tokens: Some(40960u64),
max_output_tokens: None,
input_price: Some(0.5f64),
output_price: Some(1.5f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for devstral-small-latest
pub fn get_devstralsmalllatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "mistral".to_string(),
name: "devstral-small-latest".to_string(),
max_input_tokens: Some(256000u64),
max_output_tokens: None,
input_price: Some(0.1f64),
output_price: Some(0.3f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for codestral-latest
pub fn get_codestrallatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "mistral".to_string(),
name: "codestral-latest".to_string(),
max_input_tokens: Some(256000u64),
max_output_tokens: None,
input_price: Some(0.3f64),
output_price: Some(0.9f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for mistral-embed
pub fn get_mistralembed_info() -> ModelInfoData {
ModelInfoData {
provider_name: "mistral".to_string(),
name: "mistral-embed".to_string(),
max_input_tokens: Some(8092u64),
max_output_tokens: None,
input_price: Some(0.1f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for jamba-large
pub fn get_jambalarge_info() -> ModelInfoData {
ModelInfoData {
provider_name: "ai21".to_string(),
name: "jamba-large".to_string(),
max_input_tokens: Some(256000u64),
max_output_tokens: None,
input_price: Some(2f64),
output_price: Some(8f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for jamba-mini
pub fn get_jambamini_info() -> ModelInfoData {
ModelInfoData {
provider_name: "ai21".to_string(),
name: "jamba-mini".to_string(),
max_input_tokens: Some(256000u64),
max_output_tokens: None,
input_price: Some(0.2f64),
output_price: Some(0.4f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for command-a-03-2025
pub fn get_commanda032025_info() -> ModelInfoData {
ModelInfoData {
provider_name: "cohere".to_string(),
name: "command-a-03-2025".to_string(),
max_input_tokens: Some(256000u64),
max_output_tokens: Some(8192u64),
input_price: Some(2.5f64),
output_price: Some(10f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for command-r7b-12-2024
pub fn get_commandr7b122024_info() -> ModelInfoData {
ModelInfoData {
provider_name: "cohere".to_string(),
name: "command-r7b-12-2024".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: Some(4096u64),
input_price: Some(0.0375f64),
output_price: Some(0.15f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for embed-v4.0
pub fn get_embedv40_info() -> ModelInfoData {
ModelInfoData {
provider_name: "cohere".to_string(),
name: "embed-v4.0".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0.12f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for embed-english-v3.0
pub fn get_embedenglishv30_info() -> ModelInfoData {
ModelInfoData {
provider_name: "cohere".to_string(),
name: "embed-english-v3.0".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0.1f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for embed-multilingual-v3.0
pub fn get_embedmultilingualv30_info() -> ModelInfoData {
ModelInfoData {
provider_name: "cohere".to_string(),
name: "embed-multilingual-v3.0".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0.1f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for rerank-v3.5
pub fn get_rerankv35_info() -> ModelInfoData {
ModelInfoData {
provider_name: "cohere".to_string(),
name: "rerank-v3.5".to_string(),
max_input_tokens: Some(4096u64),
max_output_tokens: None,
input_price: None,
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for rerank-english-v3.0
pub fn get_rerankenglishv30_info() -> ModelInfoData {
ModelInfoData {
provider_name: "cohere".to_string(),
name: "rerank-english-v3.0".to_string(),
max_input_tokens: Some(4096u64),
max_output_tokens: None,
input_price: None,
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for rerank-multilingual-v3.0
pub fn get_rerankmultilingualv30_info() -> ModelInfoData {
ModelInfoData {
provider_name: "cohere".to_string(),
name: "rerank-multilingual-v3.0".to_string(),
max_input_tokens: Some(4096u64),
max_output_tokens: None,
input_price: None,
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for grok-3-latest
pub fn get_grok3latest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "xai".to_string(),
name: "grok-3-latest".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(3f64),
output_price: Some(15f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for grok-3-fast-latest
pub fn get_grok3fastlatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "xai".to_string(),
name: "grok-3-fast-latest".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(5f64),
output_price: Some(25f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for grok-3-mini-latest
pub fn get_grok3minilatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "xai".to_string(),
name: "grok-3-mini-latest".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0.3f64),
output_price: Some(0.5f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for grok-3-mini-fast-latest
pub fn get_grok3minifastlatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "xai".to_string(),
name: "grok-3-mini-fast-latest".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0.6f64),
output_price: Some(4f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for sonar-pro
pub fn get_sonarpro_info() -> ModelInfoData {
ModelInfoData {
provider_name: "perplexity".to_string(),
name: "sonar-pro".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: None,
input_price: Some(3f64),
output_price: Some(15f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for sonar
pub fn get_sonar_info() -> ModelInfoData {
ModelInfoData {
provider_name: "perplexity".to_string(),
name: "sonar".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: None,
input_price: Some(1f64),
output_price: Some(1f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for sonar-reasoning-pro
pub fn get_sonarreasoningpro_info() -> ModelInfoData {
ModelInfoData {
provider_name: "perplexity".to_string(),
name: "sonar-reasoning-pro".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: None,
input_price: Some(2f64),
output_price: Some(8f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for sonar-reasoning
pub fn get_sonarreasoning_info() -> ModelInfoData {
ModelInfoData {
provider_name: "perplexity".to_string(),
name: "sonar-reasoning".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: None,
input_price: Some(1f64),
output_price: Some(5f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for sonar-deep-research
pub fn get_sonardeepresearch_info() -> ModelInfoData {
ModelInfoData {
provider_name: "perplexity".to_string(),
name: "sonar-deep-research".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: None,
input_price: Some(2f64),
output_price: Some(8f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for r1-1776
pub fn get_r11776_info() -> ModelInfoData {
ModelInfoData {
provider_name: "perplexity".to_string(),
name: "r1-1776".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: None,
input_price: Some(2f64),
output_price: Some(8f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for meta-llama/llama-4-maverick-17b-128e-instruct
pub fn get_metallamallama4maverick17b128einstruct_info() -> ModelInfoData {
ModelInfoData {
provider_name: "groq".to_string(),
name: "meta-llama/llama-4-maverick-17b-128e-instruct".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for meta-llama/llama-4-scout-17b-16e-instruct
pub fn get_metallamallama4scout17b16einstruct_info() -> ModelInfoData {
ModelInfoData {
provider_name: "groq".to_string(),
name: "meta-llama/llama-4-scout-17b-16e-instruct".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for llama-3.3-70b-versatile
pub fn get_llama3370bversatile_info() -> ModelInfoData {
ModelInfoData {
provider_name: "groq".to_string(),
name: "llama-3.3-70b-versatile".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for qwen-qwq-32b
pub fn get_qwenqwq32b_info() -> ModelInfoData {
ModelInfoData {
provider_name: "groq".to_string(),
name: "qwen-qwq-32b".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for qwen/qwen3-32b
pub fn get_qwenqwen332b_info() -> ModelInfoData {
ModelInfoData {
provider_name: "groq".to_string(),
name: "qwen/qwen3-32b".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for gemini-2.0-flash-001
pub fn get_gemini20flash001_info() -> ModelInfoData {
ModelInfoData {
provider_name: "vertexai".to_string(),
name: "gemini-2.0-flash-001".to_string(),
max_input_tokens: Some(1048576u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.15f64),
output_price: Some(0.6f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for gemini-2.0-flash-lite-001
pub fn get_gemini20flashlite001_info() -> ModelInfoData {
ModelInfoData {
provider_name: "vertexai".to_string(),
name: "gemini-2.0-flash-lite-001".to_string(),
max_input_tokens: Some(1048576u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.075f64),
output_price: Some(0.3f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for claude-3-5-sonnet-v2@20241022
pub fn get_claude35sonnetv220241022_info() -> ModelInfoData {
ModelInfoData {
provider_name: "vertexai".to_string(),
name: "claude-3-5-sonnet-v2@20241022".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(8192u64),
input_price: Some(3f64),
output_price: Some(15f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for mistral-small-2503
pub fn get_mistralsmall2503_info() -> ModelInfoData {
ModelInfoData {
provider_name: "vertexai".to_string(),
name: "mistral-small-2503".to_string(),
max_input_tokens: Some(32000u64),
max_output_tokens: None,
input_price: Some(0.1f64),
output_price: Some(0.3f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for codestral-2501
pub fn get_codestral2501_info() -> ModelInfoData {
ModelInfoData {
provider_name: "vertexai".to_string(),
name: "codestral-2501".to_string(),
max_input_tokens: Some(256000u64),
max_output_tokens: None,
input_price: Some(0.3f64),
output_price: Some(0.9f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for text-embedding-005
pub fn get_textembedding005_info() -> ModelInfoData {
ModelInfoData {
provider_name: "vertexai".to_string(),
name: "text-embedding-005".to_string(),
max_input_tokens: Some(20000u64),
max_output_tokens: None,
input_price: Some(0.025f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for text-multilingual-embedding-002
pub fn get_textmultilingualembedding002_info() -> ModelInfoData {
ModelInfoData {
provider_name: "vertexai".to_string(),
name: "text-multilingual-embedding-002".to_string(),
max_input_tokens: Some(20000u64),
max_output_tokens: None,
input_price: Some(0.2f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for us.anthropic.claude-opus-4-20250514-v1:0
pub fn get_usanthropicclaudeopus420250514v10_info() -> ModelInfoData {
ModelInfoData {
provider_name: "bedrock".to_string(),
name: "us.anthropic.claude-opus-4-20250514-v1:0".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(8192u64),
input_price: Some(15f64),
output_price: Some(75f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for us.anthropic.claude-opus-4-20250514-v1:0:thinking
pub fn get_usanthropicclaudeopus420250514v10thinking_info() -> ModelInfoData {
ModelInfoData {
provider_name: "bedrock".to_string(),
name: "us.anthropic.claude-opus-4-20250514-v1:0:thinking".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(24000u64),
input_price: Some(15f64),
output_price: Some(75f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: Some(true),
}
}

/// Get model info for us.anthropic.claude-sonnet-4-20250514-v1:0
pub fn get_usanthropicclaudesonnet420250514v10_info() -> ModelInfoData {
ModelInfoData {
provider_name: "bedrock".to_string(),
name: "us.anthropic.claude-sonnet-4-20250514-v1:0".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(8192u64),
input_price: Some(3f64),
output_price: Some(15f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for us.anthropic.claude-sonnet-4-20250514-v1:0:thinking
pub fn get_usanthropicclaudesonnet420250514v10thinking_info() -> ModelInfoData {
ModelInfoData {
provider_name: "bedrock".to_string(),
name: "us.anthropic.claude-sonnet-4-20250514-v1:0:thinking".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(24000u64),
input_price: Some(3f64),
output_price: Some(15f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: Some(true),
}
}

/// Get model info for us.anthropic.claude-3-7-sonnet-20250219-v1:0
pub fn get_usanthropicclaude37sonnet20250219v10_info() -> ModelInfoData {
ModelInfoData {
provider_name: "bedrock".to_string(),
name: "us.anthropic.claude-3-7-sonnet-20250219-v1:0".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(8192u64),
input_price: Some(3f64),
output_price: Some(15f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for us.anthropic.claude-3-7-sonnet-20250219-v1:0:thinking
pub fn get_usanthropicclaude37sonnet20250219v10thinking_info() -> ModelInfoData {
ModelInfoData {
provider_name: "bedrock".to_string(),
name: "us.anthropic.claude-3-7-sonnet-20250219-v1:0:thinking".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(24000u64),
input_price: Some(3f64),
output_price: Some(15f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: Some(true),
}
}

/// Get model info for anthropic.claude-3-5-sonnet-20241022-v2:0
pub fn get_anthropicclaude35sonnet20241022v20_info() -> ModelInfoData {
ModelInfoData {
provider_name: "bedrock".to_string(),
name: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(8192u64),
input_price: Some(3f64),
output_price: Some(15f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for anthropic.claude-3-5-haiku-20241022-v1:0
pub fn get_anthropicclaude35haiku20241022v10_info() -> ModelInfoData {
ModelInfoData {
provider_name: "bedrock".to_string(),
name: "anthropic.claude-3-5-haiku-20241022-v1:0".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.8f64),
output_price: Some(4f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for us.meta.llama4-maverick-17b-instruct-v1:0
pub fn get_usmetallama4maverick17binstructv10_info() -> ModelInfoData {
ModelInfoData {
provider_name: "bedrock".to_string(),
name: "us.meta.llama4-maverick-17b-instruct-v1:0".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.24f64),
output_price: Some(0.97f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for us.meta.llama4-scout-17b-instruct-v1:0
pub fn get_usmetallama4scout17binstructv10_info() -> ModelInfoData {
ModelInfoData {
provider_name: "bedrock".to_string(),
name: "us.meta.llama4-scout-17b-instruct-v1:0".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.17f64),
output_price: Some(0.66f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for us.meta.llama3-3-70b-instruct-v1:0
pub fn get_usmetallama3370binstructv10_info() -> ModelInfoData {
ModelInfoData {
provider_name: "bedrock".to_string(),
name: "us.meta.llama3-3-70b-instruct-v1:0".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.72f64),
output_price: Some(0.72f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for us.amazon.nova-premier-v1:0
pub fn get_usamazonnovapremierv10_info() -> ModelInfoData {
ModelInfoData {
provider_name: "bedrock".to_string(),
name: "us.amazon.nova-premier-v1:0".to_string(),
max_input_tokens: Some(300000u64),
max_output_tokens: Some(5120u64),
input_price: Some(2.5f64),
output_price: Some(12.5f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for us.amazon.nova-pro-v1:0
pub fn get_usamazonnovaprov10_info() -> ModelInfoData {
ModelInfoData {
provider_name: "bedrock".to_string(),
name: "us.amazon.nova-pro-v1:0".to_string(),
max_input_tokens: Some(300000u64),
max_output_tokens: Some(5120u64),
input_price: Some(0.8f64),
output_price: Some(3.2f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for us.amazon.nova-lite-v1:0
pub fn get_usamazonnovalitev10_info() -> ModelInfoData {
ModelInfoData {
provider_name: "bedrock".to_string(),
name: "us.amazon.nova-lite-v1:0".to_string(),
max_input_tokens: Some(300000u64),
max_output_tokens: Some(5120u64),
input_price: Some(0.06f64),
output_price: Some(0.24f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for us.amazon.nova-micro-v1:0
pub fn get_usamazonnovamicrov10_info() -> ModelInfoData {
ModelInfoData {
provider_name: "bedrock".to_string(),
name: "us.amazon.nova-micro-v1:0".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: Some(5120u64),
input_price: Some(0.035f64),
output_price: Some(0.14f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for cohere.embed-english-v3
pub fn get_cohereembedenglishv3_info() -> ModelInfoData {
ModelInfoData {
provider_name: "bedrock".to_string(),
name: "cohere.embed-english-v3".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0.1f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for cohere.embed-multilingual-v3
pub fn get_cohereembedmultilingualv3_info() -> ModelInfoData {
ModelInfoData {
provider_name: "bedrock".to_string(),
name: "cohere.embed-multilingual-v3".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0.1f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for us.deepseek.r1-v1:0
pub fn get_usdeepseekr1v10_info() -> ModelInfoData {
ModelInfoData {
provider_name: "bedrock".to_string(),
name: "us.deepseek.r1-v1:0".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: None,
input_price: Some(1.35f64),
output_price: Some(5.4f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for @cf/meta/llama-4-scout-17b-16e-instruct
pub fn get_cfmetallama4scout17b16einstruct_info() -> ModelInfoData {
ModelInfoData {
provider_name: "cloudflare".to_string(),
name: "@cf/meta/llama-4-scout-17b-16e-instruct".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: Some(2048u64),
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: Some(true),
}
}

/// Get model info for @cf/meta/llama-3.3-70b-instruct-fp8-fast
pub fn get_cfmetallama3370binstructfp8fast_info() -> ModelInfoData {
ModelInfoData {
provider_name: "cloudflare".to_string(),
name: "@cf/meta/llama-3.3-70b-instruct-fp8-fast".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: Some(2048u64),
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: Some(true),
}
}

/// Get model info for @cf/qwen/qwq-32b
pub fn get_cfqwenqwq32b_info() -> ModelInfoData {
ModelInfoData {
provider_name: "cloudflare".to_string(),
name: "@cf/qwen/qwq-32b".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: Some(2048u64),
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: Some(true),
}
}

/// Get model info for @cf/qwen/qwen2.5-coder-32b-instruct
pub fn get_cfqwenqwen25coder32binstruct_info() -> ModelInfoData {
ModelInfoData {
provider_name: "cloudflare".to_string(),
name: "@cf/qwen/qwen2.5-coder-32b-instruct".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: Some(2048u64),
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: Some(true),
}
}

/// Get model info for @cf/google/gemma-3-12b-it
pub fn get_cfgooglegemma312bit_info() -> ModelInfoData {
ModelInfoData {
provider_name: "cloudflare".to_string(),
name: "@cf/google/gemma-3-12b-it".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: Some(2048u64),
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: Some(true),
}
}

/// Get model info for @cf/mistralai/mistral-small-3.1-24b-instruct
pub fn get_cfmistralaimistralsmall3124binstruct_info() -> ModelInfoData {
ModelInfoData {
provider_name: "cloudflare".to_string(),
name: "@cf/mistralai/mistral-small-3.1-24b-instruct".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: Some(2048u64),
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: Some(true),
}
}

/// Get model info for @cf/baai/bge-large-en-v1.5
pub fn get_cfbaaibgelargeenv15_info() -> ModelInfoData {
ModelInfoData {
provider_name: "cloudflare".to_string(),
name: "@cf/baai/bge-large-en-v1.5".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for ernie-4.5-turbo-128k
pub fn get_ernie45turbo128k_info() -> ModelInfoData {
ModelInfoData {
provider_name: "ernie".to_string(),
name: "ernie-4.5-turbo-128k".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0.112f64),
output_price: Some(0.448f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for ernie-4.5-turbo-vl-32k
pub fn get_ernie45turbovl32k_info() -> ModelInfoData {
ModelInfoData {
provider_name: "ernie".to_string(),
name: "ernie-4.5-turbo-vl-32k".to_string(),
max_input_tokens: Some(32768u64),
max_output_tokens: None,
input_price: Some(0.42f64),
output_price: Some(1.26f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for ernie-x1-turbo-32k
pub fn get_erniex1turbo32k_info() -> ModelInfoData {
ModelInfoData {
provider_name: "ernie".to_string(),
name: "ernie-x1-turbo-32k".to_string(),
max_input_tokens: Some(32768u64),
max_output_tokens: None,
input_price: Some(0.14f64),
output_price: Some(0.56f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for bge-large-zh
pub fn get_bgelargezh_info() -> ModelInfoData {
ModelInfoData {
provider_name: "ernie".to_string(),
name: "bge-large-zh".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0.07f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for bge-large-en
pub fn get_bgelargeen_info() -> ModelInfoData {
ModelInfoData {
provider_name: "ernie".to_string(),
name: "bge-large-en".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0.07f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for bce-reranker-base
pub fn get_bcererankerbase_info() -> ModelInfoData {
ModelInfoData {
provider_name: "ernie".to_string(),
name: "bce-reranker-base".to_string(),
max_input_tokens: Some(1024u64),
max_output_tokens: None,
input_price: Some(0.07f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for qwen-max-latest
pub fn get_qwenmaxlatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "qianwen".to_string(),
name: "qwen-max-latest".to_string(),
max_input_tokens: Some(32678u64),
max_output_tokens: Some(8192u64),
input_price: Some(1.6f64),
output_price: Some(6.4f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for qwen-plus-latest
pub fn get_qwenpluslatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "qianwen".to_string(),
name: "qwen-plus-latest".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.112f64),
output_price: Some(0.28f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for qwen-turbo-latest
pub fn get_qwenturbolatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "qianwen".to_string(),
name: "qwen-turbo-latest".to_string(),
max_input_tokens: Some(1000000u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.042f64),
output_price: Some(0.084f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for qwen-long
pub fn get_qwenlong_info() -> ModelInfoData {
ModelInfoData {
provider_name: "qianwen".to_string(),
name: "qwen-long".to_string(),
max_input_tokens: Some(1000000u64),
max_output_tokens: None,
input_price: Some(0.07f64),
output_price: Some(0.28f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for qwen-omni-turbo-latest
pub fn get_qwenomniturbolatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "qianwen".to_string(),
name: "qwen-omni-turbo-latest".to_string(),
max_input_tokens: Some(32768u64),
max_output_tokens: Some(2048u64),
input_price: None,
output_price: None,
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for qwq-plus-latest
pub fn get_qwqpluslatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "qianwen".to_string(),
name: "qwq-plus-latest".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.224f64),
output_price: Some(0.56f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for qwen-vl-max-latest
pub fn get_qwenvlmaxlatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "qianwen".to_string(),
name: "qwen-vl-max-latest".to_string(),
max_input_tokens: Some(30720u64),
max_output_tokens: Some(2048u64),
input_price: Some(0.42f64),
output_price: Some(1.26f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for qwen-vl-plus-latest
pub fn get_qwenvlpluslatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "qianwen".to_string(),
name: "qwen-vl-plus-latest".to_string(),
max_input_tokens: Some(30000u64),
max_output_tokens: Some(2048u64),
input_price: Some(0.21f64),
output_price: Some(0.63f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for qwen3-235b-a22b
pub fn get_qwen3235ba22b_info() -> ModelInfoData {
ModelInfoData {
provider_name: "qianwen".to_string(),
name: "qwen3-235b-a22b".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.56f64),
output_price: Some(1.68f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for qwen3-30b-a3b
pub fn get_qwen330ba3b_info() -> ModelInfoData {
ModelInfoData {
provider_name: "qianwen".to_string(),
name: "qwen3-30b-a3b".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.21f64),
output_price: Some(0.84f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for qwen3-32b
pub fn get_qwen332b_info() -> ModelInfoData {
ModelInfoData {
provider_name: "qianwen".to_string(),
name: "qwen3-32b".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.28f64),
output_price: Some(1.12f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for qwq-32b
pub fn get_qwq32b_info() -> ModelInfoData {
ModelInfoData {
provider_name: "qianwen".to_string(),
name: "qwq-32b".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.28f64),
output_price: Some(0.84f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for qwen2.5-72b-instruct
pub fn get_qwen2572binstruct_info() -> ModelInfoData {
ModelInfoData {
provider_name: "qianwen".to_string(),
name: "qwen2.5-72b-instruct".to_string(),
max_input_tokens: Some(129024u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.56f64),
output_price: Some(1.68f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for qwen2.5-vl-72b-instruct
pub fn get_qwen25vl72binstruct_info() -> ModelInfoData {
ModelInfoData {
provider_name: "qianwen".to_string(),
name: "qwen2.5-vl-72b-instruct".to_string(),
max_input_tokens: Some(129024u64),
max_output_tokens: Some(8192u64),
input_price: Some(2.24f64),
output_price: Some(6.72f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for qwen2.5-coder-32b-instruct
pub fn get_qwen25coder32binstruct_info() -> ModelInfoData {
ModelInfoData {
provider_name: "qianwen".to_string(),
name: "qwen2.5-coder-32b-instruct".to_string(),
max_input_tokens: Some(129024u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.49f64),
output_price: Some(0.98f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for deepseek-v3
pub fn get_deepseekv3_info() -> ModelInfoData {
ModelInfoData {
provider_name: "qianwen".to_string(),
name: "deepseek-v3".to_string(),
max_input_tokens: Some(65536u64),
max_output_tokens: None,
input_price: Some(0.14f64),
output_price: Some(0.56f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for deepseek-r1-0528
pub fn get_deepseekr10528_info() -> ModelInfoData {
ModelInfoData {
provider_name: "qianwen".to_string(),
name: "deepseek-r1-0528".to_string(),
max_input_tokens: Some(65536u64),
max_output_tokens: None,
input_price: Some(0.28f64),
output_price: Some(1.12f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for text-embedding-v4
pub fn get_textembeddingv4_info() -> ModelInfoData {
ModelInfoData {
provider_name: "qianwen".to_string(),
name: "text-embedding-v4".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0.1f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for text-embedding-v3
pub fn get_textembeddingv3_info() -> ModelInfoData {
ModelInfoData {
provider_name: "qianwen".to_string(),
name: "text-embedding-v3".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0.1f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for hunyuan-turbos-latest
pub fn get_hunyuanturboslatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "hunyuan".to_string(),
name: "hunyuan-turbos-latest".to_string(),
max_input_tokens: Some(28000u64),
max_output_tokens: None,
input_price: Some(0.112f64),
output_price: Some(0.28f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for hunyuan-t1-latest
pub fn get_hunyuant1latest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "hunyuan".to_string(),
name: "hunyuan-t1-latest".to_string(),
max_input_tokens: Some(28000u64),
max_output_tokens: None,
input_price: Some(0.14f64),
output_price: Some(0.56f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for hunyuan-lite
pub fn get_hunyuanlite_info() -> ModelInfoData {
ModelInfoData {
provider_name: "hunyuan".to_string(),
name: "hunyuan-lite".to_string(),
max_input_tokens: Some(250000u64),
max_output_tokens: None,
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for hunyuan-turbos-vision
pub fn get_hunyuanturbosvision_info() -> ModelInfoData {
ModelInfoData {
provider_name: "hunyuan".to_string(),
name: "hunyuan-turbos-vision".to_string(),
max_input_tokens: Some(6144u64),
max_output_tokens: None,
input_price: Some(0.42f64),
output_price: Some(0.84f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for hunyuan-t1-vision
pub fn get_hunyuant1vision_info() -> ModelInfoData {
ModelInfoData {
provider_name: "hunyuan".to_string(),
name: "hunyuan-t1-vision".to_string(),
max_input_tokens: Some(24000u64),
max_output_tokens: None,
input_price: None,
output_price: None,
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for hunyuan-embedding
pub fn get_hunyuanembedding_info() -> ModelInfoData {
ModelInfoData {
provider_name: "hunyuan".to_string(),
name: "hunyuan-embedding".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0.01f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for kimi-latest
pub fn get_kimilatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "moonshot".to_string(),
name: "kimi-latest".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(1.4f64),
output_price: Some(4.2f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for kimi-thinking-preview
pub fn get_kimithinkingpreview_info() -> ModelInfoData {
ModelInfoData {
provider_name: "moonshot".to_string(),
name: "kimi-thinking-preview".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(28f64),
output_price: Some(28f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for deepseek-chat
pub fn get_deepseekchat_info() -> ModelInfoData {
ModelInfoData {
provider_name: "deepseek".to_string(),
name: "deepseek-chat".to_string(),
max_input_tokens: Some(64000u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.27f64),
output_price: Some(1.1f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for deepseek-reasoner
pub fn get_deepseekreasoner_info() -> ModelInfoData {
ModelInfoData {
provider_name: "deepseek".to_string(),
name: "deepseek-reasoner".to_string(),
max_input_tokens: Some(64000u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.55f64),
output_price: Some(2.19f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for glm-4-plus
pub fn get_glm4plus_info() -> ModelInfoData {
ModelInfoData {
provider_name: "zhipuai".to_string(),
name: "glm-4-plus".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0.7f64),
output_price: Some(0.7f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for glm-4-air
pub fn get_glm4air_info() -> ModelInfoData {
ModelInfoData {
provider_name: "zhipuai".to_string(),
name: "glm-4-air".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0.07f64),
output_price: Some(0.07f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for glm-4-long
pub fn get_glm4long_info() -> ModelInfoData {
ModelInfoData {
provider_name: "zhipuai".to_string(),
name: "glm-4-long".to_string(),
max_input_tokens: Some(1000000u64),
max_output_tokens: Some(4096u64),
input_price: Some(0.14f64),
output_price: Some(0.14f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for glm-4-flash-250414
pub fn get_glm4flash250414_info() -> ModelInfoData {
ModelInfoData {
provider_name: "zhipuai".to_string(),
name: "glm-4-flash-250414".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for glm-4v-plus-0111
pub fn get_glm4vplus0111_info() -> ModelInfoData {
ModelInfoData {
provider_name: "zhipuai".to_string(),
name: "glm-4v-plus-0111".to_string(),
max_input_tokens: Some(8192u64),
max_output_tokens: None,
input_price: Some(0.56f64),
output_price: Some(0.56f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for glm-4v-flash
pub fn get_glm4vflash_info() -> ModelInfoData {
ModelInfoData {
provider_name: "zhipuai".to_string(),
name: "glm-4v-flash".to_string(),
max_input_tokens: Some(8192u64),
max_output_tokens: None,
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for glm-z1-air
pub fn get_glmz1air_info() -> ModelInfoData {
ModelInfoData {
provider_name: "zhipuai".to_string(),
name: "glm-z1-air".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0.7f64),
output_price: Some(0.7f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for glm-z1-flash
pub fn get_glmz1flash_info() -> ModelInfoData {
ModelInfoData {
provider_name: "zhipuai".to_string(),
name: "glm-z1-flash".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0f64),
output_price: Some(0f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for embedding-3
pub fn get_embedding3_info() -> ModelInfoData {
ModelInfoData {
provider_name: "zhipuai".to_string(),
name: "embedding-3".to_string(),
max_input_tokens: Some(8192u64),
max_output_tokens: None,
input_price: Some(0.07f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for rerank
pub fn get_rerank_info() -> ModelInfoData {
ModelInfoData {
provider_name: "zhipuai".to_string(),
name: "rerank".to_string(),
max_input_tokens: Some(4096u64),
max_output_tokens: None,
input_price: Some(0.112f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for minimax-text-01
pub fn get_minimaxtext01_info() -> ModelInfoData {
ModelInfoData {
provider_name: "minimax".to_string(),
name: "minimax-text-01".to_string(),
max_input_tokens: Some(1000192u64),
max_output_tokens: None,
input_price: Some(0.14f64),
output_price: Some(1.12f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for minimax-m1
pub fn get_minimaxm1_info() -> ModelInfoData {
ModelInfoData {
provider_name: "minimax".to_string(),
name: "minimax-m1".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0.112f64),
output_price: Some(1.12f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for openai/gpt-4.1
pub fn get_openaigpt41_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "openai/gpt-4.1".to_string(),
max_input_tokens: Some(1047576u64),
max_output_tokens: Some(32768u64),
input_price: Some(2f64),
output_price: Some(8f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for openai/gpt-4.1-mini
pub fn get_openaigpt41mini_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "openai/gpt-4.1-mini".to_string(),
max_input_tokens: Some(1047576u64),
max_output_tokens: Some(32768u64),
input_price: Some(0.4f64),
output_price: Some(1.6f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for openai/gpt-4.1-nano
pub fn get_openaigpt41nano_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "openai/gpt-4.1-nano".to_string(),
max_input_tokens: Some(1047576u64),
max_output_tokens: Some(32768u64),
input_price: Some(0.1f64),
output_price: Some(0.4f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for openai/gpt-4o
pub fn get_openaigpt4o_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "openai/gpt-4o".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: None,
input_price: Some(2.5f64),
output_price: Some(10f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for openai/gpt-4o-search-preview
pub fn get_openaigpt4osearchpreview_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "openai/gpt-4o-search-preview".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: Some(16384u64),
input_price: Some(2.5f64),
output_price: Some(10f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for openai/gpt-4o-mini
pub fn get_openaigpt4omini_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "openai/gpt-4o-mini".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: None,
input_price: Some(0.15f64),
output_price: Some(0.6f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for openai/gpt-4o-mini-search-preview
pub fn get_openaigpt4ominisearchpreview_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "openai/gpt-4o-mini-search-preview".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: Some(16384u64),
input_price: Some(0.15f64),
output_price: Some(0.6f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for openai/chatgpt-4o-latest
pub fn get_openaichatgpt4olatest_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "openai/chatgpt-4o-latest".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: None,
input_price: Some(5f64),
output_price: Some(15f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for openai/o4-mini
pub fn get_openaio4mini_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "openai/o4-mini".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: None,
input_price: Some(1.1f64),
output_price: Some(4.4f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for openai/o4-mini-high
pub fn get_openaio4minihigh_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "openai/o4-mini-high".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: None,
input_price: Some(1.1f64),
output_price: Some(4.4f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for openai/o3-pro
pub fn get_openaio3pro_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "openai/o3-pro".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: None,
input_price: Some(20f64),
output_price: Some(80f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for openai/o3
pub fn get_openaio3_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "openai/o3".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: None,
input_price: Some(10f64),
output_price: Some(40f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for openai/o3-mini
pub fn get_openaio3mini_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "openai/o3-mini".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: None,
input_price: Some(1.1f64),
output_price: Some(4.4f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for openai/o3-mini-high
pub fn get_openaio3minihigh_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "openai/o3-mini-high".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: None,
input_price: Some(1.1f64),
output_price: Some(4.4f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for google/gemini-2.5-flash
pub fn get_googlegemini25flash_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "google/gemini-2.5-flash".to_string(),
max_input_tokens: Some(1048576u64),
max_output_tokens: None,
input_price: Some(0.15f64),
output_price: Some(0.6f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for google/gemini-2.5-pro
pub fn get_googlegemini25pro_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "google/gemini-2.5-pro".to_string(),
max_input_tokens: Some(1048576u64),
max_output_tokens: None,
input_price: Some(1.25f64),
output_price: Some(10f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for google/gemini-2.5-flash-lite-preview-06-17
pub fn get_googlegemini25flashlitepreview0617_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "google/gemini-2.5-flash-lite-preview-06-17".to_string(),
max_input_tokens: Some(1048576u64),
max_output_tokens: None,
input_price: Some(0.1f64),
output_price: Some(0.4f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for google/gemini-2.0-flash-001
pub fn get_googlegemini20flash001_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "google/gemini-2.0-flash-001".to_string(),
max_input_tokens: Some(1000000u64),
max_output_tokens: None,
input_price: Some(0.1f64),
output_price: Some(0.4f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for google/gemini-2.0-flash-lite-001
pub fn get_googlegemini20flashlite001_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "google/gemini-2.0-flash-lite-001".to_string(),
max_input_tokens: Some(1048576u64),
max_output_tokens: None,
input_price: Some(0.075f64),
output_price: Some(0.3f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for google/gemma-3-27b-it
pub fn get_googlegemma327bit_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "google/gemma-3-27b-it".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0.1f64),
output_price: Some(0.2f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for anthropic/claude-opus-4
pub fn get_anthropicclaudeopus4_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "anthropic/claude-opus-4".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(8192u64),
input_price: Some(15f64),
output_price: Some(75f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for anthropic/claude-sonnet-4
pub fn get_anthropicclaudesonnet4_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "anthropic/claude-sonnet-4".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(8192u64),
input_price: Some(3f64),
output_price: Some(15f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for anthropic/claude-3.7-sonnet
pub fn get_anthropicclaude37sonnet_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "anthropic/claude-3.7-sonnet".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(8192u64),
input_price: Some(3f64),
output_price: Some(15f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for anthropic/claude-3.7-sonnet:thinking
pub fn get_anthropicclaude37sonnetthinking_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "anthropic/claude-3.7-sonnet:thinking".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(24000u64),
input_price: Some(3f64),
output_price: Some(15f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: Some(true),
}
}

/// Get model info for anthropic/claude-3.5-sonnet
pub fn get_anthropicclaude35sonnet_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "anthropic/claude-3.5-sonnet".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(8192u64),
input_price: Some(3f64),
output_price: Some(15f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for anthropic/claude-3-5-haiku
pub fn get_anthropicclaude35haiku_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "anthropic/claude-3-5-haiku".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.8f64),
output_price: Some(4f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: Some(true),
}
}

/// Get model info for meta-llama/llama-4-maverick
pub fn get_metallamallama4maverick_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "meta-llama/llama-4-maverick".to_string(),
max_input_tokens: Some(1048576u64),
max_output_tokens: None,
input_price: Some(0.18f64),
output_price: Some(0.6f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for meta-llama/llama-4-scout
pub fn get_metallamallama4scout_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "meta-llama/llama-4-scout".to_string(),
max_input_tokens: Some(327680u64),
max_output_tokens: None,
input_price: Some(0.08f64),
output_price: Some(0.3f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for meta-llama/llama-3.3-70b-instruct
pub fn get_metallamallama3370binstruct_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "meta-llama/llama-3.3-70b-instruct".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0.12f64),
output_price: Some(0.3f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for mistralai/mistral-medium-3
pub fn get_mistralaimistralmedium3_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "mistralai/mistral-medium-3".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0.4f64),
output_price: Some(2f64),
supports_vision: Some(true),
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for mistralai/mistral-small-3.2-24b-instruct
pub fn get_mistralaimistralsmall3224binstruct_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "mistralai/mistral-small-3.2-24b-instruct".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0.1f64),
output_price: Some(0.3f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for mistralai/magistral-medium-2506
pub fn get_mistralaimagistralmedium2506_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "mistralai/magistral-medium-2506".to_string(),
max_input_tokens: Some(40960u64),
max_output_tokens: None,
input_price: Some(2f64),
output_price: Some(5f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for mistralai/magistral-medium-2506:thinking
pub fn get_mistralaimagistralmedium2506thinking_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "mistralai/magistral-medium-2506:thinking".to_string(),
max_input_tokens: Some(40960u64),
max_output_tokens: None,
input_price: Some(2f64),
output_price: Some(5f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for mistralai/magistral-small-2506
pub fn get_mistralaimagistralsmall2506_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "mistralai/magistral-small-2506".to_string(),
max_input_tokens: Some(40960u64),
max_output_tokens: None,
input_price: Some(0.5f64),
output_price: Some(1.5f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for mistralai/devstral-small
pub fn get_mistralaidevstralsmall_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "mistralai/devstral-small".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0.07f64),
output_price: Some(0.1f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for mistralai/codestral-2501
pub fn get_mistralaicodestral2501_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "mistralai/codestral-2501".to_string(),
max_input_tokens: Some(256000u64),
max_output_tokens: None,
input_price: Some(0.3f64),
output_price: Some(0.9f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for ai21/jamba-1.6-large
pub fn get_ai21jamba16large_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "ai21/jamba-1.6-large".to_string(),
max_input_tokens: Some(256000u64),
max_output_tokens: None,
input_price: Some(2f64),
output_price: Some(8f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for ai21/jamba-1.6-mini
pub fn get_ai21jamba16mini_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "ai21/jamba-1.6-mini".to_string(),
max_input_tokens: Some(256000u64),
max_output_tokens: None,
input_price: Some(0.2f64),
output_price: Some(0.4f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for cohere/command-a
pub fn get_coherecommanda_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "cohere/command-a".to_string(),
max_input_tokens: Some(256000u64),
max_output_tokens: None,
input_price: Some(2.5f64),
output_price: Some(10f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for cohere/command-r7b-12-2024
pub fn get_coherecommandr7b122024_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "cohere/command-r7b-12-2024".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: Some(4096u64),
input_price: Some(0.0375f64),
output_price: Some(0.15f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for deepseek/deepseek-chat-v3-0324
pub fn get_deepseekdeepseekchatv30324_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "deepseek/deepseek-chat-v3-0324".to_string(),
max_input_tokens: Some(64000u64),
max_output_tokens: None,
input_price: Some(0.27f64),
output_price: Some(1.1f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for deepseek/deepseek-r1-0528
pub fn get_deepseekdeepseekr10528_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "deepseek/deepseek-r1-0528".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: None,
input_price: Some(0.5f64),
output_price: Some(2.15f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for qwen/qwen-max
pub fn get_qwenqwenmax_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "qwen/qwen-max".to_string(),
max_input_tokens: Some(32768u64),
max_output_tokens: Some(8192u64),
input_price: Some(1.6f64),
output_price: Some(6.4f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for qwen/qwen-plus
pub fn get_qwenqwenplus_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "qwen/qwen-plus".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.4f64),
output_price: Some(1.2f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for qwen/qwen-turbo
pub fn get_qwenqwenturbo_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "qwen/qwen-turbo".to_string(),
max_input_tokens: Some(1000000u64),
max_output_tokens: Some(8192u64),
input_price: Some(0.05f64),
output_price: Some(0.2f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for qwen/qwen-vl-plus
pub fn get_qwenqwenvlplus_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "qwen/qwen-vl-plus".to_string(),
max_input_tokens: Some(7500u64),
max_output_tokens: None,
input_price: Some(0.21f64),
output_price: Some(0.63f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for qwen/qwen3-235b-a22b
pub fn get_qwenqwen3235ba22b_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "qwen/qwen3-235b-a22b".to_string(),
max_input_tokens: Some(40960u64),
max_output_tokens: None,
input_price: Some(0.15f64),
output_price: Some(0.6f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for qwen/qwen3-30b-a3b
pub fn get_qwenqwen330ba3b_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "qwen/qwen3-30b-a3b".to_string(),
max_input_tokens: Some(40960u64),
max_output_tokens: None,
input_price: Some(0.1f64),
output_price: Some(0.3f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for qwen/qwen-2.5-72b-instruct
pub fn get_qwenqwen2572binstruct_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "qwen/qwen-2.5-72b-instruct".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0.35f64),
output_price: Some(0.4f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for qwen/qwen2.5-vl-72b-instruct
pub fn get_qwenqwen25vl72binstruct_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "qwen/qwen2.5-vl-72b-instruct".to_string(),
max_input_tokens: Some(32000u64),
max_output_tokens: None,
input_price: Some(0.7f64),
output_price: Some(0.7f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for qwen/qwen-2.5-coder-32b-instruct
pub fn get_qwenqwen25coder32binstruct_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "qwen/qwen-2.5-coder-32b-instruct".to_string(),
max_input_tokens: Some(32768u64),
max_output_tokens: None,
input_price: Some(0.18f64),
output_price: Some(0.18f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for x-ai/grok-3
pub fn get_xaigrok3_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "x-ai/grok-3".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(3f64),
output_price: Some(15f64),
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for x-ai/grok-3-mini
pub fn get_xaigrok3mini_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "x-ai/grok-3-mini".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0.3f64),
output_price: Some(0.5f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for amazon/nova-pro-v1
pub fn get_amazonnovaprov1_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "amazon/nova-pro-v1".to_string(),
max_input_tokens: Some(300000u64),
max_output_tokens: Some(5120u64),
input_price: Some(0.8f64),
output_price: Some(3.2f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for amazon/nova-lite-v1
pub fn get_amazonnovalitev1_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "amazon/nova-lite-v1".to_string(),
max_input_tokens: Some(300000u64),
max_output_tokens: Some(5120u64),
input_price: Some(0.06f64),
output_price: Some(0.24f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for amazon/nova-micro-v1
pub fn get_amazonnovamicrov1_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "amazon/nova-micro-v1".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: Some(5120u64),
input_price: Some(0.035f64),
output_price: Some(0.14f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for perplexity/sonar-pro
pub fn get_perplexitysonarpro_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "perplexity/sonar-pro".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: None,
input_price: Some(3f64),
output_price: Some(15f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for perplexity/sonar
pub fn get_perplexitysonar_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "perplexity/sonar".to_string(),
max_input_tokens: Some(127072u64),
max_output_tokens: None,
input_price: Some(1f64),
output_price: Some(1f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for perplexity/sonar-reasoning-pro
pub fn get_perplexitysonarreasoningpro_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "perplexity/sonar-reasoning-pro".to_string(),
max_input_tokens: Some(128000u64),
max_output_tokens: None,
input_price: Some(2f64),
output_price: Some(8f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for perplexity/sonar-reasoning
pub fn get_perplexitysonarreasoning_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "perplexity/sonar-reasoning".to_string(),
max_input_tokens: Some(127000u64),
max_output_tokens: None,
input_price: Some(1f64),
output_price: Some(5f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for perplexity/sonar-deep-research
pub fn get_perplexitysonardeepresearch_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "perplexity/sonar-deep-research".to_string(),
max_input_tokens: Some(200000u64),
max_output_tokens: None,
input_price: Some(2f64),
output_price: Some(8f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for perplexity/r1-1776
pub fn get_perplexityr11776_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "perplexity/r1-1776".to_string(),
max_input_tokens: Some(127000u64),
max_output_tokens: None,
input_price: Some(2f64),
output_price: Some(8f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for minimax/minimax-01
pub fn get_minimaxminimax01_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "minimax/minimax-01".to_string(),
max_input_tokens: Some(1000192u64),
max_output_tokens: None,
input_price: Some(0.2f64),
output_price: Some(1.1f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for thudm/glm-4-32b
pub fn get_thudmglm432b_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "thudm/glm-4-32b".to_string(),
max_input_tokens: Some(32000u64),
max_output_tokens: None,
input_price: Some(0.24f64),
output_price: Some(0.24f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for thudm/glm-z1-32b
pub fn get_thudmglmz132b_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "thudm/glm-z1-32b".to_string(),
max_input_tokens: Some(32000u64),
max_output_tokens: None,
input_price: Some(0.24f64),
output_price: Some(0.24f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for microsoft/phi-4-reasoning-plus
pub fn get_microsoftphi4reasoningplus_info() -> ModelInfoData {
ModelInfoData {
provider_name: "openrouter".to_string(),
name: "microsoft/phi-4-reasoning-plus".to_string(),
max_input_tokens: Some(32768u64),
max_output_tokens: None,
input_price: Some(0.07f64),
output_price: Some(0.35f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for llama-4-maverick-17b-128e-instruct-fp8
pub fn get_llama4maverick17b128einstructfp8_info() -> ModelInfoData {
ModelInfoData {
provider_name: "github".to_string(),
name: "llama-4-maverick-17b-128e-instruct-fp8".to_string(),
max_input_tokens: Some(1048576u64),
max_output_tokens: None,
input_price: None,
output_price: None,
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for llama-4-scout-17b-16e-instruct
pub fn get_llama4scout17b16einstruct_info() -> ModelInfoData {
ModelInfoData {
provider_name: "github".to_string(),
name: "llama-4-scout-17b-16e-instruct".to_string(),
max_input_tokens: Some(327680u64),
max_output_tokens: None,
input_price: None,
output_price: None,
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for llama-3.3-70b-instruct
pub fn get_llama3370binstruct_info() -> ModelInfoData {
ModelInfoData {
provider_name: "github".to_string(),
name: "llama-3.3-70b-instruct".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: None,
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for mistral-medium-2505
pub fn get_mistralmedium2505_info() -> ModelInfoData {
ModelInfoData {
provider_name: "github".to_string(),
name: "mistral-medium-2505".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: None,
output_price: None,
supports_vision: None,
supports_function_calling: Some(true),
require_max_tokens: None,
}
}

/// Get model info for cohere-embed-v3-english
pub fn get_cohereembedv3english_info() -> ModelInfoData {
ModelInfoData {
provider_name: "github".to_string(),
name: "cohere-embed-v3-english".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: None,
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for cohere-embed-v3-multilingual
pub fn get_cohereembedv3multilingual_info() -> ModelInfoData {
ModelInfoData {
provider_name: "github".to_string(),
name: "cohere-embed-v3-multilingual".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: None,
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for deepseek-v3-0324
pub fn get_deepseekv30324_info() -> ModelInfoData {
ModelInfoData {
provider_name: "github".to_string(),
name: "deepseek-v3-0324".to_string(),
max_input_tokens: Some(163840u64),
max_output_tokens: None,
input_price: None,
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for mai-ds-r1
pub fn get_maidsr1_info() -> ModelInfoData {
ModelInfoData {
provider_name: "github".to_string(),
name: "mai-ds-r1".to_string(),
max_input_tokens: Some(163840u64),
max_output_tokens: None,
input_price: None,
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for phi-4
pub fn get_phi4_info() -> ModelInfoData {
ModelInfoData {
provider_name: "github".to_string(),
name: "phi-4".to_string(),
max_input_tokens: Some(16384u64),
max_output_tokens: None,
input_price: None,
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for phi-4-mini-instruct
pub fn get_phi4miniinstruct_info() -> ModelInfoData {
ModelInfoData {
provider_name: "github".to_string(),
name: "phi-4-mini-instruct".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: None,
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for phi-4-reasoning
pub fn get_phi4reasoning_info() -> ModelInfoData {
ModelInfoData {
provider_name: "github".to_string(),
name: "phi-4-reasoning".to_string(),
max_input_tokens: Some(33792u64),
max_output_tokens: None,
input_price: None,
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for phi-4-mini-reasoning
pub fn get_phi4minireasoning_info() -> ModelInfoData {
ModelInfoData {
provider_name: "github".to_string(),
name: "phi-4-mini-reasoning".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: None,
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for grok-3
pub fn get_grok3_info() -> ModelInfoData {
ModelInfoData {
provider_name: "github".to_string(),
name: "grok-3".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: None,
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for grok-3-mini
pub fn get_grok3mini_info() -> ModelInfoData {
ModelInfoData {
provider_name: "github".to_string(),
name: "grok-3-mini".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: None,
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
pub fn get_metallamallama4maverick17b128einstructfp8_info() -> ModelInfoData {
ModelInfoData {
provider_name: "deepinfra".to_string(),
name: "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8".to_string(),
max_input_tokens: Some(1048576u64),
max_output_tokens: None,
input_price: Some(0.18f64),
output_price: Some(0.6f64),
supports_vision: Some(true),
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for deepseek-ai/DeepSeek-V3-0324
pub fn get_deepseekaideepseekv30324_info() -> ModelInfoData {
ModelInfoData {
provider_name: "deepinfra".to_string(),
name: "deepseek-ai/DeepSeek-V3-0324".to_string(),
max_input_tokens: Some(163840u64),
max_output_tokens: None,
input_price: Some(0.4f64),
output_price: Some(0.89f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for deepseek-ai/DeepSeek-R1-0528
pub fn get_deepseekaideepseekr10528_info() -> ModelInfoData {
ModelInfoData {
provider_name: "deepinfra".to_string(),
name: "deepseek-ai/DeepSeek-R1-0528".to_string(),
max_input_tokens: Some(163840u64),
max_output_tokens: None,
input_price: Some(0.5f64),
output_price: Some(2.15f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for mistralai/Mistral-Small-3.2-24B-Instruct-2506
pub fn get_mistralaimistralsmall3224binstruct2506_info() -> ModelInfoData {
ModelInfoData {
provider_name: "deepinfra".to_string(),
name: "mistralai/Mistral-Small-3.2-24B-Instruct-2506".to_string(),
max_input_tokens: Some(32768u64),
max_output_tokens: None,
input_price: Some(0.06f64),
output_price: Some(0.12f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for mistralai/Devstral-Small-2505
pub fn get_mistralaidevstralsmall2505_info() -> ModelInfoData {
ModelInfoData {
provider_name: "deepinfra".to_string(),
name: "mistralai/Devstral-Small-2505".to_string(),
max_input_tokens: Some(131072u64),
max_output_tokens: None,
input_price: Some(0.06f64),
output_price: Some(0.12f64),
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for BAAI/bge-large-en-v1.5
pub fn get_baaibgelargeenv15_info() -> ModelInfoData {
ModelInfoData {
provider_name: "deepinfra".to_string(),
name: "BAAI/bge-large-en-v1.5".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0.01f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for BAAI/bge-m3
pub fn get_baaibgem3_info() -> ModelInfoData {
ModelInfoData {
provider_name: "deepinfra".to_string(),
name: "BAAI/bge-m3".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0.01f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for intfloat/e5-large-v2
pub fn get_intfloate5largev2_info() -> ModelInfoData {
ModelInfoData {
provider_name: "deepinfra".to_string(),
name: "intfloat/e5-large-v2".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0.01f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for intfloat/multilingual-e5-large
pub fn get_intfloatmultilinguale5large_info() -> ModelInfoData {
ModelInfoData {
provider_name: "deepinfra".to_string(),
name: "intfloat/multilingual-e5-large".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0.01f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for thenlper/gte-large
pub fn get_thenlpergtelarge_info() -> ModelInfoData {
ModelInfoData {
provider_name: "deepinfra".to_string(),
name: "thenlper/gte-large".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0.01f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for jina-embeddings-v3
pub fn get_jinaembeddingsv3_info() -> ModelInfoData {
ModelInfoData {
provider_name: "jina".to_string(),
name: "jina-embeddings-v3".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for jina-clip-v2
pub fn get_jinaclipv2_info() -> ModelInfoData {
ModelInfoData {
provider_name: "jina".to_string(),
name: "jina-clip-v2".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for jina-colbert-v2
pub fn get_jinacolbertv2_info() -> ModelInfoData {
ModelInfoData {
provider_name: "jina".to_string(),
name: "jina-colbert-v2".to_string(),
max_input_tokens: None,
max_output_tokens: None,
input_price: Some(0f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for jina-reranker-v2-base-multilingual
pub fn get_jinarerankerv2basemultilingual_info() -> ModelInfoData {
ModelInfoData {
provider_name: "jina".to_string(),
name: "jina-reranker-v2-base-multilingual".to_string(),
max_input_tokens: Some(8192u64),
max_output_tokens: None,
input_price: Some(0f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for voyage-3-large
pub fn get_voyage3large_info() -> ModelInfoData {
ModelInfoData {
provider_name: "voyageai".to_string(),
name: "voyage-3-large".to_string(),
max_input_tokens: Some(120000u64),
max_output_tokens: None,
input_price: Some(0.18f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for voyage-3
pub fn get_voyage3_info() -> ModelInfoData {
ModelInfoData {
provider_name: "voyageai".to_string(),
name: "voyage-3".to_string(),
max_input_tokens: Some(320000u64),
max_output_tokens: None,
input_price: Some(0.06f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for voyage-3-lite
pub fn get_voyage3lite_info() -> ModelInfoData {
ModelInfoData {
provider_name: "voyageai".to_string(),
name: "voyage-3-lite".to_string(),
max_input_tokens: Some(1000000u64),
max_output_tokens: None,
input_price: Some(0.02f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for rerank-2
pub fn get_rerank2_info() -> ModelInfoData {
ModelInfoData {
provider_name: "voyageai".to_string(),
name: "rerank-2".to_string(),
max_input_tokens: Some(16000u64),
max_output_tokens: None,
input_price: Some(0.05f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}

/// Get model info for rerank-2-lite
pub fn get_rerank2lite_info() -> ModelInfoData {
ModelInfoData {
provider_name: "voyageai".to_string(),
name: "rerank-2-lite".to_string(),
max_input_tokens: Some(8000u64),
max_output_tokens: None,
input_price: Some(0.02f64),
output_price: None,
supports_vision: None,
supports_function_calling: None,
require_max_tokens: None,
}
}
// AUTO-GENERATED END






