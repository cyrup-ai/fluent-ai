// This file is auto-generated. Do not edit manually.
use serde::{Serialize, Deserialize};

// AUTO-GENERATED START
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub supports_thinking: Option<bool>,
    pub optimal_thinking_budget: Option<u32>,
}

/// Get model info for gpt-4.1
pub fn get_gpt41_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openai".to_string(),
        name: "gpt-4.1".to_string(),
        max_input_tokens: Some(1047576),
        max_output_tokens: Some(32768),
        input_price: Some(2.0),
        output_price: Some(8.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for gpt-4.1-mini
pub fn get_gpt41mini_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openai".to_string(),
        name: "gpt-4.1-mini".to_string(),
        max_input_tokens: Some(1047576),
        max_output_tokens: Some(32768),
        input_price: Some(0.4),
        output_price: Some(1.6),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for gpt-4.1-nano
pub fn get_gpt41nano_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openai".to_string(),
        name: "gpt-4.1-nano".to_string(),
        max_input_tokens: Some(1047576),
        max_output_tokens: Some(32768),
        input_price: Some(0.1),
        output_price: Some(0.4),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for gpt-4o
pub fn get_gpt4o_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openai".to_string(),
        name: "gpt-4o".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: Some(16384),
        input_price: Some(2.5),
        output_price: Some(10.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for gpt-4o-search-preview
pub fn get_gpt4osearchpreview_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openai".to_string(),
        name: "gpt-4o-search-preview".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: Some(16384),
        input_price: Some(2.5),
        output_price: Some(10.0),
        supports_vision: Some(true),
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for gpt-4o-mini
pub fn get_gpt4omini_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openai".to_string(),
        name: "gpt-4o-mini".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: Some(16384),
        input_price: Some(0.15),
        output_price: Some(0.6),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for gpt-4o-mini-search-preview
pub fn get_gpt4ominisearchpreview_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openai".to_string(),
        name: "gpt-4o-mini-search-preview".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: Some(16384),
        input_price: Some(0.15),
        output_price: Some(0.6),
        supports_vision: Some(true),
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for chatgpt-4o-latest
pub fn get_chatgpt4olatest_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openai".to_string(),
        name: "chatgpt-4o-latest".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: Some(16384),
        input_price: Some(5.0),
        output_price: Some(15.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for o4-mini
pub fn get_o4mini_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openai".to_string(),
        name: "o4-mini".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: None,
        input_price: Some(1.1),
        output_price: Some(4.4),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for o4-mini-high
pub fn get_o4minihigh_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openai".to_string(),
        name: "o4-mini-high".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: None,
        input_price: Some(1.1),
        output_price: Some(4.4),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for o3
pub fn get_o3_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openai".to_string(),
        name: "o3".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: None,
        input_price: Some(10.0),
        output_price: Some(40.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for o3-mini
pub fn get_o3mini_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openai".to_string(),
        name: "o3-mini".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: None,
        input_price: Some(1.1),
        output_price: Some(4.4),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for o3-mini-high
pub fn get_o3minihigh_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openai".to_string(),
        name: "o3-mini-high".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: None,
        input_price: Some(1.1),
        output_price: Some(4.4),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for gpt-4-turbo
pub fn get_gpt4turbo_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openai".to_string(),
        name: "gpt-4-turbo".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: Some(4096),
        input_price: Some(10.0),
        output_price: Some(30.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for gpt-3.5-turbo
pub fn get_gpt35turbo_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openai".to_string(),
        name: "gpt-3.5-turbo".to_string(),
        max_input_tokens: Some(16385),
        max_output_tokens: Some(4096),
        input_price: Some(0.5),
        output_price: Some(1.5),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for text-embedding-3-large
pub fn get_textembedding3large_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openai".to_string(),
        name: "text-embedding-3-large".to_string(),
        max_input_tokens: None,
        max_output_tokens: None,
        input_price: Some(0.13),
        output_price: None,
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for text-embedding-3-small
pub fn get_textembedding3small_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openai".to_string(),
        name: "text-embedding-3-small".to_string(),
        max_input_tokens: None,
        max_output_tokens: None,
        input_price: Some(0.02),
        output_price: None,
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for gemini-2.5-flash
pub fn get_gemini25flash_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "gemini".to_string(),
        name: "gemini-2.5-flash".to_string(),
        max_input_tokens: Some(1048576),
        max_output_tokens: Some(65536),
        input_price: Some(0.0),
        output_price: Some(0.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for gemini-2.5-pro
pub fn get_gemini25pro_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "gemini".to_string(),
        name: "gemini-2.5-pro".to_string(),
        max_input_tokens: Some(1048576),
        max_output_tokens: Some(65536),
        input_price: Some(0.0),
        output_price: Some(0.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for gemini-2.5-flash-lite-preview-06-17
pub fn get_gemini25flashlitepreview0617_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "gemini".to_string(),
        name: "gemini-2.5-flash-lite-preview-06-17".to_string(),
        max_input_tokens: Some(1000000),
        max_output_tokens: Some(64000),
        input_price: Some(0.0),
        output_price: Some(0.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for gemini-2.0-flash
pub fn get_gemini20flash_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "gemini".to_string(),
        name: "gemini-2.0-flash".to_string(),
        max_input_tokens: Some(1048576),
        max_output_tokens: Some(8192),
        input_price: Some(0.0),
        output_price: Some(0.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for gemini-2.0-flash-lite
pub fn get_gemini20flashlite_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "gemini".to_string(),
        name: "gemini-2.0-flash-lite".to_string(),
        max_input_tokens: Some(1048576),
        max_output_tokens: Some(8192),
        input_price: Some(0.0),
        output_price: Some(0.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for gemma-3-27b-it
pub fn get_gemma327bit_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "gemini".to_string(),
        name: "gemma-3-27b-it".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: Some(8192),
        input_price: Some(0.0),
        output_price: Some(0.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for text-embedding-004
pub fn get_textembedding004_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "gemini".to_string(),
        name: "text-embedding-004".to_string(),
        max_input_tokens: None,
        max_output_tokens: None,
        input_price: Some(0.0),
        output_price: None,
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for claude-opus-4-20250514
pub fn get_claudeopus420250514_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "claude".to_string(),
        name: "claude-opus-4-20250514".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(8192),
        input_price: Some(15.0),
        output_price: Some(75.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for claude-opus-4-20250514:thinking
pub fn get_claudeopus420250514thinking_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "claude".to_string(),
        name: "claude-opus-4-20250514:thinking".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(24000),
        input_price: Some(15.0),
        output_price: Some(75.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(true),
        optimal_thinking_budget: Some(8192),
    }
}

/// Get model info for claude-sonnet-4-20250514
pub fn get_claudesonnet420250514_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "claude".to_string(),
        name: "claude-sonnet-4-20250514".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(8192),
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for claude-sonnet-4-20250514:thinking
pub fn get_claudesonnet420250514thinking_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "claude".to_string(),
        name: "claude-sonnet-4-20250514:thinking".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(24000),
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(true),
        optimal_thinking_budget: Some(8192),
    }
}

/// Get model info for claude-3-7-sonnet-20250219
pub fn get_claude37sonnet20250219_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "claude".to_string(),
        name: "claude-3-7-sonnet-20250219".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(8192),
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for claude-3-7-sonnet-20250219:thinking
pub fn get_claude37sonnet20250219thinking_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "claude".to_string(),
        name: "claude-3-7-sonnet-20250219:thinking".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(24000),
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: Some(true),
        supports_function_calling: None,
        require_max_tokens: Some(true),
        supports_thinking: Some(true),
        optimal_thinking_budget: Some(8192),
    }
}

/// Get model info for claude-3-5-sonnet-20241022
pub fn get_claude35sonnet20241022_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "claude".to_string(),
        name: "claude-3-5-sonnet-20241022".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(8192),
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for claude-3-5-haiku-20241022
pub fn get_claude35haiku20241022_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "claude".to_string(),
        name: "claude-3-5-haiku-20241022".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(8192),
        input_price: Some(0.8),
        output_price: Some(4.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for mistral-medium-latest
pub fn get_mistralmediumlatest_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "mistral".to_string(),
        name: "mistral-medium-latest".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(0.4),
        output_price: Some(2.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for mistral-small-latest
pub fn get_mistralsmalllatest_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "mistral".to_string(),
        name: "mistral-small-latest".to_string(),
        max_input_tokens: Some(32768),
        max_output_tokens: None,
        input_price: Some(0.1),
        output_price: Some(0.3),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for magistral-medium-latest
pub fn get_magistralmediumlatest_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "mistral".to_string(),
        name: "magistral-medium-latest".to_string(),
        max_input_tokens: Some(40960),
        max_output_tokens: None,
        input_price: Some(2.0),
        output_price: Some(5.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for magistral-small-latest
pub fn get_magistralsmalllatest_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "mistral".to_string(),
        name: "magistral-small-latest".to_string(),
        max_input_tokens: Some(40960),
        max_output_tokens: None,
        input_price: Some(0.5),
        output_price: Some(1.5),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for devstral-medium-latest
pub fn get_devstralmediumlatest_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "mistral".to_string(),
        name: "devstral-medium-latest".to_string(),
        max_input_tokens: Some(256000),
        max_output_tokens: None,
        input_price: Some(0.4),
        output_price: Some(2.0),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for devstral-small-latest
pub fn get_devstralsmalllatest_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "mistral".to_string(),
        name: "devstral-small-latest".to_string(),
        max_input_tokens: Some(256000),
        max_output_tokens: None,
        input_price: Some(0.1),
        output_price: Some(0.3),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for codestral-latest
pub fn get_codestrallatest_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "mistral".to_string(),
        name: "codestral-latest".to_string(),
        max_input_tokens: Some(256000),
        max_output_tokens: None,
        input_price: Some(0.3),
        output_price: Some(0.9),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for mistral-embed
pub fn get_mistralembed_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "mistral".to_string(),
        name: "mistral-embed".to_string(),
        max_input_tokens: Some(8092),
        max_output_tokens: None,
        input_price: Some(0.1),
        output_price: None,
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for jamba-large
pub fn get_jambalarge_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "ai21".to_string(),
        name: "jamba-large".to_string(),
        max_input_tokens: Some(256000),
        max_output_tokens: None,
        input_price: Some(2.0),
        output_price: Some(8.0),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for jamba-mini
pub fn get_jambamini_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "ai21".to_string(),
        name: "jamba-mini".to_string(),
        max_input_tokens: Some(256000),
        max_output_tokens: None,
        input_price: Some(0.2),
        output_price: Some(0.4),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for command-a-03-2025
pub fn get_commanda032025_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "cohere".to_string(),
        name: "command-a-03-2025".to_string(),
        max_input_tokens: Some(256000),
        max_output_tokens: Some(8192),
        input_price: Some(2.5),
        output_price: Some(10.0),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for command-r7b-12-2024
pub fn get_commandr7b122024_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "cohere".to_string(),
        name: "command-r7b-12-2024".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: Some(4096),
        input_price: Some(0.0375),
        output_price: Some(0.15),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for embed-v4.0
pub fn get_embedv40_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "cohere".to_string(),
        name: "embed-v4.0".to_string(),
        max_input_tokens: None,
        max_output_tokens: None,
        input_price: Some(0.12),
        output_price: None,
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for embed-english-v3.0
pub fn get_embedenglishv30_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "cohere".to_string(),
        name: "embed-english-v3.0".to_string(),
        max_input_tokens: None,
        max_output_tokens: None,
        input_price: Some(0.1),
        output_price: None,
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for embed-multilingual-v3.0
pub fn get_embedmultilingualv30_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "cohere".to_string(),
        name: "embed-multilingual-v3.0".to_string(),
        max_input_tokens: None,
        max_output_tokens: None,
        input_price: Some(0.1),
        output_price: None,
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for rerank-v3.5
pub fn get_rerankv35_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "cohere".to_string(),
        name: "rerank-v3.5".to_string(),
        max_input_tokens: Some(4096),
        max_output_tokens: None,
        input_price: None,
        output_price: None,
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for rerank-english-v3.0
pub fn get_rerankenglishv30_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "cohere".to_string(),
        name: "rerank-english-v3.0".to_string(),
        max_input_tokens: Some(4096),
        max_output_tokens: None,
        input_price: None,
        output_price: None,
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for rerank-multilingual-v3.0
pub fn get_rerankmultilingualv30_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "cohere".to_string(),
        name: "rerank-multilingual-v3.0".to_string(),
        max_input_tokens: Some(4096),
        max_output_tokens: None,
        input_price: None,
        output_price: None,
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for grok-3
pub fn get_grok3_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "xai".to_string(),
        name: "grok-3".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for grok-3-fast
pub fn get_grok3fast_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "xai".to_string(),
        name: "grok-3-fast".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(5.0),
        output_price: Some(25.0),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for grok-3-mini
pub fn get_grok3mini_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "xai".to_string(),
        name: "grok-3-mini".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(0.3),
        output_price: Some(0.5),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for grok-3-mini-fast
pub fn get_grok3minifast_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "xai".to_string(),
        name: "grok-3-mini-fast".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(0.6),
        output_price: Some(4.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for grok-4
pub fn get_grok4_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "xai".to_string(),
        name: "grok-4".to_string(),
        max_input_tokens: Some(256000),
        max_output_tokens: None,
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for sonar-pro
pub fn get_sonarpro_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "perplexity".to_string(),
        name: "sonar-pro".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: None,
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for sonar
pub fn get_sonar_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "perplexity".to_string(),
        name: "sonar".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: None,
        input_price: Some(1.0),
        output_price: Some(1.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for sonar-reasoning-pro
pub fn get_sonarreasoningpro_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "perplexity".to_string(),
        name: "sonar-reasoning-pro".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: None,
        input_price: Some(2.0),
        output_price: Some(8.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for sonar-reasoning
pub fn get_sonarreasoning_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "perplexity".to_string(),
        name: "sonar-reasoning".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: None,
        input_price: Some(1.0),
        output_price: Some(5.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for sonar-deep-research
pub fn get_sonardeepresearch_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "perplexity".to_string(),
        name: "sonar-deep-research".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: None,
        input_price: Some(2.0),
        output_price: Some(8.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for r1-1776
pub fn get_r11776_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "perplexity".to_string(),
        name: "r1-1776".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: None,
        input_price: Some(2.0),
        output_price: Some(8.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for meta-llama/llama-4-maverick-17b-128e-instruct
pub fn get_metallamallama4maverick17b128einstruct_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "groq".to_string(),
        name: "meta-llama/llama-4-maverick-17b-128e-instruct".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(0.0),
        output_price: Some(0.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for meta-llama/llama-4-scout-17b-16e-instruct
pub fn get_metallamallama4scout17b16einstruct_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "groq".to_string(),
        name: "meta-llama/llama-4-scout-17b-16e-instruct".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(0.0),
        output_price: Some(0.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for llama-3.3-70b-versatile
pub fn get_llama3370bversatile_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "groq".to_string(),
        name: "llama-3.3-70b-versatile".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(0.0),
        output_price: Some(0.0),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for qwen-qwq-32b
pub fn get_qwenqwq32b_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "groq".to_string(),
        name: "qwen-qwq-32b".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(0.0),
        output_price: Some(0.0),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for qwen/qwen3-32b
pub fn get_qwenqwen332b_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "groq".to_string(),
        name: "qwen/qwen3-32b".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(0.0),
        output_price: Some(0.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for compound-beta
pub fn get_compoundbeta_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "groq".to_string(),
        name: "compound-beta".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(0.0),
        output_price: Some(0.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for compound-beta-mini
pub fn get_compoundbetamini_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "groq".to_string(),
        name: "compound-beta-mini".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(0.0),
        output_price: Some(0.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for gemini-2.0-flash-001
pub fn get_gemini20flash001_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "vertexai".to_string(),
        name: "gemini-2.0-flash-001".to_string(),
        max_input_tokens: Some(1048576),
        max_output_tokens: Some(8192),
        input_price: Some(0.15),
        output_price: Some(0.6),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for gemini-2.0-flash-lite-001
pub fn get_gemini20flashlite001_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "vertexai".to_string(),
        name: "gemini-2.0-flash-lite-001".to_string(),
        max_input_tokens: Some(1048576),
        max_output_tokens: Some(8192),
        input_price: Some(0.075),
        output_price: Some(0.3),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for claude-3-5-sonnet-v2@20241022
pub fn get_claude35sonnetv220241022_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "vertexai".to_string(),
        name: "claude-3-5-sonnet-v2@20241022".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(8192),
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for mistral-small-2503
pub fn get_mistralsmall2503_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "vertexai".to_string(),
        name: "mistral-small-2503".to_string(),
        max_input_tokens: Some(32000),
        max_output_tokens: None,
        input_price: Some(0.1),
        output_price: Some(0.3),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for codestral-2501
pub fn get_codestral2501_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "vertexai".to_string(),
        name: "codestral-2501".to_string(),
        max_input_tokens: Some(256000),
        max_output_tokens: None,
        input_price: Some(0.3),
        output_price: Some(0.9),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for text-embedding-005
pub fn get_textembedding005_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "vertexai".to_string(),
        name: "text-embedding-005".to_string(),
        max_input_tokens: Some(20000),
        max_output_tokens: None,
        input_price: Some(0.025),
        output_price: None,
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for text-multilingual-embedding-002
pub fn get_textmultilingualembedding002_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "vertexai".to_string(),
        name: "text-multilingual-embedding-002".to_string(),
        max_input_tokens: Some(20000),
        max_output_tokens: None,
        input_price: Some(0.2),
        output_price: None,
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for us.anthropic.claude-opus-4-20250514-v1:0
pub fn get_usanthropicclaudeopus420250514v10_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "bedrock".to_string(),
        name: "us.anthropic.claude-opus-4-20250514-v1:0".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(8192),
        input_price: Some(15.0),
        output_price: Some(75.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for us.anthropic.claude-opus-4-20250514-v1:0:thinking
pub fn get_usanthropicclaudeopus420250514v10thinking_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "bedrock".to_string(),
        name: "us.anthropic.claude-opus-4-20250514-v1:0:thinking".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(24000),
        input_price: Some(15.0),
        output_price: Some(75.0),
        supports_vision: Some(true),
        supports_function_calling: None,
        require_max_tokens: Some(true),
        supports_thinking: Some(true),
        optimal_thinking_budget: Some(8192),
    }
}

/// Get model info for us.anthropic.claude-sonnet-4-20250514-v1:0
pub fn get_usanthropicclaudesonnet420250514v10_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "bedrock".to_string(),
        name: "us.anthropic.claude-sonnet-4-20250514-v1:0".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(8192),
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for us.anthropic.claude-sonnet-4-20250514-v1:0:thinking
pub fn get_usanthropicclaudesonnet420250514v10thinking_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "bedrock".to_string(),
        name: "us.anthropic.claude-sonnet-4-20250514-v1:0:thinking".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(24000),
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: Some(true),
        supports_function_calling: None,
        require_max_tokens: Some(true),
        supports_thinking: Some(true),
        optimal_thinking_budget: Some(8192),
    }
}

/// Get model info for us.anthropic.claude-3-7-sonnet-20250219-v1:0
pub fn get_usanthropicclaude37sonnet20250219v10_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "bedrock".to_string(),
        name: "us.anthropic.claude-3-7-sonnet-20250219-v1:0".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(8192),
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for us.anthropic.claude-3-7-sonnet-20250219-v1:0:thinking
pub fn get_usanthropicclaude37sonnet20250219v10thinking_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "bedrock".to_string(),
        name: "us.anthropic.claude-3-7-sonnet-20250219-v1:0:thinking".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(24000),
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: Some(true),
        supports_function_calling: None,
        require_max_tokens: Some(true),
        supports_thinking: Some(true),
        optimal_thinking_budget: Some(8192),
    }
}

/// Get model info for anthropic.claude-3-5-sonnet-20241022-v2:0
pub fn get_anthropicclaude35sonnet20241022v20_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "bedrock".to_string(),
        name: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(8192),
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for anthropic.claude-3-5-haiku-20241022-v1:0
pub fn get_anthropicclaude35haiku20241022v10_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "bedrock".to_string(),
        name: "anthropic.claude-3-5-haiku-20241022-v1:0".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(8192),
        input_price: Some(0.8),
        output_price: Some(4.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for us.meta.llama4-maverick-17b-instruct-v1:0
pub fn get_usmetallama4maverick17binstructv10_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "bedrock".to_string(),
        name: "us.meta.llama4-maverick-17b-instruct-v1:0".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: Some(8192),
        input_price: Some(0.24),
        output_price: Some(0.97),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for us.meta.llama4-scout-17b-instruct-v1:0
pub fn get_usmetallama4scout17binstructv10_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "bedrock".to_string(),
        name: "us.meta.llama4-scout-17b-instruct-v1:0".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: Some(8192),
        input_price: Some(0.17),
        output_price: Some(0.66),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for us.meta.llama3-3-70b-instruct-v1:0
pub fn get_usmetallama3370binstructv10_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "bedrock".to_string(),
        name: "us.meta.llama3-3-70b-instruct-v1:0".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: Some(8192),
        input_price: Some(0.72),
        output_price: Some(0.72),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for us.amazon.nova-premier-v1:0
pub fn get_usamazonnovapremierv10_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "bedrock".to_string(),
        name: "us.amazon.nova-premier-v1:0".to_string(),
        max_input_tokens: Some(300000),
        max_output_tokens: Some(5120),
        input_price: Some(2.5),
        output_price: Some(12.5),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for us.amazon.nova-pro-v1:0
pub fn get_usamazonnovaprov10_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "bedrock".to_string(),
        name: "us.amazon.nova-pro-v1:0".to_string(),
        max_input_tokens: Some(300000),
        max_output_tokens: Some(5120),
        input_price: Some(0.8),
        output_price: Some(3.2),
        supports_vision: Some(true),
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for us.amazon.nova-lite-v1:0
pub fn get_usamazonnovalitev10_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "bedrock".to_string(),
        name: "us.amazon.nova-lite-v1:0".to_string(),
        max_input_tokens: Some(300000),
        max_output_tokens: Some(5120),
        input_price: Some(0.06),
        output_price: Some(0.24),
        supports_vision: Some(true),
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for us.amazon.nova-micro-v1:0
pub fn get_usamazonnovamicrov10_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "bedrock".to_string(),
        name: "us.amazon.nova-micro-v1:0".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: Some(5120),
        input_price: Some(0.035),
        output_price: Some(0.14),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for cohere.embed-english-v3
pub fn get_cohereembedenglishv3_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "bedrock".to_string(),
        name: "cohere.embed-english-v3".to_string(),
        max_input_tokens: None,
        max_output_tokens: None,
        input_price: Some(0.1),
        output_price: None,
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for cohere.embed-multilingual-v3
pub fn get_cohereembedmultilingualv3_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "bedrock".to_string(),
        name: "cohere.embed-multilingual-v3".to_string(),
        max_input_tokens: None,
        max_output_tokens: None,
        input_price: Some(0.1),
        output_price: None,
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for us.deepseek.r1-v1:0
pub fn get_usdeepseekr1v10_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "bedrock".to_string(),
        name: "us.deepseek.r1-v1:0".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: None,
        input_price: Some(1.35),
        output_price: Some(5.4),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for deepseek-chat
pub fn get_deepseekchat_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "deepseek".to_string(),
        name: "deepseek-chat".to_string(),
        max_input_tokens: Some(64000),
        max_output_tokens: Some(8192),
        input_price: Some(0.27),
        output_price: Some(1.1),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for deepseek-reasoner
pub fn get_deepseekreasoner_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "deepseek".to_string(),
        name: "deepseek-reasoner".to_string(),
        max_input_tokens: Some(64000),
        max_output_tokens: Some(8192),
        input_price: Some(0.55),
        output_price: Some(2.19),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for openai/gpt-4.1
pub fn get_openaigpt41_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "openai/gpt-4.1".to_string(),
        max_input_tokens: Some(1047576),
        max_output_tokens: Some(32768),
        input_price: Some(2.0),
        output_price: Some(8.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for openai/gpt-4.1-mini
pub fn get_openaigpt41mini_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "openai/gpt-4.1-mini".to_string(),
        max_input_tokens: Some(1047576),
        max_output_tokens: Some(32768),
        input_price: Some(0.4),
        output_price: Some(1.6),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for openai/gpt-4.1-nano
pub fn get_openaigpt41nano_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "openai/gpt-4.1-nano".to_string(),
        max_input_tokens: Some(1047576),
        max_output_tokens: Some(32768),
        input_price: Some(0.1),
        output_price: Some(0.4),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for openai/gpt-4o
pub fn get_openaigpt4o_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "openai/gpt-4o".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: None,
        input_price: Some(2.5),
        output_price: Some(10.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for openai/gpt-4o-search-preview
pub fn get_openaigpt4osearchpreview_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "openai/gpt-4o-search-preview".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: Some(16384),
        input_price: Some(2.5),
        output_price: Some(10.0),
        supports_vision: Some(true),
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for openai/gpt-4o-mini
pub fn get_openaigpt4omini_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "openai/gpt-4o-mini".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: None,
        input_price: Some(0.15),
        output_price: Some(0.6),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for openai/gpt-4o-mini-search-preview
pub fn get_openaigpt4ominisearchpreview_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "openai/gpt-4o-mini-search-preview".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: Some(16384),
        input_price: Some(0.15),
        output_price: Some(0.6),
        supports_vision: Some(true),
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for openai/chatgpt-4o-latest
pub fn get_openaichatgpt4olatest_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "openai/chatgpt-4o-latest".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: None,
        input_price: Some(5.0),
        output_price: Some(15.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for openai/o4-mini
pub fn get_openaio4mini_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "openai/o4-mini".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: None,
        input_price: Some(1.1),
        output_price: Some(4.4),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for openai/o4-mini-high
pub fn get_openaio4minihigh_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "openai/o4-mini-high".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: None,
        input_price: Some(1.1),
        output_price: Some(4.4),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for openai/o3-pro
pub fn get_openaio3pro_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "openai/o3-pro".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: None,
        input_price: Some(20.0),
        output_price: Some(80.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for openai/o3
pub fn get_openaio3_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "openai/o3".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: None,
        input_price: Some(10.0),
        output_price: Some(40.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for openai/o3-mini
pub fn get_openaio3mini_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "openai/o3-mini".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: None,
        input_price: Some(1.1),
        output_price: Some(4.4),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for openai/o3-mini-high
pub fn get_openaio3minihigh_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "openai/o3-mini-high".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: None,
        input_price: Some(1.1),
        output_price: Some(4.4),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for google/gemini-2.5-flash
pub fn get_googlegemini25flash_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "google/gemini-2.5-flash".to_string(),
        max_input_tokens: Some(1048576),
        max_output_tokens: None,
        input_price: Some(0.15),
        output_price: Some(0.6),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for google/gemini-2.5-pro
pub fn get_googlegemini25pro_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "google/gemini-2.5-pro".to_string(),
        max_input_tokens: Some(1048576),
        max_output_tokens: None,
        input_price: Some(1.25),
        output_price: Some(10.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for google/gemini-2.5-flash-lite-preview-06-17
pub fn get_googlegemini25flashlitepreview0617_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "google/gemini-2.5-flash-lite-preview-06-17".to_string(),
        max_input_tokens: Some(1048576),
        max_output_tokens: None,
        input_price: Some(0.1),
        output_price: Some(0.4),
        supports_vision: Some(true),
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for google/gemini-2.0-flash-001
pub fn get_googlegemini20flash001_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "google/gemini-2.0-flash-001".to_string(),
        max_input_tokens: Some(1000000),
        max_output_tokens: None,
        input_price: Some(0.1),
        output_price: Some(0.4),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for google/gemini-2.0-flash-lite-001
pub fn get_googlegemini20flashlite001_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "google/gemini-2.0-flash-lite-001".to_string(),
        max_input_tokens: Some(1048576),
        max_output_tokens: None,
        input_price: Some(0.075),
        output_price: Some(0.3),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for google/gemma-3-27b-it
pub fn get_googlegemma327bit_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "google/gemma-3-27b-it".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(0.1),
        output_price: Some(0.2),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for anthropic/claude-opus-4
pub fn get_anthropicclaudeopus4_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "anthropic/claude-opus-4".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(8192),
        input_price: Some(15.0),
        output_price: Some(75.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for anthropic/claude-sonnet-4
pub fn get_anthropicclaudesonnet4_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "anthropic/claude-sonnet-4".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(8192),
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for anthropic/claude-3.7-sonnet
pub fn get_anthropicclaude37sonnet_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "anthropic/claude-3.7-sonnet".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(8192),
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for anthropic/claude-3.7-sonnet:thinking
pub fn get_anthropicclaude37sonnetthinking_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "anthropic/claude-3.7-sonnet:thinking".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(24000),
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: Some(true),
        supports_function_calling: None,
        require_max_tokens: Some(true),
        supports_thinking: Some(true),
        optimal_thinking_budget: Some(8192),
    }
}

/// Get model info for anthropic/claude-3.5-sonnet
pub fn get_anthropicclaude35sonnet_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "anthropic/claude-3.5-sonnet".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(8192),
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for anthropic/claude-3-5-haiku
pub fn get_anthropicclaude35haiku_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "anthropic/claude-3-5-haiku".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: Some(8192),
        input_price: Some(0.8),
        output_price: Some(4.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: Some(true),
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for meta-llama/llama-4-maverick
pub fn get_metallamallama4maverick_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "meta-llama/llama-4-maverick".to_string(),
        max_input_tokens: Some(1048576),
        max_output_tokens: None,
        input_price: Some(0.18),
        output_price: Some(0.6),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for meta-llama/llama-4-scout
pub fn get_metallamallama4scout_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "meta-llama/llama-4-scout".to_string(),
        max_input_tokens: Some(327680),
        max_output_tokens: None,
        input_price: Some(0.08),
        output_price: Some(0.3),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for meta-llama/llama-3.3-70b-instruct
pub fn get_metallamallama3370binstruct_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "meta-llama/llama-3.3-70b-instruct".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(0.12),
        output_price: Some(0.3),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for mistralai/mistral-medium-3
pub fn get_mistralaimistralmedium3_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "mistralai/mistral-medium-3".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(0.4),
        output_price: Some(2.0),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for mistralai/mistral-small-3.2-24b-instruct
pub fn get_mistralaimistralsmall3224binstruct_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "mistralai/mistral-small-3.2-24b-instruct".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(0.1),
        output_price: Some(0.3),
        supports_vision: Some(true),
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for mistralai/magistral-medium-2506
pub fn get_mistralaimagistralmedium2506_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "mistralai/magistral-medium-2506".to_string(),
        max_input_tokens: Some(40960),
        max_output_tokens: None,
        input_price: Some(2.0),
        output_price: Some(5.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for mistralai/magistral-medium-2506:thinking
pub fn get_mistralaimagistralmedium2506thinking_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "mistralai/magistral-medium-2506:thinking".to_string(),
        max_input_tokens: Some(40960),
        max_output_tokens: None,
        input_price: Some(2.0),
        output_price: Some(5.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(true),
        optimal_thinking_budget: Some(8192),
    }
}

/// Get model info for mistralai/magistral-small-2506
pub fn get_mistralaimagistralsmall2506_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "mistralai/magistral-small-2506".to_string(),
        max_input_tokens: Some(40960),
        max_output_tokens: None,
        input_price: Some(0.5),
        output_price: Some(1.5),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for mistralai/devstral-medium
pub fn get_mistralaidevstralmedium_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "mistralai/devstral-medium".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(0.4),
        output_price: Some(2.0),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for mistralai/devstral-small-2505
pub fn get_mistralaidevstralsmall2505_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "mistralai/devstral-small-2505".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(0.06),
        output_price: Some(0.12),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for mistralai/codestral-2501
pub fn get_mistralaicodestral2501_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "mistralai/codestral-2501".to_string(),
        max_input_tokens: Some(256000),
        max_output_tokens: None,
        input_price: Some(0.3),
        output_price: Some(0.9),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for ai21/jamba-1.6-large
pub fn get_ai21jamba16large_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "ai21/jamba-1.6-large".to_string(),
        max_input_tokens: Some(256000),
        max_output_tokens: None,
        input_price: Some(2.0),
        output_price: Some(8.0),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for ai21/jamba-1.6-mini
pub fn get_ai21jamba16mini_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "ai21/jamba-1.6-mini".to_string(),
        max_input_tokens: Some(256000),
        max_output_tokens: None,
        input_price: Some(0.2),
        output_price: Some(0.4),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for cohere/command-a
pub fn get_coherecommanda_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "cohere/command-a".to_string(),
        max_input_tokens: Some(256000),
        max_output_tokens: None,
        input_price: Some(2.5),
        output_price: Some(10.0),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for cohere/command-r7b-12-2024
pub fn get_coherecommandr7b122024_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "cohere/command-r7b-12-2024".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: Some(4096),
        input_price: Some(0.0375),
        output_price: Some(0.15),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for deepseek/deepseek-chat-v3-0324
pub fn get_deepseekdeepseekchatv30324_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "deepseek/deepseek-chat-v3-0324".to_string(),
        max_input_tokens: Some(64000),
        max_output_tokens: None,
        input_price: Some(0.27),
        output_price: Some(1.1),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for deepseek/deepseek-r1-0528
pub fn get_deepseekdeepseekr10528_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "deepseek/deepseek-r1-0528".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: None,
        input_price: Some(0.5),
        output_price: Some(2.15),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for qwen/qwen-max
pub fn get_qwenqwenmax_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "qwen/qwen-max".to_string(),
        max_input_tokens: Some(32768),
        max_output_tokens: Some(8192),
        input_price: Some(1.6),
        output_price: Some(6.4),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for qwen/qwen-plus
pub fn get_qwenqwenplus_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "qwen/qwen-plus".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: Some(8192),
        input_price: Some(0.4),
        output_price: Some(1.2),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for qwen/qwen-turbo
pub fn get_qwenqwenturbo_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "qwen/qwen-turbo".to_string(),
        max_input_tokens: Some(1000000),
        max_output_tokens: Some(8192),
        input_price: Some(0.05),
        output_price: Some(0.2),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for qwen/qwen-vl-plus
pub fn get_qwenqwenvlplus_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "qwen/qwen-vl-plus".to_string(),
        max_input_tokens: Some(7500),
        max_output_tokens: None,
        input_price: Some(0.21),
        output_price: Some(0.63),
        supports_vision: Some(true),
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for qwen/qwen3-235b-a22b
pub fn get_qwenqwen3235ba22b_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "qwen/qwen3-235b-a22b".to_string(),
        max_input_tokens: Some(40960),
        max_output_tokens: None,
        input_price: Some(0.15),
        output_price: Some(0.6),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for qwen/qwen3-30b-a3b
pub fn get_qwenqwen330ba3b_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "qwen/qwen3-30b-a3b".to_string(),
        max_input_tokens: Some(40960),
        max_output_tokens: None,
        input_price: Some(0.1),
        output_price: Some(0.3),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for qwen/qwen-2.5-72b-instruct
pub fn get_qwenqwen2572binstruct_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "qwen/qwen-2.5-72b-instruct".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(0.35),
        output_price: Some(0.4),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for qwen/qwen2.5-vl-72b-instruct
pub fn get_qwenqwen25vl72binstruct_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "qwen/qwen2.5-vl-72b-instruct".to_string(),
        max_input_tokens: Some(32000),
        max_output_tokens: None,
        input_price: Some(0.7),
        output_price: Some(0.7),
        supports_vision: Some(true),
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for qwen/qwen-2.5-coder-32b-instruct
pub fn get_qwenqwen25coder32binstruct_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "qwen/qwen-2.5-coder-32b-instruct".to_string(),
        max_input_tokens: Some(32768),
        max_output_tokens: None,
        input_price: Some(0.18),
        output_price: Some(0.18),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for moonshotai/kimi-k2
pub fn get_moonshotaikimik2_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "moonshotai/kimi-k2".to_string(),
        max_input_tokens: Some(63000),
        max_output_tokens: None,
        input_price: Some(0.14),
        output_price: Some(2.49),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for x-ai/grok-4
pub fn get_xaigrok4_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "x-ai/grok-4".to_string(),
        max_input_tokens: Some(256000),
        max_output_tokens: None,
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for x-ai/grok-3
pub fn get_xaigrok3_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "x-ai/grok-3".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: None,
        supports_function_calling: Some(true),
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for x-ai/grok-3-mini
pub fn get_xaigrok3mini_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "x-ai/grok-3-mini".to_string(),
        max_input_tokens: Some(131072),
        max_output_tokens: None,
        input_price: Some(0.3),
        output_price: Some(0.5),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for amazon/nova-pro-v1
pub fn get_amazonnovaprov1_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "amazon/nova-pro-v1".to_string(),
        max_input_tokens: Some(300000),
        max_output_tokens: Some(5120),
        input_price: Some(0.8),
        output_price: Some(3.2),
        supports_vision: Some(true),
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for amazon/nova-lite-v1
pub fn get_amazonnovalitev1_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "amazon/nova-lite-v1".to_string(),
        max_input_tokens: Some(300000),
        max_output_tokens: Some(5120),
        input_price: Some(0.06),
        output_price: Some(0.24),
        supports_vision: Some(true),
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for amazon/nova-micro-v1
pub fn get_amazonnovamicrov1_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "amazon/nova-micro-v1".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: Some(5120),
        input_price: Some(0.035),
        output_price: Some(0.14),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for perplexity/sonar-pro
pub fn get_perplexitysonarpro_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "perplexity/sonar-pro".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: None,
        input_price: Some(3.0),
        output_price: Some(15.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for perplexity/sonar
pub fn get_perplexitysonar_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "perplexity/sonar".to_string(),
        max_input_tokens: Some(127072),
        max_output_tokens: None,
        input_price: Some(1.0),
        output_price: Some(1.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for perplexity/sonar-reasoning-pro
pub fn get_perplexitysonarreasoningpro_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "perplexity/sonar-reasoning-pro".to_string(),
        max_input_tokens: Some(128000),
        max_output_tokens: None,
        input_price: Some(2.0),
        output_price: Some(8.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for perplexity/sonar-reasoning
pub fn get_perplexitysonarreasoning_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "perplexity/sonar-reasoning".to_string(),
        max_input_tokens: Some(127000),
        max_output_tokens: None,
        input_price: Some(1.0),
        output_price: Some(5.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for perplexity/sonar-deep-research
pub fn get_perplexitysonardeepresearch_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "perplexity/sonar-deep-research".to_string(),
        max_input_tokens: Some(200000),
        max_output_tokens: None,
        input_price: Some(2.0),
        output_price: Some(8.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for perplexity/r1-1776
pub fn get_perplexityr11776_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "perplexity/r1-1776".to_string(),
        max_input_tokens: Some(127000),
        max_output_tokens: None,
        input_price: Some(2.0),
        output_price: Some(8.0),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info for minimax/minimax-01
pub fn get_minimaxminimax01_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "openrouter".to_string(),
        name: "minimax/minimax-01".to_string(),
        max_input_tokens: Some(1000192),
        max_output_tokens: None,
        input_price: Some(0.2),
        output_price: Some(1.1),
        supports_vision: None,
        supports_function_calling: None,
        require_max_tokens: None,
        supports_thinking: Some(false),
        optimal_thinking_budget: Some(1024),
    }
}

/// Get model info by name - zero allocation lookup
pub fn get_model_info_by_name(name: &str) -> ModelInfoData {
    match name {
        "Gpt41" => get_gpt41_info(),
        "Gpt41Mini" => get_gpt41mini_info(),
        "Gpt41Nano" => get_gpt41nano_info(),
        "Gpt4O" => get_gpt4o_info(),
        "Gpt4OSearchPreview" => get_gpt4osearchpreview_info(),
        "Gpt4OMini" => get_gpt4omini_info(),
        "Gpt4OMiniSearchPreview" => get_gpt4ominisearchpreview_info(),
        "Chatgpt4OLatest" => get_chatgpt4olatest_info(),
        "O4Mini" => get_o4mini_info(),
        "O4MiniHigh" => get_o4minihigh_info(),
        "O3" => get_o3_info(),
        "O3Mini" => get_o3mini_info(),
        "O3MiniHigh" => get_o3minihigh_info(),
        "Gpt4Turbo" => get_gpt4turbo_info(),
        "Gpt35Turbo" => get_gpt35turbo_info(),
        "TextEmbedding3Large" => get_textembedding3large_info(),
        "TextEmbedding3Small" => get_textembedding3small_info(),
        "Gemini25Flash" => get_gemini25flash_info(),
        "Gemini25Pro" => get_gemini25pro_info(),
        "Gemini25FlashLitePreview0617" => get_gemini25flashlitepreview0617_info(),
        "Gemini20Flash" => get_gemini20flash_info(),
        "Gemini20FlashLite" => get_gemini20flashlite_info(),
        "Gemma327BIt" => get_gemma327bit_info(),
        "TextEmbedding004" => get_textembedding004_info(),
        "ClaudeOpus420250514" => get_claudeopus420250514_info(),
        "ClaudeOpus420250514Thinking" => get_claudeopus420250514thinking_info(),
        "ClaudeSonnet420250514" => get_claudesonnet420250514_info(),
        "ClaudeSonnet420250514Thinking" => get_claudesonnet420250514thinking_info(),
        "Claude37Sonnet20250219" => get_claude37sonnet20250219_info(),
        "Claude37Sonnet20250219Thinking" => get_claude37sonnet20250219thinking_info(),
        "Claude35Sonnet20241022" => get_claude35sonnet20241022_info(),
        "Claude35Haiku20241022" => get_claude35haiku20241022_info(),
        "MistralMediumLatest" => get_mistralmediumlatest_info(),
        "MistralSmallLatest" => get_mistralsmalllatest_info(),
        "MagistralMediumLatest" => get_magistralmediumlatest_info(),
        "MagistralSmallLatest" => get_magistralsmalllatest_info(),
        "DevstralMediumLatest" => get_devstralmediumlatest_info(),
        "DevstralSmallLatest" => get_devstralsmalllatest_info(),
        "CodestralLatest" => get_codestrallatest_info(),
        "MistralEmbed" => get_mistralembed_info(),
        "JambaLarge" => get_jambalarge_info(),
        "JambaMini" => get_jambamini_info(),
        "CommandA032025" => get_commanda032025_info(),
        "CommandR7b122024" => get_commandr7b122024_info(),
        "EmbedV40" => get_embedv40_info(),
        "EmbedEnglishV30" => get_embedenglishv30_info(),
        "EmbedMultilingualV30" => get_embedmultilingualv30_info(),
        "RerankV35" => get_rerankv35_info(),
        "RerankEnglishV30" => get_rerankenglishv30_info(),
        "RerankMultilingualV30" => get_rerankmultilingualv30_info(),
        "Grok3" => get_grok3_info(),
        "Grok3Fast" => get_grok3fast_info(),
        "Grok3Mini" => get_grok3mini_info(),
        "Grok3MiniFast" => get_grok3minifast_info(),
        "Grok4" => get_grok4_info(),
        "SonarPro" => get_sonarpro_info(),
        "Sonar" => get_sonar_info(),
        "SonarReasoningPro" => get_sonarreasoningpro_info(),
        "SonarReasoning" => get_sonarreasoning_info(),
        "SonarDeepResearch" => get_sonardeepresearch_info(),
        "R11776" => get_r11776_info(),
        "MetaLlamaLlama4Maverick17B128EInstruct" => get_metallamallama4maverick17b128einstruct_info(),
        "MetaLlamaLlama4Scout17B16EInstruct" => get_metallamallama4scout17b16einstruct_info(),
        "Llama3370BVersatile" => get_llama3370bversatile_info(),
        "QwenQwq32B" => get_qwenqwq32b_info(),
        "QwenQwen332B" => get_qwenqwen332b_info(),
        "CompoundBeta" => get_compoundbeta_info(),
        "CompoundBetaMini" => get_compoundbetamini_info(),
        "Gemini25Flash" => get_gemini25flash_info(),
        "Gemini25Pro" => get_gemini25pro_info(),
        "Gemini25FlashLitePreview0617" => get_gemini25flashlitepreview0617_info(),
        "Gemini20Flash001" => get_gemini20flash001_info(),
        "Gemini20FlashLite001" => get_gemini20flashlite001_info(),
        "ClaudeOpus420250514" => get_claudeopus420250514_info(),
        "ClaudeOpus420250514Thinking" => get_claudeopus420250514thinking_info(),
        "ClaudeSonnet420250514" => get_claudesonnet420250514_info(),
        "ClaudeSonnet420250514Thinking" => get_claudesonnet420250514thinking_info(),
        "Claude37Sonnet20250219" => get_claude37sonnet20250219_info(),
        "Claude37Sonnet20250219Thinking" => get_claude37sonnet20250219thinking_info(),
        "Claude35SonnetV220241022" => get_claude35sonnetv220241022_info(),
        "Claude35Haiku20241022" => get_claude35haiku20241022_info(),
        "MistralSmall2503" => get_mistralsmall2503_info(),
        "Codestral2501" => get_codestral2501_info(),
        "TextEmbedding005" => get_textembedding005_info(),
        "TextMultilingualEmbedding002" => get_textmultilingualembedding002_info(),
        "UsAnthropicClaudeOpus420250514V10" => get_usanthropicclaudeopus420250514v10_info(),
        "UsAnthropicClaudeOpus420250514V10thinking" => get_usanthropicclaudeopus420250514v10thinking_info(),
        "UsAnthropicClaudeSonnet420250514V10" => get_usanthropicclaudesonnet420250514v10_info(),
        "UsAnthropicClaudeSonnet420250514V10thinking" => get_usanthropicclaudesonnet420250514v10thinking_info(),
        "UsAnthropicClaude37Sonnet20250219V10" => get_usanthropicclaude37sonnet20250219v10_info(),
        "UsAnthropicClaude37Sonnet20250219V10thinking" => get_usanthropicclaude37sonnet20250219v10thinking_info(),
        "AnthropicClaude35Sonnet20241022V20" => get_anthropicclaude35sonnet20241022v20_info(),
        "AnthropicClaude35Haiku20241022V10" => get_anthropicclaude35haiku20241022v10_info(),
        "UsMetaLlama4Maverick17BInstructV10" => get_usmetallama4maverick17binstructv10_info(),
        "UsMetaLlama4Scout17BInstructV10" => get_usmetallama4scout17binstructv10_info(),
        "UsMetaLlama3370BInstructV10" => get_usmetallama3370binstructv10_info(),
        "UsAmazonNovaPremierV10" => get_usamazonnovapremierv10_info(),
        "UsAmazonNovaProV10" => get_usamazonnovaprov10_info(),
        "UsAmazonNovaLiteV10" => get_usamazonnovalitev10_info(),
        "UsAmazonNovaMicroV10" => get_usamazonnovamicrov10_info(),
        "CohereEmbedEnglishV3" => get_cohereembedenglishv3_info(),
        "CohereEmbedMultilingualV3" => get_cohereembedmultilingualv3_info(),
        "UsDeepseekR1V10" => get_usdeepseekr1v10_info(),
        "DeepseekChat" => get_deepseekchat_info(),
        "DeepseekReasoner" => get_deepseekreasoner_info(),
        "OpenaiGpt41" => get_openaigpt41_info(),
        "OpenaiGpt41Mini" => get_openaigpt41mini_info(),
        "OpenaiGpt41Nano" => get_openaigpt41nano_info(),
        "OpenaiGpt4O" => get_openaigpt4o_info(),
        "OpenaiGpt4OSearchPreview" => get_openaigpt4osearchpreview_info(),
        "OpenaiGpt4OMini" => get_openaigpt4omini_info(),
        "OpenaiGpt4OMiniSearchPreview" => get_openaigpt4ominisearchpreview_info(),
        "OpenaiChatgpt4OLatest" => get_openaichatgpt4olatest_info(),
        "OpenaiO4Mini" => get_openaio4mini_info(),
        "OpenaiO4MiniHigh" => get_openaio4minihigh_info(),
        "OpenaiO3Pro" => get_openaio3pro_info(),
        "OpenaiO3" => get_openaio3_info(),
        "OpenaiO3Mini" => get_openaio3mini_info(),
        "OpenaiO3MiniHigh" => get_openaio3minihigh_info(),
        "GoogleGemini25Flash" => get_googlegemini25flash_info(),
        "GoogleGemini25Pro" => get_googlegemini25pro_info(),
        "GoogleGemini25FlashLitePreview0617" => get_googlegemini25flashlitepreview0617_info(),
        "GoogleGemini20Flash001" => get_googlegemini20flash001_info(),
        "GoogleGemini20FlashLite001" => get_googlegemini20flashlite001_info(),
        "GoogleGemma327BIt" => get_googlegemma327bit_info(),
        "AnthropicClaudeOpus4" => get_anthropicclaudeopus4_info(),
        "AnthropicClaudeSonnet4" => get_anthropicclaudesonnet4_info(),
        "AnthropicClaude37Sonnet" => get_anthropicclaude37sonnet_info(),
        "AnthropicClaude37Sonnetthinking" => get_anthropicclaude37sonnetthinking_info(),
        "AnthropicClaude35Sonnet" => get_anthropicclaude35sonnet_info(),
        "AnthropicClaude35Haiku" => get_anthropicclaude35haiku_info(),
        "MetaLlamaLlama4Maverick" => get_metallamallama4maverick_info(),
        "MetaLlamaLlama4Scout" => get_metallamallama4scout_info(),
        "MetaLlamaLlama3370BInstruct" => get_metallamallama3370binstruct_info(),
        "MistralaiMistralMedium3" => get_mistralaimistralmedium3_info(),
        "MistralaiMistralSmall3224BInstruct" => get_mistralaimistralsmall3224binstruct_info(),
        "MistralaiMagistralMedium2506" => get_mistralaimagistralmedium2506_info(),
        "MistralaiMagistralMedium2506Thinking" => get_mistralaimagistralmedium2506thinking_info(),
        "MistralaiMagistralSmall2506" => get_mistralaimagistralsmall2506_info(),
        "MistralaiDevstralMedium" => get_mistralaidevstralmedium_info(),
        "MistralaiDevstralSmall2505" => get_mistralaidevstralsmall2505_info(),
        "MistralaiCodestral2501" => get_mistralaicodestral2501_info(),
        "Ai21Jamba16Large" => get_ai21jamba16large_info(),
        "Ai21Jamba16Mini" => get_ai21jamba16mini_info(),
        "CohereCommandA" => get_coherecommanda_info(),
        "CohereCommandR7b122024" => get_coherecommandr7b122024_info(),
        "DeepseekDeepseekChatV30324" => get_deepseekdeepseekchatv30324_info(),
        "DeepseekDeepseekR10528" => get_deepseekdeepseekr10528_info(),
        "QwenQwenMax" => get_qwenqwenmax_info(),
        "QwenQwenPlus" => get_qwenqwenplus_info(),
        "QwenQwenTurbo" => get_qwenqwenturbo_info(),
        "QwenQwenVlPlus" => get_qwenqwenvlplus_info(),
        "QwenQwen3235BA22b" => get_qwenqwen3235ba22b_info(),
        "QwenQwen330BA3b" => get_qwenqwen330ba3b_info(),
        "QwenQwen332B" => get_qwenqwen332b_info(),
        "QwenQwq32B" => get_qwenqwq32b_info(),
        "QwenQwen2572BInstruct" => get_qwenqwen2572binstruct_info(),
        "QwenQwen25Vl72BInstruct" => get_qwenqwen25vl72binstruct_info(),
        "QwenQwen25Coder32BInstruct" => get_qwenqwen25coder32binstruct_info(),
        "MoonshotaiKimiK2" => get_moonshotaikimik2_info(),
        "XAiGrok4" => get_xaigrok4_info(),
        "XAiGrok3" => get_xaigrok3_info(),
        "XAiGrok3Mini" => get_xaigrok3mini_info(),
        "AmazonNovaProV1" => get_amazonnovaprov1_info(),
        "AmazonNovaLiteV1" => get_amazonnovalitev1_info(),
        "AmazonNovaMicroV1" => get_amazonnovamicrov1_info(),
        "PerplexitySonarPro" => get_perplexitysonarpro_info(),
        "PerplexitySonar" => get_perplexitysonar_info(),
        "PerplexitySonarReasoningPro" => get_perplexitysonarreasoningpro_info(),
        "PerplexitySonarReasoning" => get_perplexitysonarreasoning_info(),
        "PerplexitySonarDeepResearch" => get_perplexitysonardeepresearch_info(),
        "PerplexityR11776" => get_perplexityr11776_info(),
        "MinimaxMinimax01" => get_minimaxminimax01_info(),
        _ => ModelInfoData::default(),
    }
}

impl Default for ModelInfoData {
    fn default() -> Self {
        Self {
            provider_name: String::new(),
            name: String::new(),
            max_input_tokens: None,
            max_output_tokens: None,
            input_price: None,
            output_price: None,
            supports_vision: None,
            supports_function_calling: None,
            require_max_tokens: None,
            supports_thinking: None,
            optimal_thinking_budget: None,
        }
    }
}

/// Convert ModelInfoData to ModelConfig - zero allocation lookup
pub fn model_info_to_config(info: &ModelInfoData, model_name: &'static str) -> crate::completion_provider::ModelConfig {
    crate::completion_provider::ModelConfig {
        max_tokens: info.max_output_tokens.unwrap_or(4096) as u32,
        temperature: 0.7,
        top_p: 0.9,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: info.max_input_tokens.unwrap_or(128000) as u32,
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: info.supports_function_calling.unwrap_or(false),
        supports_vision: info.supports_vision.unwrap_or(false),
        supports_audio: false,
        supports_thinking: info.supports_thinking.unwrap_or(false),
        optimal_thinking_budget: info.optimal_thinking_budget.unwrap_or(1024),
        provider: Box::leak(info.provider_name.clone().into_boxed_str()),
        model_name,
    }
}

use std::sync::OnceLock;
use std::collections::HashMap;

/// Zero-allocation caching for model configs
static MODEL_CONFIG_CACHE: OnceLock<HashMap<&'static str, crate::completion_provider::ModelConfig>> = OnceLock::new();

/// Get model configuration with zero-allocation caching
pub fn get_model_config(model_name: &'static str) -> &'static crate::completion_provider::ModelConfig {
    let cache = MODEL_CONFIG_CACHE.get_or_init(|| {
        let mut map = HashMap::new();
        let info = get_gpt41_info();
        let config = model_info_to_config(&info, "Gpt41");
        map.insert("Gpt41", config);
        let info = get_gpt41mini_info();
        let config = model_info_to_config(&info, "Gpt41Mini");
        map.insert("Gpt41Mini", config);
        let info = get_gpt41nano_info();
        let config = model_info_to_config(&info, "Gpt41Nano");
        map.insert("Gpt41Nano", config);
        let info = get_gpt4o_info();
        let config = model_info_to_config(&info, "Gpt4O");
        map.insert("Gpt4O", config);
        let info = get_gpt4osearchpreview_info();
        let config = model_info_to_config(&info, "Gpt4OSearchPreview");
        map.insert("Gpt4OSearchPreview", config);
        let info = get_gpt4omini_info();
        let config = model_info_to_config(&info, "Gpt4OMini");
        map.insert("Gpt4OMini", config);
        let info = get_gpt4ominisearchpreview_info();
        let config = model_info_to_config(&info, "Gpt4OMiniSearchPreview");
        map.insert("Gpt4OMiniSearchPreview", config);
        let info = get_chatgpt4olatest_info();
        let config = model_info_to_config(&info, "Chatgpt4OLatest");
        map.insert("Chatgpt4OLatest", config);
        let info = get_o4mini_info();
        let config = model_info_to_config(&info, "O4Mini");
        map.insert("O4Mini", config);
        let info = get_o4minihigh_info();
        let config = model_info_to_config(&info, "O4MiniHigh");
        map.insert("O4MiniHigh", config);
        let info = get_o3_info();
        let config = model_info_to_config(&info, "O3");
        map.insert("O3", config);
        let info = get_o3mini_info();
        let config = model_info_to_config(&info, "O3Mini");
        map.insert("O3Mini", config);
        let info = get_o3minihigh_info();
        let config = model_info_to_config(&info, "O3MiniHigh");
        map.insert("O3MiniHigh", config);
        let info = get_gpt4turbo_info();
        let config = model_info_to_config(&info, "Gpt4Turbo");
        map.insert("Gpt4Turbo", config);
        let info = get_gpt35turbo_info();
        let config = model_info_to_config(&info, "Gpt35Turbo");
        map.insert("Gpt35Turbo", config);
        let info = get_textembedding3large_info();
        let config = model_info_to_config(&info, "TextEmbedding3Large");
        map.insert("TextEmbedding3Large", config);
        let info = get_textembedding3small_info();
        let config = model_info_to_config(&info, "TextEmbedding3Small");
        map.insert("TextEmbedding3Small", config);
        let info = get_gemini25flash_info();
        let config = model_info_to_config(&info, "Gemini25Flash");
        map.insert("Gemini25Flash", config);
        let info = get_gemini25pro_info();
        let config = model_info_to_config(&info, "Gemini25Pro");
        map.insert("Gemini25Pro", config);
        let info = get_gemini25flashlitepreview0617_info();
        let config = model_info_to_config(&info, "Gemini25FlashLitePreview0617");
        map.insert("Gemini25FlashLitePreview0617", config);
        let info = get_gemini20flash_info();
        let config = model_info_to_config(&info, "Gemini20Flash");
        map.insert("Gemini20Flash", config);
        let info = get_gemini20flashlite_info();
        let config = model_info_to_config(&info, "Gemini20FlashLite");
        map.insert("Gemini20FlashLite", config);
        let info = get_gemma327bit_info();
        let config = model_info_to_config(&info, "Gemma327BIt");
        map.insert("Gemma327BIt", config);
        let info = get_textembedding004_info();
        let config = model_info_to_config(&info, "TextEmbedding004");
        map.insert("TextEmbedding004", config);
        let info = get_claudeopus420250514_info();
        let config = model_info_to_config(&info, "ClaudeOpus420250514");
        map.insert("ClaudeOpus420250514", config);
        let info = get_claudeopus420250514thinking_info();
        let config = model_info_to_config(&info, "ClaudeOpus420250514Thinking");
        map.insert("ClaudeOpus420250514Thinking", config);
        let info = get_claudesonnet420250514_info();
        let config = model_info_to_config(&info, "ClaudeSonnet420250514");
        map.insert("ClaudeSonnet420250514", config);
        let info = get_claudesonnet420250514thinking_info();
        let config = model_info_to_config(&info, "ClaudeSonnet420250514Thinking");
        map.insert("ClaudeSonnet420250514Thinking", config);
        let info = get_claude37sonnet20250219_info();
        let config = model_info_to_config(&info, "Claude37Sonnet20250219");
        map.insert("Claude37Sonnet20250219", config);
        let info = get_claude37sonnet20250219thinking_info();
        let config = model_info_to_config(&info, "Claude37Sonnet20250219Thinking");
        map.insert("Claude37Sonnet20250219Thinking", config);
        let info = get_claude35sonnet20241022_info();
        let config = model_info_to_config(&info, "Claude35Sonnet20241022");
        map.insert("Claude35Sonnet20241022", config);
        let info = get_claude35haiku20241022_info();
        let config = model_info_to_config(&info, "Claude35Haiku20241022");
        map.insert("Claude35Haiku20241022", config);
        let info = get_mistralmediumlatest_info();
        let config = model_info_to_config(&info, "MistralMediumLatest");
        map.insert("MistralMediumLatest", config);
        let info = get_mistralsmalllatest_info();
        let config = model_info_to_config(&info, "MistralSmallLatest");
        map.insert("MistralSmallLatest", config);
        let info = get_magistralmediumlatest_info();
        let config = model_info_to_config(&info, "MagistralMediumLatest");
        map.insert("MagistralMediumLatest", config);
        let info = get_magistralsmalllatest_info();
        let config = model_info_to_config(&info, "MagistralSmallLatest");
        map.insert("MagistralSmallLatest", config);
        let info = get_devstralmediumlatest_info();
        let config = model_info_to_config(&info, "DevstralMediumLatest");
        map.insert("DevstralMediumLatest", config);
        let info = get_devstralsmalllatest_info();
        let config = model_info_to_config(&info, "DevstralSmallLatest");
        map.insert("DevstralSmallLatest", config);
        let info = get_codestrallatest_info();
        let config = model_info_to_config(&info, "CodestralLatest");
        map.insert("CodestralLatest", config);
        let info = get_mistralembed_info();
        let config = model_info_to_config(&info, "MistralEmbed");
        map.insert("MistralEmbed", config);
        let info = get_jambalarge_info();
        let config = model_info_to_config(&info, "JambaLarge");
        map.insert("JambaLarge", config);
        let info = get_jambamini_info();
        let config = model_info_to_config(&info, "JambaMini");
        map.insert("JambaMini", config);
        let info = get_commanda032025_info();
        let config = model_info_to_config(&info, "CommandA032025");
        map.insert("CommandA032025", config);
        let info = get_commandr7b122024_info();
        let config = model_info_to_config(&info, "CommandR7b122024");
        map.insert("CommandR7b122024", config);
        let info = get_embedv40_info();
        let config = model_info_to_config(&info, "EmbedV40");
        map.insert("EmbedV40", config);
        let info = get_embedenglishv30_info();
        let config = model_info_to_config(&info, "EmbedEnglishV30");
        map.insert("EmbedEnglishV30", config);
        let info = get_embedmultilingualv30_info();
        let config = model_info_to_config(&info, "EmbedMultilingualV30");
        map.insert("EmbedMultilingualV30", config);
        let info = get_rerankv35_info();
        let config = model_info_to_config(&info, "RerankV35");
        map.insert("RerankV35", config);
        let info = get_rerankenglishv30_info();
        let config = model_info_to_config(&info, "RerankEnglishV30");
        map.insert("RerankEnglishV30", config);
        let info = get_rerankmultilingualv30_info();
        let config = model_info_to_config(&info, "RerankMultilingualV30");
        map.insert("RerankMultilingualV30", config);
        let info = get_grok3_info();
        let config = model_info_to_config(&info, "Grok3");
        map.insert("Grok3", config);
        let info = get_grok3fast_info();
        let config = model_info_to_config(&info, "Grok3Fast");
        map.insert("Grok3Fast", config);
        let info = get_grok3mini_info();
        let config = model_info_to_config(&info, "Grok3Mini");
        map.insert("Grok3Mini", config);
        let info = get_grok3minifast_info();
        let config = model_info_to_config(&info, "Grok3MiniFast");
        map.insert("Grok3MiniFast", config);
        let info = get_grok4_info();
        let config = model_info_to_config(&info, "Grok4");
        map.insert("Grok4", config);
        let info = get_sonarpro_info();
        let config = model_info_to_config(&info, "SonarPro");
        map.insert("SonarPro", config);
        let info = get_sonar_info();
        let config = model_info_to_config(&info, "Sonar");
        map.insert("Sonar", config);
        let info = get_sonarreasoningpro_info();
        let config = model_info_to_config(&info, "SonarReasoningPro");
        map.insert("SonarReasoningPro", config);
        let info = get_sonarreasoning_info();
        let config = model_info_to_config(&info, "SonarReasoning");
        map.insert("SonarReasoning", config);
        let info = get_sonardeepresearch_info();
        let config = model_info_to_config(&info, "SonarDeepResearch");
        map.insert("SonarDeepResearch", config);
        let info = get_r11776_info();
        let config = model_info_to_config(&info, "R11776");
        map.insert("R11776", config);
        let info = get_metallamallama4maverick17b128einstruct_info();
        let config = model_info_to_config(&info, "MetaLlamaLlama4Maverick17B128EInstruct");
        map.insert("MetaLlamaLlama4Maverick17B128EInstruct", config);
        let info = get_metallamallama4scout17b16einstruct_info();
        let config = model_info_to_config(&info, "MetaLlamaLlama4Scout17B16EInstruct");
        map.insert("MetaLlamaLlama4Scout17B16EInstruct", config);
        let info = get_llama3370bversatile_info();
        let config = model_info_to_config(&info, "Llama3370BVersatile");
        map.insert("Llama3370BVersatile", config);
        let info = get_qwenqwq32b_info();
        let config = model_info_to_config(&info, "QwenQwq32B");
        map.insert("QwenQwq32B", config);
        let info = get_qwenqwen332b_info();
        let config = model_info_to_config(&info, "QwenQwen332B");
        map.insert("QwenQwen332B", config);
        let info = get_compoundbeta_info();
        let config = model_info_to_config(&info, "CompoundBeta");
        map.insert("CompoundBeta", config);
        let info = get_compoundbetamini_info();
        let config = model_info_to_config(&info, "CompoundBetaMini");
        map.insert("CompoundBetaMini", config);
        let info = get_gemini25flash_info();
        let config = model_info_to_config(&info, "Gemini25Flash");
        map.insert("Gemini25Flash", config);
        let info = get_gemini25pro_info();
        let config = model_info_to_config(&info, "Gemini25Pro");
        map.insert("Gemini25Pro", config);
        let info = get_gemini25flashlitepreview0617_info();
        let config = model_info_to_config(&info, "Gemini25FlashLitePreview0617");
        map.insert("Gemini25FlashLitePreview0617", config);
        let info = get_gemini20flash001_info();
        let config = model_info_to_config(&info, "Gemini20Flash001");
        map.insert("Gemini20Flash001", config);
        let info = get_gemini20flashlite001_info();
        let config = model_info_to_config(&info, "Gemini20FlashLite001");
        map.insert("Gemini20FlashLite001", config);
        let info = get_claudeopus420250514_info();
        let config = model_info_to_config(&info, "ClaudeOpus420250514");
        map.insert("ClaudeOpus420250514", config);
        let info = get_claudeopus420250514thinking_info();
        let config = model_info_to_config(&info, "ClaudeOpus420250514Thinking");
        map.insert("ClaudeOpus420250514Thinking", config);
        let info = get_claudesonnet420250514_info();
        let config = model_info_to_config(&info, "ClaudeSonnet420250514");
        map.insert("ClaudeSonnet420250514", config);
        let info = get_claudesonnet420250514thinking_info();
        let config = model_info_to_config(&info, "ClaudeSonnet420250514Thinking");
        map.insert("ClaudeSonnet420250514Thinking", config);
        let info = get_claude37sonnet20250219_info();
        let config = model_info_to_config(&info, "Claude37Sonnet20250219");
        map.insert("Claude37Sonnet20250219", config);
        let info = get_claude37sonnet20250219thinking_info();
        let config = model_info_to_config(&info, "Claude37Sonnet20250219Thinking");
        map.insert("Claude37Sonnet20250219Thinking", config);
        let info = get_claude35sonnetv220241022_info();
        let config = model_info_to_config(&info, "Claude35SonnetV220241022");
        map.insert("Claude35SonnetV220241022", config);
        let info = get_claude35haiku20241022_info();
        let config = model_info_to_config(&info, "Claude35Haiku20241022");
        map.insert("Claude35Haiku20241022", config);
        let info = get_mistralsmall2503_info();
        let config = model_info_to_config(&info, "MistralSmall2503");
        map.insert("MistralSmall2503", config);
        let info = get_codestral2501_info();
        let config = model_info_to_config(&info, "Codestral2501");
        map.insert("Codestral2501", config);
        let info = get_textembedding005_info();
        let config = model_info_to_config(&info, "TextEmbedding005");
        map.insert("TextEmbedding005", config);
        let info = get_textmultilingualembedding002_info();
        let config = model_info_to_config(&info, "TextMultilingualEmbedding002");
        map.insert("TextMultilingualEmbedding002", config);
        let info = get_usanthropicclaudeopus420250514v10_info();
        let config = model_info_to_config(&info, "UsAnthropicClaudeOpus420250514V10");
        map.insert("UsAnthropicClaudeOpus420250514V10", config);
        let info = get_usanthropicclaudeopus420250514v10thinking_info();
        let config = model_info_to_config(&info, "UsAnthropicClaudeOpus420250514V10thinking");
        map.insert("UsAnthropicClaudeOpus420250514V10thinking", config);
        let info = get_usanthropicclaudesonnet420250514v10_info();
        let config = model_info_to_config(&info, "UsAnthropicClaudeSonnet420250514V10");
        map.insert("UsAnthropicClaudeSonnet420250514V10", config);
        let info = get_usanthropicclaudesonnet420250514v10thinking_info();
        let config = model_info_to_config(&info, "UsAnthropicClaudeSonnet420250514V10thinking");
        map.insert("UsAnthropicClaudeSonnet420250514V10thinking", config);
        let info = get_usanthropicclaude37sonnet20250219v10_info();
        let config = model_info_to_config(&info, "UsAnthropicClaude37Sonnet20250219V10");
        map.insert("UsAnthropicClaude37Sonnet20250219V10", config);
        let info = get_usanthropicclaude37sonnet20250219v10thinking_info();
        let config = model_info_to_config(&info, "UsAnthropicClaude37Sonnet20250219V10thinking");
        map.insert("UsAnthropicClaude37Sonnet20250219V10thinking", config);
        let info = get_anthropicclaude35sonnet20241022v20_info();
        let config = model_info_to_config(&info, "AnthropicClaude35Sonnet20241022V20");
        map.insert("AnthropicClaude35Sonnet20241022V20", config);
        let info = get_anthropicclaude35haiku20241022v10_info();
        let config = model_info_to_config(&info, "AnthropicClaude35Haiku20241022V10");
        map.insert("AnthropicClaude35Haiku20241022V10", config);
        let info = get_usmetallama4maverick17binstructv10_info();
        let config = model_info_to_config(&info, "UsMetaLlama4Maverick17BInstructV10");
        map.insert("UsMetaLlama4Maverick17BInstructV10", config);
        let info = get_usmetallama4scout17binstructv10_info();
        let config = model_info_to_config(&info, "UsMetaLlama4Scout17BInstructV10");
        map.insert("UsMetaLlama4Scout17BInstructV10", config);
        let info = get_usmetallama3370binstructv10_info();
        let config = model_info_to_config(&info, "UsMetaLlama3370BInstructV10");
        map.insert("UsMetaLlama3370BInstructV10", config);
        let info = get_usamazonnovapremierv10_info();
        let config = model_info_to_config(&info, "UsAmazonNovaPremierV10");
        map.insert("UsAmazonNovaPremierV10", config);
        let info = get_usamazonnovaprov10_info();
        let config = model_info_to_config(&info, "UsAmazonNovaProV10");
        map.insert("UsAmazonNovaProV10", config);
        let info = get_usamazonnovalitev10_info();
        let config = model_info_to_config(&info, "UsAmazonNovaLiteV10");
        map.insert("UsAmazonNovaLiteV10", config);
        let info = get_usamazonnovamicrov10_info();
        let config = model_info_to_config(&info, "UsAmazonNovaMicroV10");
        map.insert("UsAmazonNovaMicroV10", config);
        let info = get_cohereembedenglishv3_info();
        let config = model_info_to_config(&info, "CohereEmbedEnglishV3");
        map.insert("CohereEmbedEnglishV3", config);
        let info = get_cohereembedmultilingualv3_info();
        let config = model_info_to_config(&info, "CohereEmbedMultilingualV3");
        map.insert("CohereEmbedMultilingualV3", config);
        let info = get_usdeepseekr1v10_info();
        let config = model_info_to_config(&info, "UsDeepseekR1V10");
        map.insert("UsDeepseekR1V10", config);
        let info = get_deepseekchat_info();
        let config = model_info_to_config(&info, "DeepseekChat");
        map.insert("DeepseekChat", config);
        let info = get_deepseekreasoner_info();
        let config = model_info_to_config(&info, "DeepseekReasoner");
        map.insert("DeepseekReasoner", config);
        let info = get_openaigpt41_info();
        let config = model_info_to_config(&info, "OpenaiGpt41");
        map.insert("OpenaiGpt41", config);
        let info = get_openaigpt41mini_info();
        let config = model_info_to_config(&info, "OpenaiGpt41Mini");
        map.insert("OpenaiGpt41Mini", config);
        let info = get_openaigpt41nano_info();
        let config = model_info_to_config(&info, "OpenaiGpt41Nano");
        map.insert("OpenaiGpt41Nano", config);
        let info = get_openaigpt4o_info();
        let config = model_info_to_config(&info, "OpenaiGpt4O");
        map.insert("OpenaiGpt4O", config);
        let info = get_openaigpt4osearchpreview_info();
        let config = model_info_to_config(&info, "OpenaiGpt4OSearchPreview");
        map.insert("OpenaiGpt4OSearchPreview", config);
        let info = get_openaigpt4omini_info();
        let config = model_info_to_config(&info, "OpenaiGpt4OMini");
        map.insert("OpenaiGpt4OMini", config);
        let info = get_openaigpt4ominisearchpreview_info();
        let config = model_info_to_config(&info, "OpenaiGpt4OMiniSearchPreview");
        map.insert("OpenaiGpt4OMiniSearchPreview", config);
        let info = get_openaichatgpt4olatest_info();
        let config = model_info_to_config(&info, "OpenaiChatgpt4OLatest");
        map.insert("OpenaiChatgpt4OLatest", config);
        let info = get_openaio4mini_info();
        let config = model_info_to_config(&info, "OpenaiO4Mini");
        map.insert("OpenaiO4Mini", config);
        let info = get_openaio4minihigh_info();
        let config = model_info_to_config(&info, "OpenaiO4MiniHigh");
        map.insert("OpenaiO4MiniHigh", config);
        let info = get_openaio3pro_info();
        let config = model_info_to_config(&info, "OpenaiO3Pro");
        map.insert("OpenaiO3Pro", config);
        let info = get_openaio3_info();
        let config = model_info_to_config(&info, "OpenaiO3");
        map.insert("OpenaiO3", config);
        let info = get_openaio3mini_info();
        let config = model_info_to_config(&info, "OpenaiO3Mini");
        map.insert("OpenaiO3Mini", config);
        let info = get_openaio3minihigh_info();
        let config = model_info_to_config(&info, "OpenaiO3MiniHigh");
        map.insert("OpenaiO3MiniHigh", config);
        let info = get_googlegemini25flash_info();
        let config = model_info_to_config(&info, "GoogleGemini25Flash");
        map.insert("GoogleGemini25Flash", config);
        let info = get_googlegemini25pro_info();
        let config = model_info_to_config(&info, "GoogleGemini25Pro");
        map.insert("GoogleGemini25Pro", config);
        let info = get_googlegemini25flashlitepreview0617_info();
        let config = model_info_to_config(&info, "GoogleGemini25FlashLitePreview0617");
        map.insert("GoogleGemini25FlashLitePreview0617", config);
        let info = get_googlegemini20flash001_info();
        let config = model_info_to_config(&info, "GoogleGemini20Flash001");
        map.insert("GoogleGemini20Flash001", config);
        let info = get_googlegemini20flashlite001_info();
        let config = model_info_to_config(&info, "GoogleGemini20FlashLite001");
        map.insert("GoogleGemini20FlashLite001", config);
        let info = get_googlegemma327bit_info();
        let config = model_info_to_config(&info, "GoogleGemma327BIt");
        map.insert("GoogleGemma327BIt", config);
        let info = get_anthropicclaudeopus4_info();
        let config = model_info_to_config(&info, "AnthropicClaudeOpus4");
        map.insert("AnthropicClaudeOpus4", config);
        let info = get_anthropicclaudesonnet4_info();
        let config = model_info_to_config(&info, "AnthropicClaudeSonnet4");
        map.insert("AnthropicClaudeSonnet4", config);
        let info = get_anthropicclaude37sonnet_info();
        let config = model_info_to_config(&info, "AnthropicClaude37Sonnet");
        map.insert("AnthropicClaude37Sonnet", config);
        let info = get_anthropicclaude37sonnetthinking_info();
        let config = model_info_to_config(&info, "AnthropicClaude37Sonnetthinking");
        map.insert("AnthropicClaude37Sonnetthinking", config);
        let info = get_anthropicclaude35sonnet_info();
        let config = model_info_to_config(&info, "AnthropicClaude35Sonnet");
        map.insert("AnthropicClaude35Sonnet", config);
        let info = get_anthropicclaude35haiku_info();
        let config = model_info_to_config(&info, "AnthropicClaude35Haiku");
        map.insert("AnthropicClaude35Haiku", config);
        let info = get_metallamallama4maverick_info();
        let config = model_info_to_config(&info, "MetaLlamaLlama4Maverick");
        map.insert("MetaLlamaLlama4Maverick", config);
        let info = get_metallamallama4scout_info();
        let config = model_info_to_config(&info, "MetaLlamaLlama4Scout");
        map.insert("MetaLlamaLlama4Scout", config);
        let info = get_metallamallama3370binstruct_info();
        let config = model_info_to_config(&info, "MetaLlamaLlama3370BInstruct");
        map.insert("MetaLlamaLlama3370BInstruct", config);
        let info = get_mistralaimistralmedium3_info();
        let config = model_info_to_config(&info, "MistralaiMistralMedium3");
        map.insert("MistralaiMistralMedium3", config);
        let info = get_mistralaimistralsmall3224binstruct_info();
        let config = model_info_to_config(&info, "MistralaiMistralSmall3224BInstruct");
        map.insert("MistralaiMistralSmall3224BInstruct", config);
        let info = get_mistralaimagistralmedium2506_info();
        let config = model_info_to_config(&info, "MistralaiMagistralMedium2506");
        map.insert("MistralaiMagistralMedium2506", config);
        let info = get_mistralaimagistralmedium2506thinking_info();
        let config = model_info_to_config(&info, "MistralaiMagistralMedium2506Thinking");
        map.insert("MistralaiMagistralMedium2506Thinking", config);
        let info = get_mistralaimagistralsmall2506_info();
        let config = model_info_to_config(&info, "MistralaiMagistralSmall2506");
        map.insert("MistralaiMagistralSmall2506", config);
        let info = get_mistralaidevstralmedium_info();
        let config = model_info_to_config(&info, "MistralaiDevstralMedium");
        map.insert("MistralaiDevstralMedium", config);
        let info = get_mistralaidevstralsmall2505_info();
        let config = model_info_to_config(&info, "MistralaiDevstralSmall2505");
        map.insert("MistralaiDevstralSmall2505", config);
        let info = get_mistralaicodestral2501_info();
        let config = model_info_to_config(&info, "MistralaiCodestral2501");
        map.insert("MistralaiCodestral2501", config);
        let info = get_ai21jamba16large_info();
        let config = model_info_to_config(&info, "Ai21Jamba16Large");
        map.insert("Ai21Jamba16Large", config);
        let info = get_ai21jamba16mini_info();
        let config = model_info_to_config(&info, "Ai21Jamba16Mini");
        map.insert("Ai21Jamba16Mini", config);
        let info = get_coherecommanda_info();
        let config = model_info_to_config(&info, "CohereCommandA");
        map.insert("CohereCommandA", config);
        let info = get_coherecommandr7b122024_info();
        let config = model_info_to_config(&info, "CohereCommandR7b122024");
        map.insert("CohereCommandR7b122024", config);
        let info = get_deepseekdeepseekchatv30324_info();
        let config = model_info_to_config(&info, "DeepseekDeepseekChatV30324");
        map.insert("DeepseekDeepseekChatV30324", config);
        let info = get_deepseekdeepseekr10528_info();
        let config = model_info_to_config(&info, "DeepseekDeepseekR10528");
        map.insert("DeepseekDeepseekR10528", config);
        let info = get_qwenqwenmax_info();
        let config = model_info_to_config(&info, "QwenQwenMax");
        map.insert("QwenQwenMax", config);
        let info = get_qwenqwenplus_info();
        let config = model_info_to_config(&info, "QwenQwenPlus");
        map.insert("QwenQwenPlus", config);
        let info = get_qwenqwenturbo_info();
        let config = model_info_to_config(&info, "QwenQwenTurbo");
        map.insert("QwenQwenTurbo", config);
        let info = get_qwenqwenvlplus_info();
        let config = model_info_to_config(&info, "QwenQwenVlPlus");
        map.insert("QwenQwenVlPlus", config);
        let info = get_qwenqwen3235ba22b_info();
        let config = model_info_to_config(&info, "QwenQwen3235BA22b");
        map.insert("QwenQwen3235BA22b", config);
        let info = get_qwenqwen330ba3b_info();
        let config = model_info_to_config(&info, "QwenQwen330BA3b");
        map.insert("QwenQwen330BA3b", config);
        let info = get_qwenqwen332b_info();
        let config = model_info_to_config(&info, "QwenQwen332B");
        map.insert("QwenQwen332B", config);
        let info = get_qwenqwq32b_info();
        let config = model_info_to_config(&info, "QwenQwq32B");
        map.insert("QwenQwq32B", config);
        let info = get_qwenqwen2572binstruct_info();
        let config = model_info_to_config(&info, "QwenQwen2572BInstruct");
        map.insert("QwenQwen2572BInstruct", config);
        let info = get_qwenqwen25vl72binstruct_info();
        let config = model_info_to_config(&info, "QwenQwen25Vl72BInstruct");
        map.insert("QwenQwen25Vl72BInstruct", config);
        let info = get_qwenqwen25coder32binstruct_info();
        let config = model_info_to_config(&info, "QwenQwen25Coder32BInstruct");
        map.insert("QwenQwen25Coder32BInstruct", config);
        let info = get_moonshotaikimik2_info();
        let config = model_info_to_config(&info, "MoonshotaiKimiK2");
        map.insert("MoonshotaiKimiK2", config);
        let info = get_xaigrok4_info();
        let config = model_info_to_config(&info, "XAiGrok4");
        map.insert("XAiGrok4", config);
        let info = get_xaigrok3_info();
        let config = model_info_to_config(&info, "XAiGrok3");
        map.insert("XAiGrok3", config);
        let info = get_xaigrok3mini_info();
        let config = model_info_to_config(&info, "XAiGrok3Mini");
        map.insert("XAiGrok3Mini", config);
        let info = get_amazonnovaprov1_info();
        let config = model_info_to_config(&info, "AmazonNovaProV1");
        map.insert("AmazonNovaProV1", config);
        let info = get_amazonnovalitev1_info();
        let config = model_info_to_config(&info, "AmazonNovaLiteV1");
        map.insert("AmazonNovaLiteV1", config);
        let info = get_amazonnovamicrov1_info();
        let config = model_info_to_config(&info, "AmazonNovaMicroV1");
        map.insert("AmazonNovaMicroV1", config);
        let info = get_perplexitysonarpro_info();
        let config = model_info_to_config(&info, "PerplexitySonarPro");
        map.insert("PerplexitySonarPro", config);
        let info = get_perplexitysonar_info();
        let config = model_info_to_config(&info, "PerplexitySonar");
        map.insert("PerplexitySonar", config);
        let info = get_perplexitysonarreasoningpro_info();
        let config = model_info_to_config(&info, "PerplexitySonarReasoningPro");
        map.insert("PerplexitySonarReasoningPro", config);
        let info = get_perplexitysonarreasoning_info();
        let config = model_info_to_config(&info, "PerplexitySonarReasoning");
        map.insert("PerplexitySonarReasoning", config);
        let info = get_perplexitysonardeepresearch_info();
        let config = model_info_to_config(&info, "PerplexitySonarDeepResearch");
        map.insert("PerplexitySonarDeepResearch", config);
        let info = get_perplexityr11776_info();
        let config = model_info_to_config(&info, "PerplexityR11776");
        map.insert("PerplexityR11776", config);
        let info = get_minimaxminimax01_info();
        let config = model_info_to_config(&info, "MinimaxMinimax01");
        map.insert("MinimaxMinimax01", config);
        map
    });
    
    cache.get(model_name).unwrap_or_else(|| {
        // Fallback for unknown models
        static DEFAULT_CONFIG: crate::completion_provider::ModelConfig = crate::completion_provider::ModelConfig {
            max_tokens: 4096,
            temperature: 0.7,
            top_p: 0.9,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            context_length: 128000,
            system_prompt: "You are a helpful AI assistant.",
            supports_tools: false,
            supports_vision: false,
            supports_audio: false,
            supports_thinking: false,
            optimal_thinking_budget: 1024,
            provider: "unknown",
            model_name: "unknown",
        };
        &DEFAULT_CONFIG
    })
}

// AUTO-GENERATED END
