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

/// Get model info for gemini-2.5-flash
pub fn get_gemini25flash_info() -> ModelInfoData {
    ModelInfoData {
        provider_name: "vertexai".to_string(),
        name: "gemini-2.5-flash".to_string(),
        max_input_tokens: Some(1048576),
        max_output_tokens: Some(65536),
        input_price: Some(0.15),
        output_price: Some(0.6),
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
        provider_name: "vertexai".to_string(),
        name: "gemini-2.5-pro".to_string(),
        max_input_tokens: Some(1048576),
        max_output_tokens: Some(65536),
        input_price: Some(1.25),
        output_price: Some(10.0),
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
        provider_name: "vertexai".to_string(),
        name: "gemini-2.5-flash-lite-preview-06-17".to_string(),
        max_input_tokens: Some(1048576),
        max_output_tokens: Some(65536),
        input_price: Some(0.1),
        output_price: Some(0.4),
        supports_vision: Some(true),
        supports_function_calling: Some(true),
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
