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
        "MistralMediumLatest" => get_mistralmediumlatest_info(),
        "MistralSmallLatest" => get_mistralsmalllatest_info(),
        "MagistralMediumLatest" => get_magistralmediumlatest_info(),
        "MagistralSmallLatest" => get_magistralsmalllatest_info(),
        "DevstralMediumLatest" => get_devstralmediumlatest_info(),
        "DevstralSmallLatest" => get_devstralsmalllatest_info(),
        "CodestralLatest" => get_codestrallatest_info(),
        "MistralEmbed" => get_mistralembed_info(),
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
        }
    }
}

// AUTO-GENERATED END
