//! Bedrock model metadata for parameter validation
//!
//! Simple model metadata lookup without duplicating the build.rs enumeration.
//! Build.rs handles all model discovery and enum generation.

use arrayvec::ArrayString;
use once_cell::sync::Lazy;
use hashbrown::HashMap;

/// Model metadata for parameter validation
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Maximum context length in tokens
    pub max_tokens: u32,
    /// Supports function/tool calling
    pub supports_tools: bool,
    /// Supports image inputs
    pub supports_vision: bool,
    /// Supports streaming
    pub supports_streaming: bool,
}

impl ModelMetadata {
    const fn new(max_tokens: u32, supports_tools: bool, supports_vision: bool, supports_streaming: bool) -> Self {
        Self {
            max_tokens,
            supports_tools,
            supports_vision,
            supports_streaming,
        }
    }
}

/// Model metadata lookup - populated at runtime with known model capabilities
static MODEL_METADATA: Lazy<HashMap<&'static str, ModelMetadata>> = Lazy::new(|| {
    let mut map = HashMap::new();
    
    // Claude 4 family
    map.insert("anthropic.claude-4-opus-20250514", ModelMetadata::new(200_000, true, true, true));
    map.insert("anthropic.claude-4-sonnet-20250514", ModelMetadata::new(200_000, true, true, true));
    
    // Claude 3.5 family
    map.insert("anthropic.claude-3-5-sonnet-20241022-v2:0", ModelMetadata::new(200_000, true, true, true));
    map.insert("anthropic.claude-3-5-sonnet-20240620-v1:0", ModelMetadata::new(200_000, true, true, true));
    map.insert("anthropic.claude-3-5-haiku-20241022-v1:0", ModelMetadata::new(200_000, true, true, true));
    
    // Claude 3 family
    map.insert("anthropic.claude-3-opus-20240229-v1:0", ModelMetadata::new(200_000, true, true, true));
    map.insert("anthropic.claude-3-sonnet-20240229-v1:0", ModelMetadata::new(200_000, true, true, true));
    map.insert("anthropic.claude-3-haiku-20240307-v1:0", ModelMetadata::new(200_000, true, true, true));
    
    // Llama 4 family
    map.insert("meta.llama4-maverick-405b-instruct-v1:0", ModelMetadata::new(131_072, true, false, true));
    map.insert("meta.llama4-scout-70b-instruct-v1:0", ModelMetadata::new(131_072, true, false, true));
    map.insert("meta.llama3-3-70b-instruct-v1:0", ModelMetadata::new(131_072, true, false, true));
    
    // Nova family
    map.insert("amazon.nova-premier-v1:0", ModelMetadata::new(300_000, true, true, true));
    map.insert("amazon.nova-pro-v1:0", ModelMetadata::new(300_000, true, true, true));
    map.insert("amazon.nova-lite-v1:0", ModelMetadata::new(300_000, true, true, true));
    map.insert("amazon.nova-micro-v1:0", ModelMetadata::new(128_000, false, false, true));
    
    // DeepSeek
    map.insert("deepseek.deepseek-r1-distill-qwen-32b-instruct-v1:0", ModelMetadata::new(65_536, true, false, true));
    
    // Titan family
    map.insert("amazon.titan-text-premier-v1:0", ModelMetadata::new(32_000, false, false, true));
    map.insert("amazon.titan-embed-text-v2:0", ModelMetadata::new(8_192, false, false, false));
    
    // Mistral via Bedrock
    map.insert("mistral.mistral-large-2407-v1:0", ModelMetadata::new(131_072, true, false, true));
    map.insert("mistral.mistral-small-2402-v1:0", ModelMetadata::new(32_768, true, false, true));
    map.insert("mistral.mixtral-8x7b-instruct-v0:1", ModelMetadata::new(32_768, true, false, true));
    
    // Cohere via Bedrock
    map.insert("cohere.command-r-plus-v1:0", ModelMetadata::new(128_000, true, false, true));
    map.insert("cohere.command-r-v1:0", ModelMetadata::new(128_000, true, false, true));
    map.insert("cohere.embed-english-v3:0", ModelMetadata::new(512, false, false, false));
    
    // AI21 via Bedrock
    map.insert("ai21.jamba-large-v1:0", ModelMetadata::new(256_000, true, false, true));
    map.insert("ai21.jamba-mini-v1:0", ModelMetadata::new(256_000, true, false, true));
    
    // Stability AI
    map.insert("stability.stable-diffusion-xl-v1", ModelMetadata::new(0, false, true, false));
    
    map
});

/// Get model metadata for validation
pub fn get_model_metadata(model_id: &str) -> Option<&'static ModelMetadata> {
    MODEL_METADATA.get(model_id)
}

/// Validate model supports the requested feature
pub fn validate_model_capability(model_id: &str, capability: &str) -> Result<(), ArrayString<128>> {
    let metadata = get_model_metadata(model_id);
    
    match metadata {
        Some(meta) => {
            let supported = match capability {
                "tools" => meta.supports_tools,
                "vision" => meta.supports_vision,
                "streaming" => meta.supports_streaming,
                _ => return Err(ArrayString::from("unknown capability").unwrap_or_default()),
            };
            
            if supported {
                Ok(())
            } else {
                let mut error = ArrayString::new();
                let _ = error.try_push_str(&format!("model {} does not support {}", model_id, capability));
                Err(error)
            }
        }
        None => {
            let mut error = ArrayString::new();
            let _ = error.try_push_str(&format!("unknown model: {}", model_id));
            Err(error)
        }
    }
}