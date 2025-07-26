//! Build script for fluent_ai_provider
//! Generates provider enums and model registry from YAML files

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=models.yaml");
    
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let dest_path = Path::new(&out_dir);
    
    // Create providers.rs with basic enum
    let providers_content = r#"//! Generated provider definitions

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Provider {
    OpenAI,
    Anthropic,
    Gemini,
    Groq,
    Mistral,
    Perplexity,
    Together,
    Ollama,
    Azure,
    Candle,
    Huggingface,
    VertexAI,
    XAI,
    AI21,
    OpenRouter,
}

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Provider::OpenAI => write!(f, "openai"),
            Provider::Anthropic => write!(f, "anthropic"),
            Provider::Gemini => write!(f, "gemini"),
            Provider::Groq => write!(f, "groq"),
            Provider::Mistral => write!(f, "mistral"),
            Provider::Perplexity => write!(f, "perplexity"),
            Provider::Together => write!(f, "together"),
            Provider::Ollama => write!(f, "ollama"),
            Provider::Azure => write!(f, "azure"),
            Provider::Candle => write!(f, "candle"),
            Provider::Huggingface => write!(f, "huggingface"),
            Provider::VertexAI => write!(f, "vertexai"),
            Provider::XAI => write!(f, "xai"),
            Provider::AI21 => write!(f, "ai21"),
            Provider::OpenRouter => write!(f, "openrouter"),
        }
    }
}

impl Provider {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "openai" => Some(Provider::OpenAI),
            "anthropic" => Some(Provider::Anthropic),
            "gemini" => Some(Provider::Gemini),
            "groq" => Some(Provider::Groq),
            "mistral" => Some(Provider::Mistral),
            "perplexity" => Some(Provider::Perplexity),
            "together" => Some(Provider::Together),
            "ollama" => Some(Provider::Ollama),
            "azure" => Some(Provider::Azure),
            "candle" => Some(Provider::Candle),
            "huggingface" => Some(Provider::Huggingface),
            "vertexai" => Some(Provider::VertexAI),
            "xai" => Some(Provider::XAI),
            "ai21" => Some(Provider::AI21),
            "openrouter" => Some(Provider::OpenRouter),
            _ => None,
        }
    }
}
"#;
    
    fs::write(dest_path.join("providers.rs"), providers_content)
        .expect("Failed to write providers.rs");
    
    // Create models.rs with basic registry
    let models_content = r#"//! Generated model registry

use std::collections::HashMap;
use once_cell::sync::Lazy;

#[derive(Debug, Clone)]  
pub struct ModelInfo {
    pub name: String,
    pub provider: String,
    pub model_type: Option<String>,
    pub max_input_tokens: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub input_price: Option<f64>,
    pub output_price: Option<f64>,
    pub supports_vision: bool,
    pub supports_function_calling: bool,
    pub real_name: Option<String>,
    pub system_prompt_prefix: Option<String>,
    pub max_tokens_per_chunk: Option<u32>,
    pub default_chunk_size: Option<u32>,
    pub max_batch_size: Option<u32>,
}

pub static MODEL_REGISTRY: Lazy<HashMap<String, ModelInfo>> = Lazy::new(|| {
    let mut registry = HashMap::new();
    
    // Basic OpenAI models
    registry.insert("openai:gpt-4o".to_string(), ModelInfo {
        name: "gpt-4o".to_string(),
        provider: "openai".to_string(),
        model_type: None,
        max_input_tokens: Some(128000),
        max_output_tokens: Some(16384),
        input_price: Some(2.5),
        output_price: Some(10.0),
        supports_vision: true,
        supports_function_calling: true,
        real_name: None,
        system_prompt_prefix: None,
        max_tokens_per_chunk: None,
        default_chunk_size: None,
        max_batch_size: None,
    });
    
    registry
});
"#;
    
    fs::write(dest_path.join("models.rs"), models_content)
        .expect("Failed to write models.rs");
}