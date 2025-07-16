//! Fluent AI Provider Library
//!
//! This crate provides provider and model traits and definitions for AI services.
//! The enum variants are auto-generated from the AiChat models.yaml file.

// Module declarations
pub mod model_info;
pub mod models;
pub mod models_ext; // Extension methods for Models enum
pub mod providers;

// Provider client implementations - the actual client code moved from fluent-ai
pub mod clients;

// HTTP utilities moved from fluent-ai
pub mod http;

// Streaming utilities moved from fluent-ai
pub mod streaming;


// Re-export all types for convenience
pub use cyrup_sugars::ZeroOneOrMany;
pub use fluent_ai_domain::{Model, Provider};
pub use model_info::ModelInfoData;
pub use models::Models;
pub use providers::Providers;


// Extension methods for Providers enum
impl Providers {
    /// Create a Providers enum from a name string
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "openai" | "gpt" => Some(Providers::Openai),
            "gemini" | "google" => Some(Providers::Gemini),
            "claude" | "anthropic" => Some(Providers::Claude),
            "mistral" => Some(Providers::Mistral),
            "ai21" => Some(Providers::Ai21),
            "cohere" => Some(Providers::Cohere),
            "xai" => Some(Providers::Xai),
            "perplexity" => Some(Providers::Perplexity),
            "groq" => Some(Providers::Groq),
            "vertexai" => Some(Providers::Vertexai),
            "bedrock" => Some(Providers::Bedrock),
            "cloudflare" => Some(Providers::Cloudflare),
            "ernie" => Some(Providers::Ernie),
            "qianwen" => Some(Providers::Qianwen),
            "hunyuan" => Some(Providers::Hunyuan),
            "moonshot" => Some(Providers::Moonshot),
            "deepseek" => Some(Providers::Deepseek),
            "zhipuai" => Some(Providers::Zhipuai),
            "minimax" => Some(Providers::Minimax),
            "openrouter" => Some(Providers::Openrouter),
            "github" => Some(Providers::Github),
            "deepinfra" => Some(Providers::Deepinfra),
            "jina" => Some(Providers::Jina),
            "voyageai" => Some(Providers::Voyageai),
            _ => None,
        }
    }
}
