//! Fluent AI Provider Library
//!
//! This crate provides the dynamically generated Provider and Model enumerations
//! and client implementations for AI services.
//! The enum variants are auto-generated from the AiChat models.yaml file.

// CORE: Dynamically generated enumerations from models.yaml
pub mod model_info;
pub mod models;
pub mod models_ext; // Extension methods for Models enum
pub mod providers;

// Provider client implementations
pub mod clients;

// Client factory for provider-to-client mapping
// pub mod client_factory; // Temporarily disabled during async pattern conversion

// Client traits (NOT domain objects)
pub mod client;

// CORE EXPORTS: The dynamically generated enumerations (THE MAIN VALUE!)
// Re-export client traits for provider implementations
pub use client::*;
pub use model_info::ModelInfoData;
pub use models::Models;
pub use providers::Providers;

// Re-export fluent-ai types for provider implementations
pub use fluent_ai_domain as domain;
pub use fluent_ai_domain::AsyncTask;
pub use fluent_ai_domain::spawn_async;
pub use cyrup_sugars::{OneOrMany, ZeroOneOrMany};

// Create our own AsyncStream type for provider compatibility
pub type AsyncStream<T> = tokio_stream::wrappers::UnboundedReceiverStream<T>;
pub type AsyncStreamSender<T> = tokio::sync::mpsc::UnboundedSender<T>;

// Helper to create async stream channel
pub fn async_stream_channel<T>() -> (AsyncStreamSender<T>, AsyncStream<T>) {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    (tx, tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
}

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
