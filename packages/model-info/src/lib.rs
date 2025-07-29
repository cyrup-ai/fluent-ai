pub mod common;
pub mod providers;
pub mod generated_models;

// Re-export all common types for external use
pub use common::{
    // Core types
    ModelInfo, ModelInfoBuilder, ProviderTrait, Model,
    // Error handling
    ModelError, Result,
    // Capabilities and collections
    ModelCapabilities, ProviderModels,
};

// Re-export generated model types
pub use generated_models::{MistralModel as Mistral, AnthropicModel as Anthropic, TogetherModel as Together, OpenRouterModel as OpenRouter, HuggingFaceModel as HuggingFace};
pub use generated_models::{OpenAiModel as OpenAi, XaiModel as Xai};

use providers::{
    anthropic::AnthropicProvider, huggingface::HuggingFaceProvider, mistral::MistralProvider,
    openai::OpenAiProvider, openrouter::OpenRouterProvider, together::TogetherProvider,
    xai::XaiProvider,
};
use fluent_ai_async::AsyncStream;

#[derive(Clone, Debug, PartialEq)]
pub enum Provider {
    // Core providers with implementations
    OpenAi(OpenAiProvider),
    Mistral(MistralProvider),
    Anthropic(AnthropicProvider),
    Together(TogetherProvider),
    OpenRouter(OpenRouterProvider),
    HuggingFace(HuggingFaceProvider),
    Xai(XaiProvider),
    
    // Additional providers referenced in domain (unit variants for now)
    OpenAI,        // OpenAI API (canonical name)
    Azure,         // Azure OpenAI
    VertexAI,      // Google Vertex AI
    Gemini,        // Google Gemini
    Bedrock,       // AWS Bedrock
    Cohere,        // Cohere
    Ollama,        // Local Ollama
    Groq,          // Groq
    AI21,          // AI21 Labs
    Perplexity,    // Perplexity AI
    DeepSeek,      // DeepSeek
}

impl Provider {
    pub fn default_base_url(&self) -> &'static str {
        match self {
            // Implemented providers with concrete types
            Provider::OpenAi(_) => "https://api.openai.com/v1",
            Provider::Mistral(_) => "https://api.mistral.ai/v1",
            Provider::Anthropic(_) => "https://api.anthropic.com/v1",
            Provider::Together(_) => "https://api.together.xyz/v1",
            Provider::OpenRouter(_) => "https://openrouter.ai/api/v1",
            Provider::HuggingFace(_) => "https://api-inference.huggingface.co/models",
            Provider::Xai(_) => "https://api.x.ai/v1",
            
            // Unit variants - return static URLs
            Provider::OpenAI => "https://api.openai.com/v1",
            Provider::Azure => "https://api.openai.com/v1", // Will be overridden with deployment URL
            Provider::VertexAI => "https://aiplatform.googleapis.com/v1",
            Provider::Gemini => "https://generativelanguage.googleapis.com/v1",
            Provider::Bedrock => "https://bedrock-runtime.amazonaws.com",
            Provider::Cohere => "https://api.cohere.ai/v1",
            Provider::Ollama => "http://localhost:11434/api",
            Provider::Groq => "https://api.groq.com/openai/v1",
            Provider::AI21 => "https://api.ai21.com/studio/v1",
            Provider::Perplexity => "https://api.perplexity.ai",
            Provider::DeepSeek => "https://api.deepseek.com",
        }
    }
    
    pub fn supports_streaming(&self) -> bool {
        match self {
            // Implemented providers - check specific capabilities
            Provider::OpenAi(_) | Provider::Mistral(_) | Provider::Anthropic(_) |
            Provider::Together(_) | Provider::OpenRouter(_) | Provider::HuggingFace(_) |
            Provider::Xai(_) => true,
            
            // Unit variants - most support streaming
            Provider::OpenAI | Provider::Azure | Provider::VertexAI | Provider::Gemini |
            Provider::Cohere | Provider::Groq | Provider::AI21 | Provider::Perplexity |
            Provider::DeepSeek => true,
            
            // Some providers don't support streaming
            Provider::Bedrock | Provider::Ollama => false,
        }
    }
    
    pub fn supports_function_calling(&self) -> bool {
        match self {
            // Implemented providers - check specific capabilities
            Provider::OpenAi(_) | Provider::Anthropic(_) | Provider::Together(_) |
            Provider::OpenRouter(_) | Provider::Xai(_) => true,
            Provider::Mistral(_) | Provider::HuggingFace(_) => false,
            
            // Unit variants
            Provider::OpenAI | Provider::Azure | Provider::VertexAI | Provider::Gemini |
            Provider::Cohere | Provider::Groq => true,
            
            // Providers without function calling
            Provider::Bedrock | Provider::Ollama | Provider::AI21 | 
            Provider::Perplexity | Provider::DeepSeek => false,
        }
    }

    pub fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        match self {
            // Implemented providers with concrete types
            Provider::OpenAi(p) => p.get_model_info(model),
            Provider::Mistral(p) => p.get_model_info(model),
            Provider::Anthropic(p) => p.get_model_info(model),
            Provider::Together(p) => p.get_model_info(model),
            Provider::OpenRouter(p) => p.get_model_info(model),
            Provider::HuggingFace(p) => p.get_model_info(model),
            Provider::Xai(p) => p.get_model_info(model),
            
            // Unit variants - return empty stream for now
            Provider::OpenAI | Provider::Azure | Provider::VertexAI | Provider::Gemini |
            Provider::Bedrock | Provider::Cohere | Provider::Ollama | Provider::Groq |
            Provider::AI21 | Provider::Perplexity | Provider::DeepSeek => {
                AsyncStream::empty()
            }
        }
    }
    
    pub fn list_models(&self) -> AsyncStream<ModelInfo> {
        match self {
            // Implemented providers with concrete types
            Provider::OpenAi(p) => p.list_models(),
            Provider::Mistral(p) => p.list_models(),
            Provider::Anthropic(p) => p.list_models(),
            Provider::Together(p) => p.list_models(),
            Provider::OpenRouter(p) => p.list_models(),
            Provider::HuggingFace(p) => p.list_models(),
            Provider::Xai(p) => p.list_models(),
            
            // Unit variants - return empty stream for now
            Provider::OpenAI | Provider::Azure | Provider::VertexAI | Provider::Gemini |
            Provider::Bedrock | Provider::Cohere | Provider::Ollama | Provider::Groq |
            Provider::AI21 | Provider::Perplexity | Provider::DeepSeek => {
                AsyncStream::empty()
            }
        }
    }
    
    pub fn provider_name(&self) -> &'static str {
        match self {
            // Implemented providers with concrete types
            Provider::OpenAi(p) => p.provider_name(),
            Provider::Mistral(p) => p.provider_name(),
            Provider::Anthropic(p) => p.provider_name(),
            Provider::Together(p) => p.provider_name(),
            Provider::OpenRouter(p) => p.provider_name(),
            Provider::HuggingFace(p) => p.provider_name(),
            Provider::Xai(p) => p.provider_name(),
            
            // Unit variants - return static names
            Provider::OpenAI => "openai",
            Provider::Azure => "azure",
            Provider::VertexAI => "vertexai",
            Provider::Gemini => "gemini",
            Provider::Bedrock => "bedrock",
            Provider::Cohere => "cohere",
            Provider::Ollama => "ollama",
            Provider::Groq => "groq",
            Provider::AI21 => "ai21",
            Provider::Perplexity => "perplexity",
            Provider::DeepSeek => "deepseek",
        }
    }
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.provider_name())
    }
}