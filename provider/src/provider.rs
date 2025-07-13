//! Provider trait definitions for AI services

use crate::{ZeroOneOrMany, models::Models, providers::Providers};

/// Trait for AI service providers
pub trait Provider {
    /// Get the models supported by this provider
    fn models(&self) -> ZeroOneOrMany<Models>;
    
    /// Get the name of this provider
    fn name(&self) -> &'static str;
}

/// Permanent implementation of Provider trait for Providers enum
/// This is manually maintained permanent code - NOT auto-generated
impl Provider for Providers {
    #[inline(always)]
    fn models(&self) -> ZeroOneOrMany<Models> {
        match self {
            Providers::OpenAI => ZeroOneOrMany::Many(vec![
                Models::Gpt41,
                Models::Gpt41Mini,
                Models::Gpt41Nano,
                Models::Gpt4o,
                Models::Gpt4oMini,
                Models::O4Mini,
                Models::O4MiniHigh,
                Models::O3,
                Models::O3Mini,
                Models::O3MiniHigh,
                Models::TextEmbedding3Large,
                Models::TextEmbedding3Small,
            ]),
            Providers::Anthropic => ZeroOneOrMany::Many(vec![
                Models::Claude35Sonnet,
                Models::Claude35Haiku,
                Models::Claude3Opus,
                Models::Claude3Sonnet,
                Models::Claude3Haiku,
            ]),
            Providers::Mistral => ZeroOneOrMany::Many(vec![
                Models::MistralSmall2503,
                Models::Codestral2501,
            ]),
            Providers::Deepseek => ZeroOneOrMany::One(Models::DeepseekR10528),
            Providers::Meta => ZeroOneOrMany::Many(vec![
                Models::MetaLlamaLlama4Scout17b16eInstruct,
                Models::MetaLlamaLlama3370bInstruct,
            ]),
            Providers::Qwen => ZeroOneOrMany::Many(vec![
                Models::QwenQwen3235ba22b,
                Models::QwenQwen330ba3b,
                Models::QwenQwen332b,
                Models::QwenQwq32b,
                Models::QwenQwen2572bInstruct,
                Models::QwenQwen25Coder32bInstruct,
            ]),
            Providers::Google => ZeroOneOrMany::One(Models::GoogleGemma327bit),
            Providers::Microsoft => ZeroOneOrMany::One(Models::MicrosoftPhi4ReasoningPlus),
            Providers::Jina => ZeroOneOrMany::One(Models::JinaColbertv2),
        }
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        match self {
            Providers::OpenAI => "openai",
            Providers::Anthropic => "anthropic", 
            Providers::Mistral => "mistral",
            Providers::Deepseek => "deepseek",
            Providers::Meta => "meta",
            Providers::Qwen => "qwen",
            Providers::Google => "google",
            Providers::Microsoft => "microsoft",
            Providers::Jina => "jina",
        }
    }
}
