//! Model enumeration types for AI models
//!
//! This module contains the core model enumeration types that represent
//! different AI models across various providers.

use serde::{Deserialize, Serialize};

/// Enumeration of all supported AI models across providers
///
/// This enum provides a type-safe way to reference different AI models
/// and retrieve their metadata through the `info()` method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Models {
    // OpenAI models
    Gpt41,
    Gpt41Mini,
    Gpt41Nano,
    Gpt4O,
    Gpt4OMini,
    Gpt4OMiniSearchPreview,
    Gpt4OSearchPreview,
    Gpt4Turbo,
    Gpt35Turbo,
    TextEmbedding3Large,
    TextEmbedding3Small,
    
    // Anthropic models
    ClaudeOpus420250514,
    ClaudeOpus420250514Thinking,
    ClaudeSonnet420250514,
    ClaudeSonnet420250514Thinking,
    Claude37Sonnet20250219,
    Claude37Sonnet20250219Thinking,
    Claude35Sonnet20241022,
    Claude35Haiku20241022,
    
    // Mistral models
    MistralMediumLatest,
    MistralSmallLatest,
    MagistralMediumLatest,
    MagistralSmallLatest,
    DevstralMediumLatest,
    DevstralSmallLatest,
    CodestralLatest,
    MistralEmbed,
    
    // Jamba models
    JambaLarge,
    JambaMini,
    
    // Cohere models
    CommandA032025,
    CommandR7b122024,
    
    // Embedding models
    EmbedV40,
    EmbedEnglishV30,
    EmbedMultilingualV30,
    
    // Other models
    O4Mini,
    O4MiniHigh,
    O3,
    O3Mini,
    O3MiniHigh,
    
    // Rerank models
    RerankV35,
    RerankEnglishV30,
    RerankMultilingualV30,
    
    // Text embedding models
    TextEmbedding005,
    TextMultilingualEmbedding002,
    
    // AWS Bedrock models
    UsAmazonNovaLiteV10,
    UsAmazonNovaMicroV10,
    UsAmazonNovaPremierV10,
    UsAmazonNovaProV10,
    
    // AWS Bedrock Anthropic models
    UsAnthropicClaude37Sonnet20250219V10,
    UsAnthropicClaude37Sonnet20250219V10Thinking,
    UsAnthropicClaudeOpus420250514V10,
    UsAnthropicClaudeOpus420250514V10Thinking,
    UsAnthropicClaudeSonnet420250514V10,
    UsAnthropicClaudeSonnet420250514V10Thinking,
    
    // AWS Bedrock Meta models
    UsMetaLlama3370BInstructV10,
    UsMetaLlama4Maverick17BInstructV10,
    UsMetaLlama4Scout17BInstructV10,
    
    // AWS Bedrock Cohere models
    CohereEmbedEnglishV3,
    CohereEmbedMultilingualV3,
    
    // AWS Bedrock Deepseek models
    UsDeepseekR1V10,
    
    // AWS Bedrock Anthropic Claude 3.5 models
    AnthropicClaude35Sonnet20241022V20,
    AnthropicClaude35Haiku20241022V10,
    
    // AWS Bedrock Mistral models
    MistralSmall2503,
    
    // AWS Bedrock Codestral models
    Codestral2501,
    
    // AWS Bedrock Command models
    Chatgpt4OLatest,
    
    // AWS Bedrock Claude 3.5 models
    Claude35SonnetV220241022,
}

impl Models {
    /// Get model information with zero allocation - blazing fast lookup
    ///
    /// This method provides a type-safe way to get metadata about each model
    /// without any runtime allocation.
    #[inline]
    pub fn info(&self) -> crate::model_info::ModelInfoData {
        use crate::model_info::get_model_info_by_name;
        
        match self {
            // OpenAI models
            Models::Gpt41 => get_model_info_by_name("Gpt41"),
            Models::Gpt41Mini => get_model_info_by_name("Gpt41Mini"),
            Models::Gpt41Nano => get_model_info_by_name("Gpt41Nano"),
            Models::Gpt4O => get_model_info_by_name("Gpt4O"),
            Models::Gpt4OMini => get_model_info_by_name("Gpt4OMini"),
            Models::Gpt4OMiniSearchPreview => get_model_info_by_name("Gpt4OMiniSearchPreview"),
            Models::Gpt4OSearchPreview => get_model_info_by_name("Gpt4OSearchPreview"),
            Models::Gpt4Turbo => get_model_info_by_name("Gpt4Turbo"),
            Models::Gpt35Turbo => get_model_info_by_name("Gpt35Turbo"),
            Models::TextEmbedding3Large => get_model_info_by_name("TextEmbedding3Large"),
            Models::TextEmbedding3Small => get_model_info_by_name("TextEmbedding3Small"),
            
            // Anthropic models
            Models::ClaudeOpus420250514 => get_model_info_by_name("ClaudeOpus420250514"),
            Models::ClaudeOpus420250514Thinking => get_model_info_by_name("ClaudeOpus420250514Thinking"),
            Models::ClaudeSonnet420250514 => get_model_info_by_name("ClaudeSonnet420250514"),
            Models::ClaudeSonnet420250514Thinking => get_model_info_by_name("ClaudeSonnet420250514Thinking"),
            Models::Claude37Sonnet20250219 => get_model_info_by_name("Claude37Sonnet20250219"),
            Models::Claude37Sonnet20250219Thinking => get_model_info_by_name("Claude37Sonnet20250219Thinking"),
            Models::Claude35Sonnet20241022 => get_model_info_by_name("Claude35Sonnet20241022"),
            Models::Claude35Haiku20241022 => get_model_info_by_name("Claude35Haiku20241022"),
            
            // Mistral models
            Models::MistralMediumLatest => get_model_info_by_name("MistralMediumLatest"),
            Models::MistralSmallLatest => get_model_info_by_name("MistralSmallLatest"),
            Models::MagistralMediumLatest => get_model_info_by_name("MagistralMediumLatest"),
            Models::MagistralSmallLatest => get_model_info_by_name("MagistralSmallLatest"),
            Models::DevstralMediumLatest => get_model_info_by_name("DevstralMediumLatest"),
            Models::DevstralSmallLatest => get_model_info_by_name("DevstralSmallLatest"),
            Models::CodestralLatest => get_model_info_by_name("CodestralLatest"),
            Models::MistralEmbed => get_model_info_by_name("MistralEmbed"),
            
            // Jamba models
            Models::JambaLarge => get_model_info_by_name("JambaLarge"),
            Models::JambaMini => get_model_info_by_name("JambaMini"),
            
            // Cohere models
            Models::CommandA032025 => get_model_info_by_name("CommandA032025"),
            Models::CommandR7b122024 => get_model_info_by_name("CommandR7b122024"),
            
            // AWS Bedrock models
            Models::UsAmazonNovaLiteV10 => get_model_info_by_name("UsAmazonNovaLiteV10"),
            Models::UsAmazonNovaMicroV10 => get_model_info_by_name("UsAmazonNovaMicroV10"),
            Models::UsAmazonNovaPremierV10 => get_model_info_by_name("UsAmazonNovaPremierV10"),
            Models::UsAmazonNovaProV10 => get_model_info_by_name("UsAmazonNovaProV10"),
            
            // AWS Bedrock Anthropic models
            Models::UsAnthropicClaude37Sonnet20250219V10 => get_model_info_by_name("UsAnthropicClaude37Sonnet20250219V10"),
            Models::UsAnthropicClaude37Sonnet20250219V10Thinking => get_model_info_by_name("UsAnthropicClaude37Sonnet20250219V10Thinking"),
            Models::UsAnthropicClaudeOpus420250514V10 => get_model_info_by_name("UsAnthropicClaudeOpus420250514V10"),
            Models::UsAnthropicClaudeOpus420250514V10Thinking => get_model_info_by_name("UsAnthropicClaudeOpus420250514V10Thinking"),
            Models::UsAnthropicClaudeSonnet420250514V10 => get_model_info_by_name("UsAnthropicClaudeSonnet420250514V10"),
            Models::UsAnthropicClaudeSonnet420250514V10Thinking => get_model_info_by_name("UsAnthropicClaudeSonnet420250514V10Thinking"),
            
            // AWS Bedrock Meta models
            Models::UsMetaLlama3370BInstructV10 => get_model_info_by_name("UsMetaLlama3370BInstructV10"),
            Models::UsMetaLlama4Maverick17BInstructV10 => get_model_info_by_name("UsMetaLlama4Maverick17BInstructV10"),
            Models::UsMetaLlama4Scout17BInstructV10 => get_model_info_by_name("UsMetaLlama4Scout17BInstructV10"),
            
            // AWS Bedrock Cohere models
            Models::CohereEmbedEnglishV3 => get_model_info_by_name("CohereEmbedEnglishV3"),
            Models::CohereEmbedMultilingualV3 => get_model_info_by_name("CohereEmbedMultilingualV3"),
            
            // AWS Bedrock Deepseek models
            Models::UsDeepseekR1V10 => get_model_info_by_name("UsDeepseekR1V10"),
            
            // Embedding models
            Models::EmbedV40 => get_model_info_by_name("EmbedV40"),
            Models::EmbedEnglishV30 => get_model_info_by_name("EmbedEnglishV30"),
            Models::EmbedMultilingualV30 => get_model_info_by_name("EmbedMultilingualV30"),
            
            // Other models
            Models::O4Mini => get_model_info_by_name("O4Mini"),
            Models::O4MiniHigh => get_model_info_by_name("O4MiniHigh"),
            Models::O3 => get_model_info_by_name("O3"),
            Models::O3Mini => get_model_info_by_name("O3Mini"),
            Models::O3MiniHigh => get_model_info_by_name("O3MiniHigh"),
            
            // Rerank models
            Models::RerankV35 => get_model_info_by_name("RerankV35"),
            Models::RerankEnglishV30 => get_model_info_by_name("RerankEnglishV30"),
            Models::RerankMultilingualV30 => get_model_info_by_name("RerankMultilingualV30"),
            
            // Text embedding models
            Models::TextEmbedding005 => get_model_info_by_name("TextEmbedding005"),
            Models::TextMultilingualEmbedding002 => get_model_info_by_name("TextMultilingualEmbedding002"),
            
            // AWS Bedrock Anthropic Claude 3.5 models
            Models::AnthropicClaude35Sonnet20241022V20 => get_model_info_by_name("AnthropicClaude35Sonnet20241022V20"),
            Models::AnthropicClaude35Haiku20241022V10 => get_model_info_by_name("AnthropicClaude35Haiku20241022V10"),
            
            // AWS Bedrock Mistral models
            Models::MistralSmall2503 => get_model_info_by_name("MistralSmall2503"),
            
            // AWS Bedrock Codestral models
            Models::Codestral2501 => get_model_info_by_name("Codestral2501"),
            
            // AWS Bedrock Command models
            Models::Chatgpt4OLatest => get_model_info_by_name("Chatgpt4OLatest"),
            
            // AWS Bedrock Claude 3.5 models
            Models::Claude35SonnetV220241022 => get_model_info_by_name("Claude35SonnetV220241022"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_models_info() {
        // Test a few models to ensure the info() method works
        let gpt4 = Models::Gpt41.info();
        assert_eq!(gpt4.provider_name, "openai");
        assert_eq!(gpt4.name, "gpt-4.1");
        
        let claude = Models::ClaudeOpus420250514.info();
        assert_eq!(claude.provider_name, "anthropic");
        assert_eq!(claude.name, "claude-opus-4-20250514");
        
        let mistral = Models::MistralMediumLatest.info();
        assert_eq!(mistral.provider_name, "mistral");
        assert_eq!(mistral.name, "mistral-medium-latest");
    }
    
    #[test]
    fn test_models_serialization() {
        // Test that models can be serialized and deserialized
        let model = Models::Gpt41;
        let serialized = serde_json::to_string(&model).expect("Failed to serialize model in test");
        let deserialized: Models = serde_json::from_str(&serialized).expect("Failed to deserialize model in test");
        assert_eq!(model, deserialized);
    }
}
