// Extension methods for Models enum
// This file provides utility methods for working with model names and conversions

use crate::models::Models;

impl Models {
    /// Get the string name for this model
    pub fn name(&self) -> &'static str {
        match self {
            // OpenAI models
            Models::Gpt35Turbo => "gpt-3.5-turbo",
            Models::Gpt4O => "gpt-4o",
            Models::Gpt4OMini => "gpt-4o-mini",
            Models::Gpt4Turbo => "gpt-4-turbo",
            Models::OpenaiGpt4O => "gpt-4o",
            Models::OpenaiGpt4OMini => "gpt-4o-mini",
            Models::Chatgpt4OLatest => "chatgpt-4o-latest",
            
            // Anthropic models  
            Models::Claude35Haiku20241022 => "claude-3-5-haiku-20241022",
            Models::Claude35Sonnet20241022 => "claude-3-5-sonnet-20241022",
            Models::Claude37Sonnet20250219 => "claude-3-7-sonnet-20250219",
            Models::AnthropicClaude35Haiku => "claude-3-5-haiku",
            Models::AnthropicClaude35Sonnet => "claude-3-5-sonnet",
            
            // Mistral models
            Models::MistralSmallLatest => "mistral-small-latest",
            Models::MistralMediumLatest => "mistral-medium-latest", 
            Models::CodestralLatest => "codestral-latest",
            Models::MistralEmbed => "mistral-embed",
            
            // For all other models, use a fallback
            _ => "unknown-model",
        }
    }

    /// Create a Models variant from a string name
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            // OpenAI models
            "gpt-3.5-turbo" => Some(Models::Gpt35Turbo),
            "gpt-4o" => Some(Models::Gpt4O),
            "gpt-4o-mini" => Some(Models::Gpt4OMini),
            "gpt-4-turbo" => Some(Models::Gpt4Turbo),
            "chatgpt-4o-latest" => Some(Models::Chatgpt4OLatest),
            
            // Anthropic models
            "claude-3-5-haiku-20241022" => Some(Models::Claude35Haiku20241022),
            "claude-3-5-sonnet-20241022" => Some(Models::Claude35Sonnet20241022),
            "claude-3-7-sonnet-20250219" => Some(Models::Claude37Sonnet20250219),
            "claude-3-5-haiku" => Some(Models::AnthropicClaude35Haiku),
            "claude-3-5-sonnet" => Some(Models::AnthropicClaude35Sonnet),
            
            // Mistral models
            "mistral-small-latest" => Some(Models::MistralSmallLatest),
            "mistral-medium-latest" => Some(Models::MistralMediumLatest),
            "codestral-latest" => Some(Models::CodestralLatest),
            "mistral-embed" => Some(Models::MistralEmbed),
            
            _ => None,
        }
    }
}