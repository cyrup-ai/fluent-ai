use super::adapter::{ModelAdapter, ModelAdapterCollection};
use super::ModelInfo as DomainModelInfo;
use crate::model::{OpenAiModel, MistralModel, AnthropicModel, TogetherModel, OpenRouterModel, HuggingFaceModel, XaiModel};
use model_info::common::Model;

// OpenAI Model Adapter Implementation
impl ModelAdapter for OpenAiModel {
    fn to_model_info(&self) -> DomainModelInfo {
        DomainModelInfo {
            name: self.name().to_string(),
            max_context: self.max_context_length(),
            pricing_input: self.pricing_input(),
            pricing_output: self.pricing_output(),
            is_thinking: self.is_thinking(),
            required_temperature: self.required_temperature(),
        }
    }

    fn from_model_info(info: &DomainModelInfo) -> Self {
        Self::all_models().into_iter()
            .find(|model| model.name() == info.name)
            .unwrap_or_else(|| Self::all_models().into_iter().next()
                .unwrap_or_else(|| panic!("No OpenAI models available")))
    }

    fn model_name(&self) -> &'static str {
        self.name()
    }

    fn provider_name(&self) -> &'static str {
        "openai"
    }
}

impl ModelAdapterCollection<OpenAiModel> for OpenAiModel {
    fn all_models() -> Vec<OpenAiModel> {
        vec![
            OpenAiModel::Gpt4,
            OpenAiModel::Gpt41106Preview,
            OpenAiModel::Gpt4Turbo,
            OpenAiModel::Gpt4o,
            OpenAiModel::Gpt4oMini,
            OpenAiModel::Gpt35Turbo,
            OpenAiModel::Gpt35Turbo1106,
            OpenAiModel::O1Preview,
            OpenAiModel::O1Mini,
        ]
    }
}

// Mistral Model Adapter Implementation
impl ModelAdapter for MistralModel {
    fn to_model_info(&self) -> DomainModelInfo {
        DomainModelInfo {
            name: self.name().to_string(),
            max_context: self.max_context_length(),
            pricing_input: self.pricing_input(),
            pricing_output: self.pricing_output(),
            is_thinking: self.is_thinking(),
            required_temperature: self.required_temperature(),
        }
    }

    fn from_model_info(info: &DomainModelInfo) -> Self {
        Self::all_models().into_iter()
            .find(|model| model.name() == info.name)
            .unwrap_or_else(|| Self::all_models().into_iter().next()
                .unwrap_or_else(|| panic!("No Mistral models available")))
    }

    fn model_name(&self) -> &'static str {
        self.name()
    }

    fn provider_name(&self) -> &'static str {
        "mistral"
    }
}

impl ModelAdapterCollection<MistralModel> for MistralModel {
    fn all_models() -> Vec<MistralModel> {
        vec![
            MistralModel::MistralLargeLatest,
            MistralModel::MistralMedium,
            MistralModel::MistralSmall,
            MistralModel::MistralTiny,
            MistralModel::CodestralLatest,
            MistralModel::MistralEmbed,
        ]
    }
}

// Anthropic Model Adapter Implementation
impl ModelAdapter for AnthropicModel {
    fn to_model_info(&self) -> DomainModelInfo {
        DomainModelInfo {
            name: self.name().to_string(),
            max_context: self.max_context_length(),
            pricing_input: self.pricing_input(),
            pricing_output: self.pricing_output(),
            is_thinking: self.is_thinking(),
            required_temperature: self.required_temperature(),
        }
    }

    fn from_model_info(info: &DomainModelInfo) -> Self {
        Self::all_models().into_iter()
            .find(|model| model.name() == info.name)
            .unwrap_or_else(|| Self::all_models().into_iter().next()
                .unwrap_or_else(|| panic!("No Anthropic models available")))
    }

    fn model_name(&self) -> &'static str {
        self.name()
    }

    fn provider_name(&self) -> &'static str {
        "anthropic"
    }
}

impl ModelAdapterCollection<AnthropicModel> for AnthropicModel {
    fn all_models() -> Vec<AnthropicModel> {
        vec![
            AnthropicModel::Claude35Sonnet20240620,
            AnthropicModel::Claude3Haiku20240307,
            AnthropicModel::Claude3Opus20240229,
            AnthropicModel::Claude3Sonnet20240229,
        ]
    }
}

// Together Model Adapter Implementation
impl ModelAdapter for TogetherModel {
    fn to_model_info(&self) -> DomainModelInfo {
        DomainModelInfo {
            name: self.name().to_string(),
            max_context: self.max_context_length(),
            pricing_input: self.pricing_input(),
            pricing_output: self.pricing_output(),
            is_thinking: self.is_thinking(),
            required_temperature: self.required_temperature(),
        }
    }

    fn from_model_info(info: &DomainModelInfo) -> Self {
        Self::all_models().into_iter()
            .find(|model| model.name() == info.name)
            .unwrap_or_else(|| Self::all_models().into_iter().next()
                .unwrap_or_else(|| panic!("No Together models available")))
    }

    fn model_name(&self) -> &'static str {
        self.name()
    }

    fn provider_name(&self) -> &'static str {
        "together"
    }
}

impl ModelAdapterCollection<TogetherModel> for TogetherModel {
    fn all_models() -> Vec<TogetherModel> {
        // TODO: Add actual Together model variants when generated
        vec![]
    }
}

// OpenRouter Model Adapter Implementation
impl ModelAdapter for OpenRouterModel {
    fn to_model_info(&self) -> DomainModelInfo {
        DomainModelInfo {
            name: self.name().to_string(),
            max_context: self.max_context_length(),
            pricing_input: self.pricing_input(),
            pricing_output: self.pricing_output(),
            is_thinking: self.is_thinking(),
            required_temperature: self.required_temperature(),
        }
    }

    fn from_model_info(info: &DomainModelInfo) -> Self {
        Self::all_models().into_iter()
            .find(|model| model.name() == info.name)
            .unwrap_or_else(|| Self::all_models().into_iter().next()
                .unwrap_or_else(|| panic!("No OpenRouter models available")))
    }

    fn model_name(&self) -> &'static str {
        self.name()
    }

    fn provider_name(&self) -> &'static str {
        "openrouter"
    }
}

impl ModelAdapterCollection<OpenRouterModel> for OpenRouterModel {
    fn all_models() -> Vec<OpenRouterModel> {
        // TODO: Add actual OpenRouter model variants when generated
        vec![]
    }
}

// HuggingFace Model Adapter Implementation
impl ModelAdapter for HuggingFaceModel {
    fn to_model_info(&self) -> DomainModelInfo {
        DomainModelInfo {
            name: self.name().to_string(),
            max_context: self.max_context_length(),
            pricing_input: self.pricing_input(),
            pricing_output: self.pricing_output(),
            is_thinking: self.is_thinking(),
            required_temperature: self.required_temperature(),
        }
    }

    fn from_model_info(info: &DomainModelInfo) -> Self {
        Self::all_models().into_iter()
            .find(|model| model.name() == info.name)
            .unwrap_or_else(|| Self::all_models().into_iter().next()
                .unwrap_or_else(|| panic!("No HuggingFace models available")))
    }

    fn model_name(&self) -> &'static str {
        self.name()
    }

    fn provider_name(&self) -> &'static str {
        "huggingface"
    }
}

impl ModelAdapterCollection<HuggingFaceModel> for HuggingFaceModel {
    fn all_models() -> Vec<HuggingFaceModel> {
        // TODO: Add actual HuggingFace model variants when generated
        vec![]
    }
}

// XAI Model Adapter Implementation
impl ModelAdapter for XaiModel {
    fn to_model_info(&self) -> DomainModelInfo {
        DomainModelInfo {
            name: self.name().to_string(),
            max_context: self.max_context_length(),
            pricing_input: self.pricing_input(),
            pricing_output: self.pricing_output(),
            is_thinking: self.is_thinking(),
            required_temperature: self.required_temperature(),
        }
    }

    fn from_model_info(info: &DomainModelInfo) -> Self {
        Self::all_models().into_iter()
            .find(|model| model.name() == info.name)
            .unwrap_or_else(|| Self::all_models().into_iter().next()
                .unwrap_or_else(|| panic!("No XAI models available")))
    }

    fn model_name(&self) -> &'static str {
        self.name()
    }

    fn provider_name(&self) -> &'static str {
        "xai"
    }
}

impl ModelAdapterCollection<XaiModel> for XaiModel {
    fn all_models() -> Vec<XaiModel> {
        vec![
            XaiModel::Grok2,
            XaiModel::Grok2Mini,
            XaiModel::GrokBeta,
            XaiModel::Grok21212,
            XaiModel::Grok2VisionBeta,
            XaiModel::GrokVisionBeta,
        ]
    }
}