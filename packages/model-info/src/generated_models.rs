#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAiModel {
    Gpt41,
    Gpt41Mini,
    O3,
    O4Mini,
    Gpt4o,
    Gpt4oMini,
}

impl OpenAiModel {
    #[doc = r" Get all available model variants"]
    pub const fn all_variants() -> &'static [OpenAiModel] {
        &[
            OpenAiModel::Gpt41,
            OpenAiModel::Gpt41Mini,
            OpenAiModel::O3,
            OpenAiModel::O4Mini,
            OpenAiModel::Gpt4o,
            OpenAiModel::Gpt4oMini,
        ]
    }
    
    #[doc = r" Get all model variants as Vec"]
    pub fn all_models() -> Vec<OpenAiModel> {
        Self::all_variants().to_vec()
    }
}

impl crate::common::Model for OpenAiModel {
    fn name(&self) -> &'static str {
        match self {
            OpenAiModel::Gpt41 => "gpt-4.1",
            OpenAiModel::Gpt41Mini => "gpt-4.1-mini",
            OpenAiModel::O3 => "o3",
            OpenAiModel::O4Mini => "o4-mini",
            OpenAiModel::Gpt4o => "gpt-4o",
            OpenAiModel::Gpt4oMini => "gpt-4o-mini",
        }
    }
    
    fn provider_name(&self) -> &'static str {
        "openai"
    }
    
    fn max_input_tokens(&self) -> Option<u32> {
        match self {
            OpenAiModel::Gpt41 => Some(128000u32),
            OpenAiModel::Gpt41Mini => Some(128000u32),
            OpenAiModel::O3 => Some(200000u32),
            OpenAiModel::O4Mini => Some(128000u32),
            OpenAiModel::Gpt4o => Some(128000u32),
            OpenAiModel::Gpt4oMini => Some(128000u32),
        }
    }
    
    fn max_output_tokens(&self) -> Option<u32> {
        match self {
            OpenAiModel::Gpt41 => Some(32000u32),
            OpenAiModel::Gpt41Mini => Some(32000u32),
            OpenAiModel::O3 => Some(50000u32),
            OpenAiModel::O4Mini => Some(32000u32),
            OpenAiModel::Gpt4o => Some(32000u32),
            OpenAiModel::Gpt4oMini => Some(32000u32),
        }
    }
    
    fn pricing_input(&self) -> Option<f64> {
        match self {
            OpenAiModel::Gpt41 => Some(2f64),
            OpenAiModel::Gpt41Mini => Some(0.4f64),
            OpenAiModel::O3 => Some(3f64),
            OpenAiModel::O4Mini => Some(1.1f64),
            OpenAiModel::Gpt4o => Some(5f64),
            OpenAiModel::Gpt4oMini => Some(0.15f64),
        }
    }
    
    fn pricing_output(&self) -> Option<f64> {
        match self {
            OpenAiModel::Gpt41 => Some(8f64),
            OpenAiModel::Gpt41Mini => Some(1.6f64),
            OpenAiModel::O3 => Some(12f64),
            OpenAiModel::O4Mini => Some(4.4f64),
            OpenAiModel::Gpt4o => Some(15f64),
            OpenAiModel::Gpt4oMini => Some(0.6f64),
        }
    }
    
    fn supports_vision(&self) -> bool {
        match self {
            OpenAiModel::Gpt41 => false,
            OpenAiModel::Gpt41Mini => false,
            OpenAiModel::O3 => false,
            OpenAiModel::O4Mini => false,
            OpenAiModel::Gpt4o => true,
            OpenAiModel::Gpt4oMini => true,
        }
    }
    
    fn supports_function_calling(&self) -> bool {
        match self {
            OpenAiModel::Gpt41 => true,
            OpenAiModel::Gpt41Mini => true,
            OpenAiModel::O3 => false,
            OpenAiModel::O4Mini => false,
            OpenAiModel::Gpt4o => true,
            OpenAiModel::Gpt4oMini => true,
        }
    }
    
    fn supports_streaming(&self) -> bool {
        true
    }
    
    fn supports_embeddings(&self) -> bool {
        false
    }
    
    fn requires_max_tokens(&self) -> bool {
        false
    }
    
    fn supports_thinking(&self) -> bool {
        match self {
            OpenAiModel::Gpt41 => false,
            OpenAiModel::Gpt41Mini => false,
            OpenAiModel::O3 => true,
            OpenAiModel::O4Mini => true,
            OpenAiModel::Gpt4o => false,
            OpenAiModel::Gpt4oMini => false,
        }
    }
    
    fn required_temperature(&self) -> Option<f64> {
        match self {
            OpenAiModel::Gpt41 => None,
            OpenAiModel::Gpt41Mini => None,
            OpenAiModel::O3 => Some(1f64),
            OpenAiModel::O4Mini => Some(1f64),
            OpenAiModel::Gpt4o => None,
            OpenAiModel::Gpt4oMini => None,
        }
    }
    
    fn optimal_thinking_budget(&self) -> Option<u32> {
        match self {
            OpenAiModel::O3 => Some(100000u32),
            OpenAiModel::O4Mini => Some(100000u32),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MistralModel {
    MistralLarge2407,
    MistralLarge2312,
    MistralSmall2409,
    MistralNemo2407,
    OpenMistralNemo,
    Codestral2405,
    MistralEmbed,
    MistralTiny,
    MistralSmall,
    MistralMedium,
    MistralLarge,
}

impl MistralModel {
    #[doc = r" Get all available model variants"]
    pub const fn all_variants() -> &'static [MistralModel] {
        &[
            MistralModel::MistralLarge2407,
            MistralModel::MistralLarge2312,
            MistralModel::MistralSmall2409,
            MistralModel::MistralNemo2407,
            MistralModel::OpenMistralNemo,
            MistralModel::Codestral2405,
            MistralModel::MistralEmbed,
            MistralModel::MistralTiny,
            MistralModel::MistralSmall,
            MistralModel::MistralMedium,
            MistralModel::MistralLarge,
        ]
    }
    
    #[doc = r" Get all model variants as Vec"]
    pub fn all_models() -> Vec<MistralModel> {
        Self::all_variants().to_vec()
    }
}

impl crate::common::Model for MistralModel {
    fn name(&self) -> &'static str {
        match self {
            MistralModel::MistralLarge2407 => "mistral-large-2407",
            MistralModel::MistralLarge2312 => "mistral-large-2312",
            MistralModel::MistralSmall2409 => "mistral-small-2409",
            MistralModel::MistralNemo2407 => "mistral-nemo-2407",
            MistralModel::OpenMistralNemo => "open-mistral-nemo",
            MistralModel::Codestral2405 => "codestral-2405",
            MistralModel::MistralEmbed => "mistral-embed",
            MistralModel::MistralTiny => "mistral-tiny",
            MistralModel::MistralSmall => "mistral-small",
            MistralModel::MistralMedium => "mistral-medium",
            MistralModel::MistralLarge => "mistral-large",
        }
    }
    
    fn provider_name(&self) -> &'static str {
        "mistral"
    }
    
    fn max_input_tokens(&self) -> Option<u32> {
        match self {
            MistralModel::MistralLarge2407 => Some(128000u32),
            MistralModel::MistralLarge2312 => Some(32000u32),
            MistralModel::MistralSmall2409 => Some(128000u32),
            MistralModel::MistralNemo2407 => Some(128000u32),
            MistralModel::OpenMistralNemo => Some(128000u32),
            MistralModel::Codestral2405 => Some(32000u32),
            MistralModel::MistralEmbed => Some(8192u32),
            MistralModel::MistralTiny => Some(32000u32),
            MistralModel::MistralSmall => Some(32000u32),
            MistralModel::MistralMedium => Some(32000u32),
            MistralModel::MistralLarge => Some(32000u32),
        }
    }
    
    fn max_output_tokens(&self) -> Option<u32> {
        match self {
            MistralModel::MistralLarge2407 => Some(32000u32),
            MistralModel::MistralLarge2312 => Some(8000u32),
            MistralModel::MistralSmall2409 => Some(32000u32),
            MistralModel::MistralNemo2407 => Some(32000u32),
            MistralModel::OpenMistralNemo => Some(32000u32),
            MistralModel::Codestral2405 => Some(8000u32),
            MistralModel::MistralEmbed => None,
            MistralModel::MistralTiny => Some(8000u32),
            MistralModel::MistralSmall => Some(8000u32),
            MistralModel::MistralMedium => Some(8000u32),
            MistralModel::MistralLarge => Some(8000u32),
        }
    }
    
    fn pricing_input(&self) -> Option<f64> {
        match self {
            MistralModel::MistralLarge2407 => Some(8f64),
            MistralModel::MistralLarge2312 => Some(3f64),
            MistralModel::MistralSmall2409 => Some(0.2f64),
            MistralModel::MistralNemo2407 => Some(0.2f64),
            MistralModel::OpenMistralNemo => Some(0.3f64),
            MistralModel::Codestral2405 => Some(0.8f64),
            MistralModel::MistralEmbed => Some(0.1f64),
            MistralModel::MistralTiny => Some(0.25f64),
            MistralModel::MistralSmall => Some(2f64),
            MistralModel::MistralMedium => Some(8f64),
            MistralModel::MistralLarge => Some(20f64),
        }
    }
    
    fn pricing_output(&self) -> Option<f64> {
        match self {
            MistralModel::MistralLarge2407 => Some(24f64),
            MistralModel::MistralLarge2312 => Some(9f64),
            MistralModel::MistralSmall2409 => Some(0.6f64),
            MistralModel::MistralNemo2407 => Some(0.6f64),
            MistralModel::OpenMistralNemo => Some(1f64),
            MistralModel::Codestral2405 => Some(2.5f64),
            MistralModel::MistralEmbed => Some(0.0f64),
            MistralModel::MistralTiny => Some(0.75f64),
            MistralModel::MistralSmall => Some(6f64),
            MistralModel::MistralMedium => Some(24f64),
            MistralModel::MistralLarge => Some(60f64),
        }
    }
    
    fn supports_vision(&self) -> bool {
        match self {
            MistralModel::MistralLarge2407 => true,  // TEST EDIT: Should be preserved during regeneration
            _ => false,
        }
    }
    
    fn supports_function_calling(&self) -> bool {
        match self {
            MistralModel::MistralEmbed => false,
            MistralModel::Codestral2405 => false,
            _ => true,
        }
    }
    
    fn supports_streaming(&self) -> bool {
        true
    }
    
    fn supports_embeddings(&self) -> bool {
        match self {
            MistralModel::MistralEmbed => true,
            _ => false,
        }
    }
    
    fn requires_max_tokens(&self) -> bool {
        false
    }
    
    fn supports_thinking(&self) -> bool {
        false
    }
    
    fn required_temperature(&self) -> Option<f64> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnthropicModel {
    Claude35Sonnet20240620,
    Claude3Haiku20240307,
    Claude3Opus20240229,
}

impl AnthropicModel {
    #[doc = r" Get all available model variants"]
    pub const fn all_variants() -> &'static [AnthropicModel] {
        &[
            AnthropicModel::Claude35Sonnet20240620,
            AnthropicModel::Claude3Haiku20240307,
            AnthropicModel::Claude3Opus20240229,
        ]
    }
    
    #[doc = r" Get all model variants as Vec"]
    pub fn all_models() -> Vec<AnthropicModel> {
        Self::all_variants().to_vec()
    }
}

impl crate::common::Model for AnthropicModel {
    fn name(&self) -> &'static str {
        match self {
            AnthropicModel::Claude35Sonnet20240620 => "claude-3-5-sonnet-20240620",
            AnthropicModel::Claude3Haiku20240307 => "claude-3-haiku-20240307",
            AnthropicModel::Claude3Opus20240229 => "claude-3-opus-20240229",
        }
    }
    
    fn provider_name(&self) -> &'static str {
        "anthropic"
    }
    
    fn max_input_tokens(&self) -> Option<u32> {
        match self {
            AnthropicModel::Claude35Sonnet20240620 => Some(200000u32),
            AnthropicModel::Claude3Haiku20240307 => Some(200000u32),
            AnthropicModel::Claude3Opus20240229 => Some(200000u32),
        }
    }
    
    fn max_output_tokens(&self) -> Option<u32> {
        match self {
            AnthropicModel::Claude35Sonnet20240620 => Some(50000u32),
            AnthropicModel::Claude3Haiku20240307 => Some(50000u32),
            AnthropicModel::Claude3Opus20240229 => Some(50000u32),
        }
    }
    
    fn pricing_input(&self) -> Option<f64> {
        match self {
            AnthropicModel::Claude35Sonnet20240620 => Some(3f64),
            AnthropicModel::Claude3Haiku20240307 => Some(0.25f64),
            AnthropicModel::Claude3Opus20240229 => Some(15f64),
        }
    }
    
    fn pricing_output(&self) -> Option<f64> {
        match self {
            AnthropicModel::Claude35Sonnet20240620 => Some(15f64),
            AnthropicModel::Claude3Haiku20240307 => Some(1.25f64),
            AnthropicModel::Claude3Opus20240229 => Some(75f64),
        }
    }
    
    fn supports_vision(&self) -> bool {
        true
    }
    
    fn supports_function_calling(&self) -> bool {
        true
    }
    
    fn supports_streaming(&self) -> bool {
        true
    }
    
    fn supports_embeddings(&self) -> bool {
        false
    }
    
    fn requires_max_tokens(&self) -> bool {
        false
    }
    
    fn supports_thinking(&self) -> bool {
        false
    }
    
    fn required_temperature(&self) -> Option<f64> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TogetherModel {
    MetaLlamaLlama38bChatHf,
    MistralaiMixtral8x7bInstructV01,
    TogethercomputerCodellama34bInstruct,
}

impl TogetherModel {
    #[doc = r" Get all available model variants"]
    pub const fn all_variants() -> &'static [TogetherModel] {
        &[
            TogetherModel::MetaLlamaLlama38bChatHf,
            TogetherModel::MistralaiMixtral8x7bInstructV01,
            TogetherModel::TogethercomputerCodellama34bInstruct,
        ]
    }
    
    #[doc = r" Get all model variants as Vec"]
    pub fn all_models() -> Vec<TogetherModel> {
        Self::all_variants().to_vec()
    }
}

impl crate::common::Model for TogetherModel {
    fn name(&self) -> &'static str {
        match self {
            TogetherModel::MetaLlamaLlama38bChatHf => "meta-llama/Llama-3-8b-chat-hf",
            TogetherModel::MistralaiMixtral8x7bInstructV01 => "mistralai/Mixtral-8x7B-Instruct-v0.1",
            TogetherModel::TogethercomputerCodellama34bInstruct => "togethercomputer/CodeLlama-34b-Instruct",
        }
    }
    
    fn provider_name(&self) -> &'static str {
        "together"
    }
    
    fn max_input_tokens(&self) -> Option<u32> {
        match self {
            TogetherModel::MetaLlamaLlama38bChatHf => Some(8192u32),
            TogetherModel::MistralaiMixtral8x7bInstructV01 => Some(32768u32),
            TogetherModel::TogethercomputerCodellama34bInstruct => Some(16384u32),
        }
    }
    
    fn max_output_tokens(&self) -> Option<u32> {
        match self {
            TogetherModel::MetaLlamaLlama38bChatHf => Some(2048u32),
            TogetherModel::MistralaiMixtral8x7bInstructV01 => Some(8192u32),
            TogetherModel::TogethercomputerCodellama34bInstruct => Some(4096u32),
        }
    }
    
    fn pricing_input(&self) -> Option<f64> {
        match self {
            TogetherModel::MetaLlamaLlama38bChatHf => Some(0.2f64),
            TogetherModel::MistralaiMixtral8x7bInstructV01 => Some(0.27f64),
            TogetherModel::TogethercomputerCodellama34bInstruct => Some(0.5f64),
        }
    }
    
    fn pricing_output(&self) -> Option<f64> {
        match self {
            TogetherModel::MetaLlamaLlama38bChatHf => Some(0.2f64),
            TogetherModel::MistralaiMixtral8x7bInstructV01 => Some(0.27f64),
            TogetherModel::TogethercomputerCodellama34bInstruct => Some(0.5f64),
        }
    }
    
    fn supports_vision(&self) -> bool {
        false
    }
    
    fn supports_function_calling(&self) -> bool {
        match self {
            TogetherModel::MistralaiMixtral8x7bInstructV01 => true,
            _ => false,
        }
    }
    
    fn supports_streaming(&self) -> bool {
        true
    }
    
    fn supports_embeddings(&self) -> bool {
        false
    }
    
    fn requires_max_tokens(&self) -> bool {
        false
    }
    
    fn supports_thinking(&self) -> bool {
        false
    }
    
    fn required_temperature(&self) -> Option<f64> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenRouterModel {
    OpenaiGpt4o,
    AnthropicClaude35Sonnet,
    GoogleGeminiPro15,
}

impl OpenRouterModel {
    #[doc = r" Get all available model variants"]
    pub const fn all_variants() -> &'static [OpenRouterModel] {
        &[
            OpenRouterModel::OpenaiGpt4o,
            OpenRouterModel::AnthropicClaude35Sonnet,
            OpenRouterModel::GoogleGeminiPro15,
        ]
    }
    
    #[doc = r" Get all model variants as Vec"]
    pub fn all_models() -> Vec<OpenRouterModel> {
        Self::all_variants().to_vec()
    }
}

impl crate::common::Model for OpenRouterModel {
    fn name(&self) -> &'static str {
        match self {
            OpenRouterModel::OpenaiGpt4o => "openai/gpt-4o",
            OpenRouterModel::AnthropicClaude35Sonnet => "anthropic/claude-3.5-sonnet",
            OpenRouterModel::GoogleGeminiPro15 => "google/gemini-pro-1.5",
        }
    }
    
    fn provider_name(&self) -> &'static str {
        "openrouter"
    }
    
    fn max_input_tokens(&self) -> Option<u32> {
        match self {
            OpenRouterModel::OpenaiGpt4o => Some(128000u32),
            OpenRouterModel::AnthropicClaude35Sonnet => Some(200000u32),
            OpenRouterModel::GoogleGeminiPro15 => Some(1000000u32),
        }
    }
    
    fn max_output_tokens(&self) -> Option<u32> {
        match self {
            OpenRouterModel::OpenaiGpt4o => Some(32000u32),
            OpenRouterModel::AnthropicClaude35Sonnet => Some(50000u32),
            OpenRouterModel::GoogleGeminiPro15 => Some(250000u32),
        }
    }
    
    fn pricing_input(&self) -> Option<f64> {
        match self {
            OpenRouterModel::OpenaiGpt4o => Some(5f64),
            OpenRouterModel::AnthropicClaude35Sonnet => Some(3f64),
            OpenRouterModel::GoogleGeminiPro15 => Some(0.5f64),
        }
    }
    
    fn pricing_output(&self) -> Option<f64> {
        match self {
            OpenRouterModel::OpenaiGpt4o => Some(15f64),
            OpenRouterModel::AnthropicClaude35Sonnet => Some(15f64),
            OpenRouterModel::GoogleGeminiPro15 => Some(1.5f64),
        }
    }
    
    fn supports_vision(&self) -> bool {
        true
    }
    
    fn supports_function_calling(&self) -> bool {
        true
    }
    
    fn supports_streaming(&self) -> bool {
        true
    }
    
    fn supports_embeddings(&self) -> bool {
        false
    }
    
    fn requires_max_tokens(&self) -> bool {
        false
    }
    
    fn supports_thinking(&self) -> bool {
        false
    }
    
    fn required_temperature(&self) -> Option<f64> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HuggingFaceModel {
    MetaLlamaMetaLlama38bInstruct,
    MistralaiMistral7bInstructV03,
    GoogleGemma29bIt,
}

impl HuggingFaceModel {
    #[doc = r" Get all available model variants"]
    pub const fn all_variants() -> &'static [HuggingFaceModel] {
        &[
            HuggingFaceModel::MetaLlamaMetaLlama38bInstruct,
            HuggingFaceModel::MistralaiMistral7bInstructV03,
            HuggingFaceModel::GoogleGemma29bIt,
        ]
    }
    
    #[doc = r" Get all model variants as Vec"]
    pub fn all_models() -> Vec<HuggingFaceModel> {
        Self::all_variants().to_vec()
    }
}

impl crate::common::Model for HuggingFaceModel {
    fn name(&self) -> &'static str {
        match self {
            HuggingFaceModel::MetaLlamaMetaLlama38bInstruct => "meta-llama/Meta-Llama-3-8B-Instruct",
            HuggingFaceModel::MistralaiMistral7bInstructV03 => "mistralai/Mistral-7B-Instruct-v0.3",
            HuggingFaceModel::GoogleGemma29bIt => "google/gemma-2-9b-it",
        }
    }
    
    fn provider_name(&self) -> &'static str {
        "huggingface"
    }
    
    fn max_input_tokens(&self) -> Option<u32> {
        match self {
            HuggingFaceModel::MetaLlamaMetaLlama38bInstruct => Some(8192u32),
            HuggingFaceModel::MistralaiMistral7bInstructV03 => Some(32768u32),
            HuggingFaceModel::GoogleGemma29bIt => Some(8192u32),
        }
    }
    
    fn max_output_tokens(&self) -> Option<u32> {
        match self {
            HuggingFaceModel::MetaLlamaMetaLlama38bInstruct => Some(2048u32),
            HuggingFaceModel::MistralaiMistral7bInstructV03 => Some(8192u32),
            HuggingFaceModel::GoogleGemma29bIt => Some(2048u32),
        }
    }
    
    fn pricing_input(&self) -> Option<f64> {
        Some(0f64)
    }
    
    fn pricing_output(&self) -> Option<f64> {
        Some(0f64)
    }
    
    fn supports_vision(&self) -> bool {
        false
    }
    
    fn supports_function_calling(&self) -> bool {
        false
    }
    
    fn supports_streaming(&self) -> bool {
        true
    }
    
    fn supports_embeddings(&self) -> bool {
        false
    }
    
    fn requires_max_tokens(&self) -> bool {
        false
    }
    
    fn supports_thinking(&self) -> bool {
        false
    }
    
    fn required_temperature(&self) -> Option<f64> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum XaiModel {
    Grok4,
    Grok3,
    Grok3Mini,
}

impl XaiModel {
    #[doc = r" Get all available model variants"]
    pub const fn all_variants() -> &'static [XaiModel] {
        &[XaiModel::Grok4, XaiModel::Grok3, XaiModel::Grok3Mini]
    }
    
    #[doc = r" Get all model variants as Vec"]
    pub fn all_models() -> Vec<XaiModel> {
        Self::all_variants().to_vec()
    }
}

impl crate::common::Model for XaiModel {
    fn name(&self) -> &'static str {
        match self {
            XaiModel::Grok4 => "grok-4",
            XaiModel::Grok3 => "grok-3",
            XaiModel::Grok3Mini => "grok-3-mini",
        }
    }
    
    fn provider_name(&self) -> &'static str {
        "xai"
    }
    
    fn max_input_tokens(&self) -> Option<u32> {
        match self {
            XaiModel::Grok4 => Some(256000u32),
            XaiModel::Grok3 => Some(131072u32),
            XaiModel::Grok3Mini => Some(131072u32),
        }
    }
    
    fn max_output_tokens(&self) -> Option<u32> {
        match self {
            XaiModel::Grok4 => Some(60000u32),
            XaiModel::Grok3 => Some(30000u32),
            XaiModel::Grok3Mini => Some(30000u32),
        }
    }
    
    fn pricing_input(&self) -> Option<f64> {
        match self {
            XaiModel::Grok4 => Some(3f64),
            XaiModel::Grok3 => Some(3f64),
            XaiModel::Grok3Mini => Some(0.3f64),
        }
    }
    
    fn pricing_output(&self) -> Option<f64> {
        match self {
            XaiModel::Grok4 => Some(15f64),
            XaiModel::Grok3 => Some(15f64),
            XaiModel::Grok3Mini => Some(0.5f64),
        }
    }
    
    fn supports_vision(&self) -> bool {
        false
    }
    
    fn supports_function_calling(&self) -> bool {
        true
    }
    
    fn supports_streaming(&self) -> bool {
        true
    }
    
    fn supports_embeddings(&self) -> bool {
        false
    }
    
    fn requires_max_tokens(&self) -> bool {
        false
    }
    
    fn supports_thinking(&self) -> bool {
        true
    }
    
    fn required_temperature(&self) -> Option<f64> {
        match self {
            XaiModel::Grok4 => Some(1f64),
            XaiModel::Grok3 => Some(1f64),
            XaiModel::Grok3Mini => None,
        }
    }
    
    fn optimal_thinking_budget(&self) -> Option<u32> {
        match self {
            XaiModel::Grok4 => Some(100000u32),
            XaiModel::Grok3 => Some(100000u32),
            XaiModel::Grok3Mini => Some(100000u32),
        }
    }
}