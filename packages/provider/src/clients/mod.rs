pub mod anthropic;
pub mod azure;
pub mod candle;
pub mod deepseek;
pub mod gemini;
pub mod groq;
pub mod huggingface;
pub mod mistral;
pub mod ollama;
pub mod openai;
pub mod openrouter;
pub mod perplexity;
pub mod together;
pub mod xai;

// Remove ambiguous glob re-exports - use explicit re-exports below instead

// Re-export core provider types for convenience
pub use anthropic::{
    AnthropicProvider, AnthropicClient, AnthropicError, AnthropicResult,
    AnthropicCompletionRequest, AnthropicCompletionResponse,
};

pub use openai::{
    OpenAIProvider, OpenAIClient, OpenAIError, OpenAIResult,
    OpenAICompletionRequest, OpenAICompletionResponse,
};

// Azure OpenAI client
pub use azure::{
    AzureOpenAIAuth, Client as AzureClient, ClientBuilder as AzureClientBuilder,
};

// Gemini client
pub use gemini::{
    Client as GeminiClient, GeminiCompletionBuilder,
};

// Deepseek client
pub use deepseek::{
    Client as DeepseekClient, DeepSeekCompletionBuilder,
    DEEPSEEK_CHAT, DEEPSEEK_REASONER,
};

// Groq client
pub use groq::{
    Client as GroqClient, GroqCompletionBuilder,
    DEEPSEEK_R1_DISTILL_LLAMA_70B, GEMMA2_9B_IT, LLAMA_3_1_8B_INSTANT,
    LLAMA_3_2_11B_VISION_PREVIEW, LLAMA_3_2_1B_PREVIEW, LLAMA_3_2_3B_PREVIEW,
    LLAMA_3_2_70B_SPECDEC, LLAMA_3_2_70B_VERSATILE, LLAMA_3_2_90B_VISION_PREVIEW,
    LLAMA_3_70B_8192, LLAMA_3_8B_8192, LLAMA_GUARD_3_8B, MIXTRAL_8X7B_32768,
    WHISPER_LARGE_V3, WHISPER_LARGE_V3_TURBO, DISTIL_WHISPER_LARGE_V3,
};

// Huggingface client
pub use huggingface::{
    Client as HuggingfaceClient, HuggingfaceCompletionBuilder, SubProvider,
    GEMMA_2, META_LLAMA_3_1, PHI_4, QWEN2_5, QWEN2_5_CODER, QWEN2_VL,
    QWEN_QVQ_PREVIEW, SMALLTHINKER_PREVIEW,
};

// Mistral client
pub use mistral::{
    Client as MistralClient, MistralCompletionBuilder,
    NewMistralCompletionBuilder, mistral_completion_builder, available_mistral_models,
    CODESTRAL, CODESTRAL_MAMBA, MINISTRAL_3B, MINISTRAL_8B, MISTRAL_LARGE,
    MISTRAL_NEMO, MISTRAL_SABA, MISTRAL_SMALL, PIXTRAL_LARGE, PIXTRAL_SMALL,
    MISTRAL_EMBED,
};

// Ollama client
pub use ollama::{
    Client as OllamaClient, OllamaCompletionBuilder,
    LLAMA3_2, LLAVA, MISTRAL, MISTRAL_MAGISTRAR_SMALL,
    ALL_MINILM, NOMIC_EMBED_TEXT,
};

// OpenRouter client
pub use openrouter::{
    Client as OpenRouterClient, OpenRouterCompletionBuilder,
    CLAUDE_3_7_SONNET, GEMINI_FLASH_2_0, GPT_4_1, PERPLEXITY_SONAR_PRO,
    QWEN_QWQ_32B,
};

// Perplexity client
pub use perplexity::{
    Client as PerplexityClient, PerplexityCompletionBuilder,
    SONAR, SONAR_PRO,
};

// Together client
pub use together::{
    Client as TogetherClient, TogetherCompletionBuilder,
    ALPACA_7B, CHRONOS_HERMES_13B, CODE_LLAMA_13B_INSTRUCT,
    CODE_LLAMA_13B_INSTRUCT_TOGETHER, CODE_LLAMA_34B_INSTRUCT,
    CODE_LLAMA_34B_INSTRUCT_TOGETHER, CODE_LLAMA_70B_INSTRUCT,
    CODE_LLAMA_7B_INSTRUCT_TOGETHER, DBRX_INSTRUCT,
    DEEPSEEK_CODER_33B_INSTRUCT, DEEPSEEK_LLM_67B_CHAT, DOLPHIN_2_5_MIXTRAL_8X7B,
    GEMMA_2B_IT, GEMMA_2_27B_IT, GEMMA_2_9B_IT, GEMMA_7B_IT,
    GUANACO_13B, GUANACO_33B, GUANACO_65B, GUANACO_7B,
    HERMES_2_THETA_LLAMA_3_70B, KOALA_13B, KOALA_7B,
    LLAMA_2_13B_CHAT, LLAMA_2_13B_CHAT_TOGETHER, LLAMA_2_70B_CHAT_TOGETHER,
    LLAMA_2_7B_CHAT, LLAMA_2_7B_CHAT_TOGETHER,
    LLAMA_3_1_405B_INSTRUCT_LITE_PRO, LLAMA_3_1_405B_INSTRUCT_TURBO,
    LLAMA_3_1_70B_INSTRUCT_REFERENCE, LLAMA_3_1_70B_INSTRUCT_TURBO,
    LLAMA_3_1_8B_INSTRUCT_REFERENCE, LLAMA_3_1_8B_INSTRUCT_TURBO,
    LLAMA_3_2_11B_VISION_INSTRUCT_TURBO, LLAMA_3_2_3B_INSTRUCT_TURBO,
    LLAMA_3_2_90B_VISION_INSTRUCT_TURBO, LLAMA_3_70B_CHAT_HF,
    LLAMA_3_70B_INSTRUCT, LLAMA_3_70B_INSTRUCT_GRADIENT_1048K,
    LLAMA_3_70B_INSTRUCT_LITE, LLAMA_3_70B_INSTRUCT_TURBO,
    LLAMA_3_8B_CHAT_HF, LLAMA_3_8B_CHAT_HF_INT4, LLAMA_3_8B_CHAT_HF_INT8,
    LLAMA_3_8B_INSTRUCT, LLAMA_3_8B_INSTRUCT_LITE, LLAMA_3_8B_INSTRUCT_TURBO,
    LLAMA_VISION_FREE, LLAVA_NEXT_MISTRAL_7B,
    MISTRAL_7B_INSTRUCT_V0_1, MISTRAL_7B_INSTRUCT_V0_2, MISTRAL_7B_INSTRUCT_V0_3,
    MIXTRAL_8X22B_INSTRUCT_V0_1, MIXTRAL_8X7B_INSTRUCT_V0_1,
    ML318BR, MYTHOMAX_L2_13B, MYTHOMAX_L2_13B_LITE,
    NOUS_CAPYBARA_V1_9, NOUS_HERMES_2_MISTRAL_DPO, NOUS_HERMES_2_MIXTRAL_8X7B_DPO,
    NOUS_HERMES_2_MIXTRAL_8X7B_SFT, NOUS_HERMES_LLAMA2_13B, NOUS_HERMES_LLAMA2_70B,
    NOUS_HERMES_LLAMA2_7B, OLMO_7B_INSTRUCT, OPENCHAT_3_5,
    OPENHERMES_2_5_MISTRAL_7B, OPENHERMES_2_MISTRAL_7B, OPENORCA_MISTRAL_7B_8K,
    PLATYPUS2_70B_INSTRUCT, QWEN1_5_0_5B_CHAT, QWEN1_5_110B_CHAT,
    QWEN1_5_14B_CHAT, QWEN1_5_1_8B_CHAT, QWEN1_5_32B_CHAT,
    QWEN1_5_4B_CHAT, QWEN1_5_72B_CHAT, QWEN1_5_7B_CHAT,
    QWEN2_5_72B_INSTRUCT_TURBO, QWEN2_5_7B_INSTRUCT_TURBO,
    QWEN_2_1_5B_INSTRUCT, QWEN_2_72B_INSTRUCT, QWEN_2_7B_INSTRUCT,
    REMM_SLERP_L2_13B, SNORKEL_MISTRAL_PAIRRM_DPO, SNOWFLAKE_ARCTIC_INSTRUCT,
    SOLAR_10_7B_INSTRUCT_V1, SOLAR_10_7B_INSTRUCT_V1_INT4, TOPPY_M_7B,
    VICUNA_13B_V1_3, VICUNA_13B_V1_5, VICUNA_13B_V1_5_16K,
    VICUNA_7B_V1_3, VICUNA_7B_V1_5, WIZARDLM_13B_V1_2, WIZARDLM_2_8X22B,
    YI_34B_CHAT, ZEPHYR_7B_BETA,
    // Embedding models
    BERT_BASE_UNCASED, BGE_BASE_EN_V1_5, BGE_LARGE_EN_V1_5,
    M2_BERT_2K_RETRIEVAL_ENCODER_V1, M2_BERT_80M_2K_RETRIEVAL,
    M2_BERT_80M_32K_RETRIEVAL, M2_BERT_80M_8K_RETRIEVAL,
    SENTENCE_BERT, UAE_LARGE_V1,
};

// xAI client
pub use xai::{
    Client as XAIClient, XAICompletionBuilder,
    GROK_3, GROK_3_MINI,
};