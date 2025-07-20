pub mod anthropic;
pub mod azure;
pub mod bedrock;
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
    AnthropicClient, AnthropicCompletionRequest, AnthropicCompletionResponse, AnthropicError,
    AnthropicProvider, AnthropicResult,
};
// Azure OpenAI client
pub use azure::{AzureOpenAIAuth, Client as AzureClient, ClientBuilder as AzureClientBuilder};
pub use bedrock::{
    AwsCredentials, BedrockClient, BedrockCompletionBuilder, BedrockError, BedrockProvider,
    SigV4Signer,
};
// Deepseek client
pub use deepseek::{
    Client as DeepseekClient, DEEPSEEK_CHAT, DEEPSEEK_REASONER, DeepSeekCompletionBuilder,
};
// Gemini client
pub use gemini::{Client as GeminiClient, GeminiCompletionBuilder};
// Groq client
pub use groq::{
    Client as GroqClient, DEEPSEEK_R1_DISTILL_LLAMA_70B, DISTIL_WHISPER_LARGE_V3, GEMMA2_9B_IT,
    GroqCompletionBuilder, LLAMA_3_1_8B_INSTANT, LLAMA_3_2_1B_PREVIEW, LLAMA_3_2_3B_PREVIEW,
    LLAMA_3_2_11B_VISION_PREVIEW, LLAMA_3_2_70B_SPECDEC, LLAMA_3_2_70B_VERSATILE,
    LLAMA_3_2_90B_VISION_PREVIEW, LLAMA_3_8B_8192, LLAMA_3_70B_8192, LLAMA_GUARD_3_8B,
    MIXTRAL_8X7B_32768, WHISPER_LARGE_V3, WHISPER_LARGE_V3_TURBO,
};
// Huggingface client
pub use huggingface::{
    Client as HuggingfaceClient, GEMMA_2, HuggingfaceCompletionBuilder, META_LLAMA_3_1, PHI_4,
    QWEN_QVQ_PREVIEW, QWEN2_5, QWEN2_5_CODER, QWEN2_VL, SMALLTHINKER_PREVIEW, SubProvider,
};
// Mistral client
pub use mistral::{
    CODESTRAL, CODESTRAL_MAMBA, Client as MistralClient, MINISTRAL_3B, MINISTRAL_8B, MISTRAL_EMBED,
    MISTRAL_LARGE, MISTRAL_NEMO, MISTRAL_SABA, MISTRAL_SMALL, MistralCompletionBuilder,
    NewMistralCompletionBuilder, PIXTRAL_LARGE, PIXTRAL_SMALL, available_mistral_models,
    mistral_completion_builder,
};
// Ollama client
pub use ollama::{
    ALL_MINILM, Client as OllamaClient, LLAMA3_2, LLAVA, MISTRAL, MISTRAL_MAGISTRAR_SMALL,
    NOMIC_EMBED_TEXT, OllamaCompletionBuilder,
};
pub use openai::{
    OpenAIClient, OpenAICompletionRequest, OpenAICompletionResponse, OpenAIError, OpenAIProvider,
    OpenAIResult,
};
// OpenRouter client
pub use openrouter::{
    CLAUDE_3_7_SONNET, Client as OpenRouterClient, GEMINI_FLASH_2_0, GPT_4_1,
    OpenRouterCompletionBuilder, PERPLEXITY_SONAR_PRO, QWEN_QWQ_32B,
};
// Perplexity client
pub use perplexity::{Client as PerplexityClient, PerplexityCompletionBuilder, SONAR, SONAR_PRO};
// Together client
pub use together::{
    ALPACA_7B,
    // Embedding models
    BERT_BASE_UNCASED,
    BGE_BASE_EN_V1_5,
    BGE_LARGE_EN_V1_5,
    CHRONOS_HERMES_13B,
    CODE_LLAMA_7B_INSTRUCT_TOGETHER,
    CODE_LLAMA_13B_INSTRUCT,
    CODE_LLAMA_13B_INSTRUCT_TOGETHER,
    CODE_LLAMA_34B_INSTRUCT,
    CODE_LLAMA_34B_INSTRUCT_TOGETHER,
    CODE_LLAMA_70B_INSTRUCT,
    Client as TogetherClient,
    DBRX_INSTRUCT,
    DEEPSEEK_CODER_33B_INSTRUCT,
    DEEPSEEK_LLM_67B_CHAT,
    DOLPHIN_2_5_MIXTRAL_8X7B,
    GEMMA_2_9B_IT,
    GEMMA_2_27B_IT,
    GEMMA_2B_IT,
    GEMMA_7B_IT,
    GUANACO_7B,
    GUANACO_13B,
    GUANACO_33B,
    GUANACO_65B,
    HERMES_2_THETA_LLAMA_3_70B,
    KOALA_7B,
    KOALA_13B,
    LLAMA_2_7B_CHAT,
    LLAMA_2_7B_CHAT_TOGETHER,
    LLAMA_2_13B_CHAT,
    LLAMA_2_13B_CHAT_TOGETHER,
    LLAMA_2_70B_CHAT_TOGETHER,
    LLAMA_3_1_8B_INSTRUCT_REFERENCE,
    LLAMA_3_1_8B_INSTRUCT_TURBO,
    LLAMA_3_1_70B_INSTRUCT_REFERENCE,
    LLAMA_3_1_70B_INSTRUCT_TURBO,
    LLAMA_3_1_405B_INSTRUCT_LITE_PRO,
    LLAMA_3_1_405B_INSTRUCT_TURBO,
    LLAMA_3_2_3B_INSTRUCT_TURBO,
    LLAMA_3_2_11B_VISION_INSTRUCT_TURBO,
    LLAMA_3_2_90B_VISION_INSTRUCT_TURBO,
    LLAMA_3_8B_CHAT_HF,
    LLAMA_3_8B_CHAT_HF_INT4,
    LLAMA_3_8B_CHAT_HF_INT8,
    LLAMA_3_8B_INSTRUCT,
    LLAMA_3_8B_INSTRUCT_LITE,
    LLAMA_3_8B_INSTRUCT_TURBO,
    LLAMA_3_70B_CHAT_HF,
    LLAMA_3_70B_INSTRUCT,
    LLAMA_3_70B_INSTRUCT_GRADIENT_1048K,
    LLAMA_3_70B_INSTRUCT_LITE,
    LLAMA_3_70B_INSTRUCT_TURBO,
    LLAMA_VISION_FREE,
    LLAVA_NEXT_MISTRAL_7B,
    M2_BERT_2K_RETRIEVAL_ENCODER_V1,
    M2_BERT_80M_2K_RETRIEVAL,
    M2_BERT_80M_8K_RETRIEVAL,
    M2_BERT_80M_32K_RETRIEVAL,
    MISTRAL_7B_INSTRUCT_V0_1,
    MISTRAL_7B_INSTRUCT_V0_2,
    MISTRAL_7B_INSTRUCT_V0_3,
    MIXTRAL_8X7B_INSTRUCT_V0_1,
    MIXTRAL_8X22B_INSTRUCT_V0_1,
    ML318BR,
    MYTHOMAX_L2_13B,
    MYTHOMAX_L2_13B_LITE,
    NOUS_CAPYBARA_V1_9,
    NOUS_HERMES_2_MISTRAL_DPO,
    NOUS_HERMES_2_MIXTRAL_8X7B_DPO,
    NOUS_HERMES_2_MIXTRAL_8X7B_SFT,
    NOUS_HERMES_LLAMA2_7B,
    NOUS_HERMES_LLAMA2_13B,
    NOUS_HERMES_LLAMA2_70B,
    OLMO_7B_INSTRUCT,
    OPENCHAT_3_5,
    OPENHERMES_2_5_MISTRAL_7B,
    OPENHERMES_2_MISTRAL_7B,
    OPENORCA_MISTRAL_7B_8K,
    PLATYPUS2_70B_INSTRUCT,
    QWEN_2_1_5B_INSTRUCT,
    QWEN_2_7B_INSTRUCT,
    QWEN_2_72B_INSTRUCT,
    QWEN1_5_0_5B_CHAT,
    QWEN1_5_1_8B_CHAT,
    QWEN1_5_4B_CHAT,
    QWEN1_5_7B_CHAT,
    QWEN1_5_14B_CHAT,
    QWEN1_5_32B_CHAT,
    QWEN1_5_72B_CHAT,
    QWEN1_5_110B_CHAT,
    QWEN2_5_7B_INSTRUCT_TURBO,
    QWEN2_5_72B_INSTRUCT_TURBO,
    REMM_SLERP_L2_13B,
    SENTENCE_BERT,
    SNORKEL_MISTRAL_PAIRRM_DPO,
    SNOWFLAKE_ARCTIC_INSTRUCT,
    SOLAR_10_7B_INSTRUCT_V1,
    SOLAR_10_7B_INSTRUCT_V1_INT4,
    TOPPY_M_7B,
    TogetherCompletionBuilder,
    UAE_LARGE_V1,
    VICUNA_7B_V1_3,
    VICUNA_7B_V1_5,
    VICUNA_13B_V1_3,
    VICUNA_13B_V1_5,
    VICUNA_13B_V1_5_16K,
    WIZARDLM_2_8X22B,
    WIZARDLM_13B_V1_2,
    YI_34B_CHAT,
    ZEPHYR_7B_BETA,
};
// xAI client
pub use xai::{Client as XAIClient, GROK_3, GROK_3_MINI, XAICompletionBuilder};
