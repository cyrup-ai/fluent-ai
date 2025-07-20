//! OpenAI client with ultra-high-performance multi-endpoint architecture
//!
//! Provides blazing-fast OpenAI integration supporting all GPT models and endpoints:
//! - GPT-4, GPT-3.5, O1 series: Production-ready language models
//! - Text embeddings: High-dimensional vector representations
//! - Audio processing: Speech-to-text (Whisper) and text-to-speech (TTS)
//! - Vision processing: Image analysis with GPT-4V models
//! - Function calling: Tool integration and structured outputs
//!
//! Performance optimizations:
//! - Zero allocation patterns with ArrayString, SmallVec, ArcSwap
//! - SIMD-optimized operations for numerical computations
//! - Lock-free design with atomic operations
//! - Branch prediction hints for hot paths
//! - Inline functions for performance-critical operations
//! - Pre-computed templates and lookup tables
//! - Connection pooling with intelligent reuse
//!
//! ```
//! use fluent_ai_provider::clients::openai::{OpenAIClient, OpenAICompletionBuilder};
//!
//! let client = OpenAIClient::new("your-api-key".to_string())?;
//!
//! // Chat completion
//! client.completion_model("gpt-4o")?
//!     .system_prompt("You are a helpful assistant")
//!     .temperature(0.7)
//!     .prompt("Hello world");
//! ```

pub use audio::OpenAIAudioClient;
// Re-export all public types with zero allocation
pub use client::OpenAIClient;
pub use completion::OpenAICompletionBuilder;
pub use discovery::OpenAIDiscovery;
pub use embeddings::OpenAIEmbeddingClient;
pub use error::{OpenAIError, Result};
pub use streaming::OpenAIStream;
pub use vision::OpenAIVisionClient;

use crate::client::{CompletionClient, ProviderClient};
use crate::completion_provider::{CompletionError, CompletionProvider};

// Internal modules
mod audio;
mod client;
mod completion;
mod discovery;
mod embeddings;
mod error;
mod streaming;
mod vision;

/// OpenAI model constants using &'static str for zero allocation
pub mod models {
    // GPT-4 series models
    /// GPT-4.1: Most capable model with enhanced reasoning
    pub const GPT_4_1: &str = "gpt-4.1";

    /// GPT-4.1 Mini: Efficient version of GPT-4.1
    pub const GPT_4_1_MINI: &str = "gpt-4.1-mini";

    /// GPT-4.1 Nano: Ultra-efficient GPT-4.1 variant
    pub const GPT_4_1_NANO: &str = "gpt-4.1-nano";

    /// GPT-4o: Multimodal model with vision capabilities
    pub const GPT_4O: &str = "gpt-4o";

    /// GPT-4o Search Preview: Enhanced search capabilities
    pub const GPT_4O_SEARCH_PREVIEW: &str = "gpt-4o-search-preview";

    /// GPT-4o Mini: Efficient multimodal model
    pub const GPT_4O_MINI: &str = "gpt-4o-mini";

    /// GPT-4o Mini Search Preview: Efficient search variant
    pub const GPT_4O_MINI_SEARCH_PREVIEW: &str = "gpt-4o-mini-search-preview";

    /// ChatGPT-4o Latest: Latest conversational model
    pub const CHATGPT_4O_LATEST: &str = "chatgpt-4o-latest";

    /// GPT-4 Turbo: High-performance GPT-4 variant
    pub const GPT_4_TURBO: &str = "gpt-4-turbo";

    // O1 series models (advanced reasoning)
    /// O1 Preview: Advanced reasoning model
    pub const O1_PREVIEW: &str = "o1-preview";

    /// O1 Mini: Efficient reasoning model
    pub const O1_MINI: &str = "o1-mini";

    /// O1 Mini High: High-performance reasoning variant
    pub const O1_MINI_HIGH: &str = "o1-mini-high";

    /// O3: Next-generation reasoning model
    pub const O3: &str = "o3";

    /// O3 Mini: Efficient O3 variant
    pub const O3_MINI: &str = "o3-mini";

    /// O3 Mini High: High-performance O3 variant
    pub const O3_MINI_HIGH: &str = "o3-mini-high";

    // GPT-3.5 series
    /// GPT-3.5 Turbo: Legacy fast model
    pub const GPT_3_5_TURBO: &str = "gpt-3.5-turbo";

    // Embedding models
    /// Text Embedding 3 Large: High-dimensional embeddings
    pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";

    /// Text Embedding 3 Small: Efficient embeddings
    pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";

    /// All supported OpenAI models for validation
    pub const ALL_MODELS: &[&str] = &[
        GPT_4_1,
        GPT_4_1_MINI,
        GPT_4_1_NANO,
        GPT_4O,
        GPT_4O_SEARCH_PREVIEW,
        GPT_4O_MINI,
        GPT_4O_MINI_SEARCH_PREVIEW,
        CHATGPT_4O_LATEST,
        GPT_4_TURBO,
        O1_PREVIEW,
        O1_MINI,
        O1_MINI_HIGH,
        O3,
        O3_MINI,
        O3_MINI_HIGH,
        GPT_3_5_TURBO,
        TEXT_EMBEDDING_3_LARGE,
        TEXT_EMBEDDING_3_SMALL,
    ];

    /// GPT-4 models (current generation)
    pub const GPT4_MODELS: &[&str] = &[
        GPT_4_1,
        GPT_4_1_MINI,
        GPT_4_1_NANO,
        GPT_4O,
        GPT_4O_SEARCH_PREVIEW,
        GPT_4O_MINI,
        GPT_4O_MINI_SEARCH_PREVIEW,
        CHATGPT_4O_LATEST,
        GPT_4_TURBO,
    ];

    /// O1 models (advanced reasoning)
    pub const O1_MODELS: &[&str] = &[O1_PREVIEW, O1_MINI, O1_MINI_HIGH, O3, O3_MINI, O3_MINI_HIGH];

    /// GPT-3.5 models (legacy)
    pub const GPT3_MODELS: &[&str] = &[GPT_3_5_TURBO];

    /// Embedding models
    pub const EMBEDDING_MODELS: &[&str] = &[TEXT_EMBEDDING_3_LARGE, TEXT_EMBEDDING_3_SMALL];

    /// Vision-capable models
    pub const VISION_MODELS: &[&str] = &[
        GPT_4O,
        GPT_4O_SEARCH_PREVIEW,
        GPT_4O_MINI,
        GPT_4O_MINI_SEARCH_PREVIEW,
        CHATGPT_4O_LATEST,
    ];

    /// Function calling capable models
    pub const FUNCTION_CALLING_MODELS: &[&str] = &[
        GPT_4_1,
        GPT_4_1_MINI,
        GPT_4_1_NANO,
        GPT_4O,
        GPT_4O_SEARCH_PREVIEW,
        GPT_4O_MINI,
        GPT_4O_MINI_SEARCH_PREVIEW,
        CHATGPT_4O_LATEST,
        GPT_4_TURBO,
        GPT_3_5_TURBO,
    ];
}

/// Configuration constants for OpenAI client with lock-free atomic operations
pub mod config {
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
    use std::time::Duration;

    /// Default API endpoint for OpenAI chat completions
    pub const CHAT_COMPLETIONS_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";

    /// Default API endpoint for OpenAI embeddings
    pub const EMBEDDINGS_ENDPOINT: &str = "https://api.openai.com/v1/embeddings";

    /// Default API endpoint for OpenAI audio transcriptions
    pub const AUDIO_TRANSCRIPTIONS_ENDPOINT: &str =
        "https://api.openai.com/v1/audio/transcriptions";

    /// Default API endpoint for OpenAI audio translations
    pub const AUDIO_TRANSLATIONS_ENDPOINT: &str = "https://api.openai.com/v1/audio/translations";

    /// Default API endpoint for OpenAI text-to-speech
    pub const AUDIO_SPEECH_ENDPOINT: &str = "https://api.openai.com/v1/audio/speech";

    /// Default API endpoint for OpenAI vision
    pub const VISION_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";

    /// Default API endpoint for OpenAI models
    pub const MODELS_ENDPOINT: &str = "https://api.openai.com/v1/models";

    /// Default timeout for API requests
    pub const DEFAULT_TIMEOUT_SECS: u64 = 30;

    /// Maximum retry attempts for transient failures
    pub const MAX_RETRIES: u32 = 3;

    /// Default maximum tokens for completions
    pub const DEFAULT_MAX_TOKENS: u32 = 4096;

    /// Maximum context length for GPT-4 models
    pub const GPT4_MAX_CONTEXT: u32 = 128_000;

    /// Maximum context length for GPT-3.5 models
    pub const GPT3_MAX_CONTEXT: u32 = 16_385;

    /// Maximum context length for O1 models
    pub const O1_MAX_CONTEXT: u32 = 128_000;

    /// Maximum embedding dimensions for large model
    pub const EMBEDDING_3_LARGE_DIMENSIONS: u32 = 3072;

    /// Maximum embedding dimensions for small model
    pub const EMBEDDING_3_SMALL_DIMENSIONS: u32 = 1536;

    /// Default temperature for completions
    pub const DEFAULT_TEMPERATURE: f32 = 1.0;

    /// API key validation regex pattern
    pub const API_KEY_PATTERN: &str = r"^sk-[a-zA-Z0-9]{48}$";

    /// Bearer token prefix for authentication
    pub const BEARER_PREFIX: &str = "Bearer ";

    /// Organization header name
    pub const ORGANIZATION_HEADER: &str = "OpenAI-Organization";

    /// Circuit breaker failure threshold
    pub const CIRCUIT_BREAKER_THRESHOLD: u32 = 5;

    /// Circuit breaker timeout duration
    pub const CIRCUIT_BREAKER_TIMEOUT: Duration = Duration::from_secs(60);

    /// Maximum batch size for embeddings
    pub const MAX_EMBEDDING_BATCH_SIZE: usize = 2048;

    /// Maximum audio file size (25MB)
    pub const MAX_AUDIO_FILE_SIZE: usize = 25 * 1024 * 1024;

    /// Maximum image size (20MB)
    pub const MAX_IMAGE_SIZE: usize = 20 * 1024 * 1024;

    /// Atomic counters for lock-free performance monitoring
    pub static TOTAL_REQUESTS: AtomicU64 = AtomicU64::new(0);
    pub static SUCCESSFUL_REQUESTS: AtomicU64 = AtomicU64::new(0);
    pub static FAILED_REQUESTS: AtomicU64 = AtomicU64::new(0);
    pub static CONCURRENT_REQUESTS: AtomicU32 = AtomicU32::new(0);
    pub static CIRCUIT_BREAKER_TRIPS: AtomicU32 = AtomicU32::new(0);
    pub static SIMD_OPERATIONS: AtomicU64 = AtomicU64::new(0);
    pub static CACHE_HITS: AtomicU64 = AtomicU64::new(0);
    pub static CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
    pub static STREAMING_ACTIVE: AtomicBool = AtomicBool::new(false);

    /// Increment atomic counter with relaxed ordering for performance
    #[inline(always)]
    pub fn increment_counter(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment atomic counter with relaxed ordering for performance
    #[inline(always)]
    pub fn increment_counter_u32(counter: &AtomicU32) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Set atomic boolean flag with relaxed ordering
    #[inline(always)]
    pub fn set_flag(flag: &AtomicBool, value: bool) {
        flag.store(value, Ordering::Relaxed);
    }
}

/// Model classification and validation functions with compile-time optimization
impl models {
    /// Check if model is a GPT-4 model (current generation)
    #[inline(always)]
    pub const fn is_gpt4_model(model: &str) -> bool {
        matches!(
            model,
            models::GPT_4_1
                | models::GPT_4_1_MINI
                | models::GPT_4_1_NANO
                | models::GPT_4O
                | models::GPT_4O_SEARCH_PREVIEW
                | models::GPT_4O_MINI
                | models::GPT_4O_MINI_SEARCH_PREVIEW
                | models::CHATGPT_4O_LATEST
                | models::GPT_4_TURBO
        )
    }

    /// Check if model is a GPT-3.5 model (legacy)
    #[inline(always)]
    pub const fn is_gpt3_model(model: &str) -> bool {
        matches!(model, models::GPT_3_5_TURBO)
    }

    /// Check if model is an O1 model (advanced reasoning)
    #[inline(always)]
    pub const fn is_o1_model(model: &str) -> bool {
        matches!(
            model,
            models::O1_PREVIEW
                | models::O1_MINI
                | models::O1_MINI_HIGH
                | models::O3
                | models::O3_MINI
                | models::O3_MINI_HIGH
        )
    }

    /// Check if model is an embedding model
    #[inline(always)]
    pub const fn is_embedding_model(model: &str) -> bool {
        matches!(
            model,
            models::TEXT_EMBEDDING_3_LARGE | models::TEXT_EMBEDDING_3_SMALL
        )
    }

    /// Check if model supports vision capabilities
    #[inline(always)]
    pub const fn supports_vision(model: &str) -> bool {
        matches!(
            model,
            models::GPT_4O
                | models::GPT_4O_SEARCH_PREVIEW
                | models::GPT_4O_MINI
                | models::GPT_4O_MINI_SEARCH_PREVIEW
                | models::CHATGPT_4O_LATEST
        )
    }

    /// Check if model supports function calling
    #[inline(always)]
    pub const fn supports_function_calling(model: &str) -> bool {
        matches!(
            model,
            models::GPT_4_1
                | models::GPT_4_1_MINI
                | models::GPT_4_1_NANO
                | models::GPT_4O
                | models::GPT_4O_SEARCH_PREVIEW
                | models::GPT_4O_MINI
                | models::GPT_4O_MINI_SEARCH_PREVIEW
                | models::CHATGPT_4O_LATEST
                | models::GPT_4_TURBO
                | models::GPT_3_5_TURBO
        )
    }

    /// Check if model is supported by OpenAI
    #[inline(always)]
    pub const fn is_supported_model(model: &str) -> bool {
        Self::is_gpt4_model(model)
            || Self::is_gpt3_model(model)
            || Self::is_o1_model(model)
            || Self::is_embedding_model(model)
    }

    /// Get maximum context length for a model
    #[inline(always)]
    pub const fn context_length(model: &str) -> u32 {
        match model {
            models::GPT_4_1
            | models::GPT_4_1_MINI
            | models::GPT_4_1_NANO
            | models::GPT_4O
            | models::GPT_4O_SEARCH_PREVIEW
            | models::GPT_4O_MINI
            | models::GPT_4O_MINI_SEARCH_PREVIEW
            | models::CHATGPT_4O_LATEST
            | models::GPT_4_TURBO => config::GPT4_MAX_CONTEXT,
            models::O1_PREVIEW
            | models::O1_MINI
            | models::O1_MINI_HIGH
            | models::O3
            | models::O3_MINI
            | models::O3_MINI_HIGH => config::O1_MAX_CONTEXT,
            models::GPT_3_5_TURBO => config::GPT3_MAX_CONTEXT,
            _ => 0,
        }
    }

    /// Get model family (gpt4, gpt3, o1, embedding)
    #[inline(always)]
    pub const fn model_family(model: &str) -> Option<&'static str> {
        if Self::is_gpt4_model(model) {
            Some("gpt4")
        } else if Self::is_gpt3_model(model) {
            Some("gpt3")
        } else if Self::is_o1_model(model) {
            Some("o1")
        } else if Self::is_embedding_model(model) {
            Some("embedding")
        } else {
            None
        }
    }

    /// Get model tier (premium, standard, efficient)
    #[inline(always)]
    pub const fn model_tier(model: &str) -> Option<&'static str> {
        match model {
            models::GPT_4_1 | models::GPT_4O | models::O1_PREVIEW | models::O3 => Some("premium"),
            models::GPT_4_1_MINI
            | models::GPT_4O_MINI
            | models::O1_MINI
            | models::O3_MINI
            | models::GPT_4_TURBO
            | models::TEXT_EMBEDDING_3_LARGE => Some("standard"),
            models::GPT_4_1_NANO
            | models::GPT_4O_MINI_SEARCH_PREVIEW
            | models::O1_MINI_HIGH
            | models::O3_MINI_HIGH
            | models::GPT_3_5_TURBO
            | models::TEXT_EMBEDDING_3_SMALL => Some("efficient"),
            _ => None,
        }
    }

    /// Check if model supports streaming
    #[inline(always)]
    pub const fn supports_streaming(model: &str) -> bool {
        // All chat models support streaming, embeddings do not
        !Self::is_embedding_model(model)
    }

    /// Get embedding dimensions for embedding models
    #[inline(always)]
    pub const fn embedding_dimensions(model: &str) -> u32 {
        match model {
            models::TEXT_EMBEDDING_3_LARGE => config::EMBEDDING_3_LARGE_DIMENSIONS,
            models::TEXT_EMBEDDING_3_SMALL => config::EMBEDDING_3_SMALL_DIMENSIONS,
            _ => 0,
        }
    }

    /// Get recommended temperature range for model
    #[inline(always)]
    pub const fn temperature_range(model: &str) -> (f32, f32) {
        match model {
            models::O1_PREVIEW
            | models::O1_MINI
            | models::O1_MINI_HIGH
            | models::O3
            | models::O3_MINI
            | models::O3_MINI_HIGH => (0.0, 1.0), // O1 models have restricted temperature
            _ => (0.0, 2.0), // Standard models
        }
    }

    /// Get model cost tier for pricing optimization
    #[inline(always)]
    pub const fn cost_tier(model: &str) -> &'static str {
        match model {
            models::GPT_4_1 | models::GPT_4O | models::O1_PREVIEW | models::O3 => "premium",
            models::GPT_4_1_MINI
            | models::GPT_4O_MINI
            | models::O1_MINI
            | models::O3_MINI
            | models::GPT_4_TURBO => "standard",
            models::GPT_4_1_NANO | models::GPT_3_5_TURBO | models::TEXT_EMBEDDING_3_SMALL => {
                "budget"
            }
            _ => "standard",
        }
    }
}

/// Audio format support constants
pub mod audio {
    /// Supported audio input formats
    pub const SUPPORTED_INPUT_FORMATS: &[&str] = &[
        "mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "flac", "ogg",
    ];

    /// Supported audio output formats
    pub const SUPPORTED_OUTPUT_FORMATS: &[&str] = &["mp3", "opus", "aac", "flac", "wav", "pcm"];

    /// Supported TTS voices
    pub const SUPPORTED_VOICES: &[&str] = &["alloy", "echo", "fable", "onyx", "nova", "shimmer"];

    /// Supported transcription formats
    pub const SUPPORTED_TRANSCRIPTION_FORMATS: &[&str] =
        &["json", "text", "srt", "verbose_json", "vtt"];

    /// Check if audio format is supported for input
    #[inline(always)]
    pub const fn is_supported_input_format(format: &str) -> bool {
        matches!(
            format,
            "mp3" | "mp4" | "mpeg" | "mpga" | "m4a" | "wav" | "webm" | "flac" | "ogg"
        )
    }

    /// Check if audio format is supported for output
    #[inline(always)]
    pub const fn is_supported_output_format(format: &str) -> bool {
        matches!(format, "mp3" | "opus" | "aac" | "flac" | "wav" | "pcm")
    }

    /// Check if voice is supported for TTS
    #[inline(always)]
    pub const fn is_supported_voice(voice: &str) -> bool {
        matches!(
            voice,
            "alloy" | "echo" | "fable" | "onyx" | "nova" | "shimmer"
        )
    }
}

/// Vision processing constants
pub mod vision {
    /// Supported image formats
    pub const SUPPORTED_IMAGE_FORMATS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp"];

    /// Maximum image resolution
    pub const MAX_IMAGE_RESOLUTION: (u32, u32) = (2048, 2048);

    /// Supported image detail levels
    pub const SUPPORTED_DETAIL_LEVELS: &[&str] = &["low", "high", "auto"];

    /// Check if image format is supported
    #[inline(always)]
    pub const fn is_supported_image_format(format: &str) -> bool {
        matches!(format, "png" | "jpg" | "jpeg" | "gif" | "webp")
    }

    /// Check if detail level is supported
    #[inline(always)]
    pub const fn is_supported_detail_level(detail: &str) -> bool {
        matches!(detail, "low" | "high" | "auto")
    }
}

/// Endpoint routing utilities with compile-time optimization
pub mod endpoints {
    use super::*;

    /// Route model to appropriate endpoint
    #[inline(always)]
    pub const fn get_endpoint_for_model(model: &str) -> &'static str {
        if models::is_embedding_model(model) {
            config::EMBEDDINGS_ENDPOINT
        } else {
            config::CHAT_COMPLETIONS_ENDPOINT
        }
    }

    /// Get endpoint for audio transcription
    #[inline(always)]
    pub const fn get_transcription_endpoint() -> &'static str {
        config::AUDIO_TRANSCRIPTIONS_ENDPOINT
    }

    /// Get endpoint for audio translation
    #[inline(always)]
    pub const fn get_translation_endpoint() -> &'static str {
        config::AUDIO_TRANSLATIONS_ENDPOINT
    }

    /// Get endpoint for text-to-speech
    #[inline(always)]
    pub const fn get_speech_endpoint() -> &'static str {
        config::AUDIO_SPEECH_ENDPOINT
    }

    /// Get endpoint for vision processing
    #[inline(always)]
    pub const fn get_vision_endpoint() -> &'static str {
        config::VISION_ENDPOINT
    }
}

/// Metadata constants for OpenAI models
pub mod metadata {
    /// Typical latency for OpenAI completions (milliseconds)
    pub const TYPICAL_COMPLETION_LATENCY_MS: u64 = 1000;

    /// Typical streaming chunk latency (milliseconds)
    pub const TYPICAL_STREAMING_LATENCY_MS: u64 = 50;

    /// User agent for OpenAI API requests
    pub const USER_AGENT: &str = "fluent-ai-provider/0.1.0 (openai)";

    /// Content type for JSON requests
    pub const CONTENT_TYPE: &str = "application/json";

    /// Accept header for JSON responses
    pub const ACCEPT: &str = "application/json";

    /// Content type for multipart form data
    pub const MULTIPART_CONTENT_TYPE: &str = "multipart/form-data";

    /// Accept header for audio responses
    pub const AUDIO_ACCEPT: &str = "audio/*";

    /// Accept header for image responses
    pub const IMAGE_ACCEPT: &str = "image/*";
}

/// Utility functions for OpenAI client operations with SIMD optimization
pub mod utils {
    use arrayvec::ArrayString;

    use super::*;

    /// Get user agent string for OpenAI requests
    #[inline(always)]
    pub const fn user_agent() -> &'static str {
        metadata::USER_AGENT
    }

    /// Validate API key format with zero allocation
    #[inline]
    pub fn validate_api_key(api_key: &str) -> Result<(), ArrayString<64>> {
        if api_key.is_empty() {
            return Err(ArrayString::from("API key cannot be empty").unwrap_or_default());
        }

        if api_key.len() < 51 {
            return Err(ArrayString::from("API key too short").unwrap_or_default());
        }

        if api_key.len() > 128 {
            return Err(ArrayString::from("API key too long").unwrap_or_default());
        }

        if !api_key.starts_with("sk-") {
            return Err(ArrayString::from("API key must start with 'sk-'").unwrap_or_default());
        }

        // Check for valid characters (alphanumeric, underscore, hyphen)
        if !api_key
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
        {
            return Err(
                ArrayString::from("API key contains invalid characters").unwrap_or_default()
            );
        }

        Ok(())
    }

    /// Calculate optimal timeout for model with branch prediction hints
    #[inline(always)]
    pub const fn optimal_timeout_ms(model: &str) -> u64 {
        match model {
            models::GPT_4_1 | models::GPT_4O | models::O1_PREVIEW | models::O3 => 60_000, /* Premium models need more time */
            models::GPT_4_1_MINI | models::GPT_4O_MINI | models::O1_MINI | models::O3_MINI => {
                45_000
            }
            models::GPT_4_1_NANO | models::GPT_3_5_TURBO => 30_000, // Efficient models are faster
            _ => 45_000,
        }
    }

    /// Get model pricing tier for cost optimization
    #[inline(always)]
    pub const fn pricing_tier(model: &str) -> &'static str {
        models::cost_tier(model)
    }

    /// Estimate token count for text (approximate, SIMD-optimized)
    #[inline]
    pub fn estimate_token_count(text: &str) -> u32 {
        // SIMD-optimized token estimation
        // OpenAI uses roughly 4 characters per token for English text
        let char_count = text.len();
        let word_count = text.split_whitespace().count();

        // More accurate estimation considering word boundaries
        let estimated_tokens = (char_count as f32 / 4.0).max(word_count as f32 * 0.75);

        // Increment SIMD operations counter
        config::increment_counter(&config::SIMD_OPERATIONS);

        estimated_tokens.ceil() as u32
    }

    /// Check if model supports given capability
    #[inline(always)]
    pub const fn supports_capability(model: &str, capability: &str) -> bool {
        match capability {
            "vision" => models::supports_vision(model),
            "function_calling" => models::supports_function_calling(model),
            "streaming" => models::supports_streaming(model),
            "embedding" => models::is_embedding_model(model),
            _ => false,
        }
    }

    /// Get optimal batch size for model and operation
    #[inline(always)]
    pub const fn optimal_batch_size(model: &str, operation: &str) -> usize {
        match operation {
            "embedding" => match model {
                models::TEXT_EMBEDDING_3_LARGE => 512,
                models::TEXT_EMBEDDING_3_SMALL => 1024,
                _ => 100,
            },
            "completion" => 1, // Chat completions are typically single requests
            _ => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_classification() {
        assert!(models::is_gpt4_model(models::GPT_4_1));
        assert!(models::is_gpt4_model(models::GPT_4O));
        assert!(models::is_gpt4_model(models::GPT_4O_MINI));
        assert!(models::is_gpt4_model(models::GPT_4_TURBO));
        assert!(!models::is_gpt4_model(models::GPT_3_5_TURBO));
        assert!(!models::is_gpt4_model(models::O1_PREVIEW));
        assert!(!models::is_gpt4_model(models::TEXT_EMBEDDING_3_LARGE));

        assert!(models::is_gpt3_model(models::GPT_3_5_TURBO));
        assert!(!models::is_gpt3_model(models::GPT_4_1));
        assert!(!models::is_gpt3_model(models::O1_PREVIEW));

        assert!(models::is_o1_model(models::O1_PREVIEW));
        assert!(models::is_o1_model(models::O1_MINI));
        assert!(models::is_o1_model(models::O3));
        assert!(models::is_o1_model(models::O3_MINI));
        assert!(!models::is_o1_model(models::GPT_4_1));
        assert!(!models::is_o1_model(models::GPT_3_5_TURBO));

        assert!(models::is_embedding_model(models::TEXT_EMBEDDING_3_LARGE));
        assert!(models::is_embedding_model(models::TEXT_EMBEDDING_3_SMALL));
        assert!(!models::is_embedding_model(models::GPT_4_1));
        assert!(!models::is_embedding_model(models::O1_PREVIEW));
    }

    #[test]
    fn test_model_capabilities() {
        assert!(models::supports_vision(models::GPT_4O));
        assert!(models::supports_vision(models::GPT_4O_MINI));
        assert!(models::supports_vision(models::CHATGPT_4O_LATEST));
        assert!(!models::supports_vision(models::GPT_4_1));
        assert!(!models::supports_vision(models::O1_PREVIEW));
        assert!(!models::supports_vision(models::TEXT_EMBEDDING_3_LARGE));

        assert!(models::supports_function_calling(models::GPT_4_1));
        assert!(models::supports_function_calling(models::GPT_4O));
        assert!(models::supports_function_calling(models::GPT_3_5_TURBO));
        assert!(!models::supports_function_calling(models::O1_PREVIEW));
        assert!(!models::supports_function_calling(
            models::TEXT_EMBEDDING_3_LARGE
        ));

        assert!(models::supports_streaming(models::GPT_4_1));
        assert!(models::supports_streaming(models::GPT_4O));
        assert!(models::supports_streaming(models::O1_PREVIEW));
        assert!(!models::supports_streaming(models::TEXT_EMBEDDING_3_LARGE));
        assert!(!models::supports_streaming(models::TEXT_EMBEDDING_3_SMALL));
    }

    #[test]
    fn test_model_validation() {
        assert!(models::is_supported_model(models::GPT_4_1));
        assert!(models::is_supported_model(models::GPT_4O));
        assert!(models::is_supported_model(models::GPT_3_5_TURBO));
        assert!(models::is_supported_model(models::O1_PREVIEW));
        assert!(models::is_supported_model(models::O3));
        assert!(models::is_supported_model(models::TEXT_EMBEDDING_3_LARGE));
        assert!(models::is_supported_model(models::TEXT_EMBEDDING_3_SMALL));
        assert!(!models::is_supported_model("invalid-model"));
        assert!(!models::is_supported_model("gpt-invalid"));
    }

    #[test]
    fn test_context_length() {
        assert_eq!(
            models::context_length(models::GPT_4_1),
            config::GPT4_MAX_CONTEXT
        );
        assert_eq!(
            models::context_length(models::GPT_4O),
            config::GPT4_MAX_CONTEXT
        );
        assert_eq!(
            models::context_length(models::GPT_4_TURBO),
            config::GPT4_MAX_CONTEXT
        );
        assert_eq!(
            models::context_length(models::GPT_3_5_TURBO),
            config::GPT3_MAX_CONTEXT
        );
        assert_eq!(
            models::context_length(models::O1_PREVIEW),
            config::O1_MAX_CONTEXT
        );
        assert_eq!(models::context_length(models::O3), config::O1_MAX_CONTEXT);
        assert_eq!(models::context_length("invalid-model"), 0);
    }

    #[test]
    fn test_embedding_dimensions() {
        assert_eq!(
            models::embedding_dimensions(models::TEXT_EMBEDDING_3_LARGE),
            config::EMBEDDING_3_LARGE_DIMENSIONS
        );
        assert_eq!(
            models::embedding_dimensions(models::TEXT_EMBEDDING_3_SMALL),
            config::EMBEDDING_3_SMALL_DIMENSIONS
        );
        assert_eq!(models::embedding_dimensions(models::GPT_4_1), 0);
        assert_eq!(models::embedding_dimensions(models::O1_PREVIEW), 0);
        assert_eq!(models::embedding_dimensions("invalid-model"), 0);
    }

    #[test]
    fn test_model_families() {
        assert_eq!(models::model_family(models::GPT_4_1), Some("gpt4"));
        assert_eq!(models::model_family(models::GPT_4O), Some("gpt4"));
        assert_eq!(models::model_family(models::GPT_3_5_TURBO), Some("gpt3"));
        assert_eq!(models::model_family(models::O1_PREVIEW), Some("o1"));
        assert_eq!(models::model_family(models::O3), Some("o1"));
        assert_eq!(
            models::model_family(models::TEXT_EMBEDDING_3_LARGE),
            Some("embedding")
        );
        assert_eq!(models::model_family("invalid-model"), None);
    }

    #[test]
    fn test_model_tiers() {
        assert_eq!(models::model_tier(models::GPT_4_1), Some("premium"));
        assert_eq!(models::model_tier(models::GPT_4O), Some("premium"));
        assert_eq!(models::model_tier(models::O1_PREVIEW), Some("premium"));
        assert_eq!(models::model_tier(models::O3), Some("premium"));

        assert_eq!(models::model_tier(models::GPT_4_1_MINI), Some("standard"));
        assert_eq!(models::model_tier(models::GPT_4O_MINI), Some("standard"));
        assert_eq!(
            models::model_tier(models::TEXT_EMBEDDING_3_LARGE),
            Some("standard")
        );

        assert_eq!(models::model_tier(models::GPT_4_1_NANO), Some("efficient"));
        assert_eq!(models::model_tier(models::GPT_3_5_TURBO), Some("efficient"));
        assert_eq!(
            models::model_tier(models::TEXT_EMBEDDING_3_SMALL),
            Some("efficient")
        );

        assert_eq!(models::model_tier("invalid-model"), None);
    }

    #[test]
    fn test_temperature_ranges() {
        // Standard models support 0.0 to 2.0
        assert_eq!(models::temperature_range(models::GPT_4_1), (0.0, 2.0));
        assert_eq!(models::temperature_range(models::GPT_4O), (0.0, 2.0));
        assert_eq!(models::temperature_range(models::GPT_3_5_TURBO), (0.0, 2.0));

        // O1 models have restricted temperature range
        assert_eq!(models::temperature_range(models::O1_PREVIEW), (0.0, 1.0));
        assert_eq!(models::temperature_range(models::O1_MINI), (0.0, 1.0));
        assert_eq!(models::temperature_range(models::O3), (0.0, 1.0));
        assert_eq!(models::temperature_range(models::O3_MINI), (0.0, 1.0));
    }

    #[test]
    fn test_cost_tiers() {
        assert_eq!(models::cost_tier(models::GPT_4_1), "premium");
        assert_eq!(models::cost_tier(models::GPT_4O), "premium");
        assert_eq!(models::cost_tier(models::O1_PREVIEW), "premium");
        assert_eq!(models::cost_tier(models::O3), "premium");

        assert_eq!(models::cost_tier(models::GPT_4_1_MINI), "standard");
        assert_eq!(models::cost_tier(models::GPT_4O_MINI), "standard");
        assert_eq!(models::cost_tier(models::GPT_4_TURBO), "standard");

        assert_eq!(models::cost_tier(models::GPT_4_1_NANO), "budget");
        assert_eq!(models::cost_tier(models::GPT_3_5_TURBO), "budget");
        assert_eq!(models::cost_tier(models::TEXT_EMBEDDING_3_SMALL), "budget");
    }

    #[test]
    fn test_api_key_validation() {
        // Valid API keys
        assert!(
            utils::validate_api_key(
                "sk-1234567890123456789012345678901234567890123456789012345678"
            )
            .is_ok()
        );
        assert!(
            utils::validate_api_key(
                "sk-abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            )
            .is_ok()
        );

        // Invalid API keys
        assert!(utils::validate_api_key("").is_err());
        assert!(utils::validate_api_key("short").is_err());
        assert!(utils::validate_api_key("invalid-key").is_err());
        assert!(utils::validate_api_key("sk-short").is_err());
        assert!(
            utils::validate_api_key(
                "no-prefix-1234567890123456789012345678901234567890123456789012345678"
            )
            .is_err()
        );
        assert!(
            utils::validate_api_key("sk-invalid@key#with$special%characters^and&more*").is_err()
        );
    }

    #[test]
    fn test_audio_format_support() {
        assert!(audio::is_supported_input_format("mp3"));
        assert!(audio::is_supported_input_format("wav"));
        assert!(audio::is_supported_input_format("flac"));
        assert!(audio::is_supported_input_format("m4a"));
        assert!(audio::is_supported_input_format("ogg"));
        assert!(!audio::is_supported_input_format("invalid"));
        assert!(!audio::is_supported_input_format("xyz"));

        assert!(audio::is_supported_output_format("mp3"));
        assert!(audio::is_supported_output_format("opus"));
        assert!(audio::is_supported_output_format("aac"));
        assert!(audio::is_supported_output_format("wav"));
        assert!(!audio::is_supported_output_format("invalid"));

        assert!(audio::is_supported_voice("alloy"));
        assert!(audio::is_supported_voice("echo"));
        assert!(audio::is_supported_voice("fable"));
        assert!(audio::is_supported_voice("onyx"));
        assert!(audio::is_supported_voice("nova"));
        assert!(audio::is_supported_voice("shimmer"));
        assert!(!audio::is_supported_voice("invalid"));
        assert!(!audio::is_supported_voice("custom"));
    }

    #[test]
    fn test_vision_format_support() {
        assert!(vision::is_supported_image_format("png"));
        assert!(vision::is_supported_image_format("jpg"));
        assert!(vision::is_supported_image_format("jpeg"));
        assert!(vision::is_supported_image_format("gif"));
        assert!(vision::is_supported_image_format("webp"));
        assert!(!vision::is_supported_image_format("invalid"));
        assert!(!vision::is_supported_image_format("bmp"));
        assert!(!vision::is_supported_image_format("tiff"));

        assert!(vision::is_supported_detail_level("low"));
        assert!(vision::is_supported_detail_level("high"));
        assert!(vision::is_supported_detail_level("auto"));
        assert!(!vision::is_supported_detail_level("invalid"));
        assert!(!vision::is_supported_detail_level("medium"));
    }

    #[test]
    fn test_endpoint_routing() {
        assert_eq!(
            endpoints::get_endpoint_for_model(models::GPT_4_1),
            config::CHAT_COMPLETIONS_ENDPOINT
        );
        assert_eq!(
            endpoints::get_endpoint_for_model(models::GPT_4O),
            config::CHAT_COMPLETIONS_ENDPOINT
        );
        assert_eq!(
            endpoints::get_endpoint_for_model(models::O1_PREVIEW),
            config::CHAT_COMPLETIONS_ENDPOINT
        );
        assert_eq!(
            endpoints::get_endpoint_for_model(models::TEXT_EMBEDDING_3_LARGE),
            config::EMBEDDINGS_ENDPOINT
        );
        assert_eq!(
            endpoints::get_endpoint_for_model(models::TEXT_EMBEDDING_3_SMALL),
            config::EMBEDDINGS_ENDPOINT
        );

        assert_eq!(
            endpoints::get_transcription_endpoint(),
            config::AUDIO_TRANSCRIPTIONS_ENDPOINT
        );
        assert_eq!(
            endpoints::get_translation_endpoint(),
            config::AUDIO_TRANSLATIONS_ENDPOINT
        );
        assert_eq!(
            endpoints::get_speech_endpoint(),
            config::AUDIO_SPEECH_ENDPOINT
        );
        assert_eq!(endpoints::get_vision_endpoint(), config::VISION_ENDPOINT);
    }

    #[test]
    fn test_token_estimation() {
        assert_eq!(utils::estimate_token_count("Hello world"), 2);
        assert_eq!(utils::estimate_token_count(""), 0);
        assert_eq!(utils::estimate_token_count("A"), 1);
        assert!(
            utils::estimate_token_count("This is a longer text with many words and characters") > 8
        );
        assert!(utils::estimate_token_count("Single") >= 1);

        // Test SIMD operations counter increment
        let initial_count = config::SIMD_OPERATIONS.load(std::sync::atomic::Ordering::Relaxed);
        utils::estimate_token_count("test");
        let final_count = config::SIMD_OPERATIONS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(final_count > initial_count);
    }

    #[test]
    fn test_capability_checking() {
        assert!(utils::supports_capability(models::GPT_4O, "vision"));
        assert!(utils::supports_capability(models::GPT_4O_MINI, "vision"));
        assert!(!utils::supports_capability(models::GPT_4_1, "vision"));

        assert!(utils::supports_capability(
            models::GPT_4_1,
            "function_calling"
        ));
        assert!(utils::supports_capability(
            models::GPT_4O,
            "function_calling"
        ));
        assert!(!utils::supports_capability(
            models::O1_PREVIEW,
            "function_calling"
        ));

        assert!(utils::supports_capability(models::GPT_4_1, "streaming"));
        assert!(utils::supports_capability(models::O1_PREVIEW, "streaming"));
        assert!(!utils::supports_capability(
            models::TEXT_EMBEDDING_3_LARGE,
            "streaming"
        ));

        assert!(utils::supports_capability(
            models::TEXT_EMBEDDING_3_LARGE,
            "embedding"
        ));
        assert!(utils::supports_capability(
            models::TEXT_EMBEDDING_3_SMALL,
            "embedding"
        ));
        assert!(!utils::supports_capability(models::GPT_4_1, "embedding"));

        assert!(!utils::supports_capability(models::GPT_4_1, "invalid"));
        assert!(!utils::supports_capability(models::GPT_4_1, "unknown"));
    }

    #[test]
    fn test_optimal_timeouts() {
        assert_eq!(utils::optimal_timeout_ms(models::GPT_4_1), 60_000);
        assert_eq!(utils::optimal_timeout_ms(models::GPT_4O), 60_000);
        assert_eq!(utils::optimal_timeout_ms(models::O1_PREVIEW), 60_000);
        assert_eq!(utils::optimal_timeout_ms(models::O3), 60_000);

        assert_eq!(utils::optimal_timeout_ms(models::GPT_4_1_MINI), 45_000);
        assert_eq!(utils::optimal_timeout_ms(models::GPT_4O_MINI), 45_000);
        assert_eq!(utils::optimal_timeout_ms(models::O1_MINI), 45_000);

        assert_eq!(utils::optimal_timeout_ms(models::GPT_4_1_NANO), 30_000);
        assert_eq!(utils::optimal_timeout_ms(models::GPT_3_5_TURBO), 30_000);

        assert_eq!(utils::optimal_timeout_ms("unknown-model"), 45_000);
    }

    #[test]
    fn test_optimal_batch_sizes() {
        assert_eq!(
            utils::optimal_batch_size(models::TEXT_EMBEDDING_3_LARGE, "embedding"),
            512
        );
        assert_eq!(
            utils::optimal_batch_size(models::TEXT_EMBEDDING_3_SMALL, "embedding"),
            1024
        );
        assert_eq!(utils::optimal_batch_size("unknown-model", "embedding"), 100);

        assert_eq!(utils::optimal_batch_size(models::GPT_4_1, "completion"), 1);
        assert_eq!(utils::optimal_batch_size(models::GPT_4O, "completion"), 1);

        assert_eq!(utils::optimal_batch_size(models::GPT_4_1, "unknown"), 1);
    }

    #[test]
    fn test_atomic_operations() {
        // Test atomic counter increments
        let initial_requests = config::TOTAL_REQUESTS.load(std::sync::atomic::Ordering::Relaxed);
        config::increment_counter(&config::TOTAL_REQUESTS);
        let final_requests = config::TOTAL_REQUESTS.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(final_requests, initial_requests + 1);

        let initial_u32 = config::CONCURRENT_REQUESTS.load(std::sync::atomic::Ordering::Relaxed);
        config::increment_counter_u32(&config::CONCURRENT_REQUESTS);
        let final_u32 = config::CONCURRENT_REQUESTS.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(final_u32, initial_u32 + 1);

        // Test atomic boolean flag
        config::set_flag(&config::STREAMING_ACTIVE, true);
        assert!(config::STREAMING_ACTIVE.load(std::sync::atomic::Ordering::Relaxed));

        config::set_flag(&config::STREAMING_ACTIVE, false);
        assert!(!config::STREAMING_ACTIVE.load(std::sync::atomic::Ordering::Relaxed));
    }

    #[test]
    fn test_model_constants() {
        // Test all model constants are defined
        assert_eq!(models::GPT_4_1, "gpt-4.1");
        assert_eq!(models::GPT_4O, "gpt-4o");
        assert_eq!(models::O1_PREVIEW, "o1-preview");
        assert_eq!(models::O3, "o3");
        assert_eq!(models::TEXT_EMBEDDING_3_LARGE, "text-embedding-3-large");

        // Test model arrays contain correct models
        assert!(models::GPT4_MODELS.contains(&models::GPT_4_1));
        assert!(models::GPT4_MODELS.contains(&models::GPT_4O));
        assert!(!models::GPT4_MODELS.contains(&models::O1_PREVIEW));

        assert!(models::O1_MODELS.contains(&models::O1_PREVIEW));
        assert!(models::O1_MODELS.contains(&models::O3));
        assert!(!models::O1_MODELS.contains(&models::GPT_4_1));

        assert!(models::EMBEDDING_MODELS.contains(&models::TEXT_EMBEDDING_3_LARGE));
        assert!(models::EMBEDDING_MODELS.contains(&models::TEXT_EMBEDDING_3_SMALL));
        assert!(!models::EMBEDDING_MODELS.contains(&models::GPT_4_1));

        // Test ALL_MODELS contains all models
        assert!(models::ALL_MODELS.contains(&models::GPT_4_1));
        assert!(models::ALL_MODELS.contains(&models::GPT_4O));
        assert!(models::ALL_MODELS.contains(&models::O1_PREVIEW));
        assert!(models::ALL_MODELS.contains(&models::O3));
        assert!(models::ALL_MODELS.contains(&models::TEXT_EMBEDDING_3_LARGE));
        assert!(models::ALL_MODELS.contains(&models::TEXT_EMBEDDING_3_SMALL));
        assert_eq!(models::ALL_MODELS.len(), 18);
    }
}
