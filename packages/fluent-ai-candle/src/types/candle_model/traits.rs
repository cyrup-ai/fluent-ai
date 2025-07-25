//! Core traits for AI models and their capabilities

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::CandleError;
use crate::types::CandleModelInfo;
use crate::types::CandleUsage;

/// Core trait for all AI models
///
/// This trait provides the foundation for all AI models in the system.
/// It defines the basic functionality that all models must implement,
/// including model information and basic capabilities.
pub trait Model: Send + Sync + std::fmt::Debug + 'static {
    /// Get the model's information
    fn info(&self) -> &'static CandleModelInfo;

    /// Get the model's name
    #[inline]
    fn name(&self) -> &'static str {
        self.info().name()
    }

    /// Get the model's provider name
    #[inline]
    fn provider(&self) -> &'static str {
        self.info().provider()
    }

    /// Get the model's maximum input tokens
    #[inline]
    fn max_input_tokens(&self) -> Option<u32> {
        self.info().max_input_tokens.map(|n| n.get())
    }

    /// Get the model's maximum output tokens
    #[inline]
    fn max_output_tokens(&self) -> Option<u32> {
        self.info().max_output_tokens.map(|n| n.get())
    }

    /// Check if the model supports vision
    #[inline]
    fn supports_vision(&self) -> bool {
        self.info().has_vision()
    }

    /// Check if the model supports function calling
    #[inline]
    fn supports_function_calling(&self) -> bool {
        self.info().has_function_calling()
    }

    /// Check if the model supports streaming
    #[inline]
    fn supports_streaming(&self) -> bool {
        self.info().has_streaming()
    }

    /// Check if the model requires max_tokens to be specified
    #[inline]
    fn requires_max_tokens(&self) -> bool {
        self.info().requires_max_tokens()
    }
}

/// A message in a chat conversation
// REMOVED: Duplicate ChatMessage enum - use CandleMessage from types::candle_chat::message instead

/// A function call made by the model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    /// The name of the function to call
    pub name: String,
    /// The arguments to pass to the function (JSON-encoded string)
    pub arguments: String,
}

/// A function definition that can be called by the model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionDefinition {
    /// The name of the function to be called
    pub name: String,
    /// A description of what the function does
    pub description: Option<String>,
    /// The parameters the function accepts, described as a JSON Schema object
    pub parameters: serde_json::Value,
}

/// Parameters for text generation
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GenerationParams {
    /// The maximum number of tokens to generate
    pub max_tokens: Option<u32>,

    /// Controls randomness: lower means more deterministic
    pub temperature: Option<f32>,

    /// Nucleus sampling: limits the next token selection to a subset of the vocabulary
    pub top_p: Option<f32>,

    /// Limits the number of highest probability vocabulary tokens to consider
    pub top_k: Option<u32>,

    /// Penalty for repeating tokens in the generation
    pub frequency_penalty: Option<f32>,

    /// Penalty for repeating tokens that appear in the prompt
    pub presence_penalty: Option<f32>,

    /// Stop sequences where the API will stop generating further tokens
    pub stop_sequences: Option<Vec<String>>,

    /// Whether to stream the response
    pub stream: bool,
}

/// A chunk of generated text
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextChunk {
    /// The generated text
    pub text: String,
    /// The token IDs of the generated text
    pub token_ids: Vec<u32>,
    /// Whether this is the last chunk
    pub is_complete: bool,
    /// The reason generation stopped (if complete)
    pub finish_reason: Option<String>,
    /// Token usage for this chunk (if available)
    pub usage: Option<CandleUsage>,
}

/// Request for text generation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextGenerationRequest {
    /// The input prompt
    pub prompt: String,
    /// Generation parameters
    pub params: GenerationParams,
}

/// Request for chat completion
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionRequest {
    /// The conversation messages
    pub messages: Vec<crate::types::candle_chat::message::CandleMessage>,
    /// Generation parameters
    pub params: GenerationParams,
    /// Optional function definitions
    pub functions: Option<Vec<FunctionDefinition>>,
}

/// Request for embedding generation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingRequest {
    /// The text(s) to embed
    pub texts: Vec<String>,
}

/// An embedding result
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Embedding {
    /// The embedding vector
    pub vector: Vec<f32>,
    /// The original text that was embedded
    pub text: String,
    /// Token usage for this embedding (if available)
    pub usage: Option<CandleUsage>,
}

/// Fine-tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FineTuningConfig {
    /// Learning rate for training
    pub learning_rate: Option<f32>,
    /// Number of training epochs
    pub epochs: Option<u32>,
    /// Batch size for training
    pub batch_size: Option<u32>,
    /// Validation split ratio
    pub validation_split: Option<f32>,
}

/// Trait for models that can generate text
pub trait TextGenerationCapable: Model {
    /// Check if the model supports text generation
    fn supports_text_generation(&self) -> bool {
        true
    }

    /// Get the default generation parameters for this model
    fn default_generation_params(&self) -> GenerationParams {
        GenerationParams::default()
    }

    /// Get the maximum prompt length for this model
    fn max_prompt_length(&self) -> Option<usize> {
        self.max_input_tokens().map(|n| n as usize)
    }
}

/// Trait for models that can handle chat conversations
pub trait ChatCompletionCapable: Model {
    /// Check if the model supports chat conversations
    fn supports_chat(&self) -> bool {
        true
    }

    /// Get the maximum conversation length for this model
    fn max_conversation_length(&self) -> Option<usize> {
        self.max_input_tokens().map(|n| n as usize)
    }

    /// Get the supported message types
    fn supported_message_types(&self) -> Vec<&'static str> {
        vec!["system", "user", "assistant"]
    }

    /// Check if the model supports function calling in chat
    fn supports_function_calling_in_chat(&self) -> bool {
        self.supports_function_calling()
    }
}

/// Trait for models that can generate embeddings
pub trait EmbeddingCapable: Model {
    /// Get the dimensionality of the embeddings
    fn embedding_dimensions(&self) -> usize;

    /// Check if the model supports embeddings
    fn supports_embeddings(&self) -> bool {
        true
    }

    /// Get the maximum text length for embedding
    fn max_embedding_text_length(&self) -> Option<usize> {
        self.max_input_tokens().map(|n| n as usize)
    }

    /// Get the maximum batch size for embedding
    fn max_embedding_batch_size(&self) -> Option<usize> {
        Some(100) // Default reasonable batch size
    }

    /// Get the expected embedding range (min, max)
    fn embedding_range(&self) -> Option<(f32, f32)> {
        Some((-1.0, 1.0)) // Default normalized range
    }
}

/// Trait for models that can be fine-tuned
pub trait FineTunable: Model {
    /// Check if the model supports fine-tuning
    fn supports_fine_tuning(&self) -> bool {
        true
    }

    /// Get the supported fine-tuning data formats
    fn supported_data_formats(&self) -> Vec<&'static str> {
        vec!["json", "jsonl", "csv"]
    }

    /// Get the minimum dataset size required for fine-tuning
    fn min_dataset_size(&self) -> Option<usize> {
        Some(100) // Default minimum
    }

    /// Get the maximum dataset size supported for fine-tuning
    fn max_dataset_size(&self) -> Option<usize> {
        Some(100_000) // Default maximum
    }

    /// Get the default fine-tuning configuration
    fn default_fine_tuning_config(&self) -> FineTuningConfig {
        FineTuningConfig {
            learning_rate: Some(0.0001),
            epochs: Some(3),
            batch_size: Some(32),
            validation_split: Some(0.1),
        }
    }

    /// Check if the model supports saving/loading fine-tuned versions
    fn supports_model_persistence(&self) -> bool {
        true
    }
}

/// A model that can be used for multiple tasks
pub trait MultiTaskCapable:
    TextGenerationCapable + ChatCompletionCapable + EmbeddingCapable
{
}

// Blanket implementation for types that implement all required traits
impl<T> MultiTaskCapable for T where
    T: TextGenerationCapable + ChatCompletionCapable + EmbeddingCapable
{
}

/// A boxed model that can be used for any task
pub type AnyModel = Arc<dyn MultiTaskCapable + Send + Sync>;

/// A boxed text generation model
pub type AnyTextGenerationCapable = Arc<dyn TextGenerationCapable + Send + Sync>;

/// A boxed chat completion model
pub type AnyChatCompletionCapable = Arc<dyn ChatCompletionCapable + Send + Sync>;

/// A boxed embedding model
pub type AnyEmbeddingCapable = Arc<dyn EmbeddingCapable + Send + Sync>;

/// Trait for models that can be loaded and unloaded with zero-allocation design
pub trait CandleLoadableModel: Model {
    /// Load the model from the given path with blazing-fast synchronous operation
    fn load(&mut self, path: &str) -> Result<(), CandleError>;

    /// Unload the model with efficient resource cleanup
    fn unload(&mut self) -> Result<(), CandleError>;

    /// Check if the model is currently loaded with zero-cost inline access
    fn is_loaded(&self) -> bool;
}

/// Trait for models that track usage statistics with zero-allocation design
pub trait CandleUsageTrackingModel: Model {
    /// Get current usage statistics with zero-allocation access
    fn usage(&self) -> crate::types::CandleUsage;

    /// Reset usage statistics with efficient state management
    fn reset_usage(&mut self);
}

/// Trait for models that can perform completions with zero-allocation design
pub trait CandleCompletionModel: Model {
    /// Generate a completion from a request with blazing-fast streaming builder pattern
    fn complete(
        &self,
        request: crate::types::CandleCompletionRequest,
    ) -> crate::client::CandleCompletionBuilder;

    /// Generate streaming completions from a request with zero-allocation streaming builder
    fn stream_complete(
        &self,
        request: crate::types::CandleCompletionRequest,
    ) -> crate::types::CandleStreamingResponse;

    /// Action method: generate completion from prompt with unwrapped AsyncStream
    fn prompt<'a>(
        &'a self,
        prompt: &str,
        params: &'a crate::types::CandleCompletionParams,
    ) -> fluent_ai_async::AsyncStream<crate::types::CandleCompletionChunk>;
}

/// Trait for models that can be configured with zero-allocation design
pub trait CandleConfigurableModel: Model {
    /// Configuration type for this model - generic for flexibility
    type Config;

    /// Apply configuration to the model with blazing-fast synchronous operation
    fn configure(&mut self, config: Self::Config) -> Result<(), CandleError>;

    /// Get current configuration with zero-allocation reference access
    fn config(&self) -> &Self::Config;
}

/// Trait for models that provide tokenization with zero-allocation design
pub trait CandleTokenizerModel: Model {
    /// Tokenize text into token IDs with blazing-fast synchronous operation
    fn tokenize(&self, text: &str) -> Result<smallvec::SmallVec<u32, 64>, CandleError>;

    /// Convert token IDs back to text with efficient synchronous operation
    fn detokenize(&self, tokens: &[u32]) -> Result<Arc<str>, CandleError>;

    /// Get the vocabulary size with zero-cost inline access
    fn vocab_size(&self) -> u32;
}
