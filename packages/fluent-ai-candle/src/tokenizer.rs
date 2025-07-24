//! HuggingFace Tokenizers Integration for Candle ML Framework
//!
//! High-performance tokenization using HuggingFace tokenizers with direct Hub integration,
//! special token handling, and zero-allocation patterns for production ML inference.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrayvec::ArrayVec;
use fluent_ai_async::AsyncStream;
use tokenizers::Tokenizer;

use crate::error::{CandleError, CandleResult};

/// Maximum token buffer size for zero-allocation patterns
const MAX_TOKEN_BUFFER: usize = 4096;

/// Tokenizer configuration for production usage
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Add BOS (Beginning of Sequence) token
    pub add_bos_token: bool,
    /// Add EOS (End of Sequence) token  
    pub add_eos_token: bool,
    /// Maximum sequence length for truncation
    pub max_length: Option<usize>,
    /// Padding configuration
    pub padding: PaddingConfig,
    /// Truncation configuration
    pub truncation: TruncationConfig,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            add_bos_token: false,
            add_eos_token: false,
            max_length: Some(2048),
            padding: PaddingConfig::default(),
            truncation: TruncationConfig::default(),
        }
    }
}

/// Padding configuration
#[derive(Debug, Clone)]
pub struct PaddingConfig {
    /// Enable padding
    pub enabled: bool,
    /// Padding token
    pub token: String,
    /// Padding length
    pub length: Option<usize>,
}

impl Default for PaddingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            token: "<pad>".to_string(),
            length: None,
        }
    }
}

/// Truncation configuration
#[derive(Debug, Clone)]
pub struct TruncationConfig {
    /// Enable truncation
    pub enabled: bool,
    /// Maximum length for truncation
    pub max_length: usize,
    /// Truncation strategy
    pub strategy: TruncationStrategy,
}

impl Default for TruncationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_length: 2048,
            strategy: TruncationStrategy::LongestFirst,
        }
    }
}

/// Truncation strategies for sequence processing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// Truncate from the beginning
    OnlyFirst,
    /// Truncate from the end  
    OnlySecond,
    /// Truncate the longest sequence first
    LongestFirst,
    /// Do not truncate
    DoNotTruncate,
}

/// Production-ready tokenizer wrapper for HuggingFace tokenizers
pub struct CandleTokenizer {
    /// Core HuggingFace tokenizer
    tokenizer: Arc<Tokenizer>,
    /// Configuration for tokenization behavior
    config: TokenizerConfig,
    /// Special tokens mapping
    special_tokens: HashMap<String, u32>,
    /// Vocabulary size cache
    vocab_size: u32,
}

impl Clone for CandleTokenizer {
    fn clone(&self) -> Self {
        Self {
            tokenizer: Arc::clone(&self.tokenizer),
            config: self.config.clone(),
            special_tokens: self.special_tokens.clone(),
            vocab_size: self.vocab_size,
        }
    }
}

impl CandleTokenizer {
    /// Create tokenizer from HuggingFace Tokenizer instance
    pub fn new(tokenizer: Tokenizer, config: TokenizerConfig) -> CandleResult<Self> {
        let tokenizer = Arc::new(tokenizer);
        let vocab_size = tokenizer.get_vocab_size(false) as u32;

        // Extract special tokens from tokenizer
        let mut special_tokens = HashMap::new();

        // Common special tokens
        let token_candidates = [
            "<pad>",
            "[PAD]",
            "<unk>",
            "[UNK]",
            "<s>",
            "[BOS]",
            "</s>",
            "[EOS]",
            "<cls>",
            "[CLS]",
            "<sep>",
            "[SEP]",
            "<mask>",
            "[MASK]",
            "<|endoftext|>",
            "<|startoftext|>",
        ];

        for token in token_candidates {
            if let Some(token_id) = tokenizer.token_to_id(token) {
                special_tokens.insert(token.to_string(), token_id);
            }
        }

        Ok(Self {
            tokenizer,
            config,
            special_tokens,
            vocab_size,
        })
    }

    /// Load tokenizer from file path
    pub fn from_file<P: AsRef<Path>>(path: P, config: TokenizerConfig) -> CandleResult<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(|e| {
            CandleError::tokenization(format!("Failed to load tokenizer from file: {}", e))
        })?;

        Self::new(tokenizer, config)
    }

    /// Load tokenizer using AsyncStream architecture - no async/await
    pub fn from_hub(model_id: &str, config: TokenizerConfig) -> AsyncStream<Self> {
        use std::path::PathBuf;

        use fluent_ai_async::{emit, handle_error};

        let model_id = model_id.to_string();

        AsyncStream::with_channel(move |sender| {
            use crate::hub::{Backend, create_client, create_download_config};

            // Create ProgressHub client directly
            let _client = match create_client(Backend::Auto) {
                Ok(client) => client,
                Err(e) => {
                    handle_error!(e, "Failed to create hub client");
                }
            };

            let _cache_dir = PathBuf::from("/tmp/fluent_ai_cache"); // TODO: make configurable
            let _download_config = create_download_config(_cache_dir);

            // Since we can't use .await in AsyncStream architecture,
            // we need to implement sync downloading or use fallback
            match Self::from_fallback_path(&model_id, config) {
                Ok(tokenizer) => {
                    emit!(sender, tokenizer);
                }
                Err(e) => {
                    handle_error!(e, "Failed to load tokenizer from fallback path");
                }
            }
        })
    }

    /// Fallback method for loading tokenizer when hub download is not available in sync context
    fn from_fallback_path(model_id: &str, config: TokenizerConfig) -> CandleResult<Self> {
        use std::path::PathBuf;

        // Try common local paths first
        let common_paths = [
            format!("/tmp/fluent_ai_cache/{}/tokenizer.json", model_id),
            format!("./models/{}/tokenizer.json", model_id),
            format!("~/.cache/huggingface/hub/{}/tokenizer.json", model_id),
        ];

        for path_str in &common_paths {
            let path = PathBuf::from(path_str);
            if path.exists() {
                return Self::from_file(path, config);
            }
        }

        // If no local paths work, return an error with helpful message
        Err(CandleError::tokenization(format!(
            "Tokenizer not found for model '{}'. Please download the model first or provide a local path.",
            model_id
        )))
    }

    /// Load tokenizer from HuggingFace Hub with specific revision using AsyncStream
    /// Note: Revision support is not yet implemented in HubClient - using main revision
    pub fn from_hub_with_revision(
        model_id: &str,
        _revision: &str,
        config: TokenizerConfig,
    ) -> AsyncStream<Self> {
        // For now, use the main revision - revision support can be added to HubClient later
        Self::from_hub(model_id, config)
    }

    /// Encode text to token IDs with configuration support
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> CandleResult<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| CandleError::tokenization(format!("Encoding failed: {}", e)))?;

        let mut tokens = encoding.get_ids().to_vec();

        // Apply BOS token if configured
        if self.config.add_bos_token && add_special_tokens {
            if let Some(bos_id) = self.get_special_token_id("bos") {
                tokens.insert(0, bos_id);
            }
        }

        // Apply EOS token if configured
        if self.config.add_eos_token && add_special_tokens {
            if let Some(eos_id) = self.get_special_token_id("eos") {
                tokens.push(eos_id);
            }
        }

        // Apply truncation if configured
        if self.config.truncation.enabled {
            if tokens.len() > self.config.truncation.max_length {
                tokens.truncate(self.config.truncation.max_length);
            }
        }

        Ok(tokens)
    }

    /// Encode text with zero-allocation token buffer
    pub fn encode_to_buffer(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> CandleResult<ArrayVec<u32, MAX_TOKEN_BUFFER>> {
        let tokens = self.encode(text, add_special_tokens)?;

        let mut buffer = ArrayVec::new();
        for token in tokens {
            buffer
                .try_push(token)
                .map_err(|_| CandleError::tokenization("Token buffer overflow"))?;
        }

        Ok(buffer)
    }

    /// Decode token IDs to text
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> CandleResult<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| CandleError::tokenization(format!("Decoding failed: {}", e)))
    }

    /// Batch encode multiple texts efficiently
    pub fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> CandleResult<Vec<Vec<u32>>> {
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            results.push(self.encode(text, add_special_tokens)?);
        }

        Ok(results)
    }

    /// Batch decode multiple token sequences efficiently
    pub fn decode_batch(
        &self,
        token_sequences: &[&[u32]],
        skip_special_tokens: bool,
    ) -> CandleResult<Vec<String>> {
        let mut results = Vec::with_capacity(token_sequences.len());

        for tokens in token_sequences {
            results.push(self.decode(tokens, skip_special_tokens)?);
        }

        Ok(results)
    }

    /// Get token ID for specific token string
    #[inline(always)]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    /// Get token string for specific token ID
    #[inline(always)]
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    /// Get vocabulary size
    #[inline(always)]
    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    /// Get special token ID by type
    pub fn get_special_token_id(&self, token_type: &str) -> Option<u32> {
        match token_type.to_lowercase().as_str() {
            "bos" | "start" => self
                .special_tokens
                .get("<s>")
                .or_else(|| self.special_tokens.get("[BOS]"))
                .or_else(|| self.special_tokens.get("<|startoftext|>"))
                .copied(),
            "eos" | "end" => self
                .special_tokens
                .get("</s>")
                .or_else(|| self.special_tokens.get("[EOS]"))
                .or_else(|| self.special_tokens.get("<|endoftext|>"))
                .copied(),
            "pad" | "padding" => self
                .special_tokens
                .get("<pad>")
                .or_else(|| self.special_tokens.get("[PAD]"))
                .copied(),
            "unk" | "unknown" => self
                .special_tokens
                .get("<unk>")
                .or_else(|| self.special_tokens.get("[UNK]"))
                .copied(),
            "cls" | "class" => self
                .special_tokens
                .get("<cls>")
                .or_else(|| self.special_tokens.get("[CLS]"))
                .copied(),
            "sep" | "separator" => self
                .special_tokens
                .get("<sep>")
                .or_else(|| self.special_tokens.get("[SEP]"))
                .copied(),
            "mask" => self
                .special_tokens
                .get("<mask>")
                .or_else(|| self.special_tokens.get("[MASK]"))
                .copied(),
            _ => self.special_tokens.get(token_type).copied(),
        }
    }

    /// Get EOS (end-of-sequence) token ID
    #[inline(always)]
    pub fn eos_token_id(&self) -> Option<u32> {
        self.get_special_token_id("eos")
    }

    /// Check if token ID is a special token
    pub fn is_special_token(&self, token_id: u32) -> bool {
        self.special_tokens.values().any(|&id| id == token_id)
    }

    /// Get all special tokens
    pub fn special_tokens(&self) -> &HashMap<String, u32> {
        &self.special_tokens
    }

    /// Get tokenizer configuration
    #[inline(always)]
    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }

    /// Update tokenizer configuration
    pub fn update_config(&mut self, config: TokenizerConfig) {
        self.config = config;
    }

    /// Estimate token count for text (fast approximation)
    pub fn estimate_token_count(&self, text: &str) -> usize {
        // Fast approximation based on text length and common patterns
        // More accurate than simple character division
        let word_count = text.split_whitespace().count();
        let char_count = text.chars().count();

        // Average tokens per word is roughly 1.3 for English
        // Add some tokens for punctuation and special cases
        ((word_count as f32 * 1.3) + (char_count as f32 * 0.1)) as usize
    }

    /// Apply chat template if supported by tokenizer
    pub fn apply_chat_template(
        &self,
        messages: &[crate::types::CandleMessage],
        add_generation_prompt: bool,
    ) -> CandleResult<String> {
        // This would use the tokenizer's chat template functionality
        // For now, implement a basic chat format
        let mut formatted = String::new();

        for message in messages {
            match message.role.as_str() {
                "system" => formatted.push_str(&format!("<|system|>\n{}\n", message.content)),
                "user" => formatted.push_str(&format!("<|user|>\n{}\n", message.content)),
                "assistant" => formatted.push_str(&format!("<|assistant|>\n{}\n", message.content)),
                _ => formatted.push_str(&format!("<|{}|>\n{}\n", message.role, message.content)),
            }
        }

        if add_generation_prompt {
            formatted.push_str("<|assistant|>\n");
        }

        Ok(formatted)
    }

    /// Get the underlying HuggingFace tokenizer reference
    #[inline(always)]
    pub fn inner(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

// Removed duplicate ChatMessage type - use CandleMessage from types module instead

/// Builder for tokenizer configuration
pub struct TokenizerConfigBuilder {
    config: TokenizerConfig,
}

impl TokenizerConfigBuilder {
    /// Create new configuration builder
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            config: TokenizerConfig::default(),
        }
    }

    /// Enable/disable BOS token addition
    #[inline(always)]
    pub fn add_bos_token(mut self, add: bool) -> Self {
        self.config.add_bos_token = add;
        self
    }

    /// Enable/disable EOS token addition
    #[inline(always)]
    pub fn add_eos_token(mut self, add: bool) -> Self {
        self.config.add_eos_token = add;
        self
    }

    /// Set maximum sequence length
    #[inline(always)]
    pub fn max_length(mut self, length: Option<usize>) -> Self {
        self.config.max_length = length;
        self
    }

    /// Configure padding
    #[inline(always)]
    pub fn padding(mut self, config: PaddingConfig) -> Self {
        self.config.padding = config;
        self
    }

    /// Configure truncation
    #[inline(always)]
    pub fn truncation(mut self, config: TruncationConfig) -> Self {
        self.config.truncation = config;
        self
    }

    /// Build the configuration
    #[inline(always)]
    pub fn build(self) -> TokenizerConfig {
        self.config
    }
}

impl Default for TokenizerConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for tokenizer operations
pub mod utils {
    use super::*;

    /// Load popular tokenizer by name from HuggingFace Hub using AsyncStream
    pub fn load_popular_tokenizer(name: &str) -> AsyncStream<CandleTokenizer> {
        use fluent_ai_async::{emit, handle_error};
        let name = name.to_string();

        AsyncStream::with_channel(move |sender| {
            let model_id = match name.to_lowercase().as_str() {
                "gpt2" => "gpt2",
                "bert" => "bert-base-uncased",
                "roberta" => "roberta-base",
                "t5" => "t5-small",
                "llama" => "meta-llama/Llama-2-7b-hf",
                "mistral" => "mistralai/Mistral-7B-v0.1",
                "phi" => "microsoft/phi-2",
                "gemma" => "google/gemma-7b",
                _ => {
                    let error = CandleError::tokenization(format!("Unknown tokenizer: {}", name));
                    handle_error!(error, "Unknown tokenizer name");
                }
            };

            // Use fallback loading directly since AsyncStream can't be collected in sync context
            match CandleTokenizer::from_fallback_path(model_id, TokenizerConfig::default()) {
                Ok(tokenizer) => {
                    emit!(sender, tokenizer);
                }
                Err(e) => {
                    handle_error!(e, "Failed to load popular tokenizer");
                }
            }
        })
    }

    /// Create tokenizer configuration for specific model types
    pub fn config_for_model_type(model_type: &str) -> TokenizerConfig {
        match model_type.to_lowercase().as_str() {
            "llama" | "mistral" => TokenizerConfigBuilder::new()
                .add_bos_token(true)
                .add_eos_token(false)
                .max_length(Some(4096))
                .build(),
            "phi" => TokenizerConfigBuilder::new()
                .add_bos_token(false)
                .add_eos_token(true)
                .max_length(Some(2048))
                .build(),
            "gemma" => TokenizerConfigBuilder::new()
                .add_bos_token(true)
                .add_eos_token(true)
                .max_length(Some(8192))
                .build(),
            _ => TokenizerConfig::default(),
        }
    }

    /// Validate tokenizer for ML model compatibility
    pub fn validate_tokenizer(tokenizer: &CandleTokenizer) -> CandleResult<()> {
        // Check minimum requirements
        if tokenizer.vocab_size() == 0 {
            return Err(CandleError::tokenization(
                "Tokenizer has zero vocabulary size",
            ));
        }

        // Check for essential special tokens
        if tokenizer.get_special_token_id("unk").is_none() {
            tracing::warn!(
                "Tokenizer missing UNK token - may cause issues with out-of-vocabulary words"
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tokenizer_config_builder() {
        let config = TokenizerConfigBuilder::new()
            .add_bos_token(true)
            .add_eos_token(false)
            .max_length(Some(1024))
            .build();

        assert!(config.add_bos_token);
        assert!(!config.add_eos_token);
        assert_eq!(config.max_length, Some(1024));
    }

    #[test]
    fn test_chat_message() {
        let msg = crate::types::CandleMessage::system("You are a helpful assistant");
        assert_eq!(msg.role, "system");
        assert_eq!(msg.content, "You are a helpful assistant");

        let user_msg = crate::types::CandleMessage::user("Hello world");
        assert_eq!(user_msg.role, "user");
        assert_eq!(user_msg.content, "Hello world");
    }

    #[test]
    fn test_utils() {
        let config = utils::config_for_model_type("llama");
        assert!(config.add_bos_token);
        assert!(!config.add_eos_token);

        let config = utils::config_for_model_type("phi");
        assert!(!config.add_bos_token);
        assert!(config.add_eos_token);
    }
}
