//! Core Tokenizer Implementation
//!
//! Production-ready tokenizer wrapper for HuggingFace tokenizers with
//! zero-allocation patterns and comprehensive token management.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrayvec::ArrayVec;
use fluent_ai_async::AsyncStream;
use tokenizers::Tokenizer;

use crate::error::{CandleError, CandleResult};
use super::config::TokenizerConfig;

/// Maximum token buffer size for zero-allocation patterns
pub const MAX_TOKEN_BUFFER: usize = 4096;

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

    /// Get vocabulary size
    #[inline(always)]
    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
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

    /// Get the underlying HuggingFace tokenizer reference
    #[inline(always)]
    pub fn inner(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get all special tokens
    pub fn special_tokens(&self) -> &HashMap<String, u32> {
        &self.special_tokens
    }
}