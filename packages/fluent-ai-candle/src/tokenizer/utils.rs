//! Tokenizer Utility Functions
//!
//! Provides convenience functions for loading popular tokenizers,
//! model-specific configurations, and validation utilities.

use fluent_ai_async::{AsyncStream, emit, handle_error};

use crate::error::{CandleError, CandleResult};
use super::core::CandleTokenizer;
use super::config::{TokenizerConfig, TokenizerConfigBuilder};

/// Load popular tokenizer by name from HuggingFace Hub using AsyncStream
pub fn load_popular_tokenizer(name: &str) -> AsyncStream<CandleTokenizer> {
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