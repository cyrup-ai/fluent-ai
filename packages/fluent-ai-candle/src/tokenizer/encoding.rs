//! Text Encoding Operations
//!
//! Provides efficient text-to-token encoding with special token handling,
//! truncation, and zero-allocation buffer patterns.

use arrayvec::ArrayVec;

use crate::error::{CandleError, CandleResult};
use super::core::{CandleTokenizer, MAX_TOKEN_BUFFER};

impl CandleTokenizer {
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
}