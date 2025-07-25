//! Token Decoding Operations
//!
//! Provides efficient token-to-text decoding with special token handling
//! and batch processing capabilities.

use crate::error::{CandleError, CandleResult};
use super::core::CandleTokenizer;

impl CandleTokenizer {
    /// Decode token IDs to text
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> CandleResult<String> {
        self.inner()
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| CandleError::tokenization(format!("Decoding failed: {}", e)))
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
}