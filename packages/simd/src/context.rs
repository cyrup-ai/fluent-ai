//! Processing context for SIMD-accelerated operations

use std::time::Instant;

/// Context for processing logits with token history and configuration
#[derive(Debug, Clone)]
pub struct ProcessingContext {
    /// Current temperature for sampling (can be overridden per-call)
    pub temperature: f32,

    /// Current top-k value (can be overridden per-call)
    pub top_k: Option<usize>,

    /// Current top-p value (can be overridden per-call)
    pub top_p: Option<f32>,

    /// History of token IDs in the current sequence
    pub token_history: Vec<u32>,

    /// Optional timestamp when processing started
    pub start_time: Option<Instant>,

    /// Optional maximum number of new tokens to generate
    pub max_new_tokens: Option<usize>,

    /// Optional stop tokens that will halt generation if encountered
    pub stop_tokens: Vec<u32>,
}

impl Default for ProcessingContext {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            token_history: Vec::new(),
            start_time: None,
            max_new_tokens: None,
            stop_tokens: Vec::new(),
        }
    }
}

impl ProcessingContext {
    /// Create a new processing context with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the token history as a slice
    pub fn token_history(&self) -> &[u32] {
        &self.token_history
    }

    /// Set the temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the top-k value
    pub fn with_top_k(mut self, top_k: Option<usize>) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the top-p value
    pub fn with_top_p(mut self, top_p: Option<f32>) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set the token history
    pub fn with_token_history(mut self, history: Vec<u32>) -> Self {
        self.token_history = history;
        self
    }

    /// Add tokens to the history
    pub fn extend_history(&mut self, tokens: &[u32]) {
        self.token_history.extend_from_slice(tokens);
    }

    /// Get the number of tokens in history
    pub fn history_len(&self) -> usize {
        self.token_history.len()
    }

    /// Check if processing should stop based on stop tokens or max tokens
    pub fn should_stop(&self, new_token: u32, generated_count: usize) -> bool {
        // Check for stop tokens
        if self.stop_tokens.contains(&new_token) {
            return true;
        }

        // Check max new tokens
        if let Some(max_tokens) = self.max_new_tokens {
            if generated_count >= max_tokens {
                return true;
            }
        }

        // Check for max sequence length (safety check)
        if self.token_history.len() > 4096 {
            // Arbitrary large number
            return true;
        }

        false
    }

    /// Start timing the processing
    pub fn start_timer(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Get elapsed time since timer was started
    pub fn elapsed(&self) -> Option<std::time::Duration> {
        self.start_time.map(|start| start.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processing_context() {
        let mut ctx = ProcessingContext::new()
            .with_temperature(0.7)
            .with_top_k(Some(40))
            .with_top_p(Some(0.9))
            .with_token_history(vec![1, 2, 3]);

        assert_eq!(ctx.temperature, 0.7);
        assert_eq!(ctx.top_k, Some(40));
        assert_eq!(ctx.top_p, Some(0.9));
        assert_eq!(ctx.token_history(), &[1, 2, 3]);

        ctx.extend_history(&[4, 5]);
        assert_eq!(ctx.token_history(), &[1, 2, 3, 4, 5]);

        assert!(!ctx.should_stop(10, 0));

        let mut ctx_with_stop = ProcessingContext {
            stop_tokens: vec![10, 20],
            max_new_tokens: Some(5),
            ..Default::default()
        };

        assert!(ctx_with_stop.should_stop(10, 0)); // Stop token
        assert!(!ctx_with_stop.should_stop(15, 0)); // Not a stop token
        assert!(ctx_with_stop.should_stop(15, 5)); // Max tokens reached

        ctx_with_stop.start_timer();
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(ctx_with_stop.elapsed().unwrap() > std::time::Duration::from_millis(5));
    }
}
