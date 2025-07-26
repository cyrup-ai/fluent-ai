//! Blazing-fast autoregressive text generation with zero-allocation patterns
//!
//! This module provides high-performance text generation with advanced sampling strategies,
//! atomic statistics tracking, and memory-efficient token processing.

use std::collections::VecDeque;
use arrayvec::ArrayVec;
use std::collections::HashMap;

use crossbeam::atomic::AtomicCell;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use smallvec::SmallVec;

use super::error::{CandleError, CandleResult};
use super::models::CandleModel;
use super::tokenizer::SpecialTokens;

/// Maximum vocabulary size for efficient processing
const MAX_VOCAB_SIZE: usize = 256_000;
/// Maximum context length for generation
const MAX_CONTEXT_LENGTH: usize = 32_768;
/// Cache size for top-k/top-p sampling
const SAMPLING_CACHE_SIZE: usize = 1024;

/// Efficient logits buffer with zero-allocation patterns
pub type LogitsBuffer = SmallVec<[f32; SAMPLING_CACHE_SIZE]>;

/// Token probability for sampling operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenProb {
    /// Token ID
    pub token_id: u32,
    /// Log probability
    pub log_prob: f32,
    /// Normalized probability (0.0 to 1.0)
    pub prob: f32}

impl TokenProb {
    /// Create a new token probability
    pub fn new(token_id: u32, log_prob: f32, prob: f32) -> Self {
        Self {
            token_id,
            log_prob,
            prob}
    }
}

impl PartialOrd for TokenProb {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Sort by probability in descending order
        other.prob.partial_cmp(&self.prob)
    }
}

/// Comprehensive sampling configuration for text generation
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for softmax scaling (0.0 = deterministic, higher = more random)
    pub temperature: f32,
    /// Top-k sampling: keep only k highest probability tokens
    pub top_k: u32,
    /// Top-p (nucleus) sampling: keep tokens with cumulative probability <= p
    pub top_p: f32,
    /// Repetition penalty to discourage repeated tokens
    pub repetition_penalty: f32,
    /// Length penalty for sequence length bias
    pub length_penalty: f32,
    /// Frequency penalty to reduce common token repetition
    pub frequency_penalty: f32,
    /// Presence penalty to encourage topic diversity
    pub presence_penalty: f32,
    /// Minimum probability threshold for token inclusion
    pub min_prob_threshold: f32,
    /// Random seed for reproducible generation
    pub seed: Option<u64>,
    /// Enable deterministic sampling (disable randomness)
    pub deterministic: bool}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            length_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            min_prob_threshold: 1e-8,
            seed: None,
            deterministic: false}
    }
}

impl SamplingConfig {
    /// Create a balanced configuration suitable for most use cases
    pub fn balanced() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            length_penalty: 1.0,
            frequency_penalty: 0.1,
            presence_penalty: 0.1,
            min_prob_threshold: 1e-6,
            seed: None,
            deterministic: false}
    }

    /// Create a focused configuration for deterministic, coherent text
    pub fn focused(temperature: f32, top_k: u32) -> Self {
        Self {
            temperature,
            top_k,
            top_p: 0.95,
            repetition_penalty: 1.2,
            length_penalty: 1.0,
            frequency_penalty: 0.2,
            presence_penalty: 0.0,
            min_prob_threshold: 1e-5,
            seed: None,
            deterministic: temperature < 0.1}
    }

    /// Create a creative configuration for diverse, exploratory text
    pub fn creative(temperature: f32, top_p: f32) -> Self {
        Self {
            temperature,
            top_k: 100,
            top_p,
            repetition_penalty: 1.05,
            length_penalty: 0.95,
            frequency_penalty: 0.05,
            presence_penalty: 0.2,
            min_prob_threshold: 1e-7,
            seed: None,
            deterministic: false}
    }

    /// Create a deterministic configuration with fixed seed
    pub fn deterministic(seed: u64) -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            length_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            min_prob_threshold: 0.0,
            seed: Some(seed),
            deterministic: true}
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> CandleResult<()> {
        if self.temperature < 0.0 {
            return Err(CandleError::config(
                "Temperature must be non-negative",
                "temperature",
                ">= 0.0",
            ));
        }

        if self.top_k == 0 {
            return Err(CandleError::config(
                "Top-k must be positive",
                "top_k",
                "> 0",
            ));
        }

        if self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(CandleError::config(
                "Top-p must be in range (0.0, 1.0]",
                "top_p",
                "0.0 < top_p <= 1.0",
            ));
        }

        if self.repetition_penalty <= 0.0 {
            return Err(CandleError::config(
                "Repetition penalty must be positive",
                "repetition_penalty",
                "> 0.0",
            ));
        }

        if self.min_prob_threshold < 0.0 || self.min_prob_threshold >= 1.0 {
            return Err(CandleError::config(
                "Minimum probability threshold must be in range [0.0, 1.0)",
                "min_prob_threshold",
                "0.0 <= threshold < 1.0",
            ));
        }

        Ok(())
    }

    /// Check if configuration is deterministic
    pub fn is_deterministic(&self) -> bool {
        self.deterministic || self.temperature < 1e-6
    }
}

/// Token history for repetition penalty calculations
#[derive(Debug, Clone)]
struct TokenHistory {
    /// Recent tokens in generation order
    tokens: VecDeque<u32>,
    /// Token frequency counts
    frequency_map: std::collections::HashMap<u32, u32>,
    /// Presence set for presence penalty
    presence_set: std::collections::HashSet<u32>,
    /// Maximum history length
    max_length: usize}

impl TokenHistory {
    /// Create new token history tracker
    fn new(max_length: usize) -> Self {
        Self {
            tokens: VecDeque::with_capacity(max_length),
            frequency_map: std::collections::HashMap::new(),
            presence_set: std::collections::HashSet::new(),
            max_length}
    }

    /// Add a new token to history
    fn add_token(&mut self, token_id: u32) {
        // Add to frequency map
        *self.frequency_map.entry(token_id).or_insert(0) += 1;

        // Add to presence set
        self.presence_set.insert(token_id);

        // Add to token sequence
        if self.tokens.len() >= self.max_length {
            if let Some(old_token) = self.tokens.pop_front() {
                // Decrease frequency count for evicted token
                if let Some(count) = self.frequency_map.get_mut(&old_token) {
                    *count -= 1;
                    if *count == 0 {
                        self.frequency_map.remove(&old_token);
                        self.presence_set.remove(&old_token);
                    }
                }
            }
        }

        self.tokens.push_back(token_id);
    }

    /// Get frequency of a token
    fn get_frequency(&self, token_id: u32) -> u32 {
        self.frequency_map.get(&token_id).copied().unwrap_or(0)
    }

    /// Check if token is present in history
    fn contains(&self, token_id: u32) -> bool {
        self.presence_set.contains(&token_id)
    }

    /// Get recent tokens for context
    fn recent_tokens(&self, count: usize) -> Vec<u32> {
        self.tokens.iter().rev().take(count).copied().collect()
    }

    /// Clear all history
    fn clear(&mut self) {
        self.tokens.clear();
        self.frequency_map.clear();
        self.presence_set.clear();
    }
}

/// High-performance text generator with atomic statistics
#[derive(Debug)]
pub struct TextGenerator {
    /// Target model for generation
    model: CandleModel,
    /// Sampling configuration
    config: SamplingConfig,
    /// Random number generator
    rng: Pcg64Mcg,
    /// Token history for penalties
    token_history: TokenHistory,
    /// Generation statistics
    stats: GenerationStatistics,
    /// Cached sampling probabilities
    prob_cache: ArrayVec<TokenProb, SAMPLING_CACHE_SIZE>}

impl TextGenerator {
    /// Create a new text generator
    pub fn new(model: CandleModel, config: SamplingConfig) -> CandleResult<Self> {
        config.validate()?;

        let seed = config.seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64
        });

        let rng = Pcg64Mcg::seed_from_u64(seed);
        let token_history = TokenHistory::new(1024); // Keep track of last 1024 tokens

        Ok(Self {
            model,
            config,
            rng,
            token_history,
            stats: GenerationStatistics::default(),
            prob_cache: ArrayVec::new()})
    }

    /// Sample the next token from logits using configured sampling strategy
    pub fn sample_token(
        &mut self,
        logits: &LogitsBuffer,
        context_tokens: &[u32],
    ) -> CandleResult<u32> {
        if logits.is_empty() {
            return Err(CandleError::generation(
                "Empty logits buffer",
                "sample_token",
                "non-empty logits",
            ));
        }

        // Update statistics
        self.stats
            .total_sampling_calls
            .store(self.stats.total_sampling_calls.load() + 1);

        // Apply penalties to logits
        let mut adjusted_logits = self.apply_penalties(logits, context_tokens)?;

        // Apply temperature scaling
        if self.config.temperature > 0.0 && !self.config.is_deterministic() {
            for logit in &mut adjusted_logits {
                *logit /= self.config.temperature;
            }
        }

        // Convert logits to probabilities
        self.prob_cache.clear();
        let max_logit = adjusted_logits
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let mut sum_exp = 0.0f32;
        for (token_id, &logit) in adjusted_logits.iter().enumerate() {
            let exp_logit = (logit - max_logit).exp();
            sum_exp += exp_logit;

            if self
                .prob_cache
                .try_push(TokenProb::new(
                    token_id as u32,
                    logit - max_logit,
                    exp_logit,
                ))
                .is_err()
            {
                break; // Cache full, use what we have
            }
        }

        // Normalize probabilities
        for prob in &mut self.prob_cache {
            prob.prob /= sum_exp;
        }

        // Sort by probability for top-k/top-p sampling
        self.prob_cache.sort_by(|a, b| {
            b.prob
                .partial_cmp(&a.prob)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply top-k filtering
        let k_limit = (self.config.top_k as usize).min(self.prob_cache.len());
        self.prob_cache.truncate(k_limit);

        // Apply top-p (nucleus) filtering
        let mut cumulative_prob = 0.0f32;
        let mut p_cutoff = self.prob_cache.len();

        for (i, prob) in self.prob_cache.iter().enumerate() {
            cumulative_prob += prob.prob;
            if cumulative_prob >= self.config.top_p {
                p_cutoff = i + 1;
                break;
            }
        }

        self.prob_cache.truncate(p_cutoff);

        // Filter by minimum probability threshold
        self.prob_cache
            .retain(|prob| prob.prob >= self.config.min_prob_threshold);

        if self.prob_cache.is_empty() {
            return Err(CandleError::generation(
                "No valid tokens after filtering",
                "sample_token",
                "valid sampling candidates",
            ));
        }

        // Sample token
        let selected_token = if self.config.is_deterministic() {
            // Deterministic: select highest probability token
            self.prob_cache[0].token_id
        } else {
            // Stochastic: sample according to probability distribution
            self.sample_from_distribution()?
        };

        // Update token history
        self.token_history.add_token(selected_token);

        // Update statistics
        self.stats
            .total_tokens_generated
            .store(self.stats.total_tokens_generated.load() + 1);

        let candidates = self.prob_cache.len() as u32;
        let avg_candidates = self.stats.avg_sampling_candidates.load();
        let total_calls = self.stats.total_sampling_calls.load();

        // Update rolling average of sampling candidates
        let new_avg = if total_calls > 1 {
            (avg_candidates * (total_calls - 1) as f32 + candidates as f32) / total_calls as f32
        } else {
            candidates as f32
        };
        self.stats.avg_sampling_candidates.store(new_avg);

        Ok(selected_token)
    }

    /// Apply repetition and other penalties to logits
    fn apply_penalties(
        &self,
        logits: &LogitsBuffer,
        context_tokens: &[u32],
    ) -> CandleResult<LogitsBuffer> {
        let mut adjusted_logits = logits.clone();

        // Apply repetition penalty
        if self.config.repetition_penalty != 1.0 && !context_tokens.is_empty() {
            let recent_tokens = if context_tokens.len() > 64 {
                &context_tokens[context_tokens.len() - 64..] // Consider last 64 tokens
            } else {
                context_tokens
            };

            for &token_id in recent_tokens {
                if let Some(logit) = adjusted_logits.get_mut(token_id as usize) {
                    if *logit > 0.0 {
                        *logit /= self.config.repetition_penalty;
                    } else {
                        *logit *= self.config.repetition_penalty;
                    }
                }
            }
        }

        // Apply frequency penalty
        if self.config.frequency_penalty != 0.0 {
            for (token_id, logit) in adjusted_logits.iter_mut().enumerate() {
                let frequency = self.token_history.get_frequency(token_id as u32);
                if frequency > 0 {
                    *logit -= self.config.frequency_penalty * frequency as f32;
                }
            }
        }

        // Apply presence penalty
        if self.config.presence_penalty != 0.0 {
            for (token_id, logit) in adjusted_logits.iter_mut().enumerate() {
                if self.token_history.contains(token_id as u32) {
                    *logit -= self.config.presence_penalty;
                }
            }
        }

        Ok(adjusted_logits)
    }

    /// Sample token from probability distribution
    fn sample_from_distribution(&mut self) -> CandleResult<u32> {
        // Renormalize probabilities after filtering
        let total_prob: f32 = self.prob_cache.iter().map(|p| p.prob).sum();

        if total_prob <= 0.0 {
            return Err(CandleError::generation(
                "Invalid probability distribution",
                "sample_from_distribution",
                "positive probabilities",
            ));
        }

        let mut target = self.rng.r#gen::<f32>() * total_prob;

        for prob in &self.prob_cache {
            target -= prob.prob;
            if target <= 0.0 {
                return Ok(prob.token_id);
            }
        }

        // Fallback: return last token (should not happen with valid probabilities)
        Ok(self.prob_cache.last().unwrap().token_id)
    }

    /// Check if generation should stop
    pub fn should_stop(&self, token_id: u32, special_tokens: &SpecialTokens) -> bool {
        // Check for EOS token
        if let Some(eos_id) = special_tokens.eos_token_id {
            if token_id == eos_id {
                return true;
            }
        }

        // Check for other stop tokens
        special_tokens.is_special_token(token_id)
            && special_tokens.token_name(token_id) == Some("<EOS>")
    }

    /// Get current sampling configuration
    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }

    /// Update sampling configuration
    pub fn update_config(&mut self, new_config: SamplingConfig) -> CandleResult<()> {
        new_config.validate()?;

        // Update RNG seed if changed
        if new_config.seed != self.config.seed {
            let seed = new_config.seed.unwrap_or_else(|| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64
            });
            self.rng = Pcg64Mcg::seed_from_u64(seed);
        }

        self.config = new_config;
        Ok(())
    }

    /// Reset token history and statistics
    pub fn reset(&mut self) {
        self.token_history.clear();
        self.stats = GenerationStatistics::default();
        self.prob_cache.clear();
    }

    /// Get generation statistics
    pub fn statistics(&self) -> GenerationStatistics {
        GenerationStatistics {
            total_tokens_generated: self.stats.total_tokens_generated.load(),
            total_sampling_calls: self.stats.total_sampling_calls.load(),
            avg_sampling_candidates: self.stats.avg_sampling_candidates.load(),
            model: self.model,
            config: self.config.clone()}
    }
}

/// Comprehensive generation statistics for monitoring and optimization
#[derive(Debug, Clone)]
pub struct GenerationStatistics {
    /// Total tokens generated
    total_tokens_generated: AtomicCell<u64>,
    /// Total sampling calls made
    total_sampling_calls: AtomicCell<u64>,
    /// Average number of sampling candidates per call
    avg_sampling_candidates: AtomicCell<f32>,
    /// Model being used
    model: CandleModel,
    /// Current sampling configuration
    config: SamplingConfig}

impl Default for GenerationStatistics {
    fn default() -> Self {
        Self {
            total_tokens_generated: AtomicCell::new(0),
            total_sampling_calls: AtomicCell::new(0),
            avg_sampling_candidates: AtomicCell::new(0.0),
            model: CandleModel::Devstral_22B,
            config: SamplingConfig::default()}
    }
}

impl GenerationStatistics {
    /// Get tokens per sampling call ratio
    pub fn tokens_per_call(&self) -> f32 {
        let tokens = self.total_tokens_generated.load() as f32;
        let calls = self.total_sampling_calls.load() as f32;

        if calls > 0.0 { tokens / calls } else { 0.0 }
    }

    /// Get effective sampling diversity (lower values = more focused)
    pub fn sampling_diversity(&self) -> f32 {
        let avg_candidates = self.avg_sampling_candidates.load();
        let vocab_size = match self.model {
            CandleModel::Devstral_22B => 102400.0,
            CandleModel::Llama2_7B | CandleModel::Llama2_13B | CandleModel::Mistral_7B => 32000.0,
            CandleModel::CodeLlama_7B => 32016.0,
            CandleModel::Phi3_Mini => 32064.0,
            CandleModel::Gemma_2B | CandleModel::Gemma_7B => 256000.0};

        if vocab_size > 0.0 {
            avg_candidates / vocab_size
        } else {
            0.0
        }
    }

    /// Check if generation is running efficiently
    pub fn is_efficient(&self) -> bool {
        let tokens_per_call = self.tokens_per_call();
        let diversity = self.sampling_diversity();

        // Efficient if close to 1 token per call and reasonable diversity
        tokens_per_call >= 0.9 && diversity > 0.001 && diversity < 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_config_creation() {
        let config = SamplingConfig::balanced();
        assert!(config.validate().is_ok());
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_k, 40);

        let focused = SamplingConfig::focused(0.1, 10);
        assert!(focused.validate().is_ok());
        assert!(focused.is_deterministic());

        let creative = SamplingConfig::creative(1.2, 0.85);
        assert!(creative.validate().is_ok());
        assert!(!creative.is_deterministic());
    }

    #[test]
    fn test_text_generator_creation() {
        let config = SamplingConfig::balanced();
        let generator = TextGenerator::new(CandleModel::Mistral_7B, config);
        assert!(generator.is_ok());

        let generator_instance = generator.unwrap();
        assert_eq!(generator_instance.model, CandleModel::Mistral_7B);
    }

    #[test]
    fn test_token_history() {
        let mut history = TokenHistory::new(5);

        for i in 0..10 {
            history.add_token(i);
        }

        // Should only keep last 5 tokens
        assert_eq!(history.tokens.len(), 5);
        assert_eq!(history.recent_tokens(3), vec![9, 8, 7]);

        // Frequency should be tracked correctly
        assert_eq!(history.get_frequency(9), 1);
        assert_eq!(history.get_frequency(0), 0); // Evicted
    }

    #[test]
    fn test_logits_buffer() {
        let mut logits = LogitsBuffer::new();
        logits.extend_from_slice(&[1.0, 2.0, 3.0, 0.5]);

        assert_eq!(logits.len(), 4);
        assert_eq!(logits[2], 3.0);
    }

    #[test]
    fn test_token_prob_ordering() {
        let mut probs = vec![
            TokenProb::new(0, 1.0, 0.2),
            TokenProb::new(1, 2.0, 0.5),
            TokenProb::new(2, 0.5, 0.1),
        ];

        probs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Should be sorted by probability in descending order
        assert_eq!(probs[0].token_id, 1); // Highest prob (0.5)
        assert_eq!(probs[1].token_id, 0); // Medium prob (0.2)
        assert_eq!(probs[2].token_id, 2); // Lowest prob (0.1)
    }
}
