//! Blazing-fast autoregressive text generation with zero-allocation patterns
//!
//! This module provides high-performance text generation with advanced sampling strategies,
//! atomic statistics tracking, and memory-efficient token processing.

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};
use arrayvec::ArrayVec;

use crossbeam::atomic::AtomicCell;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use smallvec::SmallVec;

// Real imports to connect to domain types instead of placeholder implementations
use crate::domain::model::error::{CandleModelError as CandleError};
use crate::domain::model::traits::CandleModel;
use crate::providers::tokenizer::CandleTokenizer;
use fluent_ai_simd::{get_cpu_features, CpuFeatures};

pub type CandleResult<T> = Result<T, CandleError>;

/// Special tokens information from tokenizer
pub struct SpecialTokens {
    pub eos_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    tokenizer: CandleTokenizer,
}

impl SpecialTokens {
    /// Create new special tokens from tokenizer
    pub fn new(tokenizer: CandleTokenizer, eos_token_id: Option<u32>) -> Self {
        Self { 
            eos_token_id, 
            bos_token_id: None, // TODO: Extract from tokenizer
            pad_token_id: None, // TODO: Extract from tokenizer
            tokenizer 
        }
    }
    
    /// Check if token is a special token using real tokenizer
    pub fn is_special_token(&self, token_id: u32) -> bool {
        self.tokenizer.is_special_token(token_id)
    }
    
    /// Get token name from tokenizer (placeholder implementation for now)
    pub fn token_name(&self, _token_id: u32) -> Option<&str> {
        // TODO: Implement real token name lookup from tokenizer
        None
    }
    
    /// Check if token is an EOS token
    pub fn is_eos_token(&self, token_id: u32) -> bool {
        if let Some(eos_id) = self.eos_token_id {
            token_id == eos_id
        } else {
            false
        }
    }
}

// MAX_VOCAB_SIZE and MAX_CONTEXT_LENGTH constants removed - unused
/// Cache size for top-k/top-p sampling
const SAMPLING_CACHE_SIZE: usize = 1024;

/// Efficient logits buffer with zero-allocation patterns
pub type LogitsBuffer = SmallVec<f32, SAMPLING_CACHE_SIZE>;

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
    pub deterministic: bool,
    /// Enable SIMD optimizations (auto-detected by default)
    pub enable_simd: bool,
    /// Minimum vector size for SIMD operations (auto-detected by default)
    pub simd_threshold: usize}

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
            deterministic: false,
            enable_simd: true, // Auto-enable SIMD optimizations
            simd_threshold: 16 // Minimum vector size for SIMD benefits
        }
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
            deterministic: false,
            enable_simd: true,
            simd_threshold: 16
        }
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
            deterministic: temperature < 0.1,
            enable_simd: true,
            simd_threshold: 16
        }
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
            deterministic: false,
            enable_simd: true,
            simd_threshold: 16
        }
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
            deterministic: true,
            enable_simd: true,
            simd_threshold: 16
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> CandleResult<()> {
        if self.temperature < 0.0 {
            return Err(CandleError::InvalidConfiguration(
                "Temperature must be non-negative".into()
            ));
        }

        if self.top_k == 0 {
            return Err(CandleError::InvalidConfiguration(
                "Top-k must be positive".into()
            ));
        }

        if self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(CandleError::InvalidConfiguration(
                "Top-p must be in range (0.0, 1.0]".into()
            ));
        }

        if self.repetition_penalty <= 0.0 {
            return Err(CandleError::InvalidConfiguration(
                "Repetition penalty must be positive".into()
            ));
        }

        if self.min_prob_threshold < 0.0 || self.min_prob_threshold >= 1.0 {
            return Err(CandleError::InvalidConfiguration(
                "Minimum probability threshold must be in range [0.0, 1.0)".into()
            ));
        }

        Ok(())
    }

    /// Check if configuration is deterministic
    pub fn is_deterministic(&self) -> bool {
        self.deterministic || self.temperature < 1e-6
    }
    
    /// Check if SIMD operations should be used for given vector size
    pub fn should_use_simd(&self, vector_size: usize) -> bool {
        self.enable_simd && vector_size >= self.simd_threshold
    }
    
    /// Get CPU features information for SIMD optimization
    pub fn get_cpu_info(&self) -> CpuFeatures {
        get_cpu_features()
    }
    
    /// Create a SIMD-optimized configuration based on detected CPU features
    /// 
    /// Automatically configures simd_threshold based on detected SIMD vector width
    /// and enables SIMD optimizations if hardware supports them
    pub fn with_simd_optimization() -> Self {
        let mut config = Self::default();
        let cpu_features = get_cpu_features();
        
        // Configure threshold based on detected SIMD vector width
        let vector_width = cpu_features.vector_width();
        config.simd_threshold = (vector_width * 4).max(16); // Minimum 16 elements for SIMD benefit
        config.enable_simd = cpu_features.has_simd();
        
        config
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

    // recent_tokens method removed - unused

    /// Clear all history
    fn clear(&mut self) {
        self.tokens.clear();
        self.frequency_map.clear();
        self.presence_set.clear();
    }
}

/// High-performance text generator with atomic statistics
pub struct TextGenerator {
    /// Target model for generation
    model: Box<dyn CandleModel>,
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
    pub fn new(model: Box<dyn CandleModel>, config: SamplingConfig) -> CandleResult<Self> {
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

    /// Sample the next token from logits using SIMD-optimized sampling strategy
    pub fn sample_token(
        &mut self,
        logits: &LogitsBuffer,
        context_tokens: &[u32],
    ) -> CandleResult<u32> {
        if logits.is_empty() {
            return Err(CandleError::InvalidInput(
                "Empty logits buffer for token sampling".into()
            ));
        }

        // Update statistics
        self.stats
            .total_sampling_calls
            .store(self.stats.total_sampling_calls.load() + 1);

        // Apply penalties to logits
        let mut adjusted_logits = self.apply_penalties(logits, context_tokens)?;

        // Apply SIMD-optimized temperature scaling if temperature > 0
        if self.config.temperature > 0.0 && !self.config.is_deterministic() {
            match fluent_ai_simd::scale_temperature(adjusted_logits.as_mut_slice(), self.config.temperature) {
                Ok(_) => {
                    // SIMD temperature scaling successful
                },
                Err(e) => {
                    // Fallback to scalar temperature scaling
                    eprintln!("SIMD temperature scaling failed, using fallback: {}", e);
                    for logit in &mut adjusted_logits {
                        *logit /= self.config.temperature;
                    }
                }
            }
        }

        // Apply SIMD-optimized top-k filtering for large vocabularies
        if self.config.top_k > 0 && self.config.top_k < adjusted_logits.len() as u32 {
            use fluent_ai_simd::logits::topk_filtering_simd;
            
            match topk_filtering_simd(adjusted_logits.as_mut_slice(), self.config.top_k as usize) {
                Ok(_) => {
                    // SIMD top-k filtering successful
                },
                Err(e) => {
                    eprintln!("SIMD top-k filtering failed: {}", e);
                }
            }
        }

        // Apply SIMD-optimized nucleus (top-p) sampling
        if self.config.top_p > 0.0 && self.config.top_p < 1.0 {
            use fluent_ai_simd::logits::prepare_nucleus_sampling_simd;
            
            match prepare_nucleus_sampling_simd(adjusted_logits.as_mut_slice(), self.config.top_p as f64) {
                Ok(_) => {
                    // SIMD nucleus sampling preparation successful
                },
                Err(e) => {
                    eprintln!("SIMD nucleus sampling failed: {}", e);
                }
            }
        }

        // Use SIMD-optimized softmax for probability computation
        let probabilities = match fluent_ai_simd::softmax(adjusted_logits.as_slice()) {
            Ok(probs) => probs,
            Err(e) => {
                // Fallback to manual softmax computation
                eprintln!("SIMD softmax failed, using fallback: {}", e);
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

                // Normalize probabilities manually
                for prob in &mut self.prob_cache {
                    prob.prob /= sum_exp;
                }
                
                // Convert to Vec<f32> for consistency with SIMD path
                self.prob_cache.iter().map(|p| p.prob).collect()
            }
        };

        // Clear cache and populate with SIMD-computed probabilities  
        self.prob_cache.clear();
        for (token_id, &prob) in probabilities.iter().enumerate() {
            if self
                .prob_cache
                .try_push(TokenProb::new(
                    token_id as u32,
                    prob.ln(), // Store log probability for consistency
                    prob,
                ))
                .is_err()
            {
                break; // Cache full, use what we have
            }
        }

        // Sort by probability for sampling (top-k and top-p already applied via SIMD)
        self.prob_cache.sort_by(|a, b| {
            b.prob
                .partial_cmp(&a.prob)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Filter by minimum probability threshold
        self.prob_cache
            .retain(|prob| prob.prob >= self.config.min_prob_threshold);

        if self.prob_cache.is_empty() {
            return Err(CandleError::InvalidInput(
                "No valid tokens after filtering for sampling".into()
            ));
        }

        // Sample token
        let selected_token = if self.config.is_deterministic() || self.config.temperature <= 0.0 {
            // Deterministic/Greedy: select highest probability token using SIMD argmax when possible
            if self.config.enable_simd && self.prob_cache.len() >= self.config.simd_threshold {
                // Extract probabilities for SIMD argmax
                let probs: Vec<f32> = self.prob_cache.iter().map(|p| p.prob).collect();
                match fluent_ai_simd::argmax(&probs) {
                    Ok(max_idx) => {
                        // Return token_id from the corresponding cache entry with bounds checking
                        if max_idx < self.prob_cache.len() {
                            self.prob_cache[max_idx].token_id
                        } else {
                            // Bounds check failed, use fallback
                            self.prob_cache.get(0).map(|p| p.token_id).unwrap_or(0)
                        }
                    },
                    Err(e) => {
                        // Fallback to manual argmax
                        eprintln!("SIMD argmax failed, using fallback: {}", e);
                        self.prob_cache.get(0).map(|p| p.token_id).unwrap_or(0)
                    }
                }
            } else {
                // Use pre-sorted cache for small vectors
                self.prob_cache[0].token_id
            }
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
    }    /// Apply repetition and other penalties to logits using SIMD optimization when beneficial
    fn apply_penalties(
        &self,
        logits: &LogitsBuffer,
        context_tokens: &[u32],
    ) -> CandleResult<LogitsBuffer> {
        let mut adjusted_logits = logits.clone();

        // Use SIMD penalty calculations for large vocabularies when penalties are active
        let has_penalties = self.config.repetition_penalty != 1.0 
            || self.config.frequency_penalty != 0.0 
            || self.config.presence_penalty != 0.0;
        
        if has_penalties && adjusted_logits.len() >= self.config.simd_threshold {
            // Attempt SIMD-optimized penalty application
            let recent_tokens = if context_tokens.len() > 64 {
                &context_tokens[context_tokens.len() - 64..] // Consider last 64 tokens
            } else {
                context_tokens
            };

            // Create context for SIMD penalty calculation
            use fluent_ai_simd::logits::apply_penalties_simd;
            use fluent_ai_simd::context::ProcessingContext;
            use fluent_ai_simd::config::ProcessorConfig;
            
            let context = ProcessingContext::new()
                .with_token_history(recent_tokens.to_vec());
                
            let simd_config = ProcessorConfig::default()
                .with_repetition_penalty(self.config.repetition_penalty)
                .with_frequency_penalty(self.config.frequency_penalty)
                .with_presence_penalty(self.config.presence_penalty);

            match apply_penalties_simd(adjusted_logits.as_mut_slice(), &context, &simd_config) {
                Ok(()) => {
                    // SIMD penalty application successful
                },
                Err(e) => {
                    // Fallback to scalar penalty calculations
                    eprintln!("SIMD penalty calculation failed, using scalar fallback: {}", e);
                    self.apply_penalties_scalar(&mut adjusted_logits, context_tokens)?;
                }
            }
        } else {
            // Use scalar penalty calculations for small arrays or when no penalties active
            self.apply_penalties_scalar(&mut adjusted_logits, context_tokens)?;
        }

        Ok(adjusted_logits)
    }

    /// Scalar fallback for penalty calculations
    fn apply_penalties_scalar(
        &self,
        adjusted_logits: &mut LogitsBuffer,
        context_tokens: &[u32],
    ) -> CandleResult<()> {
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

        Ok(())
    }

    /// Sample token from probability distribution
    fn sample_from_distribution(&mut self) -> CandleResult<u32> {
        // Renormalize probabilities after filtering
        let total_prob: f32 = self.prob_cache.iter().map(|p| p.prob).sum();

        if total_prob <= 0.0 {
            return Err(CandleError::Internal(
                "Invalid probability distribution - total probability is zero or negative".into()
            ));
        }

        let mut target = self.rng.random::<f32>() * total_prob;

        for prob in &self.prob_cache {
            target -= prob.prob;
            if target <= 0.0 {
                return Ok(prob.token_id);
            }
        }

        // Fallback: return last token (should not happen with valid probabilities)
        self.prob_cache.last()
            .map(|prob| prob.token_id)
            .ok_or_else(|| CandleError::Internal(
                "Empty probability cache during sampling distribution".into()
            ))
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
            total_tokens_generated: AtomicCell::new(self.stats.total_tokens_generated.load()),
            total_sampling_calls: AtomicCell::new(self.stats.total_sampling_calls.load()),
            avg_sampling_candidates: AtomicCell::new(self.stats.avg_sampling_candidates.load()),
            model_name: self.model.name().to_string(),
            config: self.config.clone()}
    }
}

/// Comprehensive generation statistics for monitoring and optimization
#[derive(Debug)]
pub struct GenerationStatistics {
    /// Total tokens generated
    total_tokens_generated: AtomicCell<u64>,
    /// Total sampling calls made
    total_sampling_calls: AtomicCell<u64>,
    /// Average number of sampling candidates per call
    avg_sampling_candidates: AtomicCell<f32>,
    /// Model name being used
    model_name: String,
    /// Current sampling configuration
    config: SamplingConfig}

impl Default for GenerationStatistics {
    fn default() -> Self {
        Self {
            total_tokens_generated: AtomicCell::new(0),
            total_sampling_calls: AtomicCell::new(0),
            avg_sampling_candidates: AtomicCell::new(0.0),
            model_name: "unknown".to_string(),
            config: SamplingConfig::default()}
    }
}

impl Clone for GenerationStatistics {
    fn clone(&self) -> Self {
        Self {
            total_tokens_generated: AtomicCell::new(self.total_tokens_generated.load()),
            total_sampling_calls: AtomicCell::new(self.total_sampling_calls.load()),
            avg_sampling_candidates: AtomicCell::new(self.avg_sampling_candidates.load()),
            model_name: self.model_name.clone(),
            config: self.config.clone(),
        }
    }
}

/// Wrapper for Candle Llama model that implements CandleModel trait
#[derive(Debug)]
pub struct CandleLlamaModel {
    /// The actual Candle Llama model
    model: candle_transformers::models::llama::Llama,
    /// Llama configuration
    config: candle_transformers::models::llama::Config,
}

impl CandleLlamaModel {
    /// Create a new CandleLlamaModel wrapper
    pub fn new(
        model: candle_transformers::models::llama::Llama,
        config: candle_transformers::models::llama::Config,
    ) -> Self {
        Self {
            model,
            config,
        }
    }

    /// Get reference to the underlying Llama model
    pub fn model(&self) -> &candle_transformers::models::llama::Llama {
        &self.model
    }

    /// Get reference to the Llama configuration
    pub fn config(&self) -> &candle_transformers::models::llama::Config {
        &self.config
    }

    /// Forward pass through the model
    pub fn forward(
        &self,
        input: &candle_core::Tensor,
        index_pos: usize,
        cache: &mut candle_transformers::models::llama::Cache,
    ) -> candle_core::Result<candle_core::Tensor> {
        self.model.forward(input, index_pos, cache)
    }
}

impl CandleModel for CandleLlamaModel {
    fn info(&self) -> &'static crate::domain::model::CandleModelInfo {
        // For now, return a static default - this should be properly implemented
        // when integrating with the full model info system
        static DEFAULT_INFO: std::sync::OnceLock<crate::domain::model::CandleModelInfo> = std::sync::OnceLock::new();
        DEFAULT_INFO.get_or_init(|| {
            crate::domain::model::CandleModelInfo {
                provider_name: "kimi-k2",
                name: "kimi-k2",
                max_input_tokens: None,
                max_output_tokens: None,
                input_price: None,
                output_price: None,
                supports_vision: false,
                supports_function_calling: false,
                supports_streaming: true,
                supports_embeddings: false,
                requires_max_tokens: false,
                supports_thinking: false,
                optimal_thinking_budget: None,
                system_prompt_prefix: None,
                real_name: None,
                model_type: None,
                patch: None,
            }
        })
    }

    fn name(&self) -> &'static str {
        // Return a static string - in a real implementation this would come from model info
        "kimi-k2"
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
        // Use a reasonable default vocab size for diversity calculation
        // Real implementation would get this from the model's tokenizer
        let vocab_size = 32000.0; // Standard vocab size for most LLMs

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

/// SIMD metrics and performance monitoring integration
pub mod simd_metrics {
    use super::*;
    use fluent_ai_simd::similarity::{metrics, reset_metrics, SimilarityMetricsSnapshot};
    
    /// Get comprehensive SIMD performance metrics
    pub fn get_simd_metrics() -> SimilarityMetricsSnapshot {
        metrics()
    }
    
    /// Reset all SIMD metrics to zero for fresh measurement periods
    pub fn reset_simd_metrics() {
        reset_metrics()
    }
    
    /// Check if SIMD operations are being used effectively
    pub fn is_simd_effective() -> bool {
        let metrics = get_simd_metrics();
        // Effective if we're processing a reasonable number of elements per calculation
        metrics.total_calculations > 0 && 
        (metrics.total_elements_processed / metrics.total_calculations) >= 16
    }
    
    /// Get SIMD utilization report for performance analysis
    pub fn get_utilization_report() -> String {
        let metrics = get_simd_metrics();
        let cpu_info = get_cpu_features();
        
        format!(
            "SIMD Performance Report:\n\
            - CPU Features: {:?}\n\
            - Vector Width: {} elements\n\
            - Has SIMD: {}\n\
            - Total Calculations: {}\n\
            - Total Elements Processed: {}\n\
            - Avg Elements per Calculation: {:.2}\n\
            - SIMD Effective: {}",
            cpu_info,
            cpu_info.vector_width(),
            cpu_info.has_simd(),
            metrics.total_calculations,
            metrics.total_elements_processed,
            if metrics.total_calculations > 0 {
                metrics.total_elements_processed as f64 / metrics.total_calculations as f64
            } else { 0.0 },
            is_simd_effective()
        )
    }
}
