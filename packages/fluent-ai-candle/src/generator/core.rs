//! Core generator implementation with constructors and configuration

use std::sync::Arc;
use candle_core::Device;
use fluent_ai_async::AsyncStream;

use crate::error::CandleResult;
use crate::kv_cache::{KVCache, KVCacheConfig};
use crate::model::CandleModel;
use crate::processing::{processors::{CompositeProcessor, presets}, traits::LogitsProcessor};
use crate::sampling::{Sampling, SamplingConfig};
use crate::streaming::{StreamingConfig, TokenOutputStream};
use crate::tokenizer::CandleTokenizer;

use super::types::{GenerationConfig, GenerationState};

/// Zero-allocation text generator
pub struct CandleGenerator {
    /// The model for generation
    pub(super) model: Arc<CandleModel>,
    /// The tokenizer
    pub(super) tokenizer: Arc<CandleTokenizer>,
    /// Generation configuration
    pub(super) config: GenerationConfig,
    /// Device for computation
    pub(super) device: Device,
    /// Random number generator state
    pub(super) rng_state: parking_lot::Mutex<Option<u64>>,
    /// Cumulative log probability for current generation
    pub(super) cumulative_log_prob: parking_lot::Mutex<f64>,
    /// Sophisticated sampling configuration
    pub(super) sampling_config: Sampling,
    /// Streaming configuration for real-time output
    pub(super) streaming_config: StreamingConfig,
    /// KV cache for efficient generation
    pub(super) kv_cache: Option<Arc<parking_lot::Mutex<KVCache>>>,
    /// CompositeProcessor for sophisticated sampling
    pub(super) composite_processor: CompositeProcessor,
    /// TokenOutputStream for real-time streaming
    pub(super) token_output_stream: Option<Arc<parking_lot::Mutex<TokenOutputStream>>>,
}

impl Clone for CandleGenerator {
    fn clone(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            tokenizer: Arc::clone(&self.tokenizer),
            config: self.config.clone(),
            device: self.device.clone(),
            rng_state: parking_lot::Mutex::new(*self.rng_state.lock()),
            cumulative_log_prob: parking_lot::Mutex::new(*self.cumulative_log_prob.lock()),
            sampling_config: self.sampling_config.clone(),
            streaming_config: self.streaming_config.clone(),
            kv_cache: self.kv_cache.as_ref().map(Arc::clone),
            composite_processor: CompositeProcessor::new(),
            token_output_stream: self.token_output_stream.as_ref().map(Arc::clone),
        }
    }
}

impl CandleGenerator {
    /// Create a new generator with sophisticated features
    #[inline(always)]
    pub fn new(
        model: Arc<CandleModel>,
        tokenizer: Arc<CandleTokenizer>,
        config: GenerationConfig,
        device: Device,
    ) -> Self {
        Self {
            model,
            tokenizer,
            rng_state: parking_lot::Mutex::new(config.seed),
            config,
            device,
            cumulative_log_prob: parking_lot::Mutex::new(0.0),
            sampling_config: SamplingConfig::default().build_sampling(),
            streaming_config: StreamingConfig::default(),
            kv_cache: None,
            composite_processor: CompositeProcessor::new(),
            token_output_stream: None,
        }
    }

    /// Create a new generator with sophisticated features configured
    #[inline(always)]
    pub fn with_sophisticated_features(
        model: Arc<CandleModel>,
        tokenizer: Arc<CandleTokenizer>,
        config: GenerationConfig,
        device: Device,
        sampling_config: Sampling,
        streaming_config: StreamingConfig,
        kv_cache_config: Option<KVCacheConfig>,
    ) -> CandleResult<Self> {
        // Initialize KV cache if configured
        let kv_cache = if let Some(cache_config) = kv_cache_config {
            let cache = KVCache::with_config(cache_config)?;
            Some(Arc::new(parking_lot::Mutex::new(cache)))
        } else {
            None
        };

        // Initialize CompositeProcessor based on generation configuration
        let composite_processor =
            presets::conversation().unwrap_or_else(|_| CompositeProcessor::new());

        // Initialize streaming components
        let token_output_stream = None; // TODO: Implement proper TokenOutputStream initialization

        Ok(Self {
            model,
            tokenizer,
            rng_state: parking_lot::Mutex::new(config.seed),
            config,
            device,
            cumulative_log_prob: parking_lot::Mutex::new(0.0),
            sampling_config,
            streaming_config,
            kv_cache,
            composite_processor,
            token_output_stream,
        })
    }

    /// Update generation configuration
    #[inline(always)]
    pub fn update_config(&mut self, config: GenerationConfig) {
        self.config = config;
        *self.rng_state.lock() = config.seed;
    }

    /// Get generation configuration
    #[inline(always)]
    pub fn config(&self) -> &GenerationConfig {
        &self.config
    }

    /// Reset cumulative log probability for new generation
    #[inline(always)]
    pub fn reset_cumulative_log_prob(&self) {
        *self.cumulative_log_prob.lock() = 0.0;
    }

    /// Get current cumulative log probability
    #[inline(always)]
    pub fn cumulative_log_prob(&self) -> f64 {
        *self.cumulative_log_prob.lock()
    }

    /// Get the configured composite processor (uses composite_processor field)
    #[inline(always)]
    pub fn composite_processor(&self) -> &CompositeProcessor {
        &self.composite_processor
    }
}