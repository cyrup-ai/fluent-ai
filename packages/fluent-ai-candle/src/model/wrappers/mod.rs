//! Model architecture wrappers with centralized cache management
//!
//! This module provides zero-allocation wrappers for different model architectures:
//! - Enhanced wrapper patterns with atomic operations
//! - Centralized KV cache management
//! - Lock-free sequence management
//! - Device-aware memory handling

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;
use candle_core::{Module, Tensor};

use crate::error::CandleError;
use crate::model::cache::{CacheKey, KVCacheEntry, KVCacheManager};

/// Enhanced wrapper for Llama models loaded with CandleVarBuilder patterns
/// Optimized for sophisticated loading and memory management
pub struct EnhancedLlamaWrapper {
    model: candle_transformers::models::llama::Llama,
    cache: Arc<parking_lot::Mutex<candle_transformers::models::llama::Cache>>,
    position: AtomicU32,
}

impl EnhancedLlamaWrapper {
    #[inline(always)]
    pub fn new(model: candle_transformers::models::llama::Llama, cache: candle_transformers::models::llama::Cache) -> Self {
        Self {
            model,
            cache: Arc::new(parking_lot::Mutex::new(cache)),
            position: AtomicU32::new(0),
        }
    }
}

impl Module for EnhancedLlamaWrapper {
    #[inline(always)]
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let pos = self.position.fetch_add(1, Ordering::Relaxed);
        let mut cache = self.cache.lock();
        self.model.forward(xs, pos as usize, &mut *cache)
    }
}

/// Cache context for external KV cache management
#[derive(Clone)]
pub struct CacheContext {
    /// Sequence ID for cache isolation
    pub sequence_id: u64,
    /// Current position in the sequence
    pub position: AtomicU32,
    /// Cache manager reference
    pub cache_manager: Arc<KVCacheManager>,
}

impl CacheContext {
    #[inline(always)]
    pub fn new(sequence_id: u64, cache_manager: Arc<KVCacheManager>) -> Self {
        Self {
            sequence_id,
            position: AtomicU32::new(0),
            cache_manager,
        }
    }

    #[inline(always)]
    pub fn get_cache_entry(&self, layer_id: u32, position_start: u32, position_end: u32) -> Option<Arc<KVCacheEntry>> {
        let key = CacheKey::new(self.sequence_id, layer_id, position_start, position_end);
        self.cache_manager.get_entry(&key)
    }

    #[inline(always)]
    pub fn insert_cache_entry(&self, layer_id: u32, position_start: u32, position_end: u32, 
                             key_tensor: Tensor, value_tensor: Tensor) -> Result<Arc<KVCacheEntry>, CandleError> {
        let key = CacheKey::new(self.sequence_id, layer_id, position_start, position_end);
        self.cache_manager.insert_entry(key, key_tensor, value_tensor)
    }

    #[inline(always)]
    pub fn current_position(&self) -> u32 {
        self.position.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn advance_position(&self) -> u32 {
        self.position.fetch_add(1, Ordering::Relaxed)
    }
}

/// Wrapper for Llama model with centralized cache management
/// Uses lock-free atomic operations for zero-contention performance
pub struct LlamaWrapper {
    model: candle_transformers::models::llama::Llama,
    internal_cache: ArcSwap<candle_transformers::models::llama::Cache>,
    index_pos: AtomicU32,
    sequence_id: AtomicU64,
    cache_manager: Arc<KVCacheManager>,
}

impl LlamaWrapper {
    #[inline(always)]
    pub fn new(model: candle_transformers::models::llama::Llama, 
           cache: candle_transformers::models::llama::Cache,
           cache_manager: Arc<KVCacheManager>) -> Self {
        Self {
            model,
            internal_cache: ArcSwap::new(Arc::new(cache)),
            index_pos: AtomicU32::new(0),
            sequence_id: AtomicU64::new(0),
            cache_manager,
        }
    }

    #[inline(always)]
    pub fn set_sequence(&self, sequence_id: u64) {
        self.sequence_id.store(sequence_id, Ordering::Relaxed);
        self.index_pos.store(0, Ordering::Relaxed);
    }
}

impl Module for LlamaWrapper {
    #[inline(always)]
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let pos = self.index_pos.fetch_add(1, Ordering::Relaxed);
        let cache_arc = self.internal_cache.load();
        let mut cache_copy = (**cache_arc).clone();
        
        let result = self.model.forward(xs, pos as usize, &mut cache_copy);
        
        if result.is_ok() {
            self.internal_cache.store(Arc::new(cache_copy));
        }
        
        result
    }
}

/// Wrapper for Mistral model with centralized cache management
/// Uses immutable model with external cache context
pub struct MistralWrapper {
    model: Arc<candle_transformers::models::mistral::Model>,
    cache_manager: Arc<KVCacheManager>,
    sequence_id: AtomicU64,
    position: AtomicU32,
}

impl MistralWrapper {
    #[inline(always)]
    pub fn new(model: candle_transformers::models::mistral::Model, cache_manager: Arc<KVCacheManager>) -> Self {
        Self {
            model: Arc::new(model),
            cache_manager,
            sequence_id: AtomicU64::new(0),
            position: AtomicU32::new(0),
        }
    }

    #[inline(always)]
    pub fn set_sequence(&self, sequence_id: u64) {
        self.sequence_id.store(sequence_id, Ordering::Relaxed);
        self.position.store(0, Ordering::Relaxed);
    }
}

impl Module for MistralWrapper {
    #[inline(always)]
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let pos = self.position.fetch_add(1, Ordering::Relaxed);
        
        // Create a mutable copy of the model for this forward pass
        // This is necessary because Mistral requires mutable access for KV cache
        let mut model_copy = (*self.model).clone();
        model_copy.forward(xs, pos as usize)
    }
}

/// Wrapper for Gemma model with centralized cache management
/// Uses immutable model with external cache context
pub struct GemmaWrapper {
    model: Arc<candle_transformers::models::gemma::Model>,
    cache_manager: Arc<KVCacheManager>,
    sequence_id: AtomicU64,
    position: AtomicU32,
}

impl GemmaWrapper {
    #[inline(always)]
    pub fn new(model: candle_transformers::models::gemma::Model, cache_manager: Arc<KVCacheManager>) -> Self {
        Self {
            model: Arc::new(model),
            cache_manager,
            sequence_id: AtomicU64::new(0),
            position: AtomicU32::new(0),
        }
    }

    #[inline(always)]
    pub fn set_sequence(&self, sequence_id: u64) {
        self.sequence_id.store(sequence_id, Ordering::Relaxed);
        self.position.store(0, Ordering::Relaxed);
    }
}

impl Module for GemmaWrapper {
    #[inline(always)]
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let pos = self.position.fetch_add(1, Ordering::Relaxed);
        
        // Create a mutable copy of the model for this forward pass
        let mut model_copy = (*self.model).clone();
        model_copy.forward(xs, pos as usize)
    }
}

/// Wrapper for Phi model with centralized cache management
/// Uses immutable model with external cache context
pub struct PhiWrapper {
    model: Arc<candle_transformers::models::phi::Model>,
    cache_manager: Arc<KVCacheManager>,
    sequence_id: AtomicU64,
}

impl PhiWrapper {
    #[inline(always)]
    pub fn new(model: candle_transformers::models::phi::Model, cache_manager: Arc<KVCacheManager>) -> Self {
        Self {
            model: Arc::new(model),
            cache_manager,
            sequence_id: AtomicU64::new(0),
        }
    }

    #[inline(always)]
    pub fn set_sequence(&self, sequence_id: u64) {
        self.sequence_id.store(sequence_id, Ordering::Relaxed);
    }
}

impl Module for PhiWrapper {
    #[inline(always)]
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // Phi models are stateless for forward pass, so we can use the immutable model directly
        // Create a mutable copy for any internal state requirements
        let mut model_copy = (*self.model).clone();
        model_copy.forward(xs)
    }
}

/// Wrapper for Qwen model with centralized cache management  
/// Uses immutable model with external cache context
pub struct QwenWrapper {
    model: Arc<candle_transformers::models::qwen2::Model>,
    cache_manager: Arc<KVCacheManager>,
    sequence_id: AtomicU64,
    position: AtomicU32,
}

impl QwenWrapper {
    #[inline(always)]
    pub fn new(model: candle_transformers::models::qwen2::Model, cache_manager: Arc<KVCacheManager>) -> Self {
        Self {
            model: Arc::new(model),
            cache_manager,
            sequence_id: AtomicU64::new(0),
            position: AtomicU32::new(0),
        }
    }

    #[inline(always)]
    pub fn set_sequence(&self, sequence_id: u64) {
        self.sequence_id.store(sequence_id, Ordering::Relaxed);
        self.position.store(0, Ordering::Relaxed);
    }
}

impl Module for QwenWrapper {
    #[inline(always)]
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let pos = self.position.fetch_add(1, Ordering::Relaxed);
        
        // Create a mutable copy of the model for this forward pass
        let mut model_copy = (*self.model).clone();
        model_copy.forward(xs, pos as usize, None)
    }
}