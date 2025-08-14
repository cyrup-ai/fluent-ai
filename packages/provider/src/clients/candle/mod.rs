//! Candle-based completion client implementation
//!
//! This module provides a production-ready implementation of the CompletionClient
//! and Provider traits using the Candle ML framework for local model inference.

pub mod client;
pub mod config;
pub mod device_manager;
pub mod error;
pub mod generation;
pub mod kv_cache;
pub mod memory_pool;
pub mod model_repo;
pub mod models;
pub mod performance;
pub mod provider;
pub mod streaming;
pub mod tokenizer;

pub use client::CandleCompletionClient;
pub use config::{
    CacheConfig, CandleGlobalConfig, ComputeConfig, MetricsCollector, ModelSpecificConfig,
};
pub use device_manager::{DeviceInfo, DeviceManager, DeviceType};
pub use error::{CandleError, CandleResult, ErrorMetrics};
pub use generation::{GenerationStatistics, SamplingConfig, TextGenerator, TokenProb};
pub use kv_cache::{KvCacheStatistics, LayerCache, ModelCacheConfig, ModelKvCache};
pub use memory_pool::{MemoryPool, MemoryPoolManager, PoolConfig, PoolStatistics, PooledEntry};
pub use model_repo::{ModelArchitecture, ModelMetadata, ModelRepository, ModelState};
pub use models::{CandleModel, CandleModelInfo};
pub use performance::{
    AlignedBuffer, BenchmarkResult, PerformanceConfig, PerformanceOptimizer, SimdCapabilities,
};
pub use provider::CandleProvider;
pub use streaming::{FinishReason, StreamingChunk, StreamingCoordinator, TokenStreamer};
pub use tokenizer::{CandleTokenizer, SpecialTokens, TokenizationResult, TokenizerConfig};
