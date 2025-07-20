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
pub use config::{CandleGlobalConfig, MetricsCollector, ModelSpecificConfig, CacheConfig, ComputeConfig};
pub use device_manager::{DeviceManager, DeviceInfo, DeviceType};
pub use error::{CandleError, CandleResult, ErrorMetrics};
pub use generation::{TextGenerator, SamplingConfig, GenerationStatistics, TokenProb};
pub use kv_cache::{ModelKvCache, LayerCache, ModelCacheConfig, KvCacheStatistics};
pub use memory_pool::{MemoryPoolManager, MemoryPool, PooledEntry, PoolConfig, PoolStatistics};
pub use model_repo::{ModelRepository, ModelMetadata, ModelState, ModelArchitecture};
pub use models::{CandleModel, CandleModelInfo};
pub use performance::{PerformanceOptimizer, PerformanceConfig, SimdCapabilities, AlignedBuffer, BenchmarkResult};
pub use provider::CandleProvider;
pub use streaming::{StreamingCoordinator, StreamingChunk, FinishReason, TokenStreamer};
pub use tokenizer::{CandleTokenizer, TokenizationResult, SpecialTokens, TokenizerConfig};