//! High-performance embedding providers with zero-allocation patterns
//!
//! This module provides comprehensive embedding support for text, images, and multimodal content
//! with optimal batch processing, advanced normalization, and provider-agnostic interfaces.
//!
//! ## Features
//! - Batch processing with configurable batch sizes and parallelization
//! - Image embeddings with CLIP-style multimodal models
//! - Zero-allocation processing with static dispatch
//! - Multiple embedding providers (OpenAI, Cohere, local models)
//! - Advanced similarity operations and vector normalization
//! - Streaming embeddings for large datasets

pub mod batch;
pub mod builder;
pub mod cognitive_embedder;
pub mod distance;
pub mod embed;
pub mod embedding;
pub mod image;
pub mod metrics;
pub mod normalization;
pub mod providers;
pub mod resilience;
pub mod similarity;
pub mod tool;

// Re-export core types
pub use batch::{BatchConfig, BatchProcessor, BatchResult, EmbeddingBatch};
pub use builder::*;
pub use cognitive_embedder::{
    CognitiveEmbedder, CognitiveEmbedderConfig, CognitiveEmbedderPerformanceMetrics,
    CoherenceTracker, Complex64, QuantumMemory, QuantumRouterTrait, SuperpositionState};
pub use distance::*;
pub use embed::*;
pub use embedding::*;
pub use image::{
    CLIPEmbeddingModel, ImageData, ImageEmbeddingConfig, ImageEmbeddingModel, ImageEmbeddingResult,
    ImageFormat};
pub use metrics::{
    AlertConfig, AlertSeverity, GlobalMetrics, LatencyHistogram, PerformanceAlert,
    PerformanceMetric, PerformanceMonitor, QualityAnalysisError, QualityAnalysisMetrics,
    QualityAnalyzer, QualityDataPoint};
pub use normalization::{
    NormalizationMethod, apply_normalization, l1_norm, l2_norm, max_norm, normalize_vector};
pub use providers::{
    CognitiveEmbeddingProvider, CognitivePerformanceMetrics, CohereEmbeddingProvider,
    EmbeddingConfig, EmbeddingError, EnhancedEmbeddingModel, OpenAIEmbeddingProvider, QueryIntent,
    cohere_from_env, openai_from_env};
pub use resilience::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitBreakerRegistry,
    CircuitState, CircuitStateChange, ErrorCategory, FailureType};
pub use similarity::{
    BatchSimilarityComputer, SimilarityConfig, SimilarityMetric, SimilarityResult,
    cosine_similarity, euclidean_distance, manhattan_distance};
pub use tool::*;
