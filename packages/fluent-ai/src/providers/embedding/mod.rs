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

pub mod providers;
pub mod batch;
pub mod image;
pub mod similarity;
pub mod normalization;

// Re-export core types
pub use providers::{
    EnhancedEmbeddingModel, EmbeddingConfig, OpenAIEmbeddingProvider, CohereEmbeddingProvider,
    openai_from_env, cohere_from_env
};
pub use batch::{EmbeddingBatch, BatchConfig, BatchResult, BatchProcessor};
pub use image::{
    ImageEmbeddingModel, ImageData, ImageEmbeddingConfig, ImageEmbeddingResult, 
    CLIPEmbeddingModel, ImageFormat
};
pub use similarity::{
    SimilarityMetric, SimilarityConfig, SimilarityResult, BatchSimilarityComputer,
    cosine_similarity, euclidean_distance, manhattan_distance
};
pub use normalization::{
    normalize_vector, l2_norm, l1_norm, max_norm, NormalizationMethod, apply_normalization
};