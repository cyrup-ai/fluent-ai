//! HTTP Response Models for AI Provider Integration
//!
//! This module provides unified response structures for all AI providers with zero-allocation
//! patterns and comprehensive streaming support.
//!
//! # Response Types
//!
//! - [`completion`] - Completion responses and streaming chunks
//! - [`embedding`] - Embedding vector responses (future)
//! - [`audio`] - Audio transcription and synthesis responses (future)
//! - [`error`] - HTTP and provider-specific error types (future)
//!
//! # Architecture
//!
//! All response types follow these principles:
//! - **Zero Allocation**: Uses ArrayVec and bounded collections
//! - **Streaming First**: Built for real-time response processing
//! - **Provider Agnostic**: Unified interfaces across all providers
//! - **Type Safety**: Compile-time validation of response structures
//!
//! # Usage
//!
//! ```rust
//! use fluent_ai_domain::http::responses::completion::{
//!     CompletionResponse, CompletionChunk, StreamingResponse
//! };
//! use fluent_ai_domain::Provider;
//!
//! // Parse provider response to unified format
//! let response = CompletionResponse::from_provider_json(Provider::OpenAI, &json)?;
//!
//! // Handle streaming chunks
//! let chunk = CompletionChunk::from_provider_chunk(Provider::Anthropic, &data)?;
//!
//! // Track streaming progress
//! let mut stream = StreamingResponse::new(Provider::OpenAI, "gpt-4", "req-123")?;
//! stream.record_chunk(&chunk);
//! ```

#![allow(missing_docs)]
#![forbid(unsafe_code)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

/// Completion response models for text generation
pub mod completion;

// Future modules
// pub mod embedding;    // Embedding vector responses
// pub mod audio;        // Audio transcription and synthesis responses  
// pub mod specialized;  // Image generation, moderation, rerank responses
// pub mod error;        // HTTP and provider-specific error types