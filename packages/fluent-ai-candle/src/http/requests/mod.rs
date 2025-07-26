//! HTTP Request Models for AI Provider Integration
//!
//! This module contains request structures for all AI provider APIs,
//! designed with zero-allocation patterns and type safety.
//!
//! # Available Request Types
//!
//! - [`completion`] - Completion/chat requests for all 17 providers
//! - [`embedding`] - Text embedding requests (planned)
//! - [`audio`] - Audio transcription and TTS requests (planned)
//! - [`specialized`] - Provider-specific requests like reranking (planned)
//!
//! # Architecture
//!
//! All request models use:
//! - Zero-allocation patterns with `ArrayVec` for bounded collections
//! - Comprehensive parameter validation
//! - Provider-specific format conversion
//! - Type-safe parameter handling
//! - Proper error handling and validation

#![warn(missing_docs)]
#![forbid(unsafe_code)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod completion;

// Re-exports for convenience
pub use completion::{
    CompletionRequest, CompletionRequestError, ProviderExtensions,
    ToolDefinition, ToolChoice, FunctionDefinition, ToolType,
    OpenAIExtensions, AnthropicExtensions, GoogleExtensions,
    BedrockExtensions, CohereExtensions};