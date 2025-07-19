//! LLM integration for cognitive memory system
//!
//! This module provides production-ready LLM providers that integrate with the fluent-ai
//! provider system. Features include zero-allocation design, lock-free caching, 
//! circuit breakers, and comprehensive error handling.

pub mod production_provider;

pub use production_provider::{
    ProductionLLMProvider,
    ProviderConfig,
    ModelConfig,
    PerformanceConfig,
};