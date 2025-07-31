//! Model definitions - RE-EXPORTS ONLY
//!
//! This module ONLY re-exports from model-info package.
//! model-info is the single source of truth for all model enums.

// Re-export the Provider enum from model-info discovery module
pub use model_info::DiscoveryProvider as Provider;

// Re-export the common Model trait
pub use model_info::Model;

// Create alias for backward compatibility
pub type OpenAi = Provider;
pub type Mistral = Provider;
pub type Anthropic = Provider;
pub type Together = Provider;
pub type OpenRouter = Provider;
pub type HuggingFace = Provider;
pub type Xai = Provider;