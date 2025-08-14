//! Provider definitions - RE-EXPORTS ONLY
//!
//! This module ONLY re-exports from model-info package.
//! model-info is the single source of truth for all provider types.

// Re-export the ProviderTrait and Provider enum from model-info
pub use model_info::{DiscoveryProvider as Provider, ProviderTrait};
