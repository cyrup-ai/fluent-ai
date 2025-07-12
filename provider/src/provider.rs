//! Provider trait definitions for AI services

use crate::Model;

/// Trait for AI providers
pub trait Provider: std::fmt::Debug + Send + Sync {
    /// Get all models supported by this provider
    fn models(&self) -> Vec<Box<dyn Model>>;

    /// Get the provider name
    fn name(&self) -> &str;
}