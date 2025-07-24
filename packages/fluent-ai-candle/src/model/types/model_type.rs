//! Model type enumeration and related functionality
//!
//! Defines supported model architectures with zero-allocation design patterns.

/// Supported model types with zero-allocation inline access
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// Kimi K2 - 1T parameter MoE model with 32B activated parameters
    KimiK2 = 0,
}

impl ModelType {
    /// Get model type from string with blazing-fast inline parsing  
    #[inline(always)]
    pub fn from_str(s: &str) -> Self {
        let s = s.to_lowercase();
        if s.contains("kimi") || s.contains("k2") {
            Self::KimiK2
        } else {
            Self::KimiK2 // Default to KimiK2 as it's the only supported model
        }
    }

    /// Get default context length for model type with zero-cost inline access
    #[inline(always)]
    pub fn default_context_length(&self) -> u32 {
        match self {
            Self::KimiK2 => 131072, // Kimi K2 supports 131k context length
        }
    }
}

impl Default for ModelType {
    #[inline(always)]
    fn default() -> Self {
        Self::KimiK2
    }
}