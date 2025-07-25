//! Mirostat Configuration Module
//!
//! Configuration types and validation for Mirostat v1 and v2 algorithms.
//! Provides zero-allocation configuration with compile-time bounds checking.

use arraystring::{ArrayString, typenum::U64};
use candle_core::{Result as CandleResult};

/// Tau adjustment learning rate bounds
const MIN_LEARNING_RATE: f32 = 1e-6;
const MAX_LEARNING_RATE: f32 = 1.0;

/// Eta parameter bounds for Mirostat v2
const MIN_ETA: f32 = 1e-6;
const MAX_ETA: f32 = 10.0;

/// Minimum tau for numerical stability
const MIN_TAU: f32 = 0.1;

/// Maximum tau to prevent extreme filtering
const MAX_TAU: f32 = 20.0;

/// Ultra-compact identifier for Mirostat variants
pub type MirostatVariant = ArrayString<U64>;

/// Mirostat algorithm configuration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MirostatConfig {
    /// Mirostat v1: Direct tau adjustment
    V1 {
        /// Target surprise (tau)
        tau: f32,
        /// Learning rate for tau adjustment
        learning_rate: f32,
    },
    /// Mirostat v2: Temperature-based adjustment
    V2 {
        /// Target surprise (tau)
        tau: f32,
        /// Eta parameter for temperature scaling
        eta: f32,
    },
}

impl MirostatConfig {
    /// Create Mirostat v1 configuration with validation
    #[inline(always)]
    pub fn v1(tau: f32, learning_rate: f32) -> CandleResult<Self> {
        if tau < MIN_TAU || tau > MAX_TAU {
            return Err(candle_core::Error::Msg(
                "Tau out of valid range".to_string(),
            ));
        }

        if learning_rate < MIN_LEARNING_RATE || learning_rate > MAX_LEARNING_RATE {
            return Err(candle_core::Error::Msg(
                "Learning rate out of valid range".to_string(),
            ));
        }

        Ok(Self::V1 { tau, learning_rate })
    }

    /// Create Mirostat v2 configuration with validation
    #[inline(always)]
    pub fn v2(tau: f32, eta: f32) -> CandleResult<Self> {
        if tau < MIN_TAU || tau > MAX_TAU {
            return Err(candle_core::Error::Msg(
                "Tau out of valid range".to_string(),
            ));
        }

        if eta < MIN_ETA || eta > MAX_ETA {
            return Err(candle_core::Error::Msg(
                "Eta out of valid range".to_string(),
            ));
        }

        Ok(Self::V2 { tau, eta })
    }

    /// Get target tau value
    #[inline(always)]
    pub const fn tau(&self) -> f32 {
        match self {
            Self::V1 { tau, .. } | Self::V2 { tau, .. } => *tau,
        }
    }

    /// Get variant name for debugging
    #[inline(always)]
    pub const fn variant_name(&self) -> &'static str {
        match self {
            Self::V1 { .. } => "Mirostat v1",
            Self::V2 { .. } => "Mirostat v2",
        }
    }
}

impl Default for MirostatConfig {
    #[inline(always)]
    fn default() -> Self {
        Self::V1 {
            tau: 5.0,
            learning_rate: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let v1 = MirostatConfig::v1(5.0, 0.1).unwrap();
        assert_eq!(v1.tau(), 5.0);
        assert_eq!(v1.variant_name(), "Mirostat v1");

        let v2 = MirostatConfig::v2(5.0, 0.1).unwrap();
        assert_eq!(v2.tau(), 5.0);
        assert_eq!(v2.variant_name(), "Mirostat v2");
    }

    #[test]
    fn test_config_validation() {
        assert!(MirostatConfig::v1(-1.0, 0.1).is_err());
        assert!(MirostatConfig::v1(5.0, -1.0).is_err());
        assert!(MirostatConfig::v2(-1.0, 0.1).is_err());
        assert!(MirostatConfig::v2(5.0, -1.0).is_err());
    }

    #[test]
    fn test_default_config() {
        let config = MirostatConfig::default();
        assert_eq!(config.tau(), 5.0);
        assert_eq!(config.variant_name(), "Mirostat v1");
    }
}