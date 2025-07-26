//! Mirostat Configuration Module
//!
//! Configuration types and validation for Mirostat v1 and v2 algorithms.
//! Provides zero-allocation configuration with compile-time bounds checking.

use arraystring::{ArrayString, typenum::U64};
use candle_core::{Result as CandleResult};

/// Minimum learning rate for tau adjustment in Mirostat v1
///
/// This constant defines the lower bound for the learning rate parameter used
/// in Mirostat v1 tau adjustment. Values below this threshold can cause
/// numerical instability and extremely slow convergence.
///
/// Value: 1e-6 (0.000001)
pub const MIN_LEARNING_RATE: f32 = 1e-6;

/// Maximum learning rate for tau adjustment in Mirostat v1
///
/// This constant defines the upper bound for the learning rate parameter used
/// in Mirostat v1 tau adjustment. Values above this threshold can cause
/// oscillation and instability in the tau adjustment process.
///
/// Value: 1.0 (100% adjustment per step)
pub const MAX_LEARNING_RATE: f32 = 1.0;

/// Minimum eta parameter for Mirostat v2 temperature scaling
///
/// This constant defines the lower bound for the eta parameter used in
/// Mirostat v2 for temperature scaling. Values below this threshold can
/// cause numerical instability and ineffective temperature adjustment.
///
/// Value: 1e-6 (0.000001)
const MIN_ETA: f32 = 1e-6;

/// Maximum eta parameter for Mirostat v2 temperature scaling
///
/// This constant defines the upper bound for the eta parameter used in
/// Mirostat v2 for temperature scaling. Values above this threshold can
/// cause excessive temperature fluctuations and unstable sampling.
///
/// Value: 10.0
const MAX_ETA: f32 = 10.0;

/// Minimum tau (target surprise) for numerical stability
///
/// This constant defines the lower bound for the tau parameter (target surprise)
/// in both Mirostat v1 and v2. Values below this threshold can cause
/// overly restrictive sampling and numerical instability.
///
/// Value: 0.1
const MIN_TAU: f32 = 0.1;

/// Maximum tau (target surprise) to prevent extreme filtering
///
/// This constant defines the upper bound for the tau parameter (target surprise)
/// in both Mirostat v1 and v2. Values above this threshold can cause
/// excessively permissive sampling that defeats the purpose of Mirostat.
///
/// Value: 20.0
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
        learning_rate: f32},
    /// Mirostat v2: Temperature-based adjustment
    V2 {
        /// Target surprise (tau)
        tau: f32,
        /// Eta parameter for temperature scaling
        eta: f32}}

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
            Self::V1 { tau, .. } | Self::V2 { tau, .. } => *tau}
    }

    /// Get variant name for debugging
    #[inline(always)]
    pub const fn variant_name(&self) -> &'static str {
        match self {
            Self::V1 { .. } => "Mirostat v1",
            Self::V2 { .. } => "Mirostat v2"}
    }
}

impl Default for MirostatConfig {
    #[inline(always)]
    fn default() -> Self {
        Self::V1 {
            tau: 5.0,
            learning_rate: 0.1}
    }
}
