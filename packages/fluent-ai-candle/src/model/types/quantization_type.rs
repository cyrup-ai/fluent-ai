//! Quantization type enumeration and performance characteristics
//!
//! Defines supported quantization methods with zero-allocation performance analysis.

/// Quantization types with zero-allocation inline access
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    /// No quantization
    None = 0,
    /// 4-bit quantization (Q4_0)
    Q4_0 = 1,
    /// 4-bit quantization (Q4_1)
    Q4_1 = 2,
    /// 8-bit quantization
    Q8_0 = 3}

impl Default for QuantizationType {
    #[inline(always)]
    fn default() -> Self {
        Self::None
    }
}

impl QuantizationType {
    /// Get memory reduction factor for quantization type with blazing-fast inline calculation
    #[inline(always)]
    pub fn memory_factor(&self) -> f32 {
        match self {
            Self::None => 1.0,
            Self::Q4_0 | Self::Q4_1 => 0.25, // ~4x reduction
            Self::Q8_0 => 0.5,               // ~2x reduction
        }
    }

    /// Get performance impact factor (higher = slower) with zero-cost analysis
    #[inline(always)]
    pub fn performance_impact(&self) -> f32 {
        match self {
            Self::None => 1.0,
            Self::Q4_0 | Self::Q4_1 => 1.2, // Slight overhead
            Self::Q8_0 => 1.1,              // Minimal overhead
        }
    }

    /// Get bytes per parameter for this quantization type with inline calculation
    #[inline(always)]
    pub fn bytes_per_param(&self) -> u8 {
        match self {
            Self::None => 4,              // F32
            Self::Q8_0 => 1,              // Q8
            Self::Q4_0 | Self::Q4_1 => 1, // Q4 (packed)
        }
    }
}
