//! ModelConfig performance analysis methods
//!
//! Additional methods for ModelConfig focused on memory estimation and performance analysis.

use super::ModelConfig;

impl ModelConfig {
    /// Estimate memory requirements in bytes with blazing-fast calculation
    #[inline(always)]
    pub fn estimate_memory_bytes(&self) -> u64 {
        let param_count = self.estimate_parameter_count();
        let bytes_per_param = self.quantization.bytes_per_param() as u64;

        // Model weights + KV cache + intermediate activations
        let model_size = param_count * bytes_per_param;
        let kv_cache_size =
            (self.num_layers as u64 * self.hidden_size as u64 * self.context_length as u64 * 2)
                / 1024; // Rough estimate
        let activation_size = self.hidden_size as u64 * 1024; // Working memory

        model_size + kv_cache_size + activation_size
    }

    /// Estimate parameter count with zero-allocation calculation
    #[inline(always)]
    pub fn estimate_parameter_count(&self) -> u64 {
        let embedding_params = self.vocab_size as u64 * self.hidden_size as u64;
        let layer_params =
            self.num_layers as u64 * self.hidden_size as u64 * self.hidden_size as u64 * 4; // Rough estimate
        let output_params = self.hidden_size as u64 * self.vocab_size as u64;

        embedding_params + layer_params + output_params
    }

    /// Get memory factor for this configuration with inline optimization
    #[inline(always)]
    pub fn memory_factor(&self) -> f32 {
        self.quantization.memory_factor()
    }

    /// Get performance impact factor with zero-cost analysis
    #[inline(always)]
    pub fn performance_impact(&self) -> f32 {
        self.quantization.performance_impact()
    }
}
