//! Core Model trait for all AI models

use crate::types::CandleModelInfo as ModelInfo;

/// Core trait for all AI models with zero-allocation design
pub trait Model: Send + Sync + std::fmt::Debug + 'static {
    /// Get the model's information
    fn info(&self) -> &'static ModelInfo;

    /// Get the model's name with blazing-fast inline optimization
    #[inline(always)]
    fn name(&self) -> &'static str {
        self.info().name()
    }

    /// Get the model's provider name with blazing-fast inline optimization
    #[inline(always)]
    fn provider(&self) -> &'static str {
        self.info().provider()
    }

    /// Get the model's maximum input tokens with zero-allocation access
    #[inline(always)]
    fn max_input_tokens(&self) -> Option<u32> {
        self.info().max_input_tokens.map(|n| n.get())
    }

    /// Get the model's maximum output tokens with zero-allocation access
    #[inline(always)]
    fn max_output_tokens(&self) -> Option<u32> {
        self.info().max_output_tokens.map(|n| n.get())
    }

    /// Check if the model supports vision with blazing-fast inline optimization
    #[inline(always)]
    fn supports_vision(&self) -> bool {
        self.info().has_vision()
    }

    /// Check if the model supports function calling with blazing-fast inline optimization
    #[inline(always)]
    fn supports_function_calling(&self) -> bool {
        self.info().has_function_calling()
    }
}