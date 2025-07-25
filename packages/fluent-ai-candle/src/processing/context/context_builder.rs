//! Context builder for creating processing contexts with custom configurations
//!
//! Provides a builder pattern for constructing ProcessingContext instances
//! with various configuration options and validation.

use super::context_core::{BaseProcessingContext, ProcessingContext, DEFAULT_CONTEXT_SIZE};
use crate::processing::traits::ProcessingResult;

/// Builder for creating processing contexts with custom configurations
#[derive(Debug)]
pub struct ContextBuilder {
    vocab_size: usize,
    context_size: Option<usize>,
    base_config: Option<BaseProcessingContext>,
}

impl ContextBuilder {
    /// Create new context builder
    #[inline(always)]
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            context_size: None,
            base_config: None,
        }
    }

    /// Set context window size
    #[inline(always)]
    pub fn context_size(mut self, size: usize) -> Self {
        self.context_size = Some(size);
        self
    }

    /// Set base processing context configuration
    #[inline(always)]
    pub fn base_context(mut self, base: BaseProcessingContext) -> Self {
        self.base_config = Some(base);
        self
    }

    /// Set temperature for base context
    #[inline(always)]
    pub fn temperature(mut self, temperature: f32) -> Self {
        let mut base = self.base_config.unwrap_or_else(BaseProcessingContext::new);
        base = base.with_temperature(temperature);
        self.base_config = Some(base);
        self
    }

    /// Set top-k for base context
    #[inline(always)]
    pub fn top_k(mut self, top_k: Option<usize>) -> Self {
        let mut base = self.base_config.unwrap_or_else(BaseProcessingContext::new);
        base = base.with_top_k(top_k);
        self.base_config = Some(base);
        self
    }

    /// Set top-p for base context
    #[inline(always)]
    pub fn top_p(mut self, top_p: Option<f32>) -> Self {
        let mut base = self.base_config.unwrap_or_else(BaseProcessingContext::new);
        base = base.with_top_p(top_p);
        self.base_config = Some(base);
        self
    }

    /// Build the processing context
    pub fn build(self) -> ProcessingResult<ProcessingContext> {
        let context_size = self.context_size.unwrap_or(DEFAULT_CONTEXT_SIZE);
        let mut context = ProcessingContext::new(self.vocab_size, context_size)?;

        if let Some(base) = self.base_config {
            *context.base_context_mut() = base;
        }

        Ok(context)
    }
}