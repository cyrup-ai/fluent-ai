//! Builder pattern for creating composite processors with fluent API
//!
//! This module provides the CompositeProcessorBuilder for creating composite processors
//! with a convenient fluent API and pre-configured processor chains.

use crate::sampling::SamplingError;
use crate::processing::traits::LogitsProcessor;
use super::core::CompositeProcessor;

/// Builder for creating composite processors with fluent API
#[derive(Debug, Default)]
pub struct CompositeProcessorBuilder {
    processors: Vec<Box<dyn LogitsProcessor>>}

impl CompositeProcessorBuilder {
    /// Create a new builder
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            processors: Vec::new()}
    }

    /// Add a processor to the chain
    #[inline(always)]
    pub fn add_processor(mut self, processor: Box<dyn LogitsProcessor>) -> Self {
        self.processors.push(processor);
        self
    }

    /// Add temperature processor
    #[inline(always)]
    pub fn temperature(self, temperature: f64) -> Result<Self, SamplingError> {
        use crate::processing::processors::temperature::TemperatureProcessor;
        let processor = TemperatureProcessor::new(temperature as f32)?;
        Ok(self.add_processor(Box::new(processor)))
    }

    /// Add top-k processor
    #[inline(always)]
    pub fn top_k(self, k: usize) -> Result<Self, SamplingError> {
        use crate::processing::processors::top_k::TopKProcessor;
        let processor = TopKProcessor::new(k)?;
        Ok(self.add_processor(Box::new(processor)))
    }

    /// Add top-p processor
    #[inline(always)]
    pub fn top_p(self, p: f64) -> Result<Self, SamplingError> {
        use crate::processing::processors::top_p::TopPProcessor;
        let processor = TopPProcessor::new(p as f32)?;
        Ok(self.add_processor(Box::new(processor)))
    }

    /// Add repetition penalty processor
    #[inline(always)]
    pub fn repetition_penalty(
        self,
        penalty: f64,
        context_size: usize,
    ) -> Result<Self, SamplingError> {
        use crate::processing::processors::repetition_penalty::RepetitionPenaltyProcessor;
        let processor = RepetitionPenaltyProcessor::new(
            penalty as f32,
            0.0,
            0.0,
            context_size, /* repetition_penalty, frequency_penalty, presence_penalty, context_window */
        )?;
        Ok(self.add_processor(Box::new(processor)))
    }

    /// Add typical sampling processor
    #[inline(always)]
    pub fn typical_sampling(self, typical_p: f64) -> Result<Self, SamplingError> {
        use crate::sampling::typical::TypicalSamplingProcessor;
        let processor = TypicalSamplingProcessor::new(typical_p)?;
        Ok(self.add_processor(Box::new(processor)))
    }

    /// Add Gumbel-Softmax processor
    #[inline(always)]
    pub fn gumbel_softmax(
        self,
        temperature: f32,
        hard: bool,
        seed: u64,
        device: candle_core::Device,
    ) -> Result<Self, SamplingError> {
        use crate::sampling::gumbel::GumbelSoftmaxProcessor;
        let processor = GumbelSoftmaxProcessor::new(temperature, hard, seed, device)?;
        Ok(self.add_processor(Box::new(processor)))
    }

    /// Build the composite processor
    pub fn build(self) -> Result<CompositeProcessor, SamplingError> {
        CompositeProcessor::new(self.processors)
    }

    /// Get the current number of processors
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.processors.len()
    }

    /// Check if the builder is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }
}

/// Utility functions for creating common processor chains
pub mod presets {
    use super::*;

    /// Create a standard text generation processor chain
    ///
    /// Includes temperature scaling, repetition penalty, top-k, and top-p filtering
    /// in the optimal order for text generation.
    pub fn standard_text_generation_chain(
        temperature: f64,
        top_k: Option<usize>,
        top_p: Option<f64>,
        repetition_penalty: Option<f64>,
    ) -> Result<CompositeProcessor, SamplingError> {
        let mut builder = CompositeProcessorBuilder::new();

        // Add repetition penalty first (context-dependent)
        if let Some(penalty) = repetition_penalty {
            builder = builder.repetition_penalty(penalty, 256)?;
        }

        // Add temperature scaling
        builder = builder.temperature(temperature)?;

        // Add top-k filtering (before top-p for efficiency)
        if let Some(k) = top_k {
            builder = builder.top_k(k)?;
        }

        // Add top-p nucleus sampling
        if let Some(p) = top_p {
            builder = builder.top_p(p)?;
        }

        builder.build()
    }

    /// Create a creative writing processor chain
    ///
    /// Optimized for creative text generation with higher randomness
    /// and sophisticated repetition avoidance.
    pub fn creative_writing_chain() -> Result<CompositeProcessor, SamplingError> {
        CompositeProcessorBuilder::new()
            .repetition_penalty(1.15, 512)?
            .temperature(0.85)?
            .top_p(0.92)?
            .build()
    }

    /// Create a code generation processor chain
    ///
    /// Optimized for code generation with lower randomness
    /// and precise token selection.
    pub fn code_generation_chain() -> Result<CompositeProcessor, SamplingError> {
        CompositeProcessorBuilder::new()
            .repetition_penalty(1.05, 128)?
            .temperature(0.2)?
            .top_k(20)?
            .top_p(0.95)?
            .build()
    }

    /// Create a balanced conversation chain
    ///
    /// Optimized for conversational AI with moderate creativity
    /// and coherence maintenance.
    pub fn conversation_chain() -> Result<CompositeProcessor, SamplingError> {
        CompositeProcessorBuilder::new()
            .repetition_penalty(1.1, 256)?
            .temperature(0.7)?
            .top_k(40)?
            .top_p(0.9)?
            .build()
    }
}