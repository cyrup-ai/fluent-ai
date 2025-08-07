//! Core text generation engine with SIMD acceleration
//!
//! This module contains the TextGenerator implementation with AsyncStream token streaming,
//! SIMD-optimized sampling methods, and pure SIMD delegation without scalar fallbacks.

use fluent_ai_async::{emit, handle_error, AsyncStream};
use candle_core::{Device, Tensor}; 
use tokenizers::Tokenizer;

use super::{
    types::CandleResult,
    tokens::{SpecialTokens, TokenHistory},
    config::SamplingConfig,
    stats::GenerationStatistics,
    metrics::SimdMetrics,
    models::CandleModel,
};

/// Core text generation engine with SIMD acceleration
/// 
/// Provides AsyncStream-based token generation with comprehensive SIMD optimization,
/// statistics tracking, and configurable sampling parameters. Uses pure SIMD
/// delegation without scalar fallbacks for maximum performance.
pub struct TextGenerator {
    /// The model for text generation
    pub model: Box<dyn CandleModel>,
    
    /// Tokenizer for encoding/decoding
    pub tokenizer: Tokenizer,
    
    /// Device for tensor operations
    pub device: Device,
    
    /// Sampling configuration
    pub config: SamplingConfig,
    
    /// Token history for repetition penalties
    pub token_history: TokenHistory,
    
    /// Generation statistics tracking
    pub stats: GenerationStatistics,
    
    /// SIMD performance metrics
    pub simd_metrics: SimdMetrics,
}impl TextGenerator {
    /// Create new TextGenerator
    pub fn new(
        model: Box<dyn CandleModel>,
        tokenizer: Tokenizer,
        device: Device,
        config: SamplingConfig,
    ) -> Self {
        let max_history = config.repetition_context_length;
        
        Self {
            model,
            tokenizer,
            device,
            config,
            token_history: TokenHistory::new(max_history),
            stats: GenerationStatistics::new(),
            simd_metrics: SimdMetrics::new(),
        }
    }
    
    /// Generate text using AsyncStream with SIMD acceleration
    pub fn generate(
        mut self,
        prompt: String,
        max_tokens: u32,
        special_tokens: SpecialTokens,
    ) -> AsyncStream<String> {
        AsyncStream::with_channel(move |sender| {
            self.stats.start_generation();
            
            // Encode prompt to tokens using tokenizer
            let tokens = match self.tokenizer.encode(prompt.as_str(), true) {
                Ok(encoded) => encoded.get_ids().to_vec(),
                Err(e) => handle_error!(e, "prompt encoding"),
            };

            self.stats.set_input_tokens(tokens.len() as u64);
            let mut all_tokens = tokens.clone();
            let mut position = 0;
            
            // Initial forward pass
            let initial_input = match Tensor::new(tokens.as_slice(), &self.device) {
                Ok(tensor) => match tensor.unsqueeze(0) {
                    Ok(unsqueezed) => unsqueezed,
                    Err(e) => handle_error!(e, "initial tensor creation"),
                },
                Err(e) => handle_error!(e, "initial tensor from slice"),
            };
            
            let initial_logits = match self.model.forward(&initial_input, position) {
                Ok(logits) => match logits.squeeze(0) {
                    Ok(squeezed) => squeezed,
                    Err(e) => handle_error!(e, "initial logits squeeze"),
                },
                Err(e) => handle_error!(e, "initial forward pass"),
            };
            
            self.stats.record_forward_pass();            
            let logits_vec = match initial_logits.to_vec1::<f32>() {
                Ok(v) => v,
                Err(e) => handle_error!(e, "converting initial logits to vector"),
            };
            let mut next_token = match self.sample_token(&logits_vec, &tokens) {
                Ok(token) => token,
                Err(e) => handle_error!(e, "initial SIMD sampling"),
            };
            all_tokens.push(next_token);
            self.token_history.push(next_token);
            position += 1;
            
            // Check termination before decoding
            if self.should_stop(next_token, &special_tokens) {
                self.stats.stop_generation();
                return; // Graceful EOS termination
            }
            
            // Decode and emit initial token
            match self.tokenizer.decode(&[next_token], false) {
                Ok(token_str) => emit!(sender, token_str),
                Err(e) => handle_error!(e, "initial token decoding"),
            };
            
            // Generation loop - stream each token as generated
            for _index in 1..max_tokens {
                // Prepare input tensor for next forward pass
                let input = match Tensor::new(&[next_token], &self.device) {
                    Ok(tensor) => match tensor.unsqueeze(0) {
                        Ok(unsqueezed) => unsqueezed,
                        Err(e) => handle_error!(e, "tensor creation in loop"),
                    },
                    Err(e) => handle_error!(e, "tensor from single token"),
                };
                
                // Forward pass using model
                let logits = match self.model.forward(&input, position) {
                    Ok(logits) => match logits.squeeze(0) {
                        Ok(squeezed) => squeezed,
                        Err(e) => handle_error!(e, "logits squeeze in loop"),
                    },
                    Err(e) => handle_error!(e, "forward pass in loop"),
                };
                
                self.stats.record_forward_pass();                
                // SIMD sampling using existing infrastructure
                let logits_vec = match logits.to_vec1::<f32>() {
                    Ok(v) => v,
                    Err(e) => handle_error!(e, "converting logits to vector in loop"),
                };
                next_token = match self.sample_token(&logits_vec, &all_tokens) {
                    Ok(token) => token,
                    Err(e) => handle_error!(e, "SIMD sampling in loop"),
                };
                all_tokens.push(next_token);
                self.token_history.push(next_token);
                position += 1;
                
                // Check termination before decoding
                if self.should_stop(next_token, &special_tokens) {
                    break; // Graceful EOS termination
                }
                
                // Decode and emit individual token
                match self.tokenizer.decode(&[next_token], false) {
                    Ok(token_str) => emit!(sender, token_str), // Individual token streaming
                    Err(e) => handle_error!(e, "token decoding in loop"),
                };
                
                self.stats.add_tokens(1);
            }
            
            self.stats.stop_generation();
        })
    }    
    /// SIMD-optimized token sampling with comprehensive acceleration
    pub fn sample_token(&mut self, logits: &[f32], _context: &[u32]) -> CandleResult<u32> {
        use fluent_ai_simd::{
            scale_temperature, softmax, argmax,
            topk_filtering_simd, prepare_nucleus_sampling_simd, apply_penalties_simd
        };
        
        let mut logits = logits.to_vec();
        
        // Apply temperature scaling with SIMD
        if self.config.temperature != 1.0 {
            scale_temperature(&mut logits, self.config.temperature)
                .map_err(|e| crate::domain::model::error::CandleModelError::OperationNotSupported(e.to_string().into()))?;
            self.simd_metrics.record_temperature_op();
        }
        
        // Apply top-k filtering with SIMD
        if let Some(k) = self.config.top_k {
            topk_filtering_simd(&mut logits, k)
                .map_err(|e| crate::domain::model::error::CandleModelError::OperationNotSupported(e.to_string().into()))?;
            self.simd_metrics.record_topk_op();
        }
        
        // Apply nucleus sampling with SIMD
        if let Some(p) = self.config.top_p {
            prepare_nucleus_sampling_simd(&mut logits, p)
                .map_err(|e| crate::domain::model::error::CandleModelError::OperationNotSupported(e.to_string().into()))?;
            self.simd_metrics.record_nucleus_op();
        }
        
        // Apply penalties with SIMD
        if self.config.has_penalties() && !self.token_history.is_empty() {
            let context = fluent_ai_simd::context::ProcessingContext {
                temperature: self.config.temperature,
                top_k: self.config.top_k,
                top_p: self.config.top_p.map(|p| p as f32),
                token_history: self.token_history.as_slice(),
                start_time: None,
                max_new_tokens: None,
                stop_tokens: Vec::new(),
            };
            let processor_config = fluent_ai_simd::config::ProcessorConfig {
                temperature: self.config.temperature,
                top_k: self.config.top_k,
                top_p: self.config.top_p.map(|p| p as f32),
                repetition_penalty: self.config.repetition_penalty,
                frequency_penalty: self.config.frequency_penalty,
                presence_penalty: self.config.presence_penalty,
            };
            
            apply_penalties_simd(&mut logits, &context, &processor_config)
                .map_err(|e| crate::domain::model::error::CandleModelError::OperationNotSupported(e.to_string().into()))?;
            self.simd_metrics.record_penalty_op();
        }        
        // Convert to probabilities with SIMD softmax
        let probs = softmax(&logits)
            .map_err(|e| crate::domain::model::error::CandleModelError::OperationNotSupported(e.to_string().into()))?;
        self.simd_metrics.record_softmax_op();
        
        // Sample token with SIMD argmax (for deterministic) or weighted sampling
        let token = if self.config.is_deterministic() {
            argmax(&probs)
                .map_err(|e| crate::domain::model::error::CandleModelError::OperationNotSupported(e.to_string().into()))?
        } else {
            // Weighted sampling implementation would go here
            argmax(&probs)
                .map_err(|e| crate::domain::model::error::CandleModelError::OperationNotSupported(e.to_string().into()))?
        };
        
        self.simd_metrics.record_argmax_op();
        self.stats.record_simd_operation();
        
        Ok(token as u32)
    }
    
    /// Check if generation should stop
    pub fn should_stop(&self, token: u32, special_tokens: &SpecialTokens) -> bool {
        special_tokens.is_eos(token)
    }
    
    /// Get generation statistics
    pub fn stats(&self) -> &GenerationStatistics {
        &self.stats
    }
    
    /// Get SIMD metrics
    pub fn simd_metrics(&self) -> &SimdMetrics {
        &self.simd_metrics
    }
    
    /// Reset all statistics and metrics
    pub fn reset_stats(&mut self) {
        self.stats.reset();
        self.simd_metrics.reset();
        self.token_history.clear();
    }
}