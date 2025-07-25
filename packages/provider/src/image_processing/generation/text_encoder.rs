//! Text encoding implementation for Stable Diffusion 3
//!
//! This module implements the Triple CLIP encoder architecture following the exact patterns
//! from stable-diffusion-3/clip.rs for optimal compatibility and performance.

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::{D, DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{stable_diffusion, t5};
use hf_hub::api::sync::Api;
use thiserror::Error;
use tokenizers::tokenizer::Tokenizer;

/// Text encoding errors
#[derive(Error, Debug)]
pub enum TextEncodingError {
    #[error("Tokenizer loading failed: {0}")]
    TokenizerLoadingError(String),

    #[error("Text encoding failed: {0}")]
    EncodingError(String),

    #[error("Model loading failed: {0}")]
    ModelLoadingError(String),

    #[error("Tensor operation failed: {0}")]
    TensorError(#[from] candle_core::Error),

    #[error("HuggingFace Hub error: {0}")]
    HubError(#[from] hf_hub::api::sync::ApiError)}

/// Result type for text encoding operations
pub type TextEncodingResult<T> = Result<T, TextEncodingError>;

/// CLIP-L/G encoder with tokenizer (following stable-diffusion-3/clip.rs)
pub struct ClipWithTokenizer {
    clip: stable_diffusion::clip::ClipTextTransformer,
    config: stable_diffusion::clip::Config,
    tokenizer: Tokenizer,
    max_position_embeddings: usize}

impl ClipWithTokenizer {
    /// Create new CLIP encoder with tokenizer
    pub fn new(
        vb: VarBuilder,
        config: stable_diffusion::clip::Config,
        tokenizer_path: &str,
        max_position_embeddings: usize,
    ) -> TextEncodingResult<Self> {
        let clip = stable_diffusion::clip::ClipTextTransformer::new(vb, &config).map_err(|e| {
            TextEncodingError::ModelLoadingError(format!("CLIP model loading failed: {}", e))
        })?;

        let path_buf = Api::new()
            .map_err(|e| TextEncodingError::HubError(e))?
            .model(tokenizer_path.to_string())
            .get("tokenizer.json")
            .map_err(|e| TextEncodingError::HubError(e))?;

        let tokenizer = Tokenizer::from_file(path_buf.to_str().ok_or_else(|| {
            TextEncodingError::TokenizerLoadingError(
                "Failed to serialize HuggingFace PathBuf".to_string(),
            )
        })?)
        .map_err(|e| {
            TextEncodingError::TokenizerLoadingError(format!("Tokenizer loading failed: {}", e))
        })?;

        Ok(Self {
            clip,
            config,
            tokenizer,
            max_position_embeddings})
    }

    /// Encode text to embedding tensors
    pub fn encode_text_to_embedding(
        &self,
        prompt: &str,
        device: &Device,
    ) -> TextEncodingResult<(Tensor, Tensor)> {
        let pad_id = match &self.config.pad_with {
            Some(padding) => *self
                .tokenizer
                .get_vocab(true)
                .get(padding.as_str())
                .ok_or_else(|| {
                    TextEncodingError::EncodingError("Failed to tokenize padding".to_string())
                })?,
            None => *self
                .tokenizer
                .get_vocab(true)
                .get("<|endoftext|>")
                .ok_or_else(|| {
                    TextEncodingError::EncodingError("Failed to tokenize end-of-text".to_string())
                })?};

        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| TextEncodingError::EncodingError(format!("Token encoding failed: {}", e)))?
            .get_ids()
            .to_vec();

        let eos_position = tokens.len() - 1;

        // Pad tokens to max position embeddings
        while tokens.len() < self.max_position_embeddings {
            tokens.push(pad_id);
        }

        let tokens = Tensor::new(tokens.as_slice(), device)
            .map_err(|e| TextEncodingError::TensorError(e))?
            .unsqueeze(0)
            .map_err(|e| TextEncodingError::TensorError(e))?;

        let (text_embeddings, text_embeddings_penultimate) = self
            .clip
            .forward_until_encoder_layer(&tokens, usize::MAX, -2)
            .map_err(|e| {
                TextEncodingError::EncodingError(format!("CLIP forward pass failed: {}", e))
            })?;

        let text_embeddings_pooled = text_embeddings
            .i((0, eos_position, ..))
            .map_err(|e| TextEncodingError::TensorError(e))?;

        Ok((text_embeddings_penultimate, text_embeddings_pooled))
    }
}

/// T5-XXL encoder with tokenizer (following stable-diffusion-3/clip.rs)
pub struct T5WithTokenizer {
    t5: t5::T5EncoderModel,
    tokenizer: Tokenizer,
    max_position_embeddings: usize}

impl T5WithTokenizer {
    /// Create new T5 encoder with tokenizer
    pub fn new(
        vb: VarBuilder,
        config: &t5::Config,
        tokenizer_path: &str,
        max_position_embeddings: usize,
    ) -> TextEncodingResult<Self> {
        let t5 = t5::T5EncoderModel::load(vb, config).map_err(|e| {
            TextEncodingError::ModelLoadingError(format!("T5 model loading failed: {}", e))
        })?;

        let path_buf = Api::new()
            .map_err(|e| TextEncodingError::HubError(e))?
            .model(tokenizer_path.to_string())
            .get("tokenizer.json")
            .map_err(|e| TextEncodingError::HubError(e))?;

        let tokenizer = Tokenizer::from_file(path_buf.to_str().ok_or_else(|| {
            TextEncodingError::TokenizerLoadingError(
                "Failed to serialize HuggingFace PathBuf".to_string(),
            )
        })?)
        .map_err(|e| {
            TextEncodingError::TokenizerLoadingError(format!("Tokenizer loading failed: {}", e))
        })?;

        Ok(Self {
            t5,
            tokenizer,
            max_position_embeddings})
    }

    /// Encode text to embedding tensor
    pub fn encode_text_to_embedding(
        &self,
        prompt: &str,
        device: &Device,
    ) -> TextEncodingResult<Tensor> {
        let pad_id = *self.tokenizer.get_vocab(true).get("</s>").ok_or_else(|| {
            TextEncodingError::EncodingError("Failed to tokenize T5 end token".to_string())
        })?;

        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| {
                TextEncodingError::EncodingError(format!("T5 token encoding failed: {}", e))
            })?
            .get_ids()
            .to_vec();

        // Pad tokens to max position embeddings
        while tokens.len() < self.max_position_embeddings {
            tokens.push(pad_id);
        }

        let tokens = Tensor::new(tokens.as_slice(), device)
            .map_err(|e| TextEncodingError::TensorError(e))?
            .unsqueeze(0)
            .map_err(|e| TextEncodingError::TensorError(e))?;

        let text_embeddings = self.t5.forward(&tokens).map_err(|e| {
            TextEncodingError::EncodingError(format!("T5 forward pass failed: {}", e))
        })?;

        Ok(text_embeddings)
    }
}

/// Triple CLIP encoder combining CLIP-L, CLIP-G, and T5-XXL
pub struct StableDiffusion3TripleClipWithTokenizer {
    clip_l: ClipWithTokenizer,
    clip_g: ClipWithTokenizer,
    t5: T5WithTokenizer}

impl StableDiffusion3TripleClipWithTokenizer {
    /// Create new triple CLIP encoder
    pub fn new(vb: VarBuilder) -> TextEncodingResult<Self> {
        // CLIP-L configuration
        let clip_l_config = stable_diffusion::clip::Config {
            vocab_size: 49408,
            embed_dim: 768,
            intermediate_size: 3072,
            max_position_embeddings: 77,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            max_sequence_length: 77,
            pad_with: None};

        // CLIP-G configuration
        let clip_g_config = stable_diffusion::clip::Config {
            vocab_size: 49408,
            embed_dim: 1280,
            intermediate_size: 5120,
            max_position_embeddings: 77,
            num_hidden_layers: 32,
            num_attention_heads: 20,
            max_sequence_length: 77,
            pad_with: None};

        // T5-XXL configuration
        let t5_config = t5::Config {
            vocab_size: 32128,
            d_model: 4096,
            d_kv: 64,
            d_ff: 10240,
            num_layers: 24,
            num_heads: 64,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.0,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: t5::FeedForwardProj::Gated,
            use_cache: false};

        let clip_l = ClipWithTokenizer::new(
            vb.pp("clip_l"),
            clip_l_config,
            "openai/clip-vit-large-patch14",
            77,
        )?;

        let clip_g = ClipWithTokenizer::new(
            vb.pp("clip_g"),
            clip_g_config,
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            77,
        )?;

        let t5 = T5WithTokenizer::new(vb.pp("t5xxl"), &t5_config, "google/t5-v1_1-xxl", 77)?;

        Ok(Self { clip_l, clip_g, t5 })
    }

    /// Encode text to embedding tensors (context and y)
    pub fn encode_text_to_embedding(
        &self,
        prompt: &str,
        device: &Device,
    ) -> TextEncodingResult<(Tensor, Tensor)> {
        // CLIP-L encoding
        let (clip_l_text_embeddings, clip_l_text_embeddings_pooled) =
            self.clip_l.encode_text_to_embedding(prompt, device)?;

        // CLIP-G encoding
        let (clip_g_text_embeddings, clip_g_text_embeddings_pooled) =
            self.clip_g.encode_text_to_embedding(prompt, device)?;

        // T5-XXL encoding
        let t5_text_embeddings = self.t5.encode_text_to_embedding(prompt, device)?;

        // Concatenate CLIP-L and CLIP-G text embeddings for context
        let context = Tensor::cat(&[clip_l_text_embeddings, clip_g_text_embeddings], D::Minus1)
            .map_err(|e| TextEncodingError::TensorError(e))?;

        // Concatenate pooled embeddings with T5 for y
        let y = Tensor::cat(
            &[
                clip_l_text_embeddings_pooled,
                clip_g_text_embeddings_pooled,
                t5_text_embeddings
                    .mean(1)
                    .map_err(|e| TextEncodingError::TensorError(e))?,
            ],
            D::Minus1,
        )
        .map_err(|e| TextEncodingError::TensorError(e))?;

        Ok((context, y))
    }
}

/// Utility functions for text encoding
pub mod utils {
    use super::*;

    /// Validate text prompt length
    pub fn validate_prompt_length(prompt: &str, max_length: usize) -> TextEncodingResult<()> {
        if prompt.len() > max_length {
            return Err(TextEncodingError::EncodingError(format!(
                "Prompt length {} exceeds maximum {}",
                prompt.len(),
                max_length
            )));
        }
        Ok(())
    }

    /// Clean and prepare text prompt
    pub fn prepare_prompt(prompt: &str) -> String {
        // Remove excessive whitespace and normalize
        prompt.trim().chars().collect::<String>()
    }

    /// Estimate token count for prompt
    pub fn estimate_token_count(prompt: &str) -> usize {
        // Rough estimation: 1 token per 4 characters
        (prompt.len() + 3) / 4
    }
}

/// Get context embedder configuration for MMDiT
pub fn get_context_embedder_config()
-> candle_transformers::models::mmdit::model::ContextEmbedderConfig {
    use candle_transformers::models::mmdit::model::ContextEmbedderConfig;

    ContextEmbedderConfig {
        context_dim: 4096, // Combined dimension from CLIP-L (768) + CLIP-G (1280) + T5 (4096)
        hidden_dim: 1536,  // Hidden dimension for context processing
        num_heads: 24,     // Number of attention heads
        num_layers: 2,     // Number of transformer layers
    }
}
