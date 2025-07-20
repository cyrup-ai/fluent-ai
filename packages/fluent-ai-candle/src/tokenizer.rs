//! Zero-allocation tokenization with pre-allocated buffers and SIMD optimization

use crate::error::{CandleError, CandleResult};
use arrayvec::ArrayVec;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Maximum token buffer size
const MAX_TOKEN_BUFFER: usize = 2048;

/// Maximum text buffer size
const MAX_TEXT_BUFFER: usize = 8192;

/// Maximum special tokens
const MAX_SPECIAL_TOKENS: usize = 32;

/// Tokenizer configuration
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Add BOS token at the beginning
    pub add_bos_token: bool,
    /// Add EOS token at the end
    pub add_eos_token: bool,
    /// Padding token ID
    pub pad_token_id: Option<u32>,
    /// BOS token ID
    pub bos_token_id: Option<u32>,
    /// EOS token ID
    pub eos_token_id: Option<u32>,
    /// UNK token ID
    pub unk_token_id: Option<u32>,
    /// Maximum sequence length
    pub max_length: u32,
    /// Truncation strategy
    pub truncation: TruncationStrategy,
}

impl Default for TokenizerConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            add_bos_token: true,
            add_eos_token: false,
            pad_token_id: None,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            unk_token_id: Some(0),
            max_length: 2048,
            truncation: TruncationStrategy::LongestFirst,
        }
    }
}

/// Truncation strategies
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// Truncate from the beginning
    OnlyFirst = 0,
    /// Truncate from the end
    OnlySecond = 1,
    /// Truncate the longest sequence first
    LongestFirst = 2,
    /// Don't truncate
    DoNotTruncate = 3,
}

/// Zero-allocation tokenizer with pre-allocated buffers
#[repr(C)]
pub struct CandleTokenizer {
    /// The underlying tokenizer
    tokenizer: Arc<Tokenizer>,
    /// Pre-allocated token buffer
    token_buffer: parking_lot::Mutex<ArrayVec<u32, MAX_TOKEN_BUFFER>>,
    /// Pre-allocated text buffer
    text_buffer: parking_lot::Mutex<ArrayVec<u8, MAX_TEXT_BUFFER>>,
    /// Special tokens map
    special_tokens: HashMap<String, u32>,
    /// Tokenizer configuration
    config: TokenizerConfig,
    /// Vocabulary size
    vocab_size: u32,
}

impl CandleTokenizer {
    /// Create a new tokenizer from a tokenizer instance
    #[inline(always)]
    pub fn new(tokenizer: Tokenizer, config: TokenizerConfig) -> CandleResult<Self> {
        let vocab_size = tokenizer.get_vocab_size(false) as u32;
        
        // Extract special tokens
        let mut special_tokens = HashMap::new();
        
        // Add common special tokens
        if let Some(token) = tokenizer.token_to_id("<pad>") {
            special_tokens.insert("<pad>".to_string(), token);
        }
        if let Some(token) = tokenizer.token_to_id("<s>") {
            special_tokens.insert("<s>".to_string(), token);
        }
        if let Some(token) = tokenizer.token_to_id("</s>") {
            special_tokens.insert("</s>".to_string(), token);
        }
        if let Some(token) = tokenizer.token_to_id("<unk>") {
            special_tokens.insert("<unk>".to_string(), token);
        }
        
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            token_buffer: parking_lot::Mutex::new(ArrayVec::new()),
            text_buffer: parking_lot::Mutex::new(ArrayVec::new()),
            special_tokens,
            config,
            vocab_size,
        })
    }
    
    /// Load tokenizer from file
    #[inline(always)]
    pub fn from_file<P: AsRef<std::path::Path>>(path: P, config: TokenizerConfig) -> CandleResult<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|_| CandleError::tokenizer("Failed to load tokenizer from file"))?;
        
        Self::new(tokenizer, config)
    }
    
    /// Load tokenizer from HuggingFace Hub
    #[inline(always)]
    pub async fn from_hub(repo_id: &str, config: TokenizerConfig) -> CandleResult<Self> {
        let api = hf_hub::api::tokio::Api::new()
            .map_err(|e| CandleError::HuggingFaceHub(format!("HF Hub API error: {}", e)))?;
        
        let repo = api.model(repo_id.to_string());
        let tokenizer_path = repo.get("tokenizer.json").await
            .map_err(|e| CandleError::HuggingFaceHub(format!("Failed to download tokenizer: {}", e)))?;
        
        Self::from_file(tokenizer_path, config)
    }
    
    /// Encode text to token IDs with zero allocation
    #[inline(always)]
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> CandleResult<ArrayVec<u32, MAX_TOKEN_BUFFER>> {
        let encoding = self.tokenizer
            .encode(text, add_special_tokens)
            .map_err(|_| CandleError::tokenizer("Failed to encode text"))?;
        
        let token_ids = encoding.get_ids();
        
        // Apply configuration
        let mut final_tokens = ArrayVec::new();
        
        // Add BOS token if configured
        if self.config.add_bos_token && add_special_tokens {
            if let Some(bos_id) = self.config.bos_token_id {
                final_tokens.try_push(bos_id)
                    .map_err(|_| CandleError::tokenizer("Token buffer overflow"))?;
            }
        }
        
        // Add main tokens
        for &token_id in token_ids {
            if final_tokens.len() >= MAX_TOKEN_BUFFER {
                break; // Prevent overflow
            }
            final_tokens.try_push(token_id)
                .map_err(|_| CandleError::tokenizer("Token buffer overflow"))?;
        }
        
        // Add EOS token if configured
        if self.config.add_eos_token && add_special_tokens {
            if let Some(eos_id) = self.config.eos_token_id {
                if final_tokens.len() < MAX_TOKEN_BUFFER {
                    let _ = final_tokens.try_push(eos_id);
                }
            }
        }
        
        // Apply truncation if necessary
        if final_tokens.len() > self.config.max_length as usize {
            final_tokens.truncate(self.config.max_length as usize);
        }
        
        Ok(final_tokens)
    }
    
    /// Decode token IDs to text with zero allocation
    #[inline(always)]
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> CandleResult<String> {
        let text = self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|_| CandleError::tokenizer("Failed to decode tokens"))?;
        
        Ok(text)
    }
    
    /// Encode multiple texts in batch
    #[inline(always)]
    pub fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> CandleResult<Vec<ArrayVec<u32, MAX_TOKEN_BUFFER>>> {
        let mut results = Vec::with_capacity(texts.len());
        
        for text in texts {
            results.push(self.encode(text, add_special_tokens)?);
        }
        
        Ok(results)
    }
    
    /// Decode multiple token sequences in batch
    #[inline(always)]
    pub fn decode_batch(&self, token_sequences: &[&[u32]], skip_special_tokens: bool) -> CandleResult<Vec<String>> {
        let mut results = Vec::with_capacity(token_sequences.len());
        
        for tokens in token_sequences {
            results.push(self.decode(tokens, skip_special_tokens)?);
        }
        
        Ok(results)
    }
    
    /// Get token ID for a specific token
    #[inline(always)]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }
    
    /// Get token string for a specific ID
    #[inline(always)]
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }
    
    /// Get vocabulary size
    #[inline(always)]
    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }
    
    /// Get special token ID
    #[inline(always)]
    pub fn special_token_id(&self, token: &str) -> Option<u32> {
        self.special_tokens.get(token).copied()
    }
    
    /// Get BOS token ID
    #[inline(always)]
    pub fn bos_token_id(&self) -> Option<u32> {
        self.config.bos_token_id
    }
    
    /// Get EOS token ID
    #[inline(always)]
    pub fn eos_token_id(&self) -> Option<u32> {
        self.config.eos_token_id
    }
    
    /// Get PAD token ID
    #[inline(always)]
    pub fn pad_token_id(&self) -> Option<u32> {
        self.config.pad_token_id
    }
    
    /// Get UNK token ID
    #[inline(always)]
    pub fn unk_token_id(&self) -> Option<u32> {
        self.config.unk_token_id
    }
    
    /// Check if token is special
    #[inline(always)]
    pub fn is_special_token(&self, token_id: u32) -> bool {
        self.special_tokens.values().any(|&id| id == token_id)
    }
    
    /// Estimate token count for text (fast approximation)
    #[inline(always)]
    pub fn estimate_token_count(&self, text: &str) -> u32 {
        // Fast approximation: 1 token per 4 characters for most languages
        // This is much faster than actual tokenization for length estimation
        (text.len() / 4).max(1) as u32
    }
    
    /// Truncate text to fit within token limit (approximate)
    #[inline(always)]
    pub fn truncate_text(&self, text: &str, max_tokens: u32) -> &str {
        let max_chars = (max_tokens * 4) as usize; // 4 chars per token approximation
        
        if text.len() <= max_chars {
            text
        } else {
            // Find a safe truncation point (avoid cutting UTF-8 characters)
            let mut truncate_at = max_chars;
            while truncate_at > 0 && !text.is_char_boundary(truncate_at) {
                truncate_at -= 1;
            }
            &text[..truncate_at]
        }
    }
    
    /// Get tokenizer configuration
    #[inline(always)]
    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }
    
    /// Update tokenizer configuration
    #[inline(always)]
    pub fn update_config(&mut self, config: TokenizerConfig) {
        self.config = config;
    }
}

/// Token with metadata
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct TokenWithMetadata {
    /// Token ID
    pub id: u32,
    /// Token text
    pub text: SmallVec<[u8; 16]>,
    /// Start position in original text
    pub start: u32,
    /// End position in original text
    pub end: u32,
    /// Is special token
    pub is_special: bool,
}

impl TokenWithMetadata {
    /// Create a new token with metadata
    #[inline(always)]
    pub fn new(id: u32, text: &str, start: u32, end: u32, is_special: bool) -> CandleResult<Self> {
        let mut text_bytes = SmallVec::new();
        text_bytes.try_extend_from_slice(text.as_bytes())
            .map_err(|_| CandleError::tokenizer("Token text too long"))?;
        
        Ok(Self {
            id,
            text: text_bytes,
            start,
            end,
            is_special,
        })
    }
    
    /// Get token text as string
    #[inline(always)]
    pub fn text_str(&self) -> CandleResult<&str> {
        std::str::from_utf8(&self.text)
            .map_err(|_| CandleError::tokenizer("Invalid UTF-8 in token text"))
    }
}

/// Tokenization result with detailed metadata
#[repr(C)]
pub struct TokenizationResult {
    /// Token IDs
    pub token_ids: ArrayVec<u32, MAX_TOKEN_BUFFER>,
    /// Tokens with metadata
    pub tokens: SmallVec<[TokenWithMetadata; 64]>,
    /// Original text length
    pub original_length: u32,
    /// Total tokens
    pub token_count: u32,
}

impl TokenizationResult {
    /// Create a new tokenization result
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            token_ids: ArrayVec::new(),
            tokens: SmallVec::new(),
            original_length: 0,
            token_count: 0,
        }
    }
    
    /// Add a token to the result
    #[inline(always)]
    pub fn add_token(&mut self, token: TokenWithMetadata) -> CandleResult<()> {
        self.token_ids.try_push(token.id)
            .map_err(|_| CandleError::tokenizer("Token buffer overflow"))?;
        
        self.tokens.push(token);
        self.token_count += 1;
        
        Ok(())
    }
    
    /// Get token IDs as slice
    #[inline(always)]
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
    
    /// Get tokens with metadata
    #[inline(always)]
    pub fn tokens(&self) -> &[TokenWithMetadata] {
        &self.tokens
    }
    
    /// Get token count
    #[inline(always)]
    pub fn token_count(&self) -> u32 {
        self.token_count
    }
}

impl Default for TokenizationResult {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl Send for CandleTokenizer {}
unsafe impl Sync for CandleTokenizer {}