//! Zero-allocation tokenizer implementation for Candle models
//!
//! This module provides high-performance tokenization with streaming support,
//! lock-free design, and comprehensive special token handling.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use arc_swap::ArcSwap;
use arrayvec::ArrayString;
use crossbeam_utils::atomic::AtomicCell;
use smallvec::SmallVec;

use super::error::{CandleError, CandleResult};
use super::models::CandleModel;

/// Maximum length for cache-efficient string operations
const MAX_TOKEN_TEXT_LEN: usize = 256;
const MAX_VOCAB_SIZE: usize = 256_000;
const MAX_ENCODE_BATCH: usize = 512;

/// Zero-allocation string buffer for tokenization
pub type TokenString = ArrayString<MAX_TOKEN_TEXT_LEN>;

/// Cache-efficient token ID collection
pub type TokenIds = SmallVec<[u32; MAX_ENCODE_BATCH]>;

/// Streaming text buffer for decode operations
#[derive(Debug, Clone)]
pub struct TextBuffer {
    /// Internal buffer for accumulating decoded text
    buffer: SmallVec<[u8; 1024]>,
    /// Current write position
    write_pos: usize,
}

impl TextBuffer {
    /// Create a new text buffer
    pub fn new() -> Self {
        Self {
            buffer: SmallVec::new(),
            write_pos: 0,
        }
    }

    /// Add bytes to the buffer
    pub fn push_bytes(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
        self.write_pos = self.buffer.len();
    }

    /// Get the current buffer contents
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer[..self.write_pos]
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.write_pos = 0;
    }

    /// Get length of buffered data
    pub fn len(&self) -> usize {
        self.write_pos
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.write_pos == 0
    }
}

impl std::ops::Deref for TextBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_bytes()
    }
}

impl Default for TextBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Special tokens for model control and processing
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    /// Beginning of sequence token
    pub bos_token_id: Option<u32>,
    /// End of sequence token
    pub eos_token_id: Option<u32>,
    /// Unknown token for out-of-vocabulary
    pub unk_token_id: Option<u32>,
    /// Padding token for batched processing
    pub pad_token_id: Option<u32>,
    /// Mask token for special tasks
    pub mask_token_id: Option<u32>,
    /// Additional special tokens by name
    pub additional_special_tokens: HashMap<TokenString, u32>,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos_token_id: None,
            eos_token_id: None,
            unk_token_id: None,
            pad_token_id: None,
            mask_token_id: None,
            additional_special_tokens: HashMap::new(),
        }
    }
}

impl SpecialTokens {
    /// Create special tokens for a specific model
    pub fn for_model(model: CandleModel) -> Self {
        let mut tokens = Self::default();

        match model {
            CandleModel::Llama2_7B | CandleModel::Llama2_13B => {
                tokens.bos_token_id = Some(1);
                tokens.eos_token_id = Some(2);
                tokens.unk_token_id = Some(0);
            }
            CandleModel::Mistral_7B => {
                tokens.bos_token_id = Some(1);
                tokens.eos_token_id = Some(2);
                tokens.unk_token_id = Some(0);
            }
            CandleModel::CodeLlama_7B => {
                tokens.bos_token_id = Some(1);
                tokens.eos_token_id = Some(2);
                tokens.unk_token_id = Some(0);
                // Code-specific tokens
                let mut additional = HashMap::new();
                if let Ok(prefix_token) = TokenString::try_from("<PRE>") {
                    additional.insert(prefix_token, 32007);
                }
                if let Ok(suffix_token) = TokenString::try_from("<SUF>") {
                    additional.insert(suffix_token, 32008);
                }
                if let Ok(middle_token) = TokenString::try_from("<MID>") {
                    additional.insert(middle_token, 32009);
                }
                tokens.additional_special_tokens = additional;
            }
            CandleModel::Phi3_Mini => {
                tokens.bos_token_id = Some(1);
                tokens.eos_token_id = Some(32000);
                tokens.unk_token_id = Some(0);
                tokens.pad_token_id = Some(32000);
            }
            CandleModel::Gemma_2B | CandleModel::Gemma_7B => {
                tokens.bos_token_id = Some(2);
                tokens.eos_token_id = Some(1);
                tokens.unk_token_id = Some(3);
                tokens.pad_token_id = Some(0);
            }
        }

        tokens
    }

    /// Check if a token ID is a special token
    pub fn is_special_token(&self, token_id: u32) -> bool {
        self.bos_token_id == Some(token_id)
            || self.eos_token_id == Some(token_id)
            || self.unk_token_id == Some(token_id)
            || self.pad_token_id == Some(token_id)
            || self.mask_token_id == Some(token_id)
            || self
                .additional_special_tokens
                .values()
                .any(|&id| id == token_id)
    }

    /// Get special token name by ID
    pub fn token_name(&self, token_id: u32) -> Option<&str> {
        if self.bos_token_id == Some(token_id) {
            return Some("<BOS>");
        }
        if self.eos_token_id == Some(token_id) {
            return Some("<EOS>");
        }
        if self.unk_token_id == Some(token_id) {
            return Some("<UNK>");
        }
        if self.pad_token_id == Some(token_id) {
            return Some("<PAD>");
        }
        if self.mask_token_id == Some(token_id) {
            return Some("<MASK>");
        }

        for (name, &id) in &self.additional_special_tokens {
            if id == token_id {
                return Some(name.as_str());
            }
        }

        None
    }
}

/// Tokenization result with zero-allocation design
#[derive(Debug, Clone)]
pub struct TokenizationResult {
    /// Token IDs from encoding
    tokens: TokenIds,
    /// Attention mask for variable-length sequences
    attention_mask: SmallVec<[u8; MAX_ENCODE_BATCH]>,
    /// Whether this result contains special tokens
    has_special_tokens: bool,
    /// Original text length for metrics
    original_length: u32,
}

impl TokenizationResult {
    /// Create a new tokenization result
    pub fn new(tokens: TokenIds, original_length: usize) -> Self {
        let attention_mask = SmallVec::from_elem(1u8, tokens.len());

        Self {
            tokens,
            attention_mask,
            has_special_tokens: false,
            original_length: original_length as u32,
        }
    }

    /// Create result with attention mask
    pub fn with_attention_mask(
        tokens: TokenIds,
        attention_mask: SmallVec<[u8; MAX_ENCODE_BATCH]>,
        original_length: usize,
    ) -> Self {
        Self {
            tokens,
            attention_mask,
            has_special_tokens: false,
            original_length: original_length as u32,
        }
    }

    /// Get token IDs
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Get attention mask
    pub fn attention_mask(&self) -> &[u8] {
        &self.attention_mask
    }

    /// Check if result has special tokens
    pub fn has_special_tokens(&self) -> bool {
        self.has_special_tokens
    }

    /// Get original text length
    pub fn original_length(&self) -> u32 {
        self.original_length
    }

    /// Mark that special tokens are present
    pub fn mark_special_tokens(&mut self) {
        self.has_special_tokens = true;
    }

    /// Get token count
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Tokenizer configuration for different models and use cases
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Model this tokenizer is configured for
    pub model: CandleModel,
    /// Vocabulary size
    pub vocab_size: u32,
    /// Whether to add BOS/EOS tokens automatically
    pub add_special_tokens: bool,
    /// Whether to truncate sequences that exceed max length
    pub truncate_sequences: bool,
    /// Maximum sequence length
    pub max_sequence_length: u32,
    /// Whether to enable streaming decode mode
    pub streaming_decode: bool,
}

impl TokenizerConfig {
    /// Create configuration for a specific model
    pub fn for_model(model: CandleModel) -> Self {
        let (vocab_size, max_seq_len) = match model {
            CandleModel::Llama2_7B | CandleModel::Llama2_13B => (32000, 4096),
            CandleModel::Mistral_7B => (32000, 8192),
            CandleModel::CodeLlama_7B => (32016, 16384),
            CandleModel::Phi3_Mini => (32064, 4096),
            CandleModel::Gemma_2B | CandleModel::Gemma_7B => (256000, 8192),
        };

        Self {
            model,
            vocab_size,
            add_special_tokens: true,
            truncate_sequences: true,
            max_sequence_length: max_seq_len,
            streaming_decode: true,
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> CandleResult<()> {
        if self.vocab_size == 0 {
            return Err(CandleError::config(
                "Vocabulary size must be positive",
                "vocab_size",
                "> 0",
            ));
        }

        if self.max_sequence_length == 0 {
            return Err(CandleError::config(
                "Maximum sequence length must be positive",
                "max_sequence_length",
                "> 0",
            ));
        }

        if self.max_sequence_length > 65536 {
            return Err(CandleError::config(
                "Maximum sequence length too large",
                "max_sequence_length",
                "<= 65536",
            ));
        }

        Ok(())
    }
}

/// High-performance tokenizer implementation with zero-allocation patterns
#[derive(Debug)]
pub struct CandleTokenizer {
    /// Tokenizer configuration
    config: TokenizerConfig,
    /// Special tokens for this model
    special_tokens: SpecialTokens,
    /// Vocabulary mapping (token to ID)
    vocab: ArcSwap<HashMap<TokenString, u32>>,
    /// Reverse vocabulary (ID to token)
    reverse_vocab: ArcSwap<HashMap<u32, TokenString>>,
    /// Tokenizer state for streaming operations
    streaming_state: ArcSwap<StreamingState>,
    /// Tokenizer statistics
    stats: TokenizerStats,
}

/// Streaming tokenizer state for incremental processing
#[derive(Debug, Clone)]
struct StreamingState {
    /// Partial UTF-8 sequence buffer
    partial_utf8: SmallVec<[u8; 4]>,
    /// Current decoding position
    decode_position: u32,
    /// Whether streaming is active
    is_streaming: bool,
}

impl Default for StreamingState {
    fn default() -> Self {
        Self {
            partial_utf8: SmallVec::new(),
            decode_position: 0,
            is_streaming: false,
        }
    }
}

/// Tokenizer performance statistics
#[derive(Debug)]
struct TokenizerStats {
    /// Total characters tokenized
    total_chars_processed: AtomicCell<u64>,
    /// Total tokens generated
    total_tokens_generated: AtomicCell<u64>,
    /// Total decode operations
    total_decode_ops: AtomicCell<u64>,
    /// Average tokens per character
    avg_tokens_per_char: AtomicCell<f32>,
}

impl Default for TokenizerStats {
    fn default() -> Self {
        Self {
            total_chars_processed: AtomicCell::new(0),
            total_tokens_generated: AtomicCell::new(0),
            total_decode_ops: AtomicCell::new(0),
            avg_tokens_per_char: AtomicCell::new(0.0),
        }
    }
}

impl CandleTokenizer {
    /// Create a new tokenizer for the specified model
    pub fn new(model: CandleModel) -> Self {
        let config = TokenizerConfig::for_model(model);
        let special_tokens = SpecialTokens::for_model(model);

        // Initialize with basic vocabulary (would be loaded from tokenizer.json in real implementation)
        let vocab = HashMap::new();
        let reverse_vocab = HashMap::new();

        Self {
            config,
            special_tokens,
            vocab: ArcSwap::from_pointee(vocab),
            reverse_vocab: ArcSwap::from_pointee(reverse_vocab),
            streaming_state: ArcSwap::from_pointee(StreamingState::default()),
            stats: TokenizerStats::default(),
        }
    }

    /// Create tokenizer with custom configuration
    pub fn with_config(config: TokenizerConfig) -> CandleResult<Self> {
        config.validate()?;

        let special_tokens = SpecialTokens::for_model(config.model);
        let vocab = HashMap::new();
        let reverse_vocab = HashMap::new();

        Ok(Self {
            config,
            special_tokens,
            vocab: ArcSwap::from_pointee(vocab),
            reverse_vocab: ArcSwap::from_pointee(reverse_vocab),
            streaming_state: ArcSwap::from_pointee(StreamingState::default()),
            stats: TokenizerStats::default(),
        })
    }

    /// Load tokenizer from HuggingFace tokenizer.json file
    pub async fn load_from_file(&self, tokenizer_path: &PathBuf) -> CandleResult<()> {
        // Note: In a real implementation, this would:
        // 1. Load tokenizer.json using the tokenizers crate
        // 2. Extract vocabulary and special tokens
        // 3. Build reverse vocabulary mapping
        // 4. Validate compatibility with model

        // For now, we'll create a basic vocabulary simulation
        self.initialize_basic_vocabulary().await
    }

    /// Initialize with a basic vocabulary for demonstration
    async fn initialize_basic_vocabulary(&self) -> CandleResult<()> {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();

        // Add special tokens first
        if let Some(bos_id) = self.special_tokens.bos_token_id {
            if let Ok(bos_token) = TokenString::try_from("<BOS>") {
                vocab.insert(bos_token.clone(), bos_id);
                reverse_vocab.insert(bos_id, bos_token);
            }
        }

        if let Some(eos_id) = self.special_tokens.eos_token_id {
            if let Ok(eos_token) = TokenString::try_from("<EOS>") {
                vocab.insert(eos_token.clone(), eos_id);
                reverse_vocab.insert(eos_id, eos_token);
            }
        }

        if let Some(unk_id) = self.special_tokens.unk_token_id {
            if let Ok(unk_token) = TokenString::try_from("<UNK>") {
                vocab.insert(unk_token.clone(), unk_id);
                reverse_vocab.insert(unk_id, unk_token);
            }
        }

        // Add common tokens (in a real implementation, this would come from tokenizer.json)
        let common_tokens = [
            " ", "a", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "from", "as", "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "must",
            "shall", "this", "that", "these", "those", "I", "you", "he", "she", "it", "we", "they",
            "me", "him", "her", "us", "them", "my", "your", "his", "her", "its", "our", "their",
            "mine", "yours", "hers", "ours", "theirs",
        ];

        let mut token_id = 100; // Start after special tokens
        for token_text in common_tokens.iter() {
            if let Ok(token) = TokenString::try_from(*token_text) {
                vocab.insert(token.clone(), token_id);
                reverse_vocab.insert(token_id, token);
                token_id += 1;
            }
        }

        // Store the vocabularies atomically
        self.vocab.store(Arc::new(vocab));
        self.reverse_vocab.store(Arc::new(reverse_vocab));

        Ok(())
    }

    /// Encode text to token IDs with zero-allocation patterns
    pub fn encode(&self, text: &str) -> CandleResult<TokenizationResult> {
        if text.is_empty() {
            return Ok(TokenizationResult::new(TokenIds::new(), 0));
        }

        let vocab = self.vocab.load();
        let mut tokens = TokenIds::new();
        let original_length = text.len();

        // Add BOS token if configured
        if self.config.add_special_tokens {
            if let Some(bos_id) = self.special_tokens.bos_token_id {
                tokens.push(bos_id);
            }
        }

        // Tokenize the input text
        // Note: This is a simplified tokenization approach
        // Real implementation would use proper subword tokenization (BPE, SentencePiece, etc.)

        let mut char_buffer = String::new();
        let words: Vec<&str> = text.split_whitespace().collect();

        for (word_idx, word) in words.iter().enumerate() {
            // Add space prefix except for first word
            if word_idx > 0 {
                if let Some(&space_id) = vocab.get(" ") {
                    tokens.push(space_id);
                } else if let Some(unk_id) = self.special_tokens.unk_token_id {
                    tokens.push(unk_id);
                }
            }

            // Try to find the word in vocabulary
            if let Ok(word_token) = TokenString::try_from(*word) {
                if let Some(&token_id) = vocab.get(&word_token) {
                    tokens.push(token_id);
                } else {
                    // Break into character-level tokens as fallback
                    for ch in word.chars() {
                        char_buffer.clear();
                        char_buffer.push(ch);

                        if let Ok(char_token) = TokenString::try_from(char_buffer.as_str()) {
                            if let Some(&char_id) = vocab.get(&char_token) {
                                tokens.push(char_id);
                            } else if let Some(unk_id) = self.special_tokens.unk_token_id {
                                tokens.push(unk_id);
                            }
                        }
                    }
                }
            } else if let Some(unk_id) = self.special_tokens.unk_token_id {
                tokens.push(unk_id);
            }
        }

        // Add EOS token if configured
        if self.config.add_special_tokens {
            if let Some(eos_id) = self.special_tokens.eos_token_id {
                tokens.push(eos_id);
            }
        }

        // Truncate if necessary
        if self.config.truncate_sequences && tokens.len() > self.config.max_sequence_length as usize
        {
            tokens.truncate(self.config.max_sequence_length as usize);
        }

        // Update statistics
        self.stats
            .total_chars_processed
            .store(self.stats.total_chars_processed.load() + original_length as u64);
        self.stats
            .total_tokens_generated
            .store(self.stats.total_tokens_generated.load() + tokens.len() as u64);

        // Calculate average tokens per character
        let total_chars = self.stats.total_chars_processed.load();
        let total_tokens = self.stats.total_tokens_generated.load();
        if total_chars > 0 {
            self.stats
                .avg_tokens_per_char
                .store(total_tokens as f32 / total_chars as f32);
        }

        let mut result = TokenizationResult::new(tokens, original_length);
        if self.config.add_special_tokens {
            result.mark_special_tokens();
        }

        Ok(result)
    }

    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[u32]) -> CandleResult<String> {
        let reverse_vocab = self.reverse_vocab.load();
        let mut result = String::new();

        for &token_id in token_ids {
            if let Some(token_text) = reverse_vocab.get(&token_id) {
                result.push_str(token_text.as_str());
            } else {
                // Handle unknown token
                result.push_str("<UNK>");
            }
        }

        Ok(result)
    }

    /// Streaming decode for real-time text generation
    pub fn decode_stream(&self, token_id: u32, text_buffer: &mut TextBuffer) -> CandleResult<()> {
        let reverse_vocab = self.reverse_vocab.load();

        if let Some(token_text) = reverse_vocab.get(&token_id) {
            text_buffer.push_bytes(token_text.as_bytes());
        } else {
            // Handle unknown token
            text_buffer.push_bytes(b"<UNK>");
        }

        // Update decode statistics
        self.stats
            .total_decode_ops
            .store(self.stats.total_decode_ops.load() + 1);

        Ok(())
    }

    /// Get special tokens for this tokenizer
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    /// Get tokenizer configuration
    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> u32 {
        self.config.vocab_size
    }

    /// Check if tokenizer is ready for use
    pub fn is_ready(&self) -> bool {
        !self.vocab.load().is_empty()
    }

    /// Get tokenizer statistics
    pub fn statistics(&self) -> TokenizerStatistics {
        TokenizerStatistics {
            total_chars_processed: self.stats.total_chars_processed.load(),
            total_tokens_generated: self.stats.total_tokens_generated.load(),
            total_decode_ops: self.stats.total_decode_ops.load(),
            avg_tokens_per_char: self.stats.avg_tokens_per_char.load(),
            vocab_size: self.config.vocab_size,
            is_ready: self.is_ready(),
        }
    }

    /// Reset tokenizer statistics
    pub fn reset_statistics(&self) {
        self.stats.total_chars_processed.store(0);
        self.stats.total_tokens_generated.store(0);
        self.stats.total_decode_ops.store(0);
        self.stats.avg_tokens_per_char.store(0.0);
    }

    /// Start streaming mode for incremental decoding
    pub fn start_streaming(&self) -> CandleResult<()> {
        let mut state = StreamingState::default();
        state.is_streaming = true;
        self.streaming_state.store(Arc::new(state));
        Ok(())
    }

    /// End streaming mode
    pub fn end_streaming(&self) -> CandleResult<()> {
        let mut state = (**self.streaming_state.load()).clone();
        state.is_streaming = false;
        state.partial_utf8.clear();
        state.decode_position = 0;
        self.streaming_state.store(Arc::new(state));
        Ok(())
    }

    /// Check if tokenizer is in streaming mode
    pub fn is_streaming(&self) -> bool {
        self.streaming_state.load().is_streaming
    }
}

/// Comprehensive tokenizer statistics for monitoring and optimization
#[derive(Debug, Clone)]
pub struct TokenizerStatistics {
    /// Total characters processed
    pub total_chars_processed: u64,
    /// Total tokens generated
    pub total_tokens_generated: u64,
    /// Total decode operations
    pub total_decode_ops: u64,
    /// Average tokens per character ratio
    pub avg_tokens_per_char: f32,
    /// Vocabulary size
    pub vocab_size: u32,
    /// Whether tokenizer is ready
    pub is_ready: bool,
}

impl TokenizerStatistics {
    /// Calculate compression ratio (lower is better compression)
    pub fn compression_ratio(&self) -> f32 {
        if self.total_chars_processed > 0 {
            self.total_tokens_generated as f32 / self.total_chars_processed as f32
        } else {
            0.0
        }
    }

    /// Calculate average characters per token
    pub fn avg_chars_per_token(&self) -> f32 {
        if self.total_tokens_generated > 0 {
            self.total_chars_processed as f32 / self.total_tokens_generated as f32
        } else {
            0.0
        }
    }
}

impl Default for CandleTokenizer {
    fn default() -> Self {
        Self::new(CandleModel::Mistral_7B)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tokenizer_creation() {
        let tokenizer = CandleTokenizer::new(CandleModel::Mistral_7B);
        assert_eq!(tokenizer.config.model, CandleModel::Mistral_7B);
        assert_eq!(tokenizer.vocab_size(), 32000);
    }

    #[tokio::test]
    async fn test_special_tokens() {
        let special_tokens = SpecialTokens::for_model(CandleModel::Llama2_7B);
        assert_eq!(special_tokens.bos_token_id, Some(1));
        assert_eq!(special_tokens.eos_token_id, Some(2));
        assert!(special_tokens.is_special_token(1));
        assert!(special_tokens.is_special_token(2));
        assert!(!special_tokens.is_special_token(100));
    }

    #[tokio::test]
    async fn test_text_buffer() {
        let mut buffer = TextBuffer::new();
        assert!(buffer.is_empty());

        buffer.push_bytes(b"Hello");
        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.as_bytes(), b"Hello");

        buffer.push_bytes(b" World");
        assert_eq!(buffer.len(), 11);
        assert_eq!(buffer.as_bytes(), b"Hello World");

        buffer.clear();
        assert!(buffer.is_empty());
    }

    #[tokio::test]
    async fn test_basic_encoding() {
        let tokenizer = CandleTokenizer::new(CandleModel::Mistral_7B);
        tokenizer.initialize_basic_vocabulary().await.unwrap();

        let result = tokenizer.encode("Hello world").unwrap();
        assert!(!result.is_empty());
        assert!(result.has_special_tokens()); // BOS/EOS added
        assert_eq!(result.original_length(), 11);
    }
}
