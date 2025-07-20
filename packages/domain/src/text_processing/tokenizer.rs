//! SIMD-optimized tokenizer with zero allocation
//!
//! Provides blazing-fast tokenization using AVX2/AVX-512 and NEON SIMD instructions
//! with production-ready performance and ergonomic APIs.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
// SIMD intrinsics imports
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::sync::atomic::{AtomicU64, Ordering};

use arrayvec::ArrayVec;
use atomic_counter::{AtomicCounter, RelaxedCounter};
use crossbeam_queue::ArrayQueue;
use smallvec::SmallVec;

use super::types::*;
use crate::memory_ops::{CPU_FEATURES, CpuArchitecture, CpuFeatures, SIMD_WIDTH};

/// SIMD-optimized tokenizer with zero allocation
pub struct SIMDTokenizer {
    // Pre-allocated token storage
    token_pool: ArrayQueue<Token>,

    // Performance counters
    tokens_processed: RelaxedCounter,
    simd_operations: RelaxedCounter,
    tokenization_time_nanos: RelaxedCounter,
}

impl Default for SIMDTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl SIMDTokenizer {
    /// Create new SIMD tokenizer with pre-allocated resources
    #[inline(always)]
    pub fn new() -> Self {
        let pool_size = 1024;
        let token_pool = ArrayQueue::new(pool_size);

        // Pre-fill pool with default tokens
        for _ in 0..pool_size {
            let token = Token {
                content: ArrayVec::new(),
                token_type: TokenType::Unknown,
                start_offset: 0,
                end_offset: 0,
            };
            let _ = token_pool.push(token);
        }

        Self {
            token_pool,
            tokens_processed: RelaxedCounter::new(0),
            simd_operations: RelaxedCounter::new(0),
            tokenization_time_nanos: RelaxedCounter::new(0),
        }
    }

    /// Tokenize text with SIMD optimization
    #[inline(always)]
    pub fn tokenize(
        &self,
        text: &str,
    ) -> Result<ArrayVec<Token, MAX_TOKENS_PER_BATCH>, TextProcessingError> {
        let start_time = std::time::Instant::now();

        let result = if text.len() < 32 {
            // Use scalar tokenization for small texts
            self.tokenize_scalar(text.as_bytes())
        } else {
            // Use SIMD tokenization for larger texts
            match CPU_FEATURES.architecture {
                #[cfg(target_arch = "x86_64")]
                CpuArchitecture::X86_64 if CPU_FEATURES.features.avx2 => {
                    self.tokenize_simd_avx2(text.as_bytes())
                }
                #[cfg(target_arch = "aarch64")]
                CpuArchitecture::AArch64 if CPU_FEATURES.features.neon => {
                    self.tokenize_simd_neon(text.as_bytes())
                }
                _ => self.tokenize_scalar(text.as_bytes()),
            }
        };

        // Update performance counters
        let elapsed_nanos = start_time.elapsed().as_nanos() as usize;
        self.tokenization_time_nanos.add(elapsed_nanos);

        if let Ok(ref tokens) = result {
            self.tokens_processed.add(tokens.len());
        }

        result
    }

    /// AVX2-optimized tokenization with real intrinsics
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub fn tokenize_simd_avx2(
        &self,
        text: &[u8],
    ) -> Result<ArrayVec<Token, MAX_TOKENS_PER_BATCH>, TextProcessingError> {
        if !is_x86_feature_detected!("avx2") {
            return self.tokenize_simd_fallback(text);
        }

        self.simd_operations.inc();

        unsafe { self.tokenize_simd_avx2_unsafe(text) }
    }

    /// Unsafe AVX2 tokenization implementation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn tokenize_simd_avx2_unsafe(
        &self,
        text: &[u8],
    ) -> Result<ArrayVec<Token, MAX_TOKENS_PER_BATCH>, TextProcessingError> {
        let mut tokens = ArrayVec::new();
        let mut pos = 0;

        // Process 32-byte chunks with AVX2
        while pos + 32 <= text.len() {
            let chunk = &text[pos..pos + 32];
            let data = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

            let boundaries = self.find_word_boundaries_simd_avx2(data);

            // Create tokens from boundaries
            let mut last_boundary = 0;
            for &boundary in &boundaries {
                if boundary > last_boundary {
                    let token_start = pos + last_boundary;
                    let token_end = pos + boundary;
                    let token_content = &text[token_start..token_end];

                    if !token_content.is_empty()
                        && !token_content.iter().all(|&b| b.is_ascii_whitespace())
                    {
                        let token_type = self.determine_token_type(token_content);
                        let token = Token::new(token_content, token_type, token_start, token_end)?;

                        if tokens.try_push(token).is_err() {
                            return Err(TextProcessingError::BufferOverflow(tokens.len()));
                        }
                    }

                    last_boundary = boundary;
                }
            }

            pos += 32;
        }

        // Process remaining bytes with scalar method
        if pos < text.len() {
            let remaining_tokens = self.tokenize_scalar(&text[pos..])?;
            for mut token in remaining_tokens {
                // Adjust offsets for remaining tokens
                token.start_offset += pos;
                token.end_offset += pos;

                if tokens.try_push(token).is_err() {
                    return Err(TextProcessingError::BufferOverflow(tokens.len()));
                }
            }
        }

        Ok(tokens)
    }

    /// SIMD character classification using real intrinsics
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn find_word_boundaries_simd_avx2(&self, data: __m256i) -> SmallVec<[usize; 8]> {
        let mut boundaries = SmallVec::new();

        // Create masks for different character types
        let space_mask = _mm256_set1_epi8(b' ' as i8);
        let tab_mask = _mm256_set1_epi8(b'\t' as i8);
        let newline_mask = _mm256_set1_epi8(b'\n' as i8);
        let punct_mask_1 = _mm256_set1_epi8(b'.' as i8);
        let punct_mask_2 = _mm256_set1_epi8(b',' as i8);

        // Compare with whitespace characters
        let space_cmp = _mm256_cmpeq_epi8(data, space_mask);
        let tab_cmp = _mm256_cmpeq_epi8(data, tab_mask);
        let newline_cmp = _mm256_cmpeq_epi8(data, newline_mask);
        let punct_cmp_1 = _mm256_cmpeq_epi8(data, punct_mask_1);
        let punct_cmp_2 = _mm256_cmpeq_epi8(data, punct_mask_2);

        // Combine all boundary conditions
        let boundary_mask = _mm256_or_si256(
            _mm256_or_si256(space_cmp, tab_cmp),
            _mm256_or_si256(newline_cmp, _mm256_or_si256(punct_cmp_1, punct_cmp_2)),
        );

        // Extract boundary positions
        let mask_bits = _mm256_movemask_epi8(boundary_mask) as u32;

        for i in 0..32 {
            if (mask_bits >> i) & 1 != 0 {
                boundaries.push(i);
            }
        }

        boundaries
    }

    /// NEON-optimized tokenization for ARM processors
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub fn tokenize_simd_neon(
        &self,
        text: &[u8],
    ) -> Result<ArrayVec<Token, MAX_TOKENS_PER_BATCH>, TextProcessingError> {
        self.simd_operations.inc();

        let mut tokens = ArrayVec::new();
        let mut pos = 0;

        // Process 16-byte chunks with NEON
        while pos + 16 <= text.len() {
            let chunk = &text[pos..pos + 16];
            let boundaries = self.find_word_boundaries_simd_neon(chunk);

            // Create tokens from boundaries
            let mut last_boundary = 0;
            for &boundary in &boundaries {
                if boundary > last_boundary {
                    let token_start = pos + last_boundary;
                    let token_end = pos + boundary;
                    let token_content = &text[token_start..token_end];

                    if !token_content.is_empty()
                        && !token_content.iter().all(|&b| b.is_ascii_whitespace())
                    {
                        let token_type = self.determine_token_type(token_content);
                        let token = Token::new(token_content, token_type, token_start, token_end)?;

                        if tokens.try_push(token).is_err() {
                            return Err(TextProcessingError::BufferOverflow(tokens.len()));
                        }
                    }

                    last_boundary = boundary;
                }
            }

            pos += 16;
        }

        // Process remaining bytes
        if pos < text.len() {
            let remaining_tokens = self.tokenize_scalar(&text[pos..])?;
            for token in remaining_tokens {
                let (start, end) = token.offsets();
                let new_token =
                    Token::new(token.as_bytes(), token.token_type(), start + pos, end + pos)?;

                if tokens.try_push(new_token).is_err() {
                    return Err(TextProcessingError::BufferOverflow(tokens.len()));
                }
            }
        }

        Ok(tokens)
    }

    /// NEON-optimized word boundary detection for ARM processors
    #[cfg(target_arch = "aarch64")]
    fn find_word_boundaries_simd_neon(&self, chunk: &[u8]) -> SmallVec<[usize; 8]> {
        let mut boundaries = SmallVec::new();

        unsafe {
            let data = vld1q_u8(chunk.as_ptr());

            // Create comparison vectors for whitespace
            let space_vec = vdupq_n_u8(b' ');
            let tab_vec = vdupq_n_u8(b'\t');
            let newline_vec = vdupq_n_u8(b'\n');

            // Compare with whitespace characters
            let space_cmp = vceqq_u8(data, space_vec);
            let tab_cmp = vceqq_u8(data, tab_vec);
            let newline_cmp = vceqq_u8(data, newline_vec);

            // Combine comparisons
            let boundary_mask = vorrq_u8(vorrq_u8(space_cmp, tab_cmp), newline_cmp);

            // Extract positions (simplified extraction)
            let mask_array: [u8; 16] = std::mem::transmute(boundary_mask);
            for (i, &mask_byte) in mask_array.iter().enumerate() {
                if mask_byte != 0 {
                    boundaries.push(i);
                }
            }
        }

        boundaries
    }

    /// Fallback SIMD tokenization using portable SIMD
    #[inline(always)]
    pub fn tokenize_simd_fallback(
        &self,
        text: &[u8],
    ) -> Result<ArrayVec<Token, MAX_TOKENS_PER_BATCH>, TextProcessingError> {
        // Use scalar tokenization as fallback
        self.tokenize_scalar(text)
    }

    /// Scalar tokenization for small texts and remainder processing
    #[inline(always)]
    pub fn tokenize_scalar(
        &self,
        text: &[u8],
    ) -> Result<ArrayVec<Token, MAX_TOKENS_PER_BATCH>, TextProcessingError> {
        let mut tokens = ArrayVec::new();
        let mut start = 0;
        let mut i = 0;

        while i < text.len() {
            if self.is_word_boundary(text[i]) {
                // End current token if we have one
                if i > start {
                    let token_content = &text[start..i];
                    if !token_content.iter().all(|&b| b.is_ascii_whitespace()) {
                        let token_type = self.determine_token_type(token_content);
                        let token = Token::new(token_content, token_type, start, i)?;

                        if tokens.try_push(token).is_err() {
                            return Err(TextProcessingError::BufferOverflow(tokens.len()));
                        }
                    }
                }

                // Skip whitespace
                while i < text.len() && self.is_word_boundary(text[i]) {
                    i += 1;
                }
                start = i;
            } else {
                i += 1;
            }
        }

        // Handle final token
        if start < text.len() {
            let token_content = &text[start..];
            if !token_content.iter().all(|&b| b.is_ascii_whitespace()) {
                let token_type = self.determine_token_type(token_content);
                let token = Token::new(token_content, token_type, start, text.len())?;

                if tokens.try_push(token).is_err() {
                    return Err(TextProcessingError::BufferOverflow(tokens.len()));
                }
            }
        }

        Ok(tokens)
    }

    /// Check if character is word boundary
    #[inline(always)]
    fn is_word_boundary(&self, byte: u8) -> bool {
        byte.is_ascii_whitespace() || byte.is_ascii_punctuation()
    }

    /// Classify single character
    #[inline(always)]
    fn classify_character(&self, byte: u8) -> TokenType {
        if byte.is_ascii_alphabetic() {
            TokenType::Word
        } else if byte.is_ascii_digit() {
            TokenType::Number
        } else if byte.is_ascii_punctuation() {
            TokenType::Punctuation
        } else if byte.is_ascii_whitespace() {
            TokenType::Whitespace
        } else if byte.is_ascii_graphic() {
            TokenType::Symbol
        } else {
            TokenType::Unknown
        }
    }

    /// Determine token type from content
    #[inline(always)]
    fn determine_token_type(&self, content: &[u8]) -> TokenType {
        if content.is_empty() {
            return TokenType::Unknown;
        }

        // Check if all characters are of the same type
        let first_type = self.classify_character(content[0]);

        if content
            .iter()
            .all(|&b| self.classify_character(b) == first_type)
        {
            first_type
        } else if content.iter().all(|&b| b.is_ascii_alphanumeric()) {
            // Mixed alphanumeric is considered a word
            TokenType::Word
        } else {
            // Mixed types default to symbol
            TokenType::Symbol
        }
    }

    /// Get performance statistics
    #[inline(always)]
    pub fn get_stats(&self) -> PerformanceStats {
        let total_ops = self.simd_operations.get() + 1; // Avoid division by zero

        PerformanceStats {
            total_operations: total_ops as u64,
            total_processing_time_nanos: self.tokenization_time_nanos.get() as u64,
            average_tokenization_time_nanos: (self.tokenization_time_nanos.get() / total_ops)
                as u64,
            average_pattern_matching_time_nanos: 0,
            average_analysis_time_nanos: 0,
            simd_operations_count: self.simd_operations.get() as u64,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Reset performance counters
    #[inline(always)]
    pub fn reset_stats(&self) {
        self.tokens_processed.reset();
        self.simd_operations.reset();
        self.tokenization_time_nanos.reset();
    }
}

/// Global tokenizer instance for efficient reuse
static GLOBAL_TOKENIZER: once_cell::sync::Lazy<SIMDTokenizer> =
    once_cell::sync::Lazy::new(|| SIMDTokenizer::new());

/// Get global tokenizer instance
#[inline(always)]
pub fn get_global_tokenizer() -> &'static SIMDTokenizer {
    &GLOBAL_TOKENIZER
}

/// Convenience function for tokenizing text
#[inline(always)]
pub fn tokenize(text: &str) -> Result<ArrayVec<Token, MAX_TOKENS_PER_BATCH>, TextProcessingError> {
    get_global_tokenizer().tokenize(text)
}

/// Convenience function for getting tokenizer statistics
#[inline(always)]
pub fn get_tokenizer_stats() -> PerformanceStats {
    get_global_tokenizer().get_stats()
}

/// Reset global tokenizer statistics
#[inline(always)]
pub fn reset_tokenizer_stats() {
    get_global_tokenizer().reset_stats();
}
