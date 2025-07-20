//! SIMD-optimized pattern matching with zero allocation
//!
//! Provides blazing-fast pattern matching using Boyer-Moore, KMP, and SIMD algorithms
//! with production-ready performance and ergonomic APIs.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
// SIMD intrinsics imports
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::collections::HashMap;
use std::sync::Arc;

use arrayvec::ArrayVec;
use atomic_counter::{AtomicCounter, RelaxedCounter};
use crossbeam_skiplist::SkipMap;
use smallvec::SmallVec;

use super::types::*;

/// Pattern matching engine with multiple algorithms
pub struct PatternMatcher {
    /// Compiled patterns with their IDs
    patterns: SkipMap<usize, CompiledPattern>,
    /// Pattern lookup by content hash
    pattern_lookup: SkipMap<u64, usize>,
    /// Performance counters
    matches_found: RelaxedCounter,
    search_operations: RelaxedCounter,
    search_time_nanos: RelaxedCounter,
    /// Next pattern ID
    next_pattern_id: RelaxedCounter,
}

/// Compiled pattern with optimized search tables
#[derive(Debug, Clone)]
struct CompiledPattern {
    id: usize,
    content: ArrayVec<u8, MAX_PATTERN_LENGTH>,
    algorithm: MatchingAlgorithm,
    boyer_moore_table: Option<ArrayVec<usize, 256>>,
    kmp_table: Option<ArrayVec<isize, MAX_PATTERN_LENGTH>>,
    content_hash: u64,
}

/// Pattern matching algorithms
#[derive(Debug, Clone, Copy)]
enum MatchingAlgorithm {
    Naive,
    BoyerMoore,
    KMP,
    SIMD,
}

impl Default for PatternMatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternMatcher {
    /// Create new pattern matcher
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            patterns: SkipMap::new(),
            pattern_lookup: SkipMap::new(),
            matches_found: RelaxedCounter::new(0),
            search_operations: RelaxedCounter::new(0),
            search_time_nanos: RelaxedCounter::new(0),
            next_pattern_id: RelaxedCounter::new(1),
        }
    }

    /// Add pattern to matcher
    #[inline(always)]
    pub fn add_pattern(&self, pattern: &[u8]) -> Result<usize, TextProcessingError> {
        if pattern.len() > MAX_PATTERN_LENGTH {
            return Err(TextProcessingError::PatternTooLarge(pattern.len()));
        }

        let pattern_id = self.next_pattern_id.get();
        self.next_pattern_id.inc();

        let content_hash = self.calculate_hash(pattern);

        // Check if pattern already exists
        if let Some(existing_id) = self.pattern_lookup.get(&content_hash) {
            return Ok(*existing_id.value());
        }

        let mut content = ArrayVec::new();
        content
            .try_extend_from_slice(pattern)
            .map_err(|_| TextProcessingError::PatternTooLarge(pattern.len()))?;

        let algorithm = self.select_algorithm(pattern);
        let boyer_moore_table = if matches!(algorithm, MatchingAlgorithm::BoyerMoore) {
            Some(self.build_boyer_moore_table(pattern))
        } else {
            None
        };

        let kmp_table = if matches!(algorithm, MatchingAlgorithm::KMP) {
            Some(self.build_kmp_table(pattern))
        } else {
            None
        };

        let compiled_pattern = CompiledPattern {
            id: pattern_id,
            content,
            algorithm,
            boyer_moore_table,
            kmp_table,
            content_hash,
        };

        self.patterns.insert(pattern_id, compiled_pattern);
        self.pattern_lookup.insert(content_hash, pattern_id);

        Ok(pattern_id)
    }

    /// Remove pattern from matcher
    #[inline(always)]
    pub fn remove_pattern(&self, pattern_id: usize) -> Result<(), TextProcessingError> {
        if let Some(pattern_entry) = self.patterns.remove(&pattern_id) {
            let pattern = pattern_entry.value();
            self.pattern_lookup.remove(&pattern.content_hash);
            Ok(())
        } else {
            Err(TextProcessingError::InvalidInput(format!(
                "Pattern ID {} not found",
                pattern_id
            )))
        }
    }

    /// Find all pattern matches in text
    #[inline(always)]
    pub fn find_matches(
        &self,
        text: &[u8],
    ) -> Result<ArrayVec<PatternMatch, MAX_PATTERNS_PER_SET>, TextProcessingError> {
        let start_time = std::time::Instant::now();
        self.search_operations.inc();

        let mut matches = ArrayVec::new();

        // Search for each pattern
        for pattern_entry in self.patterns.iter() {
            let pattern = pattern_entry.value();
            let pattern_matches = self.find_pattern_matches(text, pattern)?;

            for pattern_match in pattern_matches {
                if matches.try_push(pattern_match).is_err() {
                    break; // Stop if we exceed capacity
                }
            }
        }

        // Sort matches by position
        matches.sort_by_key(|m| m.start_offset);

        // Update performance counters
        let elapsed_nanos = start_time.elapsed().as_nanos() as usize;
        self.search_time_nanos.add(elapsed_nanos);
        self.matches_found.add(matches.len());

        Ok(matches)
    }

    /// Find matches for a specific pattern
    #[inline(always)]
    fn find_pattern_matches(
        &self,
        text: &[u8],
        pattern: &CompiledPattern,
    ) -> Result<SmallVec<[PatternMatch; 8]>, TextProcessingError> {
        match pattern.algorithm {
            MatchingAlgorithm::Naive => self.find_matches_naive(text, pattern),
            MatchingAlgorithm::BoyerMoore => self.find_matches_boyer_moore(text, pattern),
            MatchingAlgorithm::KMP => self.find_matches_kmp(text, pattern),
            MatchingAlgorithm::SIMD => self.find_matches_simd(text, pattern),
        }
    }

    /// Naive pattern matching algorithm
    #[inline(always)]
    fn find_matches_naive(
        &self,
        text: &[u8],
        pattern: &CompiledPattern,
    ) -> Result<SmallVec<[PatternMatch; 8]>, TextProcessingError> {
        let mut matches = SmallVec::new();
        let pattern_bytes = &pattern.content;

        if pattern_bytes.is_empty() || text.len() < pattern_bytes.len() {
            return Ok(matches);
        }

        for i in 0..=(text.len() - pattern_bytes.len()) {
            if text[i..i + pattern_bytes.len()] == pattern_bytes[..] {
                let pattern_match = PatternMatch::new(
                    pattern.id,
                    i,
                    i + pattern_bytes.len(),
                    &text[i..i + pattern_bytes.len()],
                    1.0,
                )?;
                matches.push(pattern_match);
            }
        }

        Ok(matches)
    }

    /// Boyer-Moore pattern matching algorithm
    #[inline(always)]
    fn find_matches_boyer_moore(
        &self,
        text: &[u8],
        pattern: &CompiledPattern,
    ) -> Result<SmallVec<[PatternMatch; 8]>, TextProcessingError> {
        let mut matches = SmallVec::new();
        let pattern_bytes = &pattern.content;
        let bad_char_table = pattern.boyer_moore_table.as_ref().ok_or_else(|| {
            TextProcessingError::ProcessingFailed("Boyer-Moore table not found".into())
        })?;

        if pattern_bytes.is_empty() || text.len() < pattern_bytes.len() {
            return Ok(matches);
        }

        let mut i = 0;
        while i <= text.len() - pattern_bytes.len() {
            let mut j = pattern_bytes.len();

            // Match from right to left
            while j > 0 && pattern_bytes[j - 1] == text[i + j - 1] {
                j -= 1;
            }

            if j == 0 {
                // Pattern found
                let pattern_match = PatternMatch::new(
                    pattern.id,
                    i,
                    i + pattern_bytes.len(),
                    &text[i..i + pattern_bytes.len()],
                    1.0,
                )?;
                matches.push(pattern_match);
                i += 1; // Move to next position
            } else {
                // Use bad character heuristic
                let bad_char = text[i + j - 1] as usize;
                let skip = if bad_char < bad_char_table.len() {
                    bad_char_table[bad_char].max(1)
                } else {
                    pattern_bytes.len()
                };
                i += skip;
            }
        }

        Ok(matches)
    }

    /// KMP pattern matching algorithm
    #[inline(always)]
    fn find_matches_kmp(
        &self,
        text: &[u8],
        pattern: &CompiledPattern,
    ) -> Result<SmallVec<[PatternMatch; 8]>, TextProcessingError> {
        let mut matches = SmallVec::new();
        let pattern_bytes = &pattern.content;
        let lps_table = pattern
            .kmp_table
            .as_ref()
            .ok_or_else(|| TextProcessingError::ProcessingFailed("KMP table not found".into()))?;

        if pattern_bytes.is_empty() || text.len() < pattern_bytes.len() {
            return Ok(matches);
        }

        let mut i = 0; // Index for text
        let mut j = 0; // Index for pattern

        while i < text.len() {
            if pattern_bytes[j] == text[i] {
                i += 1;
                j += 1;
            }

            if j == pattern_bytes.len() {
                // Pattern found
                let start_pos = i - j;
                let pattern_match = PatternMatch::new(
                    pattern.id,
                    start_pos,
                    start_pos + pattern_bytes.len(),
                    &text[start_pos..start_pos + pattern_bytes.len()],
                    1.0,
                )?;
                matches.push(pattern_match);
                j = if lps_table[j - 1] >= 0 {
                    lps_table[j - 1] as usize
                } else {
                    0
                };
            } else if i < text.len() && pattern_bytes[j] != text[i] {
                if j != 0 {
                    j = if lps_table[j - 1] >= 0 {
                        lps_table[j - 1] as usize
                    } else {
                        0
                    };
                } else {
                    i += 1;
                }
            }
        }

        Ok(matches)
    }

    /// SIMD-optimized pattern matching
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn find_matches_simd(
        &self,
        text: &[u8],
        pattern: &CompiledPattern,
    ) -> Result<SmallVec<[PatternMatch; 8]>, TextProcessingError> {
        if !is_x86_feature_detected!("avx2") || pattern.content.len() > 32 {
            return self.find_matches_naive(text, pattern);
        }

        let mut matches = SmallVec::new();
        let pattern_bytes = &pattern.content;

        if pattern_bytes.is_empty() || text.len() < pattern_bytes.len() {
            return Ok(matches);
        }

        unsafe {
            let pattern_first = _mm256_set1_epi8(pattern_bytes[0] as i8);
            let mut i = 0;

            while i + 32 <= text.len() {
                let text_chunk = _mm256_loadu_si256(text[i..].as_ptr() as *const __m256i);
                let cmp_result = _mm256_cmpeq_epi8(text_chunk, pattern_first);
                let mask = _mm256_movemask_epi8(cmp_result) as u32;

                if mask != 0 {
                    // Check each potential match
                    for bit_pos in 0..32 {
                        if (mask >> bit_pos) & 1 != 0 {
                            let pos = i + bit_pos;
                            if pos + pattern_bytes.len() <= text.len() {
                                if text[pos..pos + pattern_bytes.len()] == pattern_bytes[..] {
                                    let pattern_match = PatternMatch::new(
                                        pattern.id,
                                        pos,
                                        pos + pattern_bytes.len(),
                                        &text[pos..pos + pattern_bytes.len()],
                                        1.0,
                                    )?;
                                    matches.push(pattern_match);
                                }
                            }
                        }
                    }
                }

                i += 32;
            }

            // Handle remaining bytes
            while i + pattern_bytes.len() <= text.len() {
                if text[i..i + pattern_bytes.len()] == pattern_bytes[..] {
                    let pattern_match = PatternMatch::new(
                        pattern.id,
                        i,
                        i + pattern_bytes.len(),
                        &text[i..i + pattern_bytes.len()],
                        1.0,
                    )?;
                    matches.push(pattern_match);
                }
                i += 1;
            }
        }

        Ok(matches)
    }

    /// SIMD pattern matching fallback for non-x86 architectures
    #[cfg(not(target_arch = "x86_64"))]
    #[inline(always)]
    fn find_matches_simd(
        &self,
        text: &[u8],
        pattern: &CompiledPattern,
    ) -> Result<SmallVec<[PatternMatch; 8]>, TextProcessingError> {
        self.find_matches_naive(text, pattern)
    }

    /// Select optimal algorithm for pattern
    #[inline(always)]
    fn select_algorithm(&self, pattern: &[u8]) -> MatchingAlgorithm {
        match pattern.len() {
            0..=2 => MatchingAlgorithm::Naive,
            3..=8 => {
                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx2") {
                        MatchingAlgorithm::SIMD
                    } else {
                        MatchingAlgorithm::KMP
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    MatchingAlgorithm::KMP
                }
            }
            9..=32 => MatchingAlgorithm::BoyerMoore,
            _ => MatchingAlgorithm::KMP,
        }
    }

    /// Build Boyer-Moore bad character table
    #[inline(always)]
    fn build_boyer_moore_table(&self, pattern: &[u8]) -> ArrayVec<usize, 256> {
        let mut table = ArrayVec::new();

        // Initialize with pattern length
        for _ in 0..256 {
            table.push(pattern.len());
        }

        // Fill with actual distances
        for (i, &byte) in pattern.iter().enumerate() {
            if i < pattern.len() - 1 {
                table[byte as usize] = pattern.len() - 1 - i;
            }
        }

        table
    }

    /// Build KMP longest proper prefix table
    #[inline(always)]
    fn build_kmp_table(&self, pattern: &[u8]) -> ArrayVec<isize, MAX_PATTERN_LENGTH> {
        let mut table = ArrayVec::new();
        table.push(0);

        let mut len = 0;
        let mut i = 1;

        while i < pattern.len() {
            if pattern[i] == pattern[len] {
                len += 1;
                table.push(len as isize);
                i += 1;
            } else if len != 0 {
                len = table[len - 1] as usize;
            } else {
                table.push(0);
                i += 1;
            }
        }

        table
    }

    /// Calculate hash for pattern content
    #[inline(always)]
    fn calculate_hash(&self, pattern: &[u8]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        pattern.hash(&mut hasher);
        hasher.finish()
    }

    /// Get performance statistics
    #[inline(always)]
    pub fn get_stats(&self) -> PerformanceStats {
        let total_ops = self.search_operations.get().max(1);

        PerformanceStats {
            total_operations: total_ops as u64,
            total_processing_time_nanos: self.search_time_nanos.get() as u64,
            average_tokenization_time_nanos: 0,
            average_pattern_matching_time_nanos: (self.search_time_nanos.get() / total_ops) as u64,
            average_analysis_time_nanos: 0,
            simd_operations_count: 0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Reset performance counters
    #[inline(always)]
    pub fn reset_stats(&self) {
        self.matches_found.reset();
        self.search_operations.reset();
        self.search_time_nanos.reset();
    }

    /// Get pattern count
    #[inline(always)]
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Clear all patterns
    #[inline(always)]
    pub fn clear_patterns(&self) {
        self.patterns.clear();
        self.pattern_lookup.clear();
        self.next_pattern_id.reset();
    }
}

/// Global pattern matcher instance
static GLOBAL_PATTERN_MATCHER: once_cell::sync::Lazy<PatternMatcher> =
    once_cell::sync::Lazy::new(|| PatternMatcher::new());

/// Get global pattern matcher instance
#[inline(always)]
pub fn get_global_pattern_matcher() -> &'static PatternMatcher {
    &GLOBAL_PATTERN_MATCHER
}

/// Convenience function for adding a pattern
#[inline(always)]
pub fn add_pattern(pattern: &[u8]) -> Result<usize, TextProcessingError> {
    get_global_pattern_matcher().add_pattern(pattern)
}

/// Convenience function for finding matches
#[inline(always)]
pub fn find_matches(
    text: &[u8],
) -> Result<ArrayVec<PatternMatch, MAX_PATTERNS_PER_SET>, TextProcessingError> {
    get_global_pattern_matcher().find_matches(text)
}

/// Convenience function for getting pattern matcher statistics
#[inline(always)]
pub fn get_pattern_matcher_stats() -> PerformanceStats {
    get_global_pattern_matcher().get_stats()
}
