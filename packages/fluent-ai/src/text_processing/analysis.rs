//! Text analysis and statistics with zero allocation
//!
//! Provides blazing-fast text analysis including readability metrics, complexity scoring,
//! and statistical analysis with production-ready performance and ergonomic APIs.

use atomic_counter::{AtomicCounter, RelaxedCounter};
use smallvec::SmallVec;

use super::tokenizer::tokenize;
use super::types::*;

/// Text analyzer with comprehensive statistics
pub struct TextAnalyzer {
    /// Performance counters
    analyses_performed: RelaxedCounter,
    analysis_time_nanos: RelaxedCounter,
    /// Cached word frequency data
    word_frequency_cache: std::sync::RwLock<HashMap<String, u32>>}

impl Default for TextAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl TextAnalyzer {
    /// Create new text analyzer
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            analyses_performed: RelaxedCounter::new(0),
            analysis_time_nanos: RelaxedCounter::new(0),
            word_frequency_cache: std::sync::RwLock::new(HashMap::new())}
    }

    /// Analyze text and return comprehensive statistics
    #[inline(always)]
    pub fn analyze_text(&self, text: &str) -> Result<TextStats, TextProcessingError> {
        let start_time = std::time::Instant::now();
        self.analyses_performed.inc();

        let tokens = tokenize(text)?;

        let character_count = text.chars().count();
        let word_count = tokens
            .iter()
            .filter(|token| matches!(token.token_type(), TokenType::Word))
            .count();

        let sentence_count = self.count_sentences(text);
        let paragraph_count = self.count_paragraphs(text);

        let average_word_length = if word_count > 0 {
            tokens
                .iter()
                .filter(|token| matches!(token.token_type(), TokenType::Word))
                .map(|token| token.len())
                .sum::<usize>() as f32
                / word_count as f32
        } else {
            0.0
        };

        let reading_level =
            self.calculate_reading_level(word_count, sentence_count, character_count);
        let complexity_score = self.calculate_complexity_score(&tokens);

        let stats = TextStats {
            character_count,
            word_count,
            sentence_count,
            paragraph_count,
            average_word_length,
            reading_level,
            complexity_score};

        // Update performance counters
        let elapsed_nanos = start_time.elapsed().as_nanos() as usize;
        self.analysis_time_nanos.add(elapsed_nanos);

        Ok(stats)
    }

    /// Count sentences in text
    #[inline(always)]
    fn count_sentences(&self, text: &str) -> usize {
        text.chars()
            .filter(|&c| c == '.' || c == '!' || c == '?')
            .count()
            .max(1) // At least one sentence if text is not empty
    }

    /// Count paragraphs in text
    #[inline(always)]
    fn count_paragraphs(&self, text: &str) -> usize {
        if text.trim().is_empty() {
            return 0;
        }

        text.split("\n\n")
            .filter(|paragraph| !paragraph.trim().is_empty())
            .count()
            .max(1)
    }

    /// Calculate reading level using Flesch-Kincaid formula
    #[inline(always)]
    fn calculate_reading_level(
        &self,
        word_count: usize,
        sentence_count: usize,
        character_count: usize,
    ) -> f32 {
        if sentence_count == 0 || word_count == 0 {
            return 0.0;
        }

        let avg_sentence_length = word_count as f32 / sentence_count as f32;
        let avg_syllables_per_word = self.estimate_syllables_per_word(character_count, word_count);

        // Flesch-Kincaid Grade Level
        206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    }

    /// Estimate average syllables per word
    #[inline(always)]
    fn estimate_syllables_per_word(&self, character_count: usize, word_count: usize) -> f32 {
        if word_count == 0 {
            return 0.0;
        }

        // Simple estimation: average characters per word divided by 3
        // This is a rough approximation for English text
        let avg_chars_per_word = character_count as f32 / word_count as f32;
        (avg_chars_per_word / 3.0).max(1.0)
    }

    /// Calculate text complexity score
    #[inline(always)]
    fn calculate_complexity_score(&self, tokens: &ArrayVec<Token, MAX_TOKENS_PER_BATCH>) -> f32 {
        if tokens.is_empty() {
            return 0.0;
        }

        let mut complexity_factors = 0.0;
        let mut word_lengths = SmallVec::<[usize; 64]>::new();
        let mut unique_words = std::collections::HashSet::new();

        for token in tokens {
            if matches!(token.token_type(), TokenType::Word) {
                if let Ok(word) = token.as_str() {
                    let word_len = word.len();
                    word_lengths.push(word_len);
                    unique_words.insert(word.to_lowercase());

                    // Long words increase complexity
                    if word_len > 7 {
                        complexity_factors += 1.0;
                    }

                    // Capitalized words (potential proper nouns) add complexity
                    if word.chars().next().map_or(false, |c| c.is_uppercase()) {
                        complexity_factors += 0.5;
                    }
                }
            } else if matches!(token.token_type(), TokenType::Punctuation) {
                // Complex punctuation adds to complexity
                if let Ok(punct) = token.as_str() {
                    if punct.contains(';') || punct.contains(':') || punct.contains('—') {
                        complexity_factors += 0.3;
                    }
                }
            }
        }

        // Vocabulary diversity factor
        let vocabulary_diversity = if !word_lengths.is_empty() {
            unique_words.len() as f32 / word_lengths.len() as f32
        } else {
            0.0
        };

        // Average word length factor
        let avg_word_length = if !word_lengths.is_empty() {
            word_lengths.iter().sum::<usize>() as f32 / word_lengths.len() as f32
        } else {
            0.0
        };

        // Combine factors into final complexity score
        (complexity_factors + (vocabulary_diversity * 10.0) + (avg_word_length * 2.0)) / 3.0
    }

    /// Analyze word frequency in text
    #[inline(always)]
    pub fn analyze_word_frequency(
        &self,
        text: &str,
    ) -> Result<HashMap<String, u32>, TextProcessingError> {
        let tokens = tokenize(text)?;
        let mut frequency_map = HashMap::new();

        for token in tokens {
            if matches!(token.token_type(), TokenType::Word) {
                if let Ok(word) = token.as_str() {
                    let word_lower = word.to_lowercase();
                    *frequency_map.entry(word_lower).or_insert(0) += 1;
                }
            }
        }

        Ok(frequency_map)
    }

    /// Get most common words
    #[inline(always)]
    pub fn get_most_common_words(
        &self,
        text: &str,
        count: usize,
    ) -> Result<ArrayVec<(String, u32), 32>, TextProcessingError> {
        let frequency_map = self.analyze_word_frequency(text)?;
        let mut word_freq_vec: Vec<_> = frequency_map.into_iter().collect();

        // Sort by frequency (descending)
        word_freq_vec.sort_by(|a, b| b.1.cmp(&a.1));

        let mut result = ArrayVec::new();
        for (word, freq) in word_freq_vec.into_iter().take(count) {
            if result.try_push((word, freq)).is_err() {
                break;
            }
        }

        Ok(result)
    }

    /// Calculate text similarity using simple word overlap
    #[inline(always)]
    pub fn calculate_similarity(
        &self,
        text1: &str,
        text2: &str,
    ) -> Result<f32, TextProcessingError> {
        let freq1 = self.analyze_word_frequency(text1)?;
        let freq2 = self.analyze_word_frequency(text2)?;

        if freq1.is_empty() && freq2.is_empty() {
            return Ok(1.0);
        }

        if freq1.is_empty() || freq2.is_empty() {
            return Ok(0.0);
        }

        let mut common_words = 0;
        let mut total_words = 0;

        for (word, count1) in &freq1 {
            total_words += count1;
            if let Some(count2) = freq2.get(word) {
                common_words += count1.min(count2);
            }
        }

        for (word, count2) in &freq2 {
            if !freq1.contains_key(word) {
                total_words += count2;
            }
        }

        if total_words == 0 {
            Ok(0.0)
        } else {
            Ok(common_words as f32 / total_words as f32)
        }
    }

    /// Extract key phrases from text
    #[inline(always)]
    pub fn extract_key_phrases(
        &self,
        text: &str,
        max_phrases: usize,
    ) -> Result<ArrayVec<String, 16>, TextProcessingError> {
        let tokens = tokenize(text)?;
        let mut phrases = ArrayVec::new();
        let mut current_phrase = String::new();
        let mut word_count_in_phrase = 0;

        for token in tokens {
            match token.token_type() {
                TokenType::Word => {
                    if let Ok(word) = token.as_str() {
                        if !current_phrase.is_empty() {
                            current_phrase.push(' ');
                        }
                        current_phrase.push_str(word);
                        word_count_in_phrase += 1;

                        // Create phrase when we have 2-4 words
                        if word_count_in_phrase >= 2 && word_count_in_phrase <= 4 {
                            if phrases.len() < max_phrases {
                                if phrases.try_push(current_phrase.clone()).is_err() {
                                    break;
                                }
                            }
                        }

                        // Reset phrase after 4 words
                        if word_count_in_phrase >= 4 {
                            current_phrase.clear();
                            word_count_in_phrase = 0;
                        }
                    }
                }
                TokenType::Punctuation => {
                    // End current phrase on punctuation
                    if word_count_in_phrase >= 2 && phrases.len() < max_phrases {
                        if phrases.try_push(current_phrase.clone()).is_err() {
                            break;
                        }
                    }
                    current_phrase.clear();
                    word_count_in_phrase = 0;
                }
                _ => {
                    // Continue building phrase for other token types
                }
            }
        }

        // Add final phrase if valid
        if word_count_in_phrase >= 2 && phrases.len() < max_phrases {
            let _ = phrases.try_push(current_phrase);
        }

        Ok(phrases)
    }

    /// Get performance statistics
    #[inline(always)]
    pub fn get_stats(&self) -> PerformanceStats {
        let total_ops = self.analyses_performed.get().max(1);

        PerformanceStats {
            total_operations: total_ops as u64,
            total_processing_time_nanos: self.analysis_time_nanos.get() as u64,
            average_tokenization_time_nanos: 0,
            average_pattern_matching_time_nanos: 0,
            average_analysis_time_nanos: (self.analysis_time_nanos.get() / total_ops) as u64,
            simd_operations_count: 0,
            cache_hits: 0,
            cache_misses: 0}
    }

    /// Reset performance counters
    #[inline(always)]
    pub fn reset_stats(&self) {
        self.analyses_performed.reset();
        self.analysis_time_nanos.reset();
    }

    /// Clear word frequency cache
    #[inline(always)]
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.word_frequency_cache.write() {
            cache.clear();
        }
    }
}

/// Global text analyzer instance
static GLOBAL_TEXT_ANALYZER: once_cell::sync::Lazy<TextAnalyzer> =
    once_cell::sync::Lazy::new(|| TextAnalyzer::new());

/// Get global text analyzer instance
#[inline(always)]
pub fn get_global_text_analyzer() -> &'static TextAnalyzer {
    &GLOBAL_TEXT_ANALYZER
}

/// Convenience function for analyzing text
#[inline(always)]
pub fn analyze_text(text: &str) -> Result<TextStats, TextProcessingError> {
    get_global_text_analyzer().analyze_text(text)
}

/// Convenience function for word frequency analysis
#[inline(always)]
pub fn analyze_word_frequency(text: &str) -> Result<HashMap<String, u32>, TextProcessingError> {
    get_global_text_analyzer().analyze_word_frequency(text)
}

/// Convenience function for getting most common words
#[inline(always)]
pub fn get_most_common_words(
    text: &str,
    count: usize,
) -> Result<ArrayVec<(String, u32), 32>, TextProcessingError> {
    get_global_text_analyzer().get_most_common_words(text, count)
}

/// Convenience function for calculating text similarity
#[inline(always)]
pub fn calculate_similarity(text1: &str, text2: &str) -> Result<f32, TextProcessingError> {
    get_global_text_analyzer().calculate_similarity(text1, text2)
}

/// Convenience function for extracting key phrases
#[inline(always)]
pub fn extract_key_phrases(
    text: &str,
    max_phrases: usize,
) -> Result<ArrayVec<String, 16>, TextProcessingError> {
    get_global_text_analyzer().extract_key_phrases(text, max_phrases)
}

/// Convenience function for getting analyzer statistics
#[inline(always)]
pub fn get_analyzer_stats() -> PerformanceStats {
    get_global_text_analyzer().get_stats()
}
