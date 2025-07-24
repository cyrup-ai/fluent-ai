//! Incremental UTF-8 decoder for streaming token processing
//!
//! Handles partial byte sequences and maintains state across decode operations
//! for proper UTF-8 streaming without character boundary corruption.

use std::mem;

use super::StreamingError;

/// State of the incremental decoder
#[derive(Debug, Clone, PartialEq)]
pub enum DecoderState {
    /// Ready to decode new bytes
    Ready,
    /// Waiting for more bytes to complete a multi-byte sequence
    Partial { pending_bytes: Vec<u8> },
    /// Error state requiring reset
    Error { error: String },
}

impl Default for DecoderState {
    fn default() -> Self {
        DecoderState::Ready
    }
}

/// Streaming UTF-8 decoder with incremental processing
///
/// Handles:
/// - Partial multi-byte UTF-8 sequences across token boundaries
/// - State preservation between decode calls
/// - Character boundary detection and buffering
/// - Error recovery and validation
/// - Zero-copy operations where possible
pub struct StreamingDecoder {
    state: DecoderState,
    validate_utf8: bool,
    enable_incremental: bool,
    // Statistics for monitoring
    total_bytes_processed: usize,
    total_chars_decoded: usize,
    partial_sequences_handled: usize,
    decode_errors: usize,
}

impl StreamingDecoder {
    /// Create new streaming decoder with configuration
    ///
    /// # Arguments
    /// * `validate_utf8` - Whether to perform strict UTF-8 validation
    /// * `enable_incremental` - Whether to handle partial sequences incrementally
    pub fn new(validate_utf8: bool, enable_incremental: bool) -> Self {
        Self {
            state: DecoderState::Ready,
            validate_utf8,
            enable_incremental,
            total_bytes_processed: 0,
            total_chars_decoded: 0,
            partial_sequences_handled: 0,
            decode_errors: 0,
        }
    }

    /// Decode bytes incrementally, handling partial UTF-8 sequences
    ///
    /// Returns:
    /// - `Ok((decoded_string, is_complete))` where `is_complete` indicates
    ///   whether the decoded content forms complete UTF-8 characters
    /// - `Err(StreamingError)` for validation or decoding errors
    pub fn decode_incremental(&mut self, bytes: &[u8]) -> Result<(String, bool), StreamingError> {
        if bytes.is_empty() {
            return Ok((String::new(), true));
        }

        self.total_bytes_processed += bytes.len();

        // Fast path for ASCII-only content when no partial state exists
        if self.is_ready_state() && self.is_ascii_only(bytes) {
            let result = unsafe {
                // SAFETY: We've validated these are ASCII bytes
                String::from_utf8_unchecked(bytes.to_vec())
            };
            self.total_chars_decoded += result.len();
            return Ok((result, true));
        }

        // Handle incremental decoding with potential partial sequences
        self.decode_with_state_management(bytes)
    }

    /// Decode with full state management for partial sequences
    fn decode_with_state_management(
        &mut self,
        new_bytes: &[u8],
    ) -> Result<(String, bool), StreamingError> {
        // Combine pending bytes from previous calls with new bytes
        let input_bytes = self.combine_with_pending(new_bytes);

        // Find the longest valid UTF-8 prefix
        let (complete_bytes, remaining_bytes, is_complete) =
            self.find_complete_utf8_prefix(&input_bytes)?;

        // Decode the complete portion
        let decoded = if complete_bytes.is_empty() {
            String::new()
        } else {
            self.decode_complete_bytes(&complete_bytes)?
        };

        // Update state based on remaining bytes
        self.update_state_after_decode(remaining_bytes, is_complete);

        self.total_chars_decoded += decoded.chars().count();

        Ok((decoded, is_complete))
    }

    /// Combine pending bytes from previous calls with new input
    fn combine_with_pending(&mut self, new_bytes: &[u8]) -> Vec<u8> {
        match &mut self.state {
            DecoderState::Partial { pending_bytes } => {
                let mut combined = mem::take(pending_bytes);
                combined.extend_from_slice(new_bytes);
                combined
            }
            _ => new_bytes.to_vec(),
        }
    }

    /// Find the longest valid UTF-8 prefix in the input bytes
    ///
    /// Returns (complete_bytes, remaining_bytes, is_complete)
    fn find_complete_utf8_prefix(
        &mut self,
        bytes: &[u8],
    ) -> Result<(Vec<u8>, Vec<u8>, bool), StreamingError> {
        if !self.enable_incremental {
            // Simple mode: validate entire input
            return self.validate_complete_sequence(bytes);
        }

        // Incremental mode: find the longest valid prefix
        let mut complete_end = 0;
        let mut char_start = 0;

        while char_start < bytes.len() {
            match self.get_char_byte_length(&bytes[char_start..]) {
                Ok(char_len) => {
                    if char_start + char_len <= bytes.len() {
                        // Complete character available
                        complete_end = char_start + char_len;
                        char_start = complete_end;
                    } else {
                        // Partial character at end - stop here
                        break;
                    }
                }
                Err(_) => {
                    if self.validate_utf8 {
                        self.handle_decode_error("Invalid UTF-8 sequence found")?;
                    }
                    // Skip invalid byte and continue
                    char_start += 1;
                }
            }
        }

        let complete_bytes = bytes[..complete_end].to_vec();
        let remaining_bytes = bytes[complete_end..].to_vec();
        let is_complete = remaining_bytes.is_empty();

        Ok((complete_bytes, remaining_bytes, is_complete))
    }

    /// Validate entire sequence as complete UTF-8
    fn validate_complete_sequence(
        &self,
        bytes: &[u8],
    ) -> Result<(Vec<u8>, Vec<u8>, bool), StreamingError> {
        if self.validate_utf8 {
            match std::str::from_utf8(bytes) {
                Ok(_) => Ok((bytes.to_vec(), Vec::new(), true)),
                Err(e) => {
                    // Check if error is due to incomplete sequence at end
                    let valid_up_to = e.valid_up_to();
                    if valid_up_to < bytes.len() && self.enable_incremental {
                        // Split at last valid boundary
                        let complete = bytes[..valid_up_to].to_vec();
                        let remaining = bytes[valid_up_to..].to_vec();
                        Ok((complete, remaining, false))
                    } else {
                        Err(StreamingError::Utf8Error(format!(
                            "UTF-8 validation failed: {}",
                            e
                        )))
                    }
                }
            }
        } else {
            // No validation - assume all bytes are valid
            Ok((bytes.to_vec(), Vec::new(), true))
        }
    }

    /// Get the byte length of the UTF-8 character starting at the given position
    fn get_char_byte_length(&self, bytes: &[u8]) -> Result<usize, StreamingError> {
        if bytes.is_empty() {
            return Err(StreamingError::Utf8Error("Empty byte sequence".to_string()));
        }

        let first_byte = bytes[0];
        let char_len = if first_byte < 0x80 {
            1 // ASCII
        } else if first_byte < 0xC0 {
            return Err(StreamingError::Utf8Error(
                "Invalid UTF-8 start byte".to_string(),
            ));
        } else if first_byte < 0xE0 {
            2 // 2-byte sequence
        } else if first_byte < 0xF0 {
            3 // 3-byte sequence
        } else if first_byte < 0xF8 {
            4 // 4-byte sequence
        } else {
            return Err(StreamingError::Utf8Error(
                "Invalid UTF-8 start byte".to_string(),
            ));
        };

        Ok(char_len)
    }

    /// Decode bytes that are known to be complete UTF-8 sequences
    fn decode_complete_bytes(&self, bytes: &[u8]) -> Result<String, StreamingError> {
        if self.validate_utf8 {
            String::from_utf8(bytes.to_vec())
                .map_err(|e| StreamingError::Utf8Error(format!("UTF-8 conversion failed: {}", e)))
        } else {
            // Unsafe conversion when validation is disabled (for performance)
            Ok(unsafe { String::from_utf8_unchecked(bytes.to_vec()) })
        }
    }

    /// Update decoder state after processing
    fn update_state_after_decode(&mut self, remaining_bytes: Vec<u8>, is_complete: bool) {
        if is_complete || !self.enable_incremental {
            self.state = DecoderState::Ready;
        } else if remaining_bytes.is_empty() {
            self.state = DecoderState::Ready;
        } else {
            self.state = DecoderState::Partial {
                pending_bytes: remaining_bytes,
            };
            self.partial_sequences_handled += 1;
        }
    }

    /// Handle decode error based on configuration
    fn handle_decode_error(&mut self, error_msg: &str) -> Result<(), StreamingError> {
        self.decode_errors += 1;

        if self.validate_utf8 {
            self.state = DecoderState::Error {
                error: error_msg.to_string(),
            };
            Err(StreamingError::Utf8Error(error_msg.to_string()))
        } else {
            // Continue processing in non-validating mode
            Ok(())
        }
    }

    /// Check if decoder is in ready state
    #[inline(always)]
    fn is_ready_state(&self) -> bool {
        matches!(self.state, DecoderState::Ready)
    }

    /// Check if bytes contain only ASCII characters (fast path optimization)
    #[inline(always)]
    fn is_ascii_only(&self, bytes: &[u8]) -> bool {
        bytes.iter().all(|&b| b < 0x80)
    }

    /// Force flush any pending partial sequence
    ///
    /// This should be called at end of stream to get any remaining partial content.
    /// Returns the partial content if any exists.
    pub fn flush_pending(&mut self) -> Result<Option<String>, StreamingError> {
        match mem::take(&mut self.state) {
            DecoderState::Partial { pending_bytes } => {
                self.state = DecoderState::Ready;

                if self.validate_utf8 {
                    // In validating mode, partial sequences at end are errors
                    Err(StreamingError::Utf8Error(
                        "Incomplete UTF-8 sequence at end of stream".to_string(),
                    ))
                } else {
                    // In non-validating mode, return what we can decode
                    let result = unsafe { String::from_utf8_unchecked(pending_bytes) };
                    Ok(Some(result))
                }
            }
            DecoderState::Error { error } => {
                self.state = DecoderState::Ready;
                Err(StreamingError::Utf8Error(error))
            }
            DecoderState::Ready => Ok(None),
        }
    }

    /// Reset decoder to initial state
    pub fn reset(&mut self) {
        self.state = DecoderState::Ready;
        // Keep statistics for monitoring
    }

    /// Get current decoder state
    pub fn state(&self) -> &DecoderState {
        &self.state
    }

    /// Get decoder statistics
    pub fn get_stats(&self) -> DecoderStats {
        DecoderStats {
            total_bytes_processed: self.total_bytes_processed,
            total_chars_decoded: self.total_chars_decoded,
            partial_sequences_handled: self.partial_sequences_handled,
            decode_errors: self.decode_errors,
            has_pending_bytes: matches!(self.state, DecoderState::Partial { .. }),
        }
    }

    /// Check if decoder has pending bytes from partial sequences
    pub fn has_pending(&self) -> bool {
        matches!(self.state, DecoderState::Partial { .. })
    }

    /// Get the number of pending bytes (for buffer management)
    pub fn pending_byte_count(&self) -> usize {
        match &self.state {
            DecoderState::Partial { pending_bytes } => pending_bytes.len(),
            _ => 0,
        }
    }
}

/// Statistics about decoder operations
#[derive(Debug, Clone)]
pub struct DecoderStats {
    pub total_bytes_processed: usize,
    pub total_chars_decoded: usize,
    pub partial_sequences_handled: usize,
    pub decode_errors: usize,
    pub has_pending_bytes: bool,
}

impl std::fmt::Display for DecoderStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DecoderStats(bytes: {}, chars: {}, partial: {}, errors: {}, pending: {})",
            self.total_bytes_processed,
            self.total_chars_decoded,
            self.partial_sequences_handled,
            self.decode_errors,
            self.has_pending_bytes
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ascii_decoding() {
        let mut decoder = StreamingDecoder::new(true, true);
        let (result, complete) = decoder
            .decode_incremental(b"Hello World")
            .expect("decode success");

        assert_eq!(result, "Hello World");
        assert!(complete);
        assert_eq!(decoder.state, DecoderState::Ready);
    }

    #[test]
    fn test_utf8_decoding() {
        let mut decoder = StreamingDecoder::new(true, true);
        let utf8_bytes = "Hello 世界".as_bytes();
        let (result, complete) = decoder
            .decode_incremental(utf8_bytes)
            .expect("decode success");

        assert_eq!(result, "Hello 世界");
        assert!(complete);
    }

    #[test]
    fn test_partial_utf8_sequence() {
        let mut decoder = StreamingDecoder::new(true, true);

        // Chinese character "世" in UTF-8 is [228, 184, 150]
        let full_bytes = "世".as_bytes(); // [228, 184, 150]

        // Send partial sequence
        let (result1, complete1) = decoder
            .decode_incremental(&full_bytes[..2])
            .expect("decode partial");
        assert_eq!(result1, "");
        assert!(!complete1);
        assert!(matches!(decoder.state, DecoderState::Partial { .. }));

        // Send remaining byte
        let (result2, complete2) = decoder
            .decode_incremental(&full_bytes[2..])
            .expect("decode complete");
        assert_eq!(result2, "世");
        assert!(complete2);
        assert_eq!(decoder.state, DecoderState::Ready);
    }

    #[test]
    fn test_mixed_ascii_and_utf8() {
        let mut decoder = StreamingDecoder::new(true, true);
        let mixed = "ABC世界DEF";
        let bytes = mixed.as_bytes();

        let (result, complete) = decoder.decode_incremental(bytes).expect("decode success");
        assert_eq!(result, mixed);
        assert!(complete);
    }

    #[test]
    fn test_incremental_across_multiple_calls() {
        let mut decoder = StreamingDecoder::new(true, true);
        let full_text = "Hello 世界 Testing";
        let bytes = full_text.as_bytes();

        let mut decoded_parts = Vec::new();
        let chunk_size = 3;

        for chunk in bytes.chunks(chunk_size) {
            let (part, _) = decoder.decode_incremental(chunk).expect("decode chunk");
            if !part.is_empty() {
                decoded_parts.push(part);
            }
        }

        // Flush any remaining
        if let Some(remaining) = decoder.flush_pending().expect("flush success") {
            decoded_parts.push(remaining);
        }

        let reconstructed = decoded_parts.join("");
        assert_eq!(reconstructed, full_text);
    }

    #[test]
    fn test_invalid_utf8_with_validation() {
        let mut decoder = StreamingDecoder::new(true, true);
        let invalid_bytes = vec![0xFF, 0xFE, 0xFD]; // Invalid UTF-8

        let result = decoder.decode_incremental(&invalid_bytes);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StreamingError::Utf8Error(_)));
    }

    #[test]
    fn test_invalid_utf8_without_validation() {
        let mut decoder = StreamingDecoder::new(false, true);
        let invalid_bytes = vec![0xFF, 0xFE, 0xFD]; // Invalid UTF-8

        // Should not fail when validation is disabled
        let (result, complete) = decoder
            .decode_incremental(&invalid_bytes)
            .expect("decode without validation");
        assert!(!result.is_empty()); // Will contain replacement chars or raw bytes
        assert!(complete);
    }

    #[test]
    fn test_empty_input() {
        let mut decoder = StreamingDecoder::new(true, true);
        let (result, complete) = decoder.decode_incremental(&[]).expect("empty decode");

        assert_eq!(result, "");
        assert!(complete);
        assert_eq!(decoder.state, DecoderState::Ready);
    }

    #[test]
    fn test_decoder_reset() {
        let mut decoder = StreamingDecoder::new(true, true);

        // Create partial state
        let partial_bytes = "世".as_bytes();
        let _ = decoder.decode_incremental(&partial_bytes[..2]);
        assert!(matches!(decoder.state, DecoderState::Partial { .. }));

        // Reset should clear state
        decoder.reset();
        assert_eq!(decoder.state, DecoderState::Ready);
    }

    #[test]
    fn test_decoder_stats() {
        let mut decoder = StreamingDecoder::new(true, true);

        let _ = decoder.decode_incremental(b"Hello");
        let _ = decoder.decode_incremental(" World".as_bytes());

        let stats = decoder.get_stats();
        assert_eq!(stats.total_bytes_processed, 11);
        assert_eq!(stats.total_chars_decoded, 11); // All ASCII
        assert!(!stats.has_pending_bytes);
    }

    #[test]
    fn test_4_byte_utf8_character() {
        let mut decoder = StreamingDecoder::new(true, true);

        // Emoji "🌟" is 4 bytes in UTF-8: [240, 159, 140, 159]
        let emoji_bytes = "🌟".as_bytes();

        // Send partial (3 bytes)
        let (result1, complete1) = decoder
            .decode_incremental(&emoji_bytes[..3])
            .expect("partial decode");
        assert_eq!(result1, "");
        assert!(!complete1);

        // Send final byte
        let (result2, complete2) = decoder
            .decode_incremental(&emoji_bytes[3..])
            .expect("complete decode");
        assert_eq!(result2, "🌟");
        assert!(complete2);
    }

    #[test]
    fn test_flush_pending_with_validation() {
        let mut decoder = StreamingDecoder::new(true, true);

        // Create partial state
        let partial_bytes = "世".as_bytes();
        let _ = decoder.decode_incremental(&partial_bytes[..2]);

        // Flushing partial should error in validation mode
        let result = decoder.flush_pending();
        assert!(result.is_err());
    }

    #[test]
    fn test_flush_pending_without_validation() {
        let mut decoder = StreamingDecoder::new(false, true);

        // Create partial state
        let partial_bytes = "世".as_bytes();
        let _ = decoder.decode_incremental(&partial_bytes[..2]);

        // Flushing partial should succeed in non-validation mode
        let result = decoder.flush_pending().expect("flush success");
        assert!(result.is_some());
    }

    #[test]
    fn test_pending_byte_count() {
        let mut decoder = StreamingDecoder::new(true, true);

        assert_eq!(decoder.pending_byte_count(), 0);
        assert!(!decoder.has_pending());

        // Create partial state
        let partial_bytes = "世".as_bytes(); // 3 bytes
        let _ = decoder.decode_incremental(&partial_bytes[..2]);

        assert_eq!(decoder.pending_byte_count(), 2);
        assert!(decoder.has_pending());
    }
}
