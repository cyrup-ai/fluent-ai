//! Core streaming UTF-8 decoder implementation

use super::{
    error::{DecoderError, DecoderResult},
    state::DecoderState,
    stats::DecoderStats,
    validation};

/// Configuration for the streaming decoder
#[derive(Debug, Clone, Copy)]
pub struct DecoderConfig {
    /// Whether to perform strict UTF-8 validation
    pub validate_utf8: bool,
    /// Whether to handle partial sequences incrementally
    pub enable_incremental: bool,
    /// Maximum allowed pending bytes before error
    pub max_pending_bytes: usize}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            validate_utf8: true,
            enable_incremental: true,
            max_pending_bytes: 4096, // 4KB should be enough for any valid UTF-8 sequence
        }
    }
}

/// Streaming UTF-8 decoder with configurable behavior
pub struct StreamingDecoder {
    /// Current decoder state
    state: DecoderState,
    /// Decoder configuration
    config: DecoderConfig,
    /// Statistics about decoder operations
    stats: DecoderStats}

impl Default for StreamingDecoder {
    fn default() -> Self {
        Self::new(DecoderConfig::default())
    }
}

impl StreamingDecoder {
    /// Create a new streaming decoder with default configuration
    pub fn new(config: DecoderConfig) -> Self {
        Self {
            state: DecoderState::ready(),
            config,
            stats: DecoderStats::new()}
    }

    /// Reset the decoder to its initial state
    pub fn reset(&mut self) {
        self.state = DecoderState::ready();
        self.stats = DecoderStats::new();
    }

    /// Decode a chunk of bytes into a string, handling partial sequences
    pub fn decode(&mut self, bytes: &[u8]) -> DecoderResult<String> {
        if bytes.is_empty() {
            return Ok(String::new());
        }

        self.stats.total_bytes_processed += bytes.len();

        // Fast path for ASCII-only content
        if self.state.is_ready() && validation::validate_ascii(bytes) {
            // Safe UTF-8 conversion - we've validated these are ASCII bytes
            let result = String::from_utf8(bytes.to_vec())
                .map_err(|e| DecoderError::InvalidUtf8Sequence {
                    position: 0,
                    bytes: e.into_bytes()})?;
            self.stats.total_chars_decoded += result.len();
            return Ok(result);
        }

        self.decode_with_state(bytes)
    }

    /// Get the current decoder state
    pub fn state(&self) -> &DecoderState {
        &self.state
    }

    /// Get decoder statistics
    pub fn stats(&self) -> &DecoderStats {
        &self.stats
    }

    /// Get a mutable reference to decoder statistics
    pub fn stats_mut(&mut self) -> &mut DecoderStats {
        &mut self.stats
    }

    /// Get the current configuration
    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }

    /// Get a mutable reference to the configuration
    pub fn config_mut(&mut self) -> &mut DecoderConfig {
        &mut self.config
    }

    /// Internal method to handle decoding with state management
    fn decode_with_state(&mut self, new_bytes: &[u8]) -> DecoderResult<String> {
        let combined = self.combine_with_pending(new_bytes);
        let (complete, pending, is_complete) = self.find_complete_sequence(&combined)?;

        // Update state based on whether we have pending bytes
        self.update_state(pending, is_complete)?;

        // Decode the complete sequence
        let result = self.decode_complete_bytes(&complete)?;
        self.stats.total_chars_decoded += result.chars().count();

        if !pending.is_empty() {
            self.stats.partial_sequences_handled += 1;
        }

        Ok(result)
    }

    /// Combine new bytes with any pending bytes from the current state
    fn combine_with_pending(&self, new_bytes: &[u8]) -> Vec<u8> {
        match &self.state {
            DecoderState::Partial { pending_bytes } => {
                let mut combined = Vec::with_capacity(pending_bytes.len() + new_bytes.len());
                combined.extend_from_slice(pending_bytes);
                combined.extend_from_slice(new_bytes);
                combined
            }
            _ => new_bytes.to_vec()}
    }

    /// Find the longest valid UTF-8 sequence in the input
    fn find_complete_sequence<'a>(
        &self,
        bytes: &'a [u8],
    ) -> DecoderResult<(&'a [u8], &'a [u8], bool)> {
        if bytes.is_empty() {
            return Ok((&[], &[], true));
        }

        let mut end = bytes.len();
        let mut is_complete = true;

        // Find the last valid UTF-8 sequence boundary
        while end > 0 {
            if validation::is_valid_start_byte(bytes[end - 1]) {
                let seq_len = validation::expected_sequence_length(bytes[end - 1]);
                if end + seq_len - 1 <= bytes.len() {
                    // We have a complete sequence
                    break;
                } else if self.config.enable_incremental {
                    // Incomplete sequence at the end
                    is_complete = false;
                    break;
                } else {
                    return Err(DecoderError::UnexpectedEof {
                        expected: seq_len,
                        actual: bytes.len() - end + 1});
                }
            } else if !validation::is_continuation_byte(bytes[end - 1]) {
                // Invalid UTF-8 sequence
                return Err(DecoderError::InvalidUtf8Sequence {
                    position: end - 1,
                    bytes: bytes.to_vec()});
            }
            end -= 1;
        }

        let (complete, pending) = bytes.split_at(end);
        Ok((complete, pending, is_complete))
    }

    /// Update the decoder state based on the current operation
    fn update_state(&mut self, pending: &[u8], is_complete: bool) -> DecoderResult<()> {
        if !pending.is_empty() {
            if pending.len() > self.config.max_pending_bytes {
                return Err(DecoderError::InvalidUtf8Sequence {
                    position: 0,
                    bytes: pending.to_vec()});
            }

            if self.config.enable_incremental && !is_complete {
                self.state = DecoderState::partial(pending.to_vec());
            } else if self.config.validate_utf8 {
                validation::validate_utf8_sequence(pending)?;
            }
        } else {
            self.state = DecoderState::ready();
        }

        Ok(())
    }

    /// Decode a complete UTF-8 sequence
    fn decode_complete_bytes(&self, bytes: &[u8]) -> DecoderResult<String> {
        if self.config.validate_utf8 {
            validation::validate_utf8_sequence(bytes)?;
        }

        String::from_utf8(bytes.to_vec()).map_err(|e| DecoderError::InvalidUtf8Sequence {
            position: e.utf8_error().valid_up_to(),
            bytes: bytes.to_vec()})
    }
}
