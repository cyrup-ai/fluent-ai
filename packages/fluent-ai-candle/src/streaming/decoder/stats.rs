//! Statistics tracking for the streaming decoder

use std::fmt;

/// Statistics about decoder operations
#[derive(Debug, Clone, Default)]
pub struct DecoderStats {
    /// Total bytes processed by the decoder
    pub total_bytes_processed: usize,
    /// Total characters successfully decoded
    pub total_chars_decoded: usize,
    /// Number of partial sequences handled
    pub partial_sequences_handled: usize,
    /// Number of decode errors encountered
    pub decode_errors: usize,
}

impl fmt::Display for DecoderStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DecoderStats {{ bytes: {}, chars: {}, partials: {}, errors: {} }}",
            self.total_bytes_processed,
            self.total_chars_decoded,
            self.partial_sequences_handled,
            self.decode_errors
        )
    }
}

impl DecoderStats {
    /// Create a new instance with all counters set to zero
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new instance with the specified values
    pub fn with_counts(
        total_bytes_processed: usize,
        total_chars_decoded: usize,
        partial_sequences_handled: usize,
        decode_errors: usize,
    ) -> Self {
        Self {
            total_bytes_processed,
            total_chars_decoded,
            partial_sequences_handled,
            decode_errors,
        }
    }

    /// Reset all counters to zero
    pub fn reset(&mut self) {
        self.total_bytes_processed = 0;
        self.total_chars_decoded = 0;
        self.partial_sequences_handled = 0;
        self.decode_errors = 0;
    }

    /// Merge another DecoderStats into this one
    pub fn merge(&mut self, other: &Self) {
        self.total_bytes_processed += other.total_bytes_processed;
        self.total_chars_decoded += other.total_chars_decoded;
        self.partial_sequences_handled += other.partial_sequences_handled;
        self.decode_errors += other.decode_errors;
    }
}
