//! Error types for the streaming decoder

use std::fmt;

/// Errors that can occur during streaming decoding
#[derive(Debug, Clone, PartialEq)]
pub enum DecoderError {
    /// Invalid UTF-8 sequence encountered
    InvalidUtf8Sequence {
        /// The position where the error occurred
        position: usize,
        /// The invalid bytes that caused the error
        bytes: Vec<u8>,
    },
    /// Unexpected end of input in the middle of a sequence
    UnexpectedEof {
        /// The expected number of bytes
        expected: usize,
        /// The actual number of bytes available
        actual: usize,
    },
    /// Invalid continuation byte
    InvalidContinuationByte {
        /// The position of the invalid byte
        position: usize,
        /// The invalid byte value
        byte: u8,
    },
    /// Overlong encoding detected
    OverlongEncoding {
        /// The position where the overlong sequence starts
        position: usize,
        /// The overlong encoded codepoint
        codepoint: u32,
    },
    /// Invalid codepoint
    InvalidCodepoint {
        /// The position where the invalid codepoint was found
        position: usize,
        /// The invalid codepoint value
        codepoint: u32,
    },
}

impl fmt::Display for DecoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUtf8Sequence { position, bytes } => {
                write!(
                    f,
                    "Invalid UTF-8 sequence at position {}: {:?}",
                    position, bytes
                )
            }
            Self::UnexpectedEof { expected, actual } => {
                write!(
                    f,
                    "Unexpected end of input: expected {} bytes, got {}",
                    expected, actual
                )
            }
            Self::InvalidContinuationByte { position, byte } => {
                write!(
                    f,
                    "Invalid continuation byte 0x{:02x} at position {}",
                    byte, position
                )
            }
            Self::OverlongEncoding { position, codepoint } => {
                write!(
                    f,
                    "Overlong encoding at position {} for codepoint U+{:04X}",
                    position, codepoint
                )
            }
            Self::InvalidCodepoint { position, codepoint } => {
                write!(
                    f,
                    "Invalid Unicode codepoint at position {}: U+{:04X}",
                    position, codepoint
                )
            }
        }
    }
}

impl std::error::Error for DecoderError {}

/// A specialized `Result` type for decoder operations
pub type Result<T> = std::result::Result<T, DecoderError>;

/// Extension trait for converting between different error types
pub trait ErrorExt<T> {
    /// Convert an error into a decoder error
    fn into_decoder_error(self, context: &'static str) -> std::result::Result<T, DecoderError>;
}

impl<T, E> ErrorExt<T> for std::result::Result<T, E>
where
    E: std::error::Error,
{
    fn into_decoder_error(self, context: &'static str) -> std::result::Result<T, DecoderError> {
        self.map_err(|e| DecoderError::InvalidUtf8Sequence {
            position: 0,
            bytes: format!("{}: {}", context, e).into_bytes(),
        })
    }
}
