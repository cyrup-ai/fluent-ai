//! UTF-8 validation utilities for the streaming decoder

use super::error::{DecoderError, Result};

/// Validates if a byte is a valid UTF-8 start byte
#[inline(always)]
pub fn is_valid_start_byte(byte: u8) -> bool {
    // 0xxxxxxx or 11xxxxxx
    byte < 0x80 || (byte & 0xC0) == 0xC0
}

/// Gets the expected length of a UTF-8 sequence from its first byte
#[inline(always)]
pub fn expected_sequence_length(first_byte: u8) -> usize {
    match first_byte {
        b if b < 0x80 => 1,       // 0xxxxxxx
        b if b & 0xE0 == 0xC0 => 2, // 110xxxxx
        b if b & 0xF0 == 0xE0 => 3, // 1110xxxx
        b if b & 0xF8 == 0xF0 => 4, // 11110xxx
        _ => 1,                    // Invalid, but we'll handle it as a single byte
    }
}

/// Validates a complete UTF-8 sequence
pub fn validate_utf8_sequence(bytes: &[u8]) -> Result<()> {
    if bytes.is_empty() {
        return Ok(());
    }

    let expected_len = expected_sequence_length(bytes[0]);
    
    if bytes.len() < expected_len {
        return Err(DecoderError::UnexpectedEof {
            expected: expected_len,
            actual: bytes.len()});
    }

    // Check continuation bytes
    for (i, &byte) in bytes[1..expected_len].iter().enumerate() {
        if (byte & 0xC0) != 0x80 {
            return Err(DecoderError::InvalidContinuationByte {
                position: i + 1,
                byte});
        }
    }

    // Check for overlong encodings
    if let Some(codepoint) = decode_codepoint(&bytes[..expected_len])? {
        // Check for overlong encoding
        let min_bytes = match codepoint {
            n if n <= 0x7F => 1,
            n if n <= 0x7FF => 2,
            n if n <= 0xFFFF => 3,
            _ => 4};

        if expected_len > min_bytes {
            return Err(DecoderError::OverlongEncoding {
                position: 0,
                codepoint});
        }

        // Check for invalid codepoints
        if (0xD800..=0xDFFF).contains(&codepoint) || codepoint > 0x10FFFF {
            return Err(DecoderError::InvalidCodepoint {
                position: 0,
                codepoint});
        }
    }

    Ok(())
}

/// Decodes a codepoint from a UTF-8 sequence
pub fn decode_codepoint(bytes: &[u8]) -> Result<Option<u32>> {
    if bytes.is_empty() {
        return Ok(None);
    }

    let first_byte = bytes[0];
    let len = expected_sequence_length(first_byte);

    if bytes.len() < len {
        return Ok(None);
    }

    let codepoint = match len {
        1 => u32::from(first_byte),
        2 => ((u32::from(first_byte) & 0x1F) << 6) | (u32::from(bytes[1]) & 0x3F),
        3 => ((u32::from(first_byte) & 0x0F) << 12)
            | ((u32::from(bytes[1]) & 0x3F) << 6)
            | (u32::from(bytes[2]) & 0x3F),
        4 => ((u32::from(first_byte) & 0x07) << 18)
            | ((u32::from(bytes[1]) & 0x3F) << 12)
            | ((u32::from(bytes[2]) & 0x3F) << 6)
            | (u32::from(bytes[3]) & 0x3F),
        _ => {
            return Err(DecoderError::InvalidUtf8Sequence {
                position: 0,
                bytes: bytes.to_vec()})
        }
    };

    Ok(Some(codepoint))
}

/// Checks if a byte is an ASCII character
#[inline(always)]
pub fn is_ascii(byte: u8) -> bool {
    byte < 0x80
}

/// Checks if a byte is a continuation byte
#[inline(always)]
pub fn is_continuation_byte(byte: u8) -> bool {
    (byte & 0xC0) == 0x80
}

/// Validates that a slice contains only ASCII characters
pub fn validate_ascii(bytes: &[u8]) -> bool {
    bytes.iter().all(|&b| is_ascii(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expected_sequence_length() {
        assert_eq!(expected_sequence_length(0x00), 1);
        assert_eq!(expected_sequence_length(0x7F), 1);
        assert_eq!(expected_sequence_length(0xC2), 2);
        assert_eq!(expected_sequence_length(0xE0), 3);
        assert_eq!(expected_sequence_length(0xF0), 4);
    }

    #[test]
    fn test_validate_utf8_sequence() {
        // Valid sequences
        assert!(validate_utf8_sequence(b"A").is_ok());
        assert!(validate_utf8_sequence("ß".as_bytes()).is_ok());
        assert!(validate_utf8_sequence("→".as_bytes()).is_ok());
        assert!(validate_utf8_sequence("𠮷".as_bytes()).is_ok());

        // Invalid sequences
        assert!(validate_utf8_sequence(&[0x80]).is_err()); // Invalid start byte
        assert!(validate_utf8_sequence(&[0xC2]).is_err()); // Incomplete sequence
        assert!(validate_utf8_sequence(&[0xE0, 0x80]).is_err()); // Overlong encoding
    }

    #[test]
    fn test_decode_codepoint() {
        assert_eq!(decode_codepoint(b"A").unwrap(), Some(0x41));
        assert_eq!(decode_codepoint("ß".as_bytes()).unwrap(), Some(0xDF));
        assert_eq!(decode_codepoint("→".as_bytes()).unwrap(), Some(0x2192));
        assert_eq!(decode_codepoint("𠮷".as_bytes()).unwrap(), Some(0x20BB7));
    }
}
