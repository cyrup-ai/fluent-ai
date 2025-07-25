//! Tests for the streaming UTF-8 decoder

use fluent_ai_candle::streaming::decoder::{
    decode_codepoint, expected_sequence_length, is_ascii, is_continuation_byte, is_valid_start_byte,
    validate_ascii, validate_utf8_sequence, DecoderConfig, DecoderError, DecoderState, DecoderStats,
    StreamingDecoder};

#[test]
fn test_decoder_stats() {
    let mut stats = DecoderStats::new();
    assert_eq!(stats.total_bytes_processed, 0);
    assert_eq!(stats.total_chars_decoded, 0);
    assert_eq!(stats.partial_sequences_handled, 0);
    assert_eq!(stats.decode_errors, 0);

    stats.total_bytes_processed = 100;
    stats.total_chars_decoded = 50;
    stats.partial_sequences_handled = 2;
    stats.decode_errors = 1;

    let stats_str = format!("{}", stats);
    assert!(stats_str.contains("bytes: 100"));
    assert!(stats_str.contains("chars: 50"));
    assert!(stats_str.contains("partials: 2"));
    assert!(stats_str.contains("errors: 1"));
}

#[test]
fn test_decoder_state() {
    let ready = DecoderState::ready();
    assert!(ready.is_ready());
    assert!(!ready.is_partial());
    assert!(!ready.is_error());
    assert!(ready.pending_bytes().is_none());
    assert!(ready.error_message().is_none());

    let partial = DecoderState::partial(vec![0xC3, 0xA9]);
    assert!(!partial.is_ready());
    assert!(partial.is_partial());
    assert!(!partial.is_error());
    assert_eq!(partial.pending_bytes(), Some(&[0xC3, 0xA9][..]));
    assert!(partial.error_message().is_none());

    let error = DecoderState::error("test error");
    assert!(!error.is_ready());
    assert!(!error.is_partial());
    assert!(error.is_error());
    assert!(error.pending_bytes().is_none());
    assert_eq!(error.error_message(), Some("test error"));
}

#[test]
fn test_validation() {
    // Test ASCII detection
    assert!(is_ascii(b'A'));
    assert!(!is_ascii(0x80));

    // Test continuation byte detection
    assert!(is_continuation_byte(0x80));
    assert!(!is_continuation_byte(0x40));

    // Test start byte validation
    assert!(is_valid_start_byte(0x00)); // Single byte
    assert!(is_valid_start_byte(0xC2)); // Two-byte start
    assert!(is_valid_start_byte(0xE0)); // Three-byte start
    assert!(is_valid_start_byte(0xF0)); // Four-byte start
    assert!(!is_valid_start_byte(0x80)); // Continuation byte

    // Test sequence length detection
    assert_eq!(expected_sequence_length(0x41), 1);     // 'A'
    assert_eq!(expected_sequence_length(0xC3), 2);     // 'é'
    assert_eq!(expected_sequence_length(0xE2), 3);     // '→'
    assert_eq!(expected_sequence_length(0xF0), 4);     // '𠮷'
    assert_eq!(expected_sequence_length(0x80), 1);     // Invalid, treated as single byte

    // Test ASCII validation
    assert!(validate_ascii(b"Hello, world!"));
    assert!(!validate_ascii("Hello, 世界!".as_bytes()));
}

#[test]
fn test_utf8_validation() {
    // Valid sequences
    assert!(validate_utf8_sequence("A".as_bytes()).is_ok());
    assert!(validate_utf8_sequence("ß".as_bytes()).is_ok());
    assert!(validate_utf8_sequence("→".as_bytes()).is_ok());
    assert!(validate_utf8_sequence("𠮷".as_bytes()).is_ok());

    // Invalid sequences
    assert!(validate_utf8_sequence(&[0x80]).is_err()); // Invalid start byte
    assert!(validate_utf8_sequence(&[0xC2]).is_err()); // Incomplete sequence
    assert!(validate_utf8_sequence(&[0xE0, 0x80]).is_err()); // Overlong encoding
    assert!(validate_utf8_sequence(&[0xED, 0xA0, 0x80]).is_err()); // Surrogate pair
}

#[test]
fn test_codepoint_decoding() {
    assert_eq!(decode_codepoint(b"A").unwrap(), Some(0x41));
    assert_eq!(decode_codepoint("ß".as_bytes()).unwrap(), Some(0xDF));
    assert_eq!(decode_codepoint("→".as_bytes()).unwrap(), Some(0x2192));
    assert_eq!(decode_codepoint("𠮷".as_bytes()).unwrap(), Some(0x20BB7));
    assert!(decode_codepoint(&[0xC2]).is_ok_and(|o| o.is_none())); // Incomplete
}

#[test]
fn test_streaming_decoder() {
    let config = DecoderConfig {
        validate_utf8: true,
        enable_incremental: true,
        ..Default::default()
    };

    let mut decoder = StreamingDecoder::new(config);

    // Test ASCII
    let result = decoder.decode(b"Hello, ").unwrap();
    assert_eq!(result, "Hello, ");

    // Test multi-byte character split across chunks
    let euro = [0xE2, 0x82, 0xAC]; // '€' in UTF-8
    let first_half = &euro[..2];
    let second_half = &euro[2..];

    let result = decoder.decode(first_half).unwrap();
    assert!(result.is_empty()); // No complete characters yet
    assert!(matches!(*decoder.state(), DecoderState::Partial { .. }));

    let result = decoder.decode(second_half).unwrap();
    assert_eq!(result, "€");
    assert!(matches!(*decoder.state(), DecoderState::Ready));

    // Test stats
    let stats = decoder.stats();
    assert!(stats.total_bytes_processed > 0);
    assert_eq!(stats.partial_sequences_handled, 1);
}

#[test]
fn test_decoder_error_handling() {
    let config = DecoderConfig {
        validate_utf8: true,
        enable_incremental: false, // Disable incremental to test error handling
        ..Default::default()
    };

    let mut decoder = StreamingDecoder::new(config);

    // Test invalid UTF-8 sequence
    let result = decoder.decode(&[0xC3, 0x28]); // Invalid UTF-8
    assert!(matches!(
        result,
        Err(DecoderError::InvalidUtf8Sequence { .. })
    ));
    assert!(matches!(*decoder.state(), DecoderState::Error { .. }));

    // Test reset
    decoder.reset();
    assert!(matches!(*decoder.state(), DecoderState::Ready));
}

#[test]
fn test_decoder_edge_cases() {
    let config = DecoderConfig {
        validate_utf8: true,
        enable_incremental: true,
        max_pending_bytes: 4, // Small limit for testing
    };

    let mut decoder = StreamingDecoder::new(config);

    // Test max pending bytes
    let result = decoder.decode(&[0xF0, 0x9F, 0x8D]); // Start of a 4-byte sequence
    assert!(result.is_ok());
    assert!(matches!(*decoder.state(), DecoderState::Partial { .. }));

    // Add one more byte to exceed max_pending_bytes
    let result = decoder.decode(&[0x95, 0xF0, 0x9F, 0x8D, 0x95]);
    assert!(matches!(
        result,
        Err(DecoderError::InvalidUtf8Sequence { .. })
    ));
}

#[test]
fn test_decoder_configuration() {
    // Test with validation disabled
    let config = DecoderConfig {
        validate_utf8: false,
        enable_incremental: true,
        ..Default::default()
    };

    let mut decoder = StreamingDecoder::new(config);
    let result = decoder.decode(&[0xC3, 0x28]); // Invalid UTF-8
    assert!(result.is_ok()); // Should pass with validation disabled
}

// Property-based tests for more thorough validation
#[cfg(feature = "proptest")]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_roundtrip_ascii(s in "[ -~]+") {
            let mut decoder = StreamingDecoder::default();
            let result = decoder.decode(s.as_bytes()).unwrap();
            assert_eq!(result, s);
        }

        #[test]
        fn test_roundtrip_unicode(s in "[\u{0}-\u{10FFFF}]+") {
            let mut decoder = StreamingDecoder::default();
            let result = decoder.decode(s.as_bytes()).unwrap();
            assert_eq!(result, s);
        }
    }
}
