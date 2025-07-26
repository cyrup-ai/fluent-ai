use fluent_ai_candle::streaming::decoder::validation::*;
use fluent_ai_candle::streaming::decoder::*;

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
