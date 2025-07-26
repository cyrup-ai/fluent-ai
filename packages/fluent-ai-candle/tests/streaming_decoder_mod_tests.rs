use fluent_ai_candle::streaming::decoder::mod::*;
use fluent_ai_candle::streaming::decoder::*;

#[test]
    fn test_decoder_integration() {
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

        let result = decoder.decode(second_half).unwrap();
        assert_eq!(result, "€");

        // Test stats
        let stats = decoder.stats();
        assert!(stats.total_bytes_processed > 0);
        assert_eq!(stats.partial_sequences_handled, 1);
    }
