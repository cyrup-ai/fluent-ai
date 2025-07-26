use fluent_ai_candle::types::candle_chat::chat::formatting::error::*;
use fluent_ai_candle::types::candle_chat::chat::formatting::*;

#[test]
    fn test_format_error_creation() {
        let error = FormatError::invalid_markdown("Missing closing tag");
        assert_eq!(error.category(), "markdown");
        assert!(error.is_user_error());
        assert!(!error.is_recoverable());
    }

    #[test]
    fn test_timeout_error() {
        let error = FormatError::timeout();
        assert_eq!(error.category(), "timeout");
        assert!(!error.is_user_error());
        assert!(error.is_recoverable());
    }

    #[test]
    fn test_error_conversion() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let format_error: FormatError = io_error.into();
        assert_eq!(format_error.category(), "io");
    }

    #[test]
    fn test_error_display() {
        let error = FormatError::unsupported_language("cobol");
        let error_string = format!("{}", error);
        assert!(error_string.contains("Unsupported language: cobol"));
    }

    #[test]
    fn test_error_classification() {
        let user_errors = vec![
            FormatError::invalid_markdown("test"),
            FormatError::parse_error("test"),
            FormatError::configuration_error("test"),
        ];

        for error in user_errors {
            assert!(error.is_user_error());
            assert!(!error.is_recoverable());
        }

        let recoverable_errors = vec![
            FormatError::timeout(),
            FormatError::io_error("test"),
            FormatError::resource_not_found("test"),
        ];

        for error in recoverable_errors {
            assert!(error.is_recoverable());
        }
    }
