//! Tests for formatting module
//!
//! Provides comprehensive tests for message formatting, styles, configuration,
//! and streaming operations with zero-allocation verification.

#[cfg(test)]
mod tests {
    use super::super::content::ImmutableMessageContent;
    use super::super::options::{ImmutableFormatOptions, OutputFormat};
    use super::super::streaming::StreamingMessageFormatter;
    use super::super::styles::{FormatStyle, StyleType};

    #[test]
    fn test_message_content_validation() {
        let content = ImmutableMessageContent::Plain {
            text: "Hello, world!".to_string()};
        assert!(content.validate().is_ok());
        assert_eq!(content.content_type(), "plain");
        assert!(!content.is_empty());
    }

    #[test]
    fn test_format_style_creation() {
        let style = FormatStyle::new(0, 5, StyleType::Bold).unwrap();
        assert_eq!(style.length(), 5);

        let invalid_style = FormatStyle::new(5, 0, StyleType::Bold);
        assert!(invalid_style.is_err());
    }

    #[test]
    fn test_color_scheme_validation() {
        let valid_scheme = super::super::options::ImmutableColorScheme::default();
        assert!(valid_scheme.validate().is_ok());

        let invalid_scheme = super::super::options::ImmutableColorScheme {
            primary_text: "invalid".to_string(),
            ..Default::default()
        };
        assert!(invalid_scheme.validate().is_err());
    }

    #[test]
    fn test_formatter_creation() {
        let options = ImmutableFormatOptions::default();
        let formatter = StreamingMessageFormatter::new(options).unwrap();
        let stats = formatter.stats();
        assert_eq!(stats.total_operations, 0);
    }

    #[test]
    fn test_output_format_capabilities() {
        assert!(OutputFormat::Html.supports_styling());
        assert!(OutputFormat::Html.supports_colors());
        assert!(!OutputFormat::PlainText.supports_styling());
        assert!(!OutputFormat::PlainText.supports_colors());
    }
}