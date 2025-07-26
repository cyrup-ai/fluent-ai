use fluent_ai_candle::types::candle_chat::chat::formatting::options::*;
use fluent_ai_candle::types::candle_chat::chat::formatting::*;

#[test]
    fn test_format_options_creation() {
        let options = ImmutableFormatOptions::new();
        assert!(options.validate().is_ok());

        let terminal_options = ImmutableFormatOptions::terminal();
        assert_eq!(terminal_options.output_format, OutputFormat::AnsiTerminal);
        assert_eq!(terminal_options.max_line_length, 120);

        let html_options = ImmutableFormatOptions::html();
        assert_eq!(html_options.output_format, OutputFormat::Html);
        assert!(html_options.enable_markdown);
    }

    #[test]
    fn test_syntax_theme() {
        let theme = SyntaxTheme::Dark;
        assert_eq!(theme.theme_name(), "dark");
        assert!(theme.is_dark());

        let light_theme = SyntaxTheme::Light;
        assert!(!light_theme.is_dark());
    }

    #[test]
    fn test_color_scheme_validation() {
        let valid_scheme = ImmutableColorScheme::default();
        assert!(valid_scheme.validate().is_ok());

        let invalid_scheme = ImmutableColorScheme {
            primary_text: "invalid".to_string(),
            ..Default::default()
        };
        assert!(invalid_scheme.validate().is_err());
    }

    #[test]
    fn test_output_format_capabilities() {
        assert!(OutputFormat::Html.supports_styling());
        assert!(OutputFormat::Html.supports_colors());

        assert!(!OutputFormat::PlainText.supports_styling());
        assert!(!OutputFormat::PlainText.supports_colors());

        assert!(OutputFormat::AnsiTerminal.supports_colors());
        assert_eq!(OutputFormat::Markdown.format_name(), "markdown");
    }

    #[test]
    fn test_custom_format_rule() {
        let rule = ImmutableCustomFormatRule::new(
            "test".to_string(),
            r"\b[A-Z]+\b".to_string(),
            "<strong>$0</strong>".to_string(),
            10,
        );
        assert!(rule.is_ok());

        let invalid_rule = ImmutableCustomFormatRule::new(
            "".to_string(),
            "pattern".to_string(),
            "replacement".to_string(),
            0,
        );
        assert!(invalid_rule.is_err());
    }

    #[test]
    fn test_options_validation() {
        let mut options = ImmutableFormatOptions::plain_text();
        options.enable_syntax_highlighting = true;
        assert!(options.validate().is_err());

        options = ImmutableFormatOptions::new();
        options.max_line_length = 10; // Too short
        assert!(options.validate().is_err());
    }
