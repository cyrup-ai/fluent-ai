//! Format options, themes, and color schemes for message formatting
//!
//! Comprehensive configuration for formatting behavior including syntax highlighting,
//! color schemes, output formats, and custom formatting rules.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use super::error::{FormatError, FormatResult};
use super::themes::{ImmutableColorScheme, SyntaxTheme, OutputFormat};
use super::styles::ImmutableCustomFormatRule;

/// Immutable formatting options with owned strings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImmutableFormatOptions {
    /// Enable markdown parsing and rendering
    pub enable_markdown: bool,
    /// Enable syntax highlighting for code blocks
    pub enable_syntax_highlighting: bool,
    /// Enable inline formatting (bold, italic, etc.)
    pub enable_inline_formatting: bool,
    /// Enable link detection and formatting
    pub enable_link_detection: bool,
    /// Enable emoji rendering
    pub enable_emoji: bool,
    /// Maximum line length for text wrapping (0 = no wrapping)
    pub max_line_length: usize,
    /// Indentation size for nested content
    pub indent_size: usize,
    /// Theme for syntax highlighting
    pub syntax_theme: SyntaxTheme,
    /// Color scheme for formatting
    pub color_scheme: ImmutableColorScheme,
    /// Output format target
    pub output_format: OutputFormat,
    /// Include metadata in formatted output
    pub include_metadata: bool,
    /// Enable performance optimizations
    pub enable_optimizations: bool,
    /// Custom CSS classes for HTML output
    pub custom_css_classes: HashMap<String, String>,
    /// Custom formatting rules
    pub custom_rules: Vec<ImmutableCustomFormatRule>}

impl ImmutableFormatOptions {
    /// Create new format options with default values
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create format options optimized for terminal output
    #[inline]
    pub fn terminal() -> Self {
        Self {
            output_format: OutputFormat::AnsiTerminal,
            max_line_length: 120,
            enable_optimizations: true,
            ..Default::default()
        }
    }

    /// Create format options optimized for HTML output
    #[inline]
    pub fn html() -> Self {
        Self {
            output_format: OutputFormat::Html,
            enable_markdown: true,
            enable_syntax_highlighting: true,
            include_metadata: true,
            ..Default::default()
        }
    }

    /// Create format options optimized for plain text
    #[inline]
    pub fn plain_text() -> Self {
        Self {
            output_format: OutputFormat::PlainText,
            enable_markdown: false,
            enable_syntax_highlighting: false,
            enable_inline_formatting: false,
            enable_optimizations: true,
            ..Default::default()
        }
    }

    /// Validate formatting options
    #[inline]
    pub fn validate(&self) -> FormatResult<()> {
        // Validate color scheme
        self.color_scheme.validate()?;

        // Validate line length
        if self.max_line_length > 0 && self.max_line_length < 20 {
            return Err(FormatError::ConfigurationError {
                detail: "Max line length must be at least 20 characters or 0 for no wrapping".to_string()});
        }

        // Validate custom rules
        for rule in &self.custom_rules {
            rule.validate()?;
        }

        // Check compatibility
        if self.output_format == OutputFormat::PlainText {
            if self.enable_syntax_highlighting {
                return Err(FormatError::ConfigurationError {
                    detail: "Syntax highlighting not supported for plain text output".to_string()});
            }
            if !self.custom_css_classes.is_empty() {
                return Err(FormatError::ConfigurationError {
                    detail: "CSS classes not supported for plain text output".to_string()});
            }
        }

        Ok(())
    }

    /// Get effective line length (considering format limits)
    #[inline]
    pub fn effective_line_length(&self) -> Option<usize> {
        match self.max_line_length {
            0 => None,
            len => Some(len)}
    }
    
    /// Get color scheme reference
    #[inline]
    pub fn color_scheme(&self) -> &ImmutableColorScheme {
        &self.color_scheme
    }
    
    /// Get format rules reference
    #[inline]
    pub fn format_rules(&self) -> &[ImmutableCustomFormatRule] {
        &self.custom_rules
    }
    
    /// Get syntax theme
    #[inline]
    pub fn syntax_theme(&self) -> SyntaxTheme {
        self.syntax_theme
    }
    
    /// Get output format
    #[inline]
    pub fn output_format(&self) -> OutputFormat {
        self.output_format
    }
}

impl Default for ImmutableFormatOptions {
    #[inline]
    fn default() -> Self {
        Self {
            enable_markdown: true,
            enable_syntax_highlighting: true,
            enable_inline_formatting: true,
            enable_link_detection: true,
            enable_emoji: false,
            max_line_length: 0, // No wrapping by default
            indent_size: 2,
            syntax_theme: Default::default(),
            color_scheme: ImmutableColorScheme::default(),
            output_format: OutputFormat::Html,
            include_metadata: false,
            enable_optimizations: true,
            custom_css_classes: HashMap::new(),
            custom_rules: Vec::new()}
    }
}

// SyntaxTheme moved to themes.rs - use that version instead
// This duplicate has been eliminated to resolve type conflicts

// ImmutableColorScheme moved to themes.rs - use that version instead
// This duplicate has been eliminated to resolve type conflicts

// OutputFormat moved to themes.rs - use that version instead
// This duplicate has been eliminated to resolve type conflicts

// ImmutableCustomFormatRule moved to styles.rs - use that version instead
// This duplicate has been eliminated to resolve type conflicts

#[cfg(test)]
mod tests {
    use super::*;

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
}