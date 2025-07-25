//! Format options, themes, and color schemes for message formatting
//!
//! Comprehensive configuration for formatting behavior including syntax highlighting,
//! color schemes, output formats, and custom formatting rules.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use super::error::{FormatError, FormatResult};

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
    pub custom_rules: Vec<ImmutableCustomFormatRule>,
}

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
                detail: "Max line length must be at least 20 characters or 0 for no wrapping".to_string(),
            });
        }

        // Validate custom rules
        for rule in &self.custom_rules {
            rule.validate()?;
        }

        // Check compatibility
        if self.output_format == OutputFormat::PlainText {
            if self.enable_syntax_highlighting {
                return Err(FormatError::ConfigurationError {
                    detail: "Syntax highlighting not supported for plain text output".to_string(),
                });
            }
            if !self.custom_css_classes.is_empty() {
                return Err(FormatError::ConfigurationError {
                    detail: "CSS classes not supported for plain text output".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Get effective line length (considering format limits)
    #[inline]
    pub fn effective_line_length(&self) -> Option<usize> {
        match self.max_line_length {
            0 => None,
            len => Some(len),
        }
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
            syntax_theme: SyntaxTheme::default(),
            color_scheme: ImmutableColorScheme::default(),
            output_format: OutputFormat::Html,
            include_metadata: false,
            enable_optimizations: true,
            custom_css_classes: HashMap::new(),
            custom_rules: Vec::new(),
        }
    }
}

/// Syntax highlighting themes
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SyntaxTheme {
    /// Light theme with dark text
    Light,
    /// Dark theme with light text
    Dark,
    /// High contrast theme
    HighContrast,
    /// Solarized light theme
    SolarizedLight,
    /// Solarized dark theme
    SolarizedDark,
    /// GitHub theme
    GitHub,
    /// VS Code theme
    VSCode,
    /// Custom theme
    Custom,
}

impl SyntaxTheme {
    /// Get theme name as static string (zero allocation)
    #[inline]
    pub fn theme_name(&self) -> &'static str {
        match self {
            Self::Light => "light",
            Self::Dark => "dark",
            Self::HighContrast => "high-contrast",
            Self::SolarizedLight => "solarized-light",
            Self::SolarizedDark => "solarized-dark",
            Self::GitHub => "github",
            Self::VSCode => "vscode",
            Self::Custom => "custom",
        }
    }

    /// Check if theme is dark
    #[inline]
    pub fn is_dark(&self) -> bool {
        matches!(self, Self::Dark | Self::SolarizedDark)
    }
}

impl Default for SyntaxTheme {
    fn default() -> Self {
        Self::Light
    }
}

/// Immutable color scheme with owned strings
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImmutableColorScheme {
    /// Primary text color
    pub primary_text: String,
    /// Secondary text color
    pub secondary_text: String,
    /// Background color
    pub background: String,
    /// Accent color
    pub accent: String,
    /// Error color
    pub error: String,
    /// Warning color
    pub warning: String,
    /// Success color
    pub success: String,
    /// Link color
    pub link: String,
}

impl ImmutableColorScheme {
    /// Create new color scheme with validation
    #[inline]
    pub fn new(
        primary_text: String,
        secondary_text: String,
        background: String,
        accent: String,
        error: String,
        warning: String,
        success: String,
        link: String,
    ) -> FormatResult<Self> {
        let scheme = Self {
            primary_text,
            secondary_text,
            background,
            accent,
            error,
            warning,
            success,
            link,
        };
        scheme.validate()?;
        Ok(scheme)
    }

    /// Validate color values
    #[inline]
    pub fn validate(&self) -> FormatResult<()> {
        let colors = [
            &self.primary_text,
            &self.secondary_text,
            &self.background,
            &self.accent,
            &self.error,
            &self.warning,
            &self.success,
            &self.link,
        ];

        for color in &colors {
            if !Self::is_valid_color(color) {
                return Err(FormatError::ConfigurationError {
                    detail: format!("Invalid color format: {}", color),
                });
            }
        }
        Ok(())
    }

    /// Check if color string is valid (hex format)
    #[inline]
    fn is_valid_color(color: &str) -> bool {
        if !color.starts_with('#') || color.len() != 7 {
            return false;
        }
        color[1..].chars().all(|c| c.is_ascii_hexdigit())
    }
}

impl Default for ImmutableColorScheme {
    #[inline]
    fn default() -> Self {
        Self {
            primary_text: "#333333".to_string(),
            secondary_text: "#666666".to_string(),
            background: "#ffffff".to_string(),
            accent: "#0066cc".to_string(),
            error: "#cc0000".to_string(),
            warning: "#ff9900".to_string(),
            success: "#00cc00".to_string(),
            link: "#0066cc".to_string(),
        }
    }
}

/// Output format targets
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OutputFormat {
    /// Plain text output
    PlainText,
    /// HTML output
    Html,
    /// Markdown output
    Markdown,
    /// ANSI colored terminal output
    AnsiTerminal,
    /// Rich text format
    RichText,
    /// LaTeX output
    LaTeX,
}

impl OutputFormat {
    /// Get format name as static string (zero allocation)
    #[inline]
    pub fn format_name(&self) -> &'static str {
        match self {
            Self::PlainText => "plain-text",
            Self::Html => "html",
            Self::Markdown => "markdown",
            Self::AnsiTerminal => "ansi-terminal",
            Self::RichText => "rich-text",
            Self::LaTeX => "latex",
        }
    }

    /// Check if format supports styling
    #[inline]
    pub fn supports_styling(&self) -> bool {
        !matches!(self, Self::PlainText)
    }

    /// Check if format supports colors
    #[inline]
    pub fn supports_colors(&self) -> bool {
        matches!(self, Self::Html | Self::AnsiTerminal | Self::RichText)
    }
}

/// Immutable custom formatting rules with owned strings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImmutableCustomFormatRule {
    /// Rule name/identifier
    pub name: String,
    /// Pattern to match (regex)
    pub pattern: String,
    /// Replacement template
    pub replacement: String,
    /// Priority (higher values processed first)
    pub priority: i32,
    /// Whether rule is enabled
    pub enabled: bool,
}

impl ImmutableCustomFormatRule {
    /// Create new custom format rule
    #[inline]
    pub fn new(
        name: String,
        pattern: String,
        replacement: String,
        priority: i32,
    ) -> FormatResult<Self> {
        let rule = Self {
            name,
            pattern,
            replacement,
            priority,
            enabled: true,
        };
        rule.validate()?;
        Ok(rule)
    }

    /// Validate custom format rule
    #[inline]
    pub fn validate(&self) -> FormatResult<()> {
        if self.name.is_empty() {
            return Err(FormatError::ConfigurationError {
                detail: "Custom rule name cannot be empty".to_string(),
            });
        }

        if self.pattern.is_empty() {
            return Err(FormatError::ConfigurationError {
                detail: "Custom rule pattern cannot be empty".to_string(),
            });
        }

        // Validate regex pattern
        if let Err(e) = regex::Regex::new(&self.pattern) {
            return Err(FormatError::ConfigurationError {
                detail: format!("Invalid regex pattern '{}': {}", self.pattern, e),
            });
        }

        Ok(())
    }
}

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