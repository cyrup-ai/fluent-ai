//! Theme and color scheme definitions
//!
//! Provides syntax highlighting themes and color schemes for formatting
//! operations with zero-allocation patterns and validation.

use serde::{Deserialize, Serialize};

use super::error::{FormatError, FormatResult};

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

/// Legacy compatibility type aliases (deprecated)
#[deprecated(note = "Use ImmutableColorScheme instead for zero-allocation streaming")]
pub type ColorScheme = ImmutableColorScheme;