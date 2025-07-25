//! Formatting styles and custom formatting rules
//!
//! Provides comprehensive styling system with inline formatting support,
//! custom rules, and zero-allocation style operations.

use serde::{Deserialize, Serialize};

use super::error::{FormatError, FormatResult};

/// Formatting styles for inline text
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FormatStyle {
    /// Start position in the text
    pub start: usize,
    /// End position in the text
    pub end: usize,
    /// Style type
    pub style: StyleType}

impl FormatStyle {
    /// Create new format style with validation
    #[inline]
    pub fn new(start: usize, end: usize, style: StyleType) -> FormatResult<Self> {
        if start >= end {
            return Err(FormatError::InvalidContent {
                detail: "Style start must be less than end".to_string()});
        }
        Ok(Self { start, end, style })
    }

    /// Get style length
    #[inline]
    pub fn length(&self) -> usize {
        self.end - self.start
    }

    /// Check if style overlaps with another
    #[inline]
    pub fn overlaps_with(&self, other: &FormatStyle) -> bool {
        !(self.end <= other.start || other.end <= self.start)
    }
}

/// Available style types with owned strings
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StyleType {
    Bold,
    Italic,
    Underline,
    Strikethrough,
    Code,
    Link { url: String },
    Color { rgb: u32 },
    Background { rgb: u32 }}

impl StyleType {
    /// Get style name as static string (zero allocation)
    #[inline]
    pub fn style_name(&self) -> &'static str {
        match self {
            Self::Bold => "bold",
            Self::Italic => "italic",
            Self::Underline => "underline",
            Self::Strikethrough => "strikethrough",
            Self::Code => "code",
            Self::Link { .. } => "link",
            Self::Color { .. } => "color",
            Self::Background { .. } => "background"}
    }

    /// Check if style requires additional data
    #[inline]
    pub fn requires_data(&self) -> bool {
        matches!(
            self,
            Self::Link { .. } | Self::Color { .. } | Self::Background { .. }
        )
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
    /// Rule priority (higher = applied first)
    pub priority: u32,
    /// Whether rule is enabled
    pub enabled: bool}

impl ImmutableCustomFormatRule {
    /// Create new custom format rule with validation
    #[inline]
    pub fn new(
        name: String,
        pattern: String,
        replacement: String,
        priority: u32,
        enabled: bool,
    ) -> FormatResult<Self> {
        if name.is_empty() {
            return Err(FormatError::ConfigurationError {
                detail: "Rule name cannot be empty".to_string()});
        }
        if pattern.is_empty() {
            return Err(FormatError::ConfigurationError {
                detail: "Rule pattern cannot be empty".to_string()});
        }

        Ok(Self {
            name,
            pattern,
            replacement,
            priority,
            enabled})
    }

    /// Validate rule configuration
    #[inline]
    pub fn validate(&self) -> FormatResult<()> {
        if self.name.is_empty() {
            return Err(FormatError::ConfigurationError {
                detail: "Rule name cannot be empty".to_string()});
        }
        if self.pattern.is_empty() {
            return Err(FormatError::ConfigurationError {
                detail: "Rule pattern cannot be empty".to_string()});
        }
        // TODO: Validate regex pattern syntax
        Ok(())
    }
}

/// Legacy compatibility alias
#[deprecated(note = "Use ImmutableCustomFormatRule instead for zero-allocation streaming")]
pub type CustomFormatRule = ImmutableCustomFormatRule;