//! Immutable message content structures and validation
//!
//! Provides zero-allocation message content types with comprehensive validation
//! and owned string patterns for blazing-fast performance.

use serde::{Deserialize, Serialize};

use super::error::{FormatError, FormatResult};
use super::styles::FormatStyle;

/// Immutable message content with owned strings
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImmutableMessageContent {
    /// Plain text content
    Plain { text: String },
    /// Markdown formatted content
    Markdown {
        content: String,
        rendered_html: Option<String>,
    },
    /// Code block with syntax highlighting
    Code {
        content: String,
        language: String,
        highlighted: Option<String>,
    },
    /// Formatted content with inline styling
    Formatted {
        content: String,
        styles: Vec<FormatStyle>,
    },
    /// Composite content with multiple parts
    Composite { parts: Vec<ImmutableMessageContent> },
}

impl ImmutableMessageContent {
    /// Get content as borrowed string (zero allocation)
    #[inline]
    pub fn as_text(&self) -> &str {
        match self {
            Self::Plain { text } => text,
            Self::Markdown { content, .. } => content,
            Self::Code { content, .. } => content,
            Self::Formatted { content, .. } => content,
            Self::Composite { .. } => "", // Composite content needs rendering
        }
    }

    /// Get content type as static string (zero allocation)
    #[inline]
    pub fn content_type(&self) -> &'static str {
        match self {
            Self::Plain { .. } => "plain",
            Self::Markdown { .. } => "markdown",
            Self::Code { .. } => "code",
            Self::Formatted { .. } => "formatted",
            Self::Composite { .. } => "composite",
        }
    }

    /// Check if content is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Plain { text } => text.is_empty(),
            Self::Markdown { content, .. } => content.is_empty(),
            Self::Code { content, .. } => content.is_empty(),
            Self::Formatted { content, .. } => content.is_empty(),
            Self::Composite { parts } => parts.is_empty(),
        }
    }

    /// Get estimated character count
    #[inline]
    pub fn char_count(&self) -> usize {
        match self {
            Self::Plain { text } => text.chars().count(),
            Self::Markdown { content, .. } => content.chars().count(),
            Self::Code { content, .. } => content.chars().count(),
            Self::Formatted { content, .. } => content.chars().count(),
            Self::Composite { parts } => parts.iter().map(|p| p.char_count()).sum(),
        }
    }

    /// Validate content structure
    #[inline]
    pub fn validate(&self) -> FormatResult<()> {
        match self {
            Self::Code { language, .. } => {
                if language.is_empty() {
                    return Err(FormatError::InvalidContent {
                        detail: "Code language cannot be empty".to_string(),
                    });
                }
            }
            Self::Formatted { content, styles } => {
                for style in styles {
                    if style.end > content.len() {
                        return Err(FormatError::InvalidContent {
                            detail: "Style range exceeds content length".to_string(),
                        });
                    }
                    if style.start >= style.end {
                        return Err(FormatError::InvalidContent {
                            detail: "Invalid style range".to_string(),
                        });
                    }
                }
            }
            Self::Composite { parts } => {
                for part in parts {
                    part.validate()?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Legacy compatibility type alias (deprecated)
#[deprecated(note = "Use ImmutableMessageContent instead for zero-allocation streaming")]
pub type MessageContent = ImmutableMessageContent;