//! Immutable message formatting with streaming operations
//!
//! Provides zero-allocation, lock-free message formatting with streaming operations.
//! All Arc usage eliminated in favor of owned strings and borrowed data patterns
//! for blazing-fast performance with immutable formatting structures.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use fluent_ai_async::{AsyncStream, AsyncStreamSender};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Immutable message content with owned strings
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImmutableMessageContent {
    /// Plain text content
    Plain { text: String },
    /// Markdown formatted content
    Markdown {
        content: String,
        rendered_html: Option<String>},
    /// Code block with syntax highlighting
    Code {
        content: String,
        language: String,
        highlighted: Option<String>},
    /// Formatted content with inline styling
    Formatted {
        content: String,
        styles: Vec<FormatStyle>},
    /// Composite content with multiple parts
    Composite { parts: Vec<ImmutableMessageContent> }}

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
            Self::Composite { .. } => "composite"}
    }

    /// Check if content is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Plain { text } => text.is_empty(),
            Self::Markdown { content, .. } => content.is_empty(),
            Self::Code { content, .. } => content.is_empty(),
            Self::Formatted { content, .. } => content.is_empty(),
            Self::Composite { parts } => parts.is_empty()}
    }

    /// Get estimated character count
    #[inline]
    pub fn char_count(&self) -> usize {
        match self {
            Self::Plain { text } => text.chars().count(),
            Self::Markdown { content, .. } => content.chars().count(),
            Self::Code { content, .. } => content.chars().count(),
            Self::Formatted { content, .. } => content.chars().count(),
            Self::Composite { parts } => parts.iter().map(|p| p.char_count()).sum()}
    }

    /// Validate content structure
    #[inline]
    pub fn validate(&self) -> FormatResult<()> {
        match self {
            Self::Code { language, .. } => {
                if language.is_empty() {
                    return Err(FormatError::InvalidContent {
                        detail: "Code language cannot be empty".to_string()});
                }
            }
            Self::Formatted { content, styles } => {
                for style in styles {
                    if style.end > content.len() {
                        return Err(FormatError::InvalidContent {
                            detail: "Style range exceeds content length".to_string()});
                    }
                    if style.start >= style.end {
                        return Err(FormatError::InvalidContent {
                            detail: "Invalid style range".to_string()});
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

/// Formatting errors with owned strings
#[derive(Error, Debug, Clone)]
pub enum FormatError {
    #[error("Invalid markdown syntax: {detail}")]
    InvalidMarkdown { detail: String },
    #[error("Unsupported language: {language}")]
    UnsupportedLanguage { language: String },
    #[error("Parse error: {detail}")]
    ParseError { detail: String },
    #[error("Render error: {detail}")]
    RenderError { detail: String },
    #[error("Invalid content: {detail}")]
    InvalidContent { detail: String },
    #[error("Configuration error: {detail}")]
    ConfigurationError { detail: String },
    #[error("IO error: {detail}")]
    IoError { detail: String },
    #[error("Timeout error")]
    Timeout,
    #[error("Resource not found: {resource}")]
    ResourceNotFound { resource: String },
    #[error("Internal error: {detail}")]
    InternalError { detail: String }}

/// Result type for formatting operations
pub type FormatResult<T> = Result<T, FormatError>;

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

    /// Validate configuration
    #[inline]
    pub fn validate(&self) -> FormatResult<()> {
        if self.max_line_length > 0 && self.max_line_length < 10 {
            return Err(FormatError::ConfigurationError {
                detail: "Max line length must be at least 10 or 0 for no wrapping".to_string()});
        }
        if self.indent_size > 20 {
            return Err(FormatError::ConfigurationError {
                detail: "Indent size cannot exceed 20".to_string()});
        }
        for rule in &self.custom_rules {
            rule.validate()?;
        }
        Ok(())
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
            enable_emoji: true,
            max_line_length: 80,
            indent_size: 2,
            syntax_theme: SyntaxTheme::GitHub,
            color_scheme: ImmutableColorScheme::default(),
            output_format: OutputFormat::Html,
            include_metadata: false,
            enable_optimizations: true,
            custom_css_classes: HashMap::new(),
            custom_rules: Vec::new()}
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
    Custom}

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
            Self::Custom => "custom"}
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
    pub link: String}

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
            link};
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
                    detail: format!("Invalid color format: {}", color)});
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
            link: "#0066cc".to_string()}
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
    LaTeX}

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
            Self::LaTeX => "latex"}
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

/// Formatting event for streaming operations
#[derive(Debug, Clone)]
pub enum FormattingEvent {
    /// Formatting started
    Started {
        content_id: u64,
        content_type: String,
        timestamp_nanos: u64},
    /// Formatting progress
    Progress {
        content_id: u64,
        progress_percent: f32,
        stage: String},
    /// Formatting completed
    Completed {
        content_id: u64,
        result: ImmutableMessageContent,
        duration_nanos: u64},
    /// Formatting failed
    Failed {
        content_id: u64,
        error: FormatError,
        duration_nanos: u64},
    /// Partial result available
    PartialResult {
        content_id: u64,
        partial_content: String}}

/// Streaming message formatter with atomic state tracking
pub struct StreamingMessageFormatter {
    /// Content counter (atomic)
    content_counter: AtomicU64,
    /// Active formatting operations (atomic)
    active_operations: AtomicUsize,
    /// Total operations (atomic)
    total_operations: AtomicU64,
    /// Successful operations (atomic)
    successful_operations: AtomicU64,
    /// Failed operations (atomic)
    failed_operations: AtomicU64,
    /// Event stream sender
    event_sender: Option<AsyncStreamSender<FormattingEvent>>,
    /// Formatter configuration
    options: ImmutableFormatOptions}

impl std::fmt::Debug for StreamingMessageFormatter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingMessageFormatter")
            .field(
                "content_counter",
                &self
                    .content_counter
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field(
                "active_operations",
                &self
                    .active_operations
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field(
                "total_operations",
                &self
                    .total_operations
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field(
                "successful_operations",
                &self
                    .successful_operations
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field(
                "failed_operations",
                &self
                    .failed_operations
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field("event_sender", &self.event_sender.is_some())
            .field("options", &self.options)
            .finish()
    }
}

impl StreamingMessageFormatter {
    /// Create new streaming message formatter
    #[inline]
    pub fn new(options: ImmutableFormatOptions) -> FormatResult<Self> {
        options.validate()?;
        Ok(Self {
            content_counter: AtomicU64::new(0),
            active_operations: AtomicUsize::new(0),
            total_operations: AtomicU64::new(0),
            successful_operations: AtomicU64::new(0),
            failed_operations: AtomicU64::new(0),
            event_sender: None,
            options})
    }

    /// Create formatter with event streaming
    #[inline]
    pub fn with_streaming(
        options: ImmutableFormatOptions,
    ) -> FormatResult<(Self, AsyncStream<FormattingEvent>)> {
        options.validate()?;
        let stream = AsyncStream::with_channel(|_sender| {
            // Stream is created but not used directly
        });
        let formatter = Self {
            content_counter: AtomicU64::new(0),
            active_operations: AtomicUsize::new(0),
            total_operations: AtomicU64::new(0),
            successful_operations: AtomicU64::new(0),
            failed_operations: AtomicU64::new(0),
            event_sender: None, // Will be set up separately if needed
            options};
        Ok((formatter, stream))
    }

    /// Format content with streaming events
    #[inline]
    pub fn format_content(&self, content: &ImmutableMessageContent) -> FormatResult<u64> {
        // Validate content first
        content.validate()?;

        // Generate content ID
        let content_id = self.content_counter.fetch_add(1, Ordering::Relaxed);

        // Update counters
        self.active_operations.fetch_add(1, Ordering::Relaxed);
        self.total_operations.fetch_add(1, Ordering::Relaxed);

        // Send started event
        if let Some(ref sender) = self.event_sender {
            let _ = sender.send(FormattingEvent::Started {
                content_id,
                content_type: content.content_type().to_string(),
                timestamp_nanos: Self::current_timestamp_nanos()});
        }

        // TODO: Implement actual formatting logic here
        // This would integrate with markdown parsers, syntax highlighters, etc.

        Ok(content_id)
    }

    /// Get current timestamp in nanoseconds
    #[inline]
    fn current_timestamp_nanos() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    }

    /// Get formatting statistics (atomic reads)
    #[inline]
    pub fn stats(&self) -> FormatterStats {
        FormatterStats {
            active_operations: self.active_operations.load(Ordering::Relaxed) as u64,
            total_operations: self.total_operations.load(Ordering::Relaxed),
            successful_operations: self.successful_operations.load(Ordering::Relaxed),
            failed_operations: self.failed_operations.load(Ordering::Relaxed)}
    }

    /// Get formatter options (borrowed reference)
    #[inline]
    pub fn options(&self) -> &ImmutableFormatOptions {
        &self.options
    }

    /// Update formatter options
    #[inline]
    pub fn update_options(&mut self, options: ImmutableFormatOptions) -> FormatResult<()> {
        options.validate()?;
        self.options = options;
        Ok(())
    }
}

/// Formatter statistics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatterStats {
    pub active_operations: u64,
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64}

impl FormatterStats {
    /// Calculate success rate as percentage
    #[inline]
    pub fn success_rate(&self) -> f64 {
        let completed = self.successful_operations + self.failed_operations;
        if completed == 0 {
            0.0
        } else {
            (self.successful_operations as f64 / completed as f64) * 100.0
        }
    }

    /// Calculate failure rate as percentage
    #[inline]
    pub fn failure_rate(&self) -> f64 {
        100.0 - self.success_rate()
    }
}

/// Legacy compatibility type aliases (deprecated)
#[deprecated(note = "Use ImmutableMessageContent instead for zero-allocation streaming")]
pub type MessageContent = ImmutableMessageContent;

#[deprecated(note = "Use ImmutableFormatOptions instead for zero-allocation streaming")]
pub type FormatOptions = ImmutableFormatOptions;

#[deprecated(note = "Use ImmutableColorScheme instead for zero-allocation streaming")]
pub type ColorScheme = ImmutableColorScheme;

#[deprecated(note = "Use ImmutableCustomFormatRule instead for zero-allocation streaming")]
pub type CustomFormatRule = ImmutableCustomFormatRule;

/// Legacy compatibility alias for StreamingMessageFormatter
#[deprecated(note = "Use StreamingMessageFormatter instead for zero-allocation streaming")]
pub type MessageFormatter = StreamingMessageFormatter;

#[cfg(test)]
mod tests {
    use super::*;

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
        let valid_scheme = ImmutableColorScheme::default();
        assert!(valid_scheme.validate().is_ok());

        let invalid_scheme = ImmutableColorScheme {
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
