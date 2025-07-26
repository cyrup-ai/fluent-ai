//! Chat loop and terminal UI utilities for interactive conversations
//!
//! This module provides the foundation for building interactive chat interfaces
//! with styled terminal output, conversation flow control, and user interaction
//! management. It includes the Cyrup.ai color theme and utilities for creating
//! polished command-line chat experiences.

use std::io;
use std::io::Write;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

/// Cyrup.ai brand color theme for consistent terminal styling
///
/// Provides a cohesive color palette for terminal UI elements including
/// primary branding colors, semantic colors for different message types,
/// and accessibility-conscious muted tones.
#[allow(dead_code)]
mod theme {
    use termcolor::Color;
    
    /// Primary brand color (cyan) for headers and key UI elements
    pub const PRIMARY: Color = Color::Cyan;
    /// Secondary brand color (blue) for supporting UI elements
    pub const SECONDARY: Color = Color::Blue;
    /// Accent color (magenta) for highlights and special emphasis
    pub const ACCENT: Color = Color::Magenta;
    /// Success state color (green) for positive feedback and confirmations
    pub const SUCCESS: Color = Color::Green;
    /// Warning state color (yellow) for cautionary messages and alerts
    pub const WARNING: Color = Color::Yellow;
    /// Error state color (red) for error messages and failures
    pub const ERROR: Color = Color::Red;
    /// Default text color (white) for primary content
    pub const TEXT: Color = Color::White;
    /// Muted text color (gray) for secondary information and metadata
    pub const MUTED: Color = Color::Rgb(150, 150, 150);
}

/// Print text to stdout with specified styling options
///
/// Outputs text with customizable color and formatting to the terminal.
/// Automatically resets terminal styling after output to prevent
/// formatting from bleeding into subsequent output.
///
/// # Arguments
/// * `text` - The text content to display
/// * `color` - Terminal color for the text
/// * `bold` - Whether to apply bold formatting
///
/// # Returns
/// Result indicating success or IO error
///
/// # Examples
/// ```ignore
/// print_styled("Success!", Color::Green, true)?;
/// print_styled("Normal text", Color::White, false)?;
/// ```
#[allow(dead_code)]
fn print_styled(text: &str, color: Color, bold: bool) -> io::Result<()> {
    let mut stdout = StandardStream::stdout(ColorChoice::Always);
    stdout.set_color(
        ColorSpec::new()
            .set_fg(Some(color))
            .set_bold(bold)
            .set_intense(true),
    )?;
    writeln!(&mut stdout, "{text}")?;
    stdout.reset()
}

/// Print a styled header with Cyrup.ai branding
///
/// Outputs a prominently styled header using the primary brand color
/// with bold and underlined formatting. Includes padding and separators
/// for visual hierarchy and readability.
///
/// # Arguments
/// * `title` - The header text to display
///
/// # Returns
/// Result indicating success or IO error
///
/// # Examples
/// ```ignore
/// print_header("Conversation Started")?;
/// // Outputs: === Conversation Started ===
/// ```
#[allow(dead_code)]
fn print_header(title: &str) -> io::Result<()> {
    let mut stdout = StandardStream::stdout(ColorChoice::Always);
    stdout.set_color(
        ColorSpec::new()
            .set_fg(Some(theme::PRIMARY))
            .set_bold(true)
            .set_underline(true),
    )?;
    writeln!(&mut stdout, "\n=== {title} ===\n")?;
    stdout.reset()
}

/// Control flow enumeration for managing interactive chat conversations
///
/// Provides explicit conversation flow control with three distinct states:
/// continuing with AI responses, requesting user input, or terminating the
/// conversation. This enables clean separation of conversation logic from
/// UI rendering and input handling.
///
/// # Design Philosophy
/// 
/// The ChatLoop enum follows a declarative approach where conversation
/// logic specifies *what* should happen next, while the builder handles
/// *how* it's implemented (formatting, rendering, input collection).
#[derive(Debug, Clone, PartialEq)]
pub enum ChatLoop {
    /// Continue the conversation with an AI response message
    ///
    /// The builder automatically handles formatting, styling, and rendering
    /// of the response. The message content is displayed using the configured
    /// theme and styling options.
    Reprompt(
        /// The response message content to display
        String
    ),

    /// Request user input with an optional custom prompt
    ///
    /// The builder handles stdin/stdout interaction, input validation,
    /// and prompt styling automatically. If no custom prompt is provided,
    /// a default prompt will be used.
    UserPrompt(
        /// Optional custom prompt message for user input
        Option<String>
    ),

    /// Terminate the chat conversation loop
    ///
    /// Signals that the conversation should end gracefully. The builder
    /// will perform any necessary cleanup and exit the interaction loop.
    Break
}

impl ChatLoop {
    /// Check if the conversation loop should continue
    ///
    /// Returns true for states that indicate the conversation should continue
    /// (Reprompt and UserPrompt), false for termination states (Break).
    ///
    /// # Returns
    /// true if the conversation should continue, false if it should terminate
    ///
    /// # Examples
    /// ```
    /// # use fluent_ai_candle::builders::candle_chat::candle_chat_loop::ChatLoop;
    /// assert!(ChatLoop::Reprompt("Hello".to_string()).should_continue());
    /// assert!(ChatLoop::UserPrompt(None).should_continue());
    /// assert!(!ChatLoop::Break.should_continue());
    /// ```
    pub fn should_continue(&self) -> bool {
        matches!(self, ChatLoop::Reprompt(_) | ChatLoop::UserPrompt(_))
    }

    /// Extract the response message content from a Reprompt variant
    ///
    /// Returns the message content if this is a Reprompt, None otherwise.
    /// Useful for accessing the AI response content without pattern matching.
    ///
    /// # Returns
    /// Some(message) for Reprompt variants, None for other variants
    ///
    /// # Examples
    /// ```
    /// # use fluent_ai_candle::builders::candle_chat::candle_chat_loop::ChatLoop;
    /// let reprompt = ChatLoop::Reprompt("AI response".to_string());
    /// assert_eq!(reprompt.message(), Some("AI response"));
    /// assert_eq!(ChatLoop::Break.message(), None);
    /// ```
    pub fn message(&self) -> Option<&str> {
        match self {
            ChatLoop::Reprompt(msg) => Some(msg),
            ChatLoop::UserPrompt(_) | ChatLoop::Break => None
        }
    }

    /// Extract the custom prompt message from a UserPrompt variant
    ///
    /// Returns the custom prompt message if this is a UserPrompt with a
    /// custom prompt, None otherwise. Note that UserPrompt(None) also
    /// returns None since no custom prompt was specified.
    ///
    /// # Returns
    /// Some(prompt) for UserPrompt with custom message, None otherwise
    ///
    /// # Examples
    /// ```
    /// # use fluent_ai_candle::builders::candle_chat::candle_chat_loop::ChatLoop;
    /// let custom_prompt = ChatLoop::UserPrompt(Some("Enter your name:".to_string()));
    /// assert_eq!(custom_prompt.user_prompt(), Some("Enter your name:"));
    /// 
    /// let default_prompt = ChatLoop::UserPrompt(None);
    /// assert_eq!(default_prompt.user_prompt(), None);
    /// ```
    pub fn user_prompt(&self) -> Option<&str> {
        match self {
            ChatLoop::UserPrompt(Some(prompt)) => Some(prompt),
            ChatLoop::UserPrompt(None) | ChatLoop::Reprompt(_) | ChatLoop::Break => None
        }
    }

    /// Check if this state requires user input
    ///
    /// Returns true for UserPrompt variants (both with and without custom
    /// prompts), false for response and termination states.
    ///
    /// # Returns
    /// true if user input is needed, false otherwise
    ///
    /// # Examples
    /// ```
    /// # use fluent_ai_candle::builders::candle_chat::candle_chat_loop::ChatLoop;
    /// assert!(ChatLoop::UserPrompt(None).needs_user_input());
    /// assert!(ChatLoop::UserPrompt(Some("Custom".to_string())).needs_user_input());
    /// assert!(!ChatLoop::Reprompt("Response".to_string()).needs_user_input());
    /// assert!(!ChatLoop::Break.needs_user_input());
    /// ```
    pub fn needs_user_input(&self) -> bool {
        matches!(self, ChatLoop::UserPrompt(_))
    }
}

impl From<String> for ChatLoop {
    /// Convert a String into a Reprompt ChatLoop variant
    ///
    /// Provides convenient conversion from owned strings to ChatLoop::Reprompt.
    /// This enables natural usage patterns where AI responses can be directly
    /// converted to conversation flow control.
    ///
    /// # Arguments
    /// * `msg` - The message string to convert
    ///
    /// # Returns
    /// ChatLoop::Reprompt containing the message
    ///
    /// # Examples
    /// ```
    /// # use fluent_ai_candle::builders::candle_chat::candle_chat_loop::ChatLoop;
    /// let response = "Hello, how can I help?".to_string();
    /// let chat_loop: ChatLoop = response.into();
    /// assert_eq!(chat_loop.message(), Some("Hello, how can I help?"));
    /// ```
    fn from(msg: String) -> Self {
        ChatLoop::Reprompt(msg)
    }
}

impl From<&str> for ChatLoop {
    /// Convert a string slice into a Reprompt ChatLoop variant
    ///
    /// Provides convenient conversion from string literals and borrowed strings
    /// to ChatLoop::Reprompt. The string slice is converted to an owned String.
    ///
    /// # Arguments
    /// * `msg` - The message string slice to convert
    ///
    /// # Returns
    /// ChatLoop::Reprompt containing the owned message string
    ///
    /// # Examples
    /// ```
    /// # use fluent_ai_candle::builders::candle_chat::candle_chat_loop::ChatLoop;
    /// let chat_loop: ChatLoop = "Hello, world!".into();
    /// assert_eq!(chat_loop.message(), Some("Hello, world!"));
    /// ```
    fn from(msg: &str) -> Self {
        ChatLoop::Reprompt(msg.to_string())
    }
}
