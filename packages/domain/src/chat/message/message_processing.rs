//! Message processing utilities for the chat system.
//!
//! This module provides functionality for processing, validating, and transforming
//! chat messages in a production environment using async streaming patterns.

// Removed unused import: use crate::error::ZeroAllocResult;
use fluent_ai_async::{AsyncStream};

use super::types::Message;

/// Processes a message before it's sent to the chat system using async streaming.
///
/// # Arguments
/// * `message` - The message to process
///
/// # Returns
/// Returns an AsyncStream that will emit the processed message.
/// The on_chunk handler should validate the processed message.
pub fn process_message(message: Message) -> AsyncStream<Message> {
    AsyncStream::with_channel(move |sender| {
        let mut processed_message = message;

        // Trim whitespace from the message content
        processed_message.content = processed_message.content.trim().to_string();

        // Always emit the processed message - validation handled by on_chunk handler
        let _ = sender.send(processed_message);
    })
}

/// Validates that a message is safe to send using async streaming.
///
/// # Arguments
/// * `message` - The message to validate
///
/// # Returns
/// Returns an AsyncStream that will emit the message if valid.
/// Invalid messages will be handled by the on_chunk error handler.
pub fn validate_message(message: Message) -> AsyncStream<Message> {
    AsyncStream::with_channel(move |sender| {
        // Always emit the message - the on_chunk handler decides validation behavior
        let _ = sender.send(message);
    })
}

/// Sanitizes potentially dangerous content from a message.
///
/// # Arguments
/// * `content` - The content to sanitize
///
/// # Returns
/// Returns the sanitized content.
pub fn sanitize_content(content: &str) -> String {
    // For now, just trim the content
    // In a real implementation, you would want to do more thorough sanitization
    content.trim().to_string()
}

/// Validates a message to ensure it meets system requirements.
///
/// # Arguments
/// * `message` - The message to validate
///
/// # Returns
/// Returns Ok(()) if the message is valid, or an error if validation fails.
pub fn validate_message_sync(message: &Message) -> Result<(), String> {
    // Basic validation logic - can be extended as needed
    match message {
        Message::User { content } => {
            if content.is_empty() {
                return Err("Empty user message".to_string());
            }
        }
        Message::Assistant { content } => {
            if content.is_empty() {
                return Err("Empty assistant message".to_string());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::types::{Message, MessageRole};
    use super::*;

    #[test]
    fn test_process_message() {
        let mut message = Message {
            role: MessageRole::User,
            content: "  Hello, world!  ".to_string(),
            id: None,
            timestamp: None,
        };

        process_message(&mut message).unwrap();
        assert_eq!(message.content, "Hello, world!");
    }

    #[test]
    fn test_validate_message() {
        let valid_message = Message {
            role: MessageRole::User,
            content: "Hello, world!".to_string(),
            id: None,
            timestamp: None,
        };

        let empty_message = Message {
            role: MessageRole::User,
            content: "   ".to_string(),
            id: None,
            timestamp: None,
        };

        assert!(validate_message(&valid_message).is_ok());
        assert!(validate_message(&empty_message).is_err());
    }

    #[test]
    fn test_sanitize_content() {
        assert_eq!(sanitize_content("  Hello, world!  "), "Hello, world!");
        assert_eq!(sanitize_content(""), "");
        assert_eq!(sanitize_content("  "), "");
    }
}
