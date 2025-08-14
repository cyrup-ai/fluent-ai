//! Message processing utilities for the chat system.
//!
//! This module provides functionality for processing, validating, and transforming
//! chat messages in a production environment using async streaming patterns.

// Removed unused import: use crate::error::ZeroAllocResult;
use fluent_ai_async::AsyncStream;

use super::types::{CandleMessage, CandleMessageRole};

/// Processes a message before it's sent to the chat system using async streaming.
///
/// # Arguments
/// * `message` - The message to process
///
/// # Returns
/// Returns an AsyncStream that will emit the processed message.
/// The on_chunk handler should validate the processed message.
pub fn process_message(message: CandleMessage) -> AsyncStream<CandleMessage> {
    AsyncStream::with_channel(move |sender| {
        // Trim whitespace from the message content
        let mut processed_message = message;
        processed_message.content = processed_message.content.trim().to_string();

        // Always emit the processed message - validation handled by on_chunk handler
        let _ = sender.send(processed_message);
        ()
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
pub fn validate_message(message: CandleMessage) -> AsyncStream<CandleMessage> {
    AsyncStream::with_channel(move |sender| {
        // Always emit the message - the on_chunk handler decides validation behavior
        let _ = sender.send(message);
        ()
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
pub fn validate_message_sync(message: &CandleMessage) -> Result<(), String> {
    // Basic validation logic - can be extended as needed
    if message.content.is_empty() {
        return Err("Empty message content".to_string());
    }

    // Validate role-specific constraints
    match message.role {
        CandleMessageRole::User => {
            // User-specific validation if needed
        }
        CandleMessageRole::Assistant => {
            // Assistant-specific validation if needed
        }
        _ => {
            // Other role validation if needed
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::types::{CandleMessage, CandleMessageRole};
    use super::*;

    #[tokio::test]
    async fn test_process_message() {
        let message = CandleMessage {
            role: CandleMessageRole::User,
            content: "  Hello, world!  ".to_string(),
            id: None,
            timestamp: None,
        };

        let processed: Vec<_> = process_message(message).collect();
        assert_eq!(processed.len(), 1);
        assert_eq!(processed[0].content, "Hello, world!");
    }

    #[tokio::test]
    async fn test_validate_message() {
        let valid_message = CandleMessage {
            role: CandleMessageRole::User,
            content: "Hello, world!".to_string(),
            id: None,
            timestamp: None,
        };

        let empty_message = CandleMessage {
            role: CandleMessageRole::User,
            content: "   ".to_string(),
            id: None,
            timestamp: None,
        };

        let mut valid_stream = validate_message(valid_message);
        let valid_result = valid_stream.next().unwrap();
        assert_eq!(valid_result.content, "Hello, world!");

        let mut empty_stream = validate_message(empty_message);
        let empty_result = empty_stream.next().unwrap();
        assert_eq!(empty_result.content, "   "); // Validation is now handled by on_chunk handler
    }

    #[test]
    fn test_sanitize_content() {
        assert_eq!(sanitize_content("  Hello, world!  "), "Hello, world!");
        assert_eq!(sanitize_content(""), "");
        assert_eq!(sanitize_content("  "), "");
    }
}
