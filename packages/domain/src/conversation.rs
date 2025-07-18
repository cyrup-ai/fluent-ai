//! Conversation domain trait and implementation
//!
//! Provides trait-based conversation management with builder pattern

use std::fmt;

use crate::ZeroOneOrMany;
use crate::{AsyncTask, spawn_async};

/// Core conversation trait for managing message history
pub trait Conversation: Send + Sync + fmt::Debug + Clone {
    /// Get the latest user message
    fn latest_user_message(&self) -> &str;

    /// Add a new user message to the conversation
    fn add_user_message(&mut self, message: impl Into<String>);

    /// Add an assistant response to the conversation  
    fn add_assistant_response(&mut self, response: impl Into<String>);

    /// Get all messages in the conversation
    fn messages(&self) -> ZeroOneOrMany<String>;

    /// Get the number of messages in the conversation
    fn message_count(&self) -> usize;

    /// Create a new conversation with initial user message
    fn new(user_message: impl Into<String>) -> Self;
}

/// Default implementation of the Conversation trait
#[derive(Debug, Clone)]
pub struct ConversationImpl {
    messages: Vec<String>,
    latest_user_message: String,
}

impl Conversation for ConversationImpl {
    fn latest_user_message(&self) -> &str {
        &self.latest_user_message
    }

    fn add_user_message(&mut self, message: impl Into<String>) {
        let message = message.into();
        self.messages.push(message.clone());
        self.latest_user_message = message;
    }

    fn add_assistant_response(&mut self, response: impl Into<String>) {
        self.messages.push(response.into());
    }

    fn messages(&self) -> ZeroOneOrMany<String> {
        match self.messages.len() {
            0 => ZeroOneOrMany::None,
            1 => {
                if let Some(message) = self.messages.first() {
                    ZeroOneOrMany::One(message.clone())
                } else {
                    ZeroOneOrMany::None
                }
            }
            _ => ZeroOneOrMany::many(self.messages.clone()),
        }
    }

    fn message_count(&self) -> usize {
        self.messages.len()
    }

    fn new(user_message: impl Into<String>) -> Self {
        let message = user_message.into();
        Self {
            messages: vec![message.clone()],
            latest_user_message: message,
        }
    }
}

// Builder implementations moved to fluent_ai/src/builders/conversation.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversation_impl_new() {
        let conversation = ConversationImpl::new("Hello, world!");
        assert_eq!(conversation.latest_user_message(), "Hello, world!");
        assert_eq!(conversation.message_count(), 1);
    }

    #[test]
    fn test_conversation_impl_add_messages() {
        let mut conversation = ConversationImpl::new("First message");
        conversation.add_assistant_response("AI response");
        conversation.add_user_message("Second message");

        assert_eq!(conversation.latest_user_message(), "Second message");
        assert_eq!(conversation.message_count(), 3);
    }

    #[test]
    fn test_conversation_impl_messages_access() {
        let mut conversation = ConversationImpl::new("User 1");
        conversation.add_assistant_response("Assistant 1");
        conversation.add_user_message("User 2");

        let messages = conversation.messages();
        assert_eq!(messages.len(), 3);
        let messages_vec: Vec<_> = messages.into_iter().collect();
        assert_eq!(messages_vec[0], "User 1");
        assert_eq!(messages_vec[1], "Assistant 1");
        assert_eq!(messages_vec[2], "User 2");
    }

    #[test]
    fn test_conversation_builder() {
        let conversation = ConversationImpl::new("Test message");

        assert_eq!(conversation.latest_user_message(), "Test message");
        assert_eq!(conversation.message_count(), 1);
    }
}
