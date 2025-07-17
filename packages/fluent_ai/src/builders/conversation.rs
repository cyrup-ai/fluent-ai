//! Conversation builder implementations
//!
//! All conversation construction logic and builder patterns.

use fluent_ai_domain::{AsyncTask, spawn_async, ZeroOneOrMany};
use fluent_ai_domain::conversation::Conversation;
use std::fmt;

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
            },
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

/// Builder for creating and configuring conversations
pub struct ConversationBuilder {
    initial_message: Option<String>,
    error_handler: Option<Box<dyn FnMut(String) + Send + 'static>>,
}

/// Builder with error handler - exposes terminal methods
pub struct ConversationBuilderWithHandler {
    initial_message: Option<String>,
    error_handler: Box<dyn FnMut(String) + Send + 'static>,
    result_handler: Option<Box<dyn FnOnce(ConversationImpl) -> ConversationImpl + Send + 'static>>,
    chunk_handler: Option<Box<dyn FnMut(ConversationImpl) -> ConversationImpl + Send + 'static>>,
}

impl ConversationBuilder {
    /// Create a new conversation builder
    pub fn new() -> Self {
        Self {
            initial_message: None,
            error_handler: None,
        }
    }

    /// Set the initial message for the conversation
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.initial_message = Some(message.into());
        self
    }

    /// Set error handler - required before terminal methods
    pub fn on_error<F>(self, handler: F) -> ConversationBuilderWithHandler
    where
        F: FnMut(String) + Send + 'static,
    {
        ConversationBuilderWithHandler {
            initial_message: self.initial_message,
            error_handler: Box::new(handler),
            result_handler: None,
            chunk_handler: None,
        }
    }
}

impl ConversationBuilderWithHandler {
    /// Terminal method - create conversation
    pub fn build(self) -> ConversationImpl {
        let initial_message = self.initial_message.unwrap_or_else(|| "".to_string());
        ConversationImpl::new(initial_message)
    }

    /// Terminal method - create conversation with async handling
    pub fn build_async(self) -> AsyncTask<ConversationImpl> {
        let initial_message = self.initial_message.unwrap_or_else(|| "".to_string());
        spawn_async(async move {
            ConversationImpl::new(initial_message)
        })
    }
}