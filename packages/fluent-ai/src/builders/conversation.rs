use crate::domain::conversation::{ConversationImpl, Conversation as ConversationTrait};
use crate::domain::message::{Message, MessageRole};

/// Builder for Conversation objects
pub struct ConversationBuilder {
    messages: Vec<Message>,
}

impl ConversationBuilder {
    /// Create a new ConversationBuilder
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// Add a message to the conversation
    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// Add a user message to the conversation
    pub fn user_message(mut self, content: String) -> Self {
        let message = Message {
            role: MessageRole::User,
            content: content.into(),
            ..Default::default()
        };
        self.messages.push(message);
        self
    }

    /// Add an assistant message to the conversation
    pub fn assistant_message(mut self, content: String) -> Self {
        let message = Message {
            role: MessageRole::Assistant,
            content: content.into(),
            ..Default::default()
        };
        self.messages.push(message);
        self
    }

    /// Add a system message to the conversation
    pub fn system_message(mut self, content: String) -> Self {
        let message = Message {
            role: MessageRole::System,
            content: content.into(),
            ..Default::default()
        };
        self.messages.push(message);
        self
    }

    /// Build the Conversation object
    pub fn build(self) -> ConversationImpl {
        ConversationImpl::new(self.messages)
    }
}

impl Default for ConversationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversationImpl {
    /// Create a new ConversationBuilder
    pub fn builder() -> ConversationBuilder {
        ConversationBuilder::new()
    }
}