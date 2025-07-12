/// Conversation context provided to chat closures
/// Gives access to message history and latest user input
#[derive(Debug, Clone)]
pub struct Conversation {
    messages: Vec<String>,
    latest_user_message: String,
}

impl Conversation {
    /// Create a new conversation with initial user message
    pub fn new(user_message: impl Into<String>) -> Self {
        let message = user_message.into();
        Self {
            messages: vec![message.clone()],
            latest_user_message: message,
        }
    }
    
    /// Get the latest user message
    pub fn latest_user_message(&self) -> &str {
        &self.latest_user_message
    }
    
    /// Add a new user message to the conversation
    pub fn add_user_message(&mut self, message: impl Into<String>) {
        let message = message.into();
        self.messages.push(message.clone());
        self.latest_user_message = message;
    }
    
    /// Add an assistant response to the conversation
    pub fn add_assistant_response(&mut self, response: impl Into<String>) {
        self.messages.push(response.into());
    }
    
    /// Get all messages in the conversation
    pub fn messages(&self) -> &[String] {
        &self.messages
    }
    
    /// Get the number of messages in the conversation
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversation_new() {
        let conversation = Conversation::new("Hello, world!");
        assert_eq!(conversation.latest_user_message(), "Hello, world!");
        assert_eq!(conversation.message_count(), 1);
    }

    #[test]
    fn test_add_messages() {
        let mut conversation = Conversation::new("First message");
        conversation.add_assistant_response("AI response");
        conversation.add_user_message("Second message");
        
        assert_eq!(conversation.latest_user_message(), "Second message");
        assert_eq!(conversation.message_count(), 3);
    }

    #[test]
    fn test_messages_access() {
        let mut conversation = Conversation::new("User 1");
        conversation.add_assistant_response("Assistant 1");
        conversation.add_user_message("User 2");
        
        let messages = conversation.messages();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0], "User 1");
        assert_eq!(messages[1], "Assistant 1");
        assert_eq!(messages[2], "User 2");
    }
}
