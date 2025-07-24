/// Control flow enum for chat closures - provides explicit conversation management
#[derive(Debug, Clone, PartialEq)]
pub enum ChatLoop {
    /// Continue the conversation with a response message
    /// All formatting and rendering happens automatically by the builder
    Reprompt(String),

    /// Request user input with an optional prompt message
    /// Builder handles stdin/stdout automatically behind the scenes
    UserPrompt(Option<String>),

    /// Terminate the chat loop
    Break,
}

impl ChatLoop {
    /// Helper method to check if the loop should continue
    pub fn should_continue(&self) -> bool {
        matches!(self, ChatLoop::Reprompt(_) | ChatLoop::UserPrompt(_))
    }

    /// Extract the response message if this is a Reprompt
    pub fn message(&self) -> Option<&str> {
        match self {
            ChatLoop::Reprompt(msg) => Some(msg),
            ChatLoop::UserPrompt(_) | ChatLoop::Break => None,
        }
    }

    /// Extract the user prompt message if this is a UserPrompt
    pub fn user_prompt(&self) -> Option<&str> {
        match self {
            ChatLoop::UserPrompt(Some(prompt)) => Some(prompt),
            ChatLoop::UserPrompt(None) | ChatLoop::Reprompt(_) | ChatLoop::Break => None,
        }
    }

    /// Check if this requests user input
    pub fn needs_user_input(&self) -> bool {
        matches!(self, ChatLoop::UserPrompt(_))
    }
}

impl From<String> for ChatLoop {
    fn from(msg: String) -> Self {
        ChatLoop::Reprompt(msg)
    }
}

impl From<&str> for ChatLoop {
    fn from(msg: &str) -> Self {
        ChatLoop::Reprompt(msg.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_loop_reprompt() {
        let loop_val = ChatLoop::Reprompt("Hello".to_string());
        assert!(loop_val.should_continue());
        assert_eq!(loop_val.message(), Some("Hello"));
        assert_eq!(loop_val.user_prompt(), None);
        assert!(!loop_val.needs_user_input());
    }

    #[test]
    fn test_chat_loop_user_prompt_with_message() {
        let loop_val = ChatLoop::UserPrompt(Some("Enter your choice:".to_string()));
        assert!(loop_val.should_continue());
        assert_eq!(loop_val.message(), None);
        assert_eq!(loop_val.user_prompt(), Some("Enter your choice:"));
        assert!(loop_val.needs_user_input());
    }

    #[test]
    fn test_chat_loop_user_prompt_without_message() {
        let loop_val = ChatLoop::UserPrompt(None);
        assert!(loop_val.should_continue());
        assert_eq!(loop_val.message(), None);
        assert_eq!(loop_val.user_prompt(), None);
        assert!(loop_val.needs_user_input());
    }

    #[test]
    fn test_chat_loop_break() {
        let loop_val = ChatLoop::Break;
        assert!(!loop_val.should_continue());
        assert_eq!(loop_val.message(), None);
        assert_eq!(loop_val.user_prompt(), None);
        assert!(!loop_val.needs_user_input());
    }

    #[test]
    fn test_from_string() {
        let loop_val: ChatLoop = "test message".into();
        assert_eq!(loop_val, ChatLoop::Reprompt("test message".to_string()));
    }

    #[test]
    fn test_from_str() {
        let loop_val: ChatLoop = String::from("test message").into();
        assert_eq!(loop_val, ChatLoop::Reprompt("test message".to_string()));
    }
}
