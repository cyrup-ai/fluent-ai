//! Chat domain module
//!
//! Provides chat-related functionality

use crate::AsyncTask;

/// Chat completion function
pub async fn complete_chat(_input: &str) -> Result<String, String> {
    // Placeholder implementation
    Ok("Chat response".to_string())
}

/// Chat stream function
pub fn stream_chat(_input: &str) -> AsyncTask<String> {
    crate::spawn_async(async move {
        "Streaming chat response".to_string()
    })
}

/// Chat loop function
pub async fn chat_loop(_input: &str) -> Result<String, String> {
    // Placeholder implementation
    Ok("Chat loop response".to_string())
}

/// Chat loop control flow module
pub mod chat_loop {
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
}