use serde::{Deserialize, Serialize};

use crate::domain::chat::message::types::CandleMessageRole as MessageRole;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prompt {
    pub content: String,
    #[serde(default = "default_role")]
    pub role: MessageRole}

fn default_role() -> MessageRole {
    MessageRole::User
}

impl Prompt {
    pub fn new(content: impl Into<String>) -> Self {
        Prompt {
            content: content.into(),
            role: MessageRole::User}
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

// PromptBuilder moved to fluent-ai/src/builders/prompt.rs

impl Into<String> for Prompt {
    fn into(self) -> String {
        self.content
    }
}

// PromptBuilder implementation moved to fluent-ai/src/builders/prompt.rs

/// Candle-prefixed type alias for domain compatibility
pub type CandlePrompt = Prompt;
