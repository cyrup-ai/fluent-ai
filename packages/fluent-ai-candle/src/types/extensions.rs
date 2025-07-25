use std::time::{SystemTime, UNIX_EPOCH};

use fluent_ai_domain::{Message, MessageRole};

pub trait MessageExt {
    fn user(content: impl Into<String>) -> Self;
    fn assistant(content: impl Into<String>) -> Self;
    fn system(content: impl Into<String>) -> Self;
}

impl MessageExt for Message {
    #[inline]
    fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            id: None,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .ok(),
        }
    }

    #[inline]
    fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            id: None,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .ok(),
        }
    }

    #[inline]
    fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
            id: None,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .ok(),
        }
    }
}

pub trait RoleExt {
    fn as_str(&self) -> &'static str;
}

impl RoleExt for MessageRole {
    #[inline]
    fn as_str(&self) -> &'static str {
        match self {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        }
    }
}
