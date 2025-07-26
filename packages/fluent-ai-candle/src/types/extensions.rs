//! Message and role extension traits for convenient message creation
//!
//! This module provides extension traits for `Message` and `MessageRole` types from
//! the fluent_ai_domain crate, adding convenient factory methods and utility functions
//! for common message operations.
//!
//! # Key Features
//!
//! - **Factory Methods**: Create messages with specific roles using simple builder patterns
//! - **Automatic Timestamps**: Messages are automatically timestamped with Unix epoch seconds
//! - **Type Conversion**: Flexible content input accepting any type implementing `Into<String>`
//! - **Zero Allocation**: Uses inline functions for optimal performance
//!
//! # Usage Examples
//!
//! ```rust
//! use fluent_ai_candle::types::extensions::{MessageExt, RoleExt};
//! use fluent_ai_domain::{Message, MessageRole};
//!
//! // Create messages with convenient factory methods
//! let user_msg = Message::user("Hello, how are you?");
//! let assistant_msg = Message::assistant("I'm doing well, thank you!");
//! let system_msg = Message::system("You are a helpful assistant.");
//!
//! // Convert roles to string representations
//! let role_str = MessageRole::User.as_str(); // "user"
//! ```

use std::time::{SystemTime, UNIX_EPOCH};

use fluent_ai_domain::{Message, MessageRole};

/// Extension trait for Message providing convenient factory methods
///
/// This trait extends the `Message` type with factory methods for creating messages
/// with specific roles. Each method automatically sets the appropriate role and
/// current timestamp while allowing flexible content input.
///
/// # Performance
///
/// All methods are marked `#[inline]` for zero-cost abstractions and optimal
/// performance in hot paths. The timestamp calculation uses system time with
/// graceful handling of potential errors.
///
/// # Timestamp Behavior
///
/// Timestamps are automatically set to the current Unix epoch time in seconds.
/// If system time is unavailable or invalid, the timestamp will be `None`.
///
/// # Examples
///
/// ```rust
/// use fluent_ai_candle::types::extensions::MessageExt;
/// use fluent_ai_domain::Message;
///
/// // Create messages with different roles
/// let user_message = Message::user("What's the weather like?");
/// let bot_response = Message::assistant("It's sunny and 75°F today.");
/// let instructions = Message::system("Be helpful and concise.");
///
/// // Content can be any type implementing Into<String>
/// let from_string = Message::user(String::from("Hello"));
/// let from_str = Message::user("Hello");
/// ```
pub trait MessageExt {
    /// Create a user message with the given content
    ///
    /// Creates a new message with `MessageRole::User` and the provided content.
    /// The message is automatically timestamped with the current Unix epoch time.
    ///
    /// # Arguments
    ///
    /// * `content` - The message content, accepting any type that implements `Into<String>`
    ///
    /// # Returns
    ///
    /// A new `Message` with user role, provided content, no ID, and current timestamp
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fluent_ai_candle::types::extensions::MessageExt;
    /// use fluent_ai_domain::Message;
    ///
    /// let message = Message::user("Hello, I need help with something.");
    /// assert_eq!(message.role, MessageRole::User);
    /// assert_eq!(message.content, "Hello, I need help with something.");
    /// assert!(message.timestamp.is_some());
    /// ```
    fn user(content: impl Into<String>) -> Self;
    
    /// Create an assistant message with the given content
    ///
    /// Creates a new message with `MessageRole::Assistant` and the provided content.
    /// This is typically used for AI-generated responses in conversations.
    ///
    /// # Arguments
    ///
    /// * `content` - The response content, accepting any type that implements `Into<String>`
    ///
    /// # Returns
    ///
    /// A new `Message` with assistant role, provided content, no ID, and current timestamp
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fluent_ai_candle::types::extensions::MessageExt;
    /// use fluent_ai_domain::Message;
    ///
    /// let response = Message::assistant("I'd be happy to help you with that!");
    /// assert_eq!(response.role, MessageRole::Assistant);
    /// ```
    fn assistant(content: impl Into<String>) -> Self;
    
    /// Create a system message with the given content
    ///
    /// Creates a new message with `MessageRole::System` and the provided content.
    /// System messages are typically used for providing context, instructions,
    /// or configuration to AI models.
    ///
    /// # Arguments
    ///
    /// * `content` - The system instruction content, accepting any type that implements `Into<String>`
    ///
    /// # Returns
    ///
    /// A new `Message` with system role, provided content, no ID, and current timestamp
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fluent_ai_candle::types::extensions::MessageExt;
    /// use fluent_ai_domain::Message;
    ///
    /// let system_prompt = Message::system("You are a helpful coding assistant.");
    /// assert_eq!(system_prompt.role, MessageRole::System);
    /// ```
    fn system(content: impl Into<String>) -> Self;
}

/// Implementation of MessageExt for the Message type
///
/// Provides concrete implementations of the factory methods for creating
/// messages with different roles. All methods use inline optimization
/// and consistent timestamp generation.
impl MessageExt for Message {
    /// Create a user message with automatic timestamp
    ///
    /// Implementation of the user message factory method. Sets the role to User,
    /// converts the content, and generates a current timestamp.
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

    /// Create an assistant message with automatic timestamp
    ///
    /// Implementation of the assistant message factory method. Sets the role to Assistant,
    /// converts the content, and generates a current timestamp.
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

    /// Create a system message with automatic timestamp
    ///
    /// Implementation of the system message factory method. Sets the role to System,
    /// converts the content, and generates a current timestamp.
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

/// Extension trait for MessageRole providing string conversion utilities
///
/// This trait extends the `MessageRole` enum with utility methods for converting
/// roles to their string representations. Useful for serialization, logging,
/// and API interactions where string representations are required.
///
/// # Performance
///
/// The `as_str` method returns static string slices, ensuring zero allocation
/// and optimal performance for role-to-string conversions.
///
/// # String Representations
///
/// The string representations follow common conventions:
/// - `MessageRole::System` → `"system"`
/// - `MessageRole::User` → `"user"`
/// - `MessageRole::Assistant` → `"assistant"`
/// - `MessageRole::Tool` → `"tool"`
///
/// # Examples
///
/// ```rust
/// use fluent_ai_candle::types::extensions::RoleExt;
/// use fluent_ai_domain::MessageRole;
///
/// assert_eq!(MessageRole::User.as_str(), "user");
/// assert_eq!(MessageRole::Assistant.as_str(), "assistant");
/// assert_eq!(MessageRole::System.as_str(), "system");
/// assert_eq!(MessageRole::Tool.as_str(), "tool");
/// ```
pub trait RoleExt {
    /// Convert the message role to its string representation
    ///
    /// Returns a static string slice representing the role. This is useful
    /// for serialization, API calls, and human-readable output.
    ///
    /// # Returns
    ///
    /// A static string slice (`&'static str`) representing the role:
    /// - System role returns `"system"`
    /// - User role returns `"user"`
    /// - Assistant role returns `"assistant"`
    /// - Tool role returns `"tool"`
    ///
    /// # Performance
    ///
    /// This method has zero allocation overhead as it returns static strings.
    /// The inline optimization ensures minimal call overhead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fluent_ai_candle::types::extensions::RoleExt;
    /// use fluent_ai_domain::MessageRole;
    ///
    /// let role = MessageRole::User;
    /// let role_string = role.as_str();
    /// println!("Message role: {}", role_string); // "Message role: user"
    /// ```
    fn as_str(&self) -> &'static str;
}

/// Implementation of RoleExt for the MessageRole enum
///
/// Provides the concrete implementation for converting MessageRole variants
/// to their string representations using pattern matching.
impl RoleExt for MessageRole {
    /// Convert MessageRole to string representation
    ///
    /// Maps each MessageRole variant to its corresponding string representation
    /// using exhaustive pattern matching to ensure all variants are handled.
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
