//! Immutable conversation management for chat interactions
//!
//! This module provides streaming-only, zero-allocation conversation management with
//! immutable message storage. All operations use borrowed data and atomic operations
//! for blazing-fast, lock-free performance.

use std::sync::atomic::{AtomicUsize, Ordering};

use fluent_ai_async::{AsyncStream, AsyncStreamSender};
// REMOVED: use fluent_ai_async::AsyncStream::with_channel;

use crate::ZeroOneOrMany;

/// Immutable message in a conversation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImmutableMessage {
    /// Message content (owned once, never mutated)
    pub content: String,
    /// Message role
    pub role: MessageRole,
    /// Message timestamp (nanoseconds since epoch)
    pub timestamp_nanos: u64,
    /// Message sequence number
    pub sequence: u64,
}

/// Message role in conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

impl ImmutableMessage {
    /// Create a new immutable message
    #[inline]
    pub fn new(content: impl Into<String>, role: MessageRole, sequence: u64) -> Self {
        Self {
            content: content.into(),
            role,
            timestamp_nanos: Self::current_timestamp_nanos(),
            sequence,
        }
    }

    /// Create user message
    #[inline]
    pub fn user(content: impl Into<String>, sequence: u64) -> Self {
        Self::new(content, MessageRole::User, sequence)
    }

    /// Create assistant message
    #[inline]
    pub fn assistant(content: impl Into<String>, sequence: u64) -> Self {
        Self::new(content, MessageRole::Assistant, sequence)
    }

    /// Create system message
    #[inline]
    pub fn system(content: impl Into<String>, sequence: u64) -> Self {
        Self::new(content, MessageRole::System, sequence)
    }

    /// Get current timestamp in nanoseconds
    #[inline]
    fn current_timestamp_nanos() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    }

    /// Get message content as borrowed string
    #[inline]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Check if message is from user
    #[inline]
    pub fn is_user(&self) -> bool {
        matches!(self.role, MessageRole::User)
    }

    /// Check if message is from assistant
    #[inline]
    pub fn is_assistant(&self) -> bool {
        matches!(self.role, MessageRole::Assistant)
    }

    /// Check if message is system message
    #[inline]
    pub fn is_system(&self) -> bool {
        matches!(self.role, MessageRole::System)
    }
}

/// Streaming conversation event
#[derive(Debug, Clone)]
pub enum ConversationEvent {
    /// New message added to conversation
    MessageAdded(ImmutableMessage),
    /// Conversation cleared
    Cleared,
    /// Conversation statistics updated
    StatsUpdated {
        total_messages: u64,
        user_messages: u64,
        assistant_messages: u64,
        system_messages: u64,
    },
}

/// Immutable conversation with streaming updates
pub struct StreamingConversation {
    /// Immutable message history (append-only)
    messages: Vec<ImmutableMessage>,
    /// Message sequence counter (atomic)
    sequence_counter: AtomicUsize,
    /// Total message count (atomic)
    total_messages: AtomicUsize,
    /// User message count (atomic)
    user_messages: AtomicUsize,
    /// Assistant message count (atomic)
    assistant_messages: AtomicUsize,
    /// System message count (atomic)
    system_messages: AtomicUsize,
    /// Event stream sender
    event_sender: Option<AsyncStreamSender<ConversationEvent>>,
}

impl std::fmt::Debug for StreamingConversation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingConversation")
            .field("messages", &self.messages)
            .field("sequence_counter", &self.sequence_counter.load(std::sync::atomic::Ordering::Relaxed))
            .field("total_messages", &self.total_messages.load(std::sync::atomic::Ordering::Relaxed))
            .field("user_messages", &self.user_messages.load(std::sync::atomic::Ordering::Relaxed))
            .field("assistant_messages", &self.assistant_messages.load(std::sync::atomic::Ordering::Relaxed))
            .field("system_messages", &self.system_messages.load(std::sync::atomic::Ordering::Relaxed))
            .field("event_sender", &self.event_sender.is_some())
            .finish()
    }
}

impl StreamingConversation {
    /// Create a new streaming conversation
    #[inline]
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            sequence_counter: AtomicUsize::new(0),
            total_messages: AtomicUsize::new(0),
            user_messages: AtomicUsize::new(0),
            assistant_messages: AtomicUsize::new(0),
            system_messages: AtomicUsize::new(0),
            event_sender: None,
        }
    }

    /// Create conversation with event streaming
    #[inline]
    pub fn with_streaming() -> (Self, AsyncStream<ConversationEvent>) {
        let (sender, stream) = AsyncStream::with_channel();
        let mut conversation = Self::new();
        conversation.event_sender = Some(sender);
        (conversation, stream)
    }

    /// Add user message (creates new immutable message)
    #[inline]
    pub fn add_user_message(&mut self, content: impl Into<String>) -> &ImmutableMessage {
        let sequence = self.sequence_counter.fetch_add(1, Ordering::Relaxed) as u64;
        let message = ImmutableMessage::user(content, sequence);

        self.messages.push(message.clone());
        self.total_messages.fetch_add(1, Ordering::Relaxed);
        self.user_messages.fetch_add(1, Ordering::Relaxed);

        // Send event if streaming enabled
        if let Some(ref sender) = self.event_sender {
            let _ = sender.send(ConversationEvent::MessageAdded(message.clone()));
        }

        // Safety: We just pushed a message, so messages cannot be empty
        match self.messages.last() {
            Some(msg) => msg,
            None => panic!("Critical error: message vector empty after push - possible memory corruption"),
        }
    }

    /// Add assistant message (creates new immutable message)
    #[inline]
    pub fn add_assistant_message(&mut self, content: impl Into<String>) -> &ImmutableMessage {
        let sequence = self.sequence_counter.fetch_add(1, Ordering::Relaxed) as u64;
        let message = ImmutableMessage::assistant(content, sequence);

        self.messages.push(message.clone());
        self.total_messages.fetch_add(1, Ordering::Relaxed);
        self.assistant_messages.fetch_add(1, Ordering::Relaxed);

        // Send event if streaming enabled
        if let Some(ref sender) = self.event_sender {
            let _ = sender.send(ConversationEvent::MessageAdded(message.clone()));
        }

        // Safety: We just pushed a message, so messages cannot be empty
        match self.messages.last() {
            Some(msg) => msg,
            None => panic!("Critical error: message vector empty after push - possible memory corruption"),
        }
    }

    /// Add system message (creates new immutable message)
    #[inline]
    pub fn add_system_message(&mut self, content: impl Into<String>) -> &ImmutableMessage {
        let sequence = self.sequence_counter.fetch_add(1, Ordering::Relaxed) as u64;
        let message = ImmutableMessage::system(content, sequence);

        self.messages.push(message.clone());
        self.total_messages.fetch_add(1, Ordering::Relaxed);
        self.system_messages.fetch_add(1, Ordering::Relaxed);

        // Send event if streaming enabled
        if let Some(ref sender) = self.event_sender {
            let _ = sender.send(ConversationEvent::MessageAdded(message.clone()));
        }

        // Safety: We just pushed a message, so messages cannot be empty
        match self.messages.last() {
            Some(msg) => msg,
            None => panic!("Critical error: message vector empty after push - possible memory corruption"),
        }
    }

    /// Get all messages as borrowed slice (zero allocation)
    #[inline]
    pub fn messages(&self) -> &[ImmutableMessage] {
        &self.messages
    }

    /// Get messages by role (zero allocation iterator)
    #[inline]
    pub fn messages_by_role(&self, role: MessageRole) -> impl Iterator<Item = &ImmutableMessage> {
        self.messages.iter().filter(move |msg| msg.role == role)
    }

    /// Get user messages (zero allocation iterator)
    #[inline]
    pub fn user_messages(&self) -> impl Iterator<Item = &ImmutableMessage> {
        self.messages_by_role(MessageRole::User)
    }

    /// Get assistant messages (zero allocation iterator)
    #[inline]
    pub fn assistant_messages(&self) -> impl Iterator<Item = &ImmutableMessage> {
        self.messages_by_role(MessageRole::Assistant)
    }

    /// Get system messages (zero allocation iterator)
    #[inline]
    pub fn system_messages(&self) -> impl Iterator<Item = &ImmutableMessage> {
        self.messages_by_role(MessageRole::System)
    }

    /// Get latest user message
    #[inline]
    pub fn latest_user_message(&self) -> Option<&ImmutableMessage> {
        self.user_messages().last()
    }

    /// Get latest assistant message
    #[inline]
    pub fn latest_assistant_message(&self) -> Option<&ImmutableMessage> {
        self.assistant_messages().last()
    }

    /// Get latest message of any type
    #[inline]
    pub fn latest_message(&self) -> Option<&ImmutableMessage> {
        self.messages.last()
    }

    /// Get message count (atomic read)
    #[inline]
    pub fn message_count(&self) -> usize {
        self.total_messages.load(Ordering::Relaxed)
    }

    /// Get user message count (atomic read)
    #[inline]
    pub fn user_message_count(&self) -> usize {
        self.user_messages.load(Ordering::Relaxed)
    }

    /// Get assistant message count (atomic read)
    #[inline]
    pub fn assistant_message_count(&self) -> usize {
        self.assistant_messages.load(Ordering::Relaxed)
    }

    /// Get system message count (atomic read)
    #[inline]
    pub fn system_message_count(&self) -> usize {
        self.system_messages.load(Ordering::Relaxed)
    }

    /// Check if conversation is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.message_count() == 0
    }

    /// Clear all messages (creates new empty conversation)
    #[inline]
    pub fn clear(&mut self) {
        self.messages.clear();
        self.sequence_counter.store(0, Ordering::Relaxed);
        self.total_messages.store(0, Ordering::Relaxed);
        self.user_messages.store(0, Ordering::Relaxed);
        self.assistant_messages.store(0, Ordering::Relaxed);
        self.system_messages.store(0, Ordering::Relaxed);

        // Send clear event if streaming enabled
        if let Some(ref sender) = self.event_sender {
            let _ = sender.send(ConversationEvent::Cleared);
        }
    }

    /// Get conversation statistics
    #[inline]
    pub fn stats(&self) -> ConversationStats {
        ConversationStats {
            total_messages: self.total_messages.load(Ordering::Relaxed) as u64,
            user_messages: self.user_messages.load(Ordering::Relaxed) as u64,
            assistant_messages: self.assistant_messages.load(Ordering::Relaxed) as u64,
            system_messages: self.system_messages.load(Ordering::Relaxed) as u64,
        }
    }

    /// Stream conversation statistics updates
    #[inline]
    pub fn stream_stats_updates(&self) {
        if let Some(ref sender) = self.event_sender {
            let stats = self.stats();
            let _ = sender.send(ConversationEvent::StatsUpdated {
                total_messages: stats.total_messages,
                user_messages: stats.user_messages,
                assistant_messages: stats.assistant_messages,
                system_messages: stats.system_messages,
            });
        }
    }
}

impl Default for StreamingConversation {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Conversation statistics snapshot
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConversationStats {
    pub total_messages: u64,
    pub user_messages: u64,
    pub assistant_messages: u64,
    pub system_messages: u64,
}

impl ConversationStats {
    /// Calculate user message percentage
    #[inline]
    pub fn user_percentage(&self) -> f64 {
        if self.total_messages == 0 {
            0.0
        } else {
            (self.user_messages as f64 / self.total_messages as f64) * 100.0
        }
    }

    /// Calculate assistant message percentage
    #[inline]
    pub fn assistant_percentage(&self) -> f64 {
        if self.total_messages == 0 {
            0.0
        } else {
            (self.assistant_messages as f64 / self.total_messages as f64) * 100.0
        }
    }

    /// Calculate system message percentage
    #[inline]
    pub fn system_percentage(&self) -> f64 {
        if self.total_messages == 0 {
            0.0
        } else {
            (self.system_messages as f64 / self.total_messages as f64) * 100.0
        }
    }
}

/// Legacy conversation trait for backward compatibility
pub trait Conversation: Send + Sync + std::fmt::Debug + Clone {
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

/// Legacy conversation implementation (deprecated - use StreamingConversation)
#[derive(Debug, Clone)]
pub struct ConversationImpl {
    messages: Vec<String>,
    latest_user_message: String,
}

impl Conversation for ConversationImpl {
    #[inline]
    fn latest_user_message(&self) -> &str {
        &self.latest_user_message
    }

    #[inline]
    fn add_user_message(&mut self, message: impl Into<String>) {
        let message = message.into();
        self.messages.push(message.clone());
        self.latest_user_message = message;
    }

    #[inline]
    fn add_assistant_response(&mut self, response: impl Into<String>) {
        self.messages.push(response.into());
    }

    #[inline]
    fn messages(&self) -> ZeroOneOrMany<String> {
        match self.messages.len() {
            0 => ZeroOneOrMany::None,
            1 => ZeroOneOrMany::One(self.messages[0].clone()),
            _ => ZeroOneOrMany::Many(self.messages.clone()),
        }
    }

    #[inline]
    fn message_count(&self) -> usize {
        self.messages.len()
    }

    #[inline]
    fn new(user_message: impl Into<String>) -> Self {
        let message = user_message.into();
        Self {
            latest_user_message: message.clone(),
            messages: vec![message],
        }
    }
}

/// Builder for creating streaming conversations
#[derive(Debug, Default)]
pub struct ConversationBuilder {
    enable_streaming: bool,
    initial_messages: Vec<(String, MessageRole)>,
}

impl ConversationBuilder {
    /// Create a new conversation builder
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable event streaming
    #[inline]
    pub fn with_streaming(mut self) -> Self {
        self.enable_streaming = true;
        self
    }

    /// Add initial user message
    #[inline]
    pub fn with_user_message(mut self, message: impl Into<String>) -> Self {
        self.initial_messages
            .push((message.into(), MessageRole::User));
        self
    }

    /// Add initial assistant message
    #[inline]
    pub fn with_assistant_message(mut self, message: impl Into<String>) -> Self {
        self.initial_messages
            .push((message.into(), MessageRole::Assistant));
        self
    }

    /// Add initial system message
    #[inline]
    pub fn with_system_message(mut self, message: impl Into<String>) -> Self {
        self.initial_messages
            .push((message.into(), MessageRole::System));
        self
    }

    /// Build the conversation
    #[inline]
    pub fn build(self) -> StreamingConversation {
        let mut conversation = if self.enable_streaming {
            let (conv, _stream) = StreamingConversation::with_streaming();
            conv
        } else {
            StreamingConversation::new()
        };

        // Add initial messages
        for (content, role) in self.initial_messages {
            match role {
                MessageRole::User => {
                    conversation.add_user_message(content);
                }
                MessageRole::Assistant => {
                    conversation.add_assistant_message(content);
                }
                MessageRole::System => {
                    conversation.add_system_message(content);
                }
            }
        }

        conversation
    }

    /// Build with streaming enabled, returning both conversation and event stream
    #[inline]
    pub fn build_with_stream(mut self) -> (StreamingConversation, AsyncStream<ConversationEvent>) {
        self.enable_streaming = true;
        let (mut conversation, stream) = StreamingConversation::with_streaming();

        // Add initial messages
        for (content, role) in self.initial_messages {
            match role {
                MessageRole::User => {
                    conversation.add_user_message(content);
                }
                MessageRole::Assistant => {
                    conversation.add_assistant_message(content);
                }
                MessageRole::System => {
                    conversation.add_system_message(content);
                }
            }
        }

        (conversation, stream)
    }
}
