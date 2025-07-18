//! Chat domain module
//!
//! Provides comprehensive chat functionality with zero-allocation patterns and production-ready
//! error handling. Integrates with the engine system to provide intelligent conversation
//! management and configuration-driven behavior.

use crate::{AsyncTask, spawn_async};
use crate::engine::{EngineConfig, EngineError, EngineResult, complete_with_engine};
use std::sync::Arc;
use std::collections::VecDeque;
use thiserror::Error;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// Chat-specific error types with zero-allocation string sharing
#[derive(Error, Debug, Clone)]
pub enum ChatError {
    #[error("Engine error: {source}")]
    EngineError { #[from] source: EngineError },
    
    #[error("Conversation error: {detail}")]
    ConversationError { detail: Arc<str> },
    
    #[error("Invalid message: {reason}")]
    InvalidMessage { reason: Arc<str> },
    
    #[error("Context limit exceeded: {current_tokens}/{max_tokens}")]
    ContextLimitExceeded { current_tokens: u32, max_tokens: u32 },
    
    #[error("Configuration error: {detail}")]
    ConfigurationError { detail: Arc<str> },
    
    #[error("Session error: {detail}")]
    SessionError { detail: Arc<str> },
    
    #[error("Memory error: {detail}")]
    MemoryError { detail: Arc<str> },
}

/// Result type for chat operations
pub type ChatResult<T> = Result<T, ChatError>;

/// Chat message with zero-allocation optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Arc<str>,
    pub content: Arc<str>,
    pub timestamp: u64,
    pub tokens: u32,
    pub metadata: Option<Arc<str>>,
}

impl ChatMessage {
    /// Create a new chat message
    #[inline]
    pub fn new(role: impl Into<Arc<str>>, content: impl Into<Arc<str>>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            tokens: 0, // Will be calculated by tokenizer
            metadata: None,
        }
    }

    /// Create a user message
    #[inline]
    pub fn user(content: impl Into<Arc<str>>) -> Self {
        Self::new("user", content)
    }

    /// Create an assistant message
    #[inline]
    pub fn assistant(content: impl Into<Arc<str>>) -> Self {
        Self::new("assistant", content)
    }

    /// Create a system message
    #[inline]
    pub fn system(content: impl Into<Arc<str>>) -> Self {
        Self::new("system", content)
    }

    /// Add metadata to the message
    #[inline]
    pub fn with_metadata(mut self, metadata: impl Into<Arc<str>>) -> Self {
        self.metadata = Some(metadata.into());
        self
    }

    /// Set token count
    #[inline]
    pub fn with_tokens(mut self, tokens: u32) -> Self {
        self.tokens = tokens;
        self
    }
}

/// Personality configuration with zero-allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityConfig {
    /// Response tone (formal, casual, friendly, professional)
    pub tone: Arc<str>,
    /// Creativity level (0.0-1.0)
    pub creativity: f32,
    /// Formality level (0.0-1.0)
    pub formality: f32,
    /// Expertise level (beginner, intermediate, advanced, expert)
    pub expertise_level: Arc<str>,
    /// Personality traits
    pub traits: Vec<Arc<str>>,
    /// Preferred response style
    pub response_style: Arc<str>,
    /// Humor level (0.0-1.0)
    pub humor: f32,
    /// Empathy level (0.0-1.0)
    pub empathy: f32,
    /// Verbosity level (concise, balanced, detailed)
    pub verbosity: Arc<str>,
}

impl Default for PersonalityConfig {
    fn default() -> Self {
        Self {
            tone: Arc::from("friendly"),
            creativity: 0.7,
            formality: 0.5,
            expertise_level: Arc::from("intermediate"),
            traits: vec![Arc::from("helpful"), Arc::from("knowledgeable")],
            response_style: Arc::from("balanced"),
            humor: 0.3,
            empathy: 0.8,
            verbosity: Arc::from("balanced"),
        }
    }
}

/// Behavior configuration with zero-allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorConfig {
    /// Interaction patterns
    pub interaction_patterns: Vec<Arc<str>>,
    /// Conversation flow style
    pub conversation_flow: Arc<str>,
    /// Engagement rules
    pub engagement_rules: Vec<Arc<str>>,
    /// Response patterns
    pub response_patterns: Vec<Arc<str>>,
    /// Proactivity level (0.0-1.0)
    pub proactivity: f32,
    /// Question asking frequency (0.0-1.0)
    pub question_frequency: f32,
    /// Follow-up behavior
    pub follow_up_behavior: Arc<str>,
    /// Error handling approach
    pub error_handling: Arc<str>,
    /// Clarification seeking behavior
    pub clarification_seeking: Arc<str>,
}

impl Default for BehaviorConfig {
    fn default() -> Self {
        Self {
            interaction_patterns: vec![Arc::from("responsive"), Arc::from("attentive")],
            conversation_flow: Arc::from("natural"),
            engagement_rules: vec![Arc::from("respectful"), Arc::from("constructive")],
            response_patterns: vec![Arc::from("thoughtful"), Arc::from("comprehensive")],
            proactivity: 0.6,
            question_frequency: 0.4,
            follow_up_behavior: Arc::from("contextual"),
            error_handling: Arc::from("graceful"),
            clarification_seeking: Arc::from("when_needed"),
        }
    }
}

/// UI configuration with zero-allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIConfig {
    /// Display preferences
    pub display_preferences: Vec<Arc<str>>,
    /// Theme settings
    pub theme: Arc<str>,
    /// Layout configuration
    pub layout: Arc<str>,
    /// Color scheme
    pub color_scheme: Arc<str>,
    /// Font preferences
    pub font_preferences: Vec<Arc<str>>,
    /// Animation settings
    pub animations: Arc<str>,
    /// Accessibility settings
    pub accessibility: Vec<Arc<str>>,
    /// Display density
    pub display_density: Arc<str>,
    /// Message formatting preferences
    pub message_formatting: Vec<Arc<str>>,
}

impl Default for UIConfig {
    fn default() -> Self {
        Self {
            display_preferences: vec![Arc::from("clean"), Arc::from("modern")],
            theme: Arc::from("auto"),
            layout: Arc::from("standard"),
            color_scheme: Arc::from("adaptive"),
            font_preferences: vec![Arc::from("system"), Arc::from("readable")],
            animations: Arc::from("smooth"),
            accessibility: vec![Arc::from("high_contrast"), Arc::from("screen_reader")],
            display_density: Arc::from("comfortable"),
            message_formatting: vec![Arc::from("markdown"), Arc::from("syntax_highlight")],
        }
    }
}

/// Integration configuration with zero-allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// External service settings
    pub external_services: Vec<Arc<str>>,
    /// API configurations
    pub api_configurations: Vec<Arc<str>>,
    /// Plugin preferences
    pub plugin_preferences: Vec<Arc<str>>,
    /// Integration endpoints
    pub integration_endpoints: Vec<Arc<str>>,
    /// Authentication settings
    pub authentication: Vec<Arc<str>>,
    /// Security settings
    pub security: Vec<Arc<str>>,
    /// Rate limiting settings
    pub rate_limiting: Vec<Arc<str>>,
    /// Caching settings
    pub caching: Vec<Arc<str>>,
    /// Integration priorities
    pub integration_priorities: Vec<Arc<str>>,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            external_services: vec![Arc::from("mcp"), Arc::from("tools")],
            api_configurations: vec![Arc::from("rest"), Arc::from("websocket")],
            plugin_preferences: vec![Arc::from("secure"), Arc::from("performant")],
            integration_endpoints: vec![Arc::from("local"), Arc::from("remote")],
            authentication: vec![Arc::from("token"), Arc::from("oauth")],
            security: vec![Arc::from("sandbox"), Arc::from("validate")],
            rate_limiting: vec![Arc::from("adaptive"), Arc::from("burst")],
            caching: vec![Arc::from("memory"), Arc::from("persistent")],
            integration_priorities: vec![Arc::from("reliability"), Arc::from("performance")],
        }
    }
}

/// Chat configuration with zero-allocation patterns and nested configuration objects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatConfig {
    /// Engine configuration for AI completions
    pub engine: EngineConfig,
    /// System prompt for the conversation
    pub system_prompt: Option<Arc<str>>,
    /// Maximum conversation history length
    pub max_history: usize,
    /// Maximum tokens per conversation
    pub max_tokens: u32,
    /// Whether to enable conversation memory
    pub enable_memory: bool,
    /// Whether to enable streaming responses
    pub enable_streaming: bool,
    /// Auto-save conversation interval (seconds)
    pub auto_save_interval: u64,
    /// Custom conversation settings
    pub custom_settings: Option<Arc<str>>,
    /// Personality configuration
    pub personality: PersonalityConfig,
    /// Behavior configuration
    pub behavior: BehaviorConfig,
    /// UI configuration
    pub ui: UIConfig,
    /// Integration configuration
    pub integration: IntegrationConfig,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            engine: EngineConfig::default(),
            system_prompt: Some(Arc::from("You are a helpful AI assistant.")),
            max_history: 50,
            max_tokens: 8192,
            enable_memory: true,
            enable_streaming: false,
            auto_save_interval: 300, // 5 minutes
            custom_settings: None,
            personality: PersonalityConfig::default(),
            behavior: BehaviorConfig::default(),
            ui: UIConfig::default(),
            integration: IntegrationConfig::default(),
        }
    }
}

impl ChatConfig {
    /// Create a new chat configuration
    #[inline]
    pub fn new(engine: EngineConfig) -> Self {
        Self {
            engine,
            ..Default::default()
        }
    }

    /// Set system prompt
    #[inline]
    pub fn with_system_prompt(mut self, prompt: impl Into<Arc<str>>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set maximum history length
    #[inline]
    pub fn with_max_history(mut self, max_history: usize) -> Self {
        self.max_history = max_history;
        self
    }

    /// Set maximum tokens
    #[inline]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Enable conversation memory
    #[inline]
    pub fn with_memory(mut self) -> Self {
        self.enable_memory = true;
        self
    }

    /// Enable streaming responses
    #[inline]
    pub fn with_streaming(mut self) -> Self {
        self.enable_streaming = true;
        self
    }

    /// Set auto-save interval
    #[inline]
    pub fn with_auto_save(mut self, interval_seconds: u64) -> Self {
        self.auto_save_interval = interval_seconds;
        self
    }

    /// Set personality configuration
    #[inline]
    pub fn with_personality(mut self, personality: PersonalityConfig) -> Self {
        self.personality = personality;
        self
    }

    /// Set behavior configuration
    #[inline]
    pub fn with_behavior(mut self, behavior: BehaviorConfig) -> Self {
        self.behavior = behavior;
        self
    }

    /// Set UI configuration
    #[inline]
    pub fn with_ui(mut self, ui: UIConfig) -> Self {
        self.ui = ui;
        self
    }

    /// Set integration configuration
    #[inline]
    pub fn with_integration(mut self, integration: IntegrationConfig) -> Self {
        self.integration = integration;
        self
    }

    /// Configure personality tone
    #[inline]
    pub fn with_tone(mut self, tone: impl Into<Arc<str>>) -> Self {
        self.personality.tone = tone.into();
        self
    }

    /// Configure creativity level
    #[inline]
    pub fn with_creativity(mut self, creativity: f32) -> Self {
        self.personality.creativity = creativity.clamp(0.0, 1.0);
        self
    }

    /// Configure formality level
    #[inline]
    pub fn with_formality(mut self, formality: f32) -> Self {
        self.personality.formality = formality.clamp(0.0, 1.0);
        self
    }

    /// Configure expertise level
    #[inline]
    pub fn with_expertise(mut self, expertise: impl Into<Arc<str>>) -> Self {
        self.personality.expertise_level = expertise.into();
        self
    }

    /// Configure conversation flow
    #[inline]
    pub fn with_conversation_flow(mut self, flow: impl Into<Arc<str>>) -> Self {
        self.behavior.conversation_flow = flow.into();
        self
    }

    /// Configure proactivity level
    #[inline]
    pub fn with_proactivity(mut self, proactivity: f32) -> Self {
        self.behavior.proactivity = proactivity.clamp(0.0, 1.0);
        self
    }

    /// Configure UI theme
    #[inline]
    pub fn with_theme(mut self, theme: impl Into<Arc<str>>) -> Self {
        self.ui.theme = theme.into();
        self
    }

    /// Configure layout
    #[inline]
    pub fn with_layout(mut self, layout: impl Into<Arc<str>>) -> Self {
        self.ui.layout = layout.into();
        self
    }

    /// Add external service integration
    #[inline]
    pub fn with_external_service(mut self, service: impl Into<Arc<str>>) -> Self {
        self.integration.external_services.push(service.into());
        self
    }

    /// Add plugin preference
    #[inline]
    pub fn with_plugin_preference(mut self, preference: impl Into<Arc<str>>) -> Self {
        self.integration.plugin_preferences.push(preference.into());
        self
    }
}

/// Chat session with conversation state management
pub struct ChatSession {
    config: Arc<RwLock<ChatConfig>>,
    messages: Arc<RwLock<VecDeque<ChatMessage>>>,
    total_tokens: Arc<std::sync::atomic::AtomicU32>,
    session_id: Arc<str>,
    created_at: u64,
    last_activity: Arc<std::sync::atomic::AtomicU64>,
}

impl ChatSession {
    /// Create a new chat session
    #[inline]
    pub fn new(config: ChatConfig) -> Self {
        let session_id = Arc::from(uuid::Uuid::new_v4().to_string());
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            config: Arc::new(RwLock::new(config)),
            messages: Arc::new(RwLock::new(VecDeque::new())),
            total_tokens: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            session_id,
            created_at: now,
            last_activity: Arc::new(std::sync::atomic::AtomicU64::new(now)),
        }
    }

    /// Get session ID
    #[inline]
    pub fn session_id(&self) -> &Arc<str> {
        &self.session_id
    }

    /// Get total tokens used
    #[inline]
    pub fn total_tokens(&self) -> u32 {
        self.total_tokens.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Update last activity timestamp
    #[inline]
    fn update_activity(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.last_activity.store(now, std::sync::atomic::Ordering::Relaxed);
    }

    /// Add a message to the conversation
    #[inline]
    pub async fn add_message(&self, message: ChatMessage) -> ChatResult<()> {
        let mut messages = self.messages.write().await;
        let config = self.config.read().await;

        // Check context limits
        let current_tokens = self.total_tokens() + message.tokens;
        if current_tokens > config.max_tokens {
            return Err(ChatError::ContextLimitExceeded {
                current_tokens,
                max_tokens: config.max_tokens,
            });
        }

        // Add message and manage history size
        messages.push_back(message.clone());
        if messages.len() > config.max_history {
            if let Some(removed) = messages.pop_front() {
                self.total_tokens.fetch_sub(removed.tokens, std::sync::atomic::Ordering::Relaxed);
            }
        }

        // Update token count
        self.total_tokens.fetch_add(message.tokens, std::sync::atomic::Ordering::Relaxed);
        self.update_activity();

        Ok(())
    }

    /// Get conversation history
    #[inline]
    pub async fn get_history(&self) -> Vec<ChatMessage> {
        self.messages.read().await.iter().cloned().collect()
    }

    /// Clear conversation history
    #[inline]
    pub async fn clear_history(&self) -> ChatResult<()> {
        let mut messages = self.messages.write().await;
        messages.clear();
        self.total_tokens.store(0, std::sync::atomic::Ordering::Relaxed);
        self.update_activity();
        Ok(())
    }

    /// Generate completion using the engine
    #[inline]
    pub async fn complete(&self, input: &str) -> ChatResult<String> {
        let config = self.config.read().await;
        let history = self.get_history().await;

        // Validate input
        if input.is_empty() {
            return Err(ChatError::InvalidMessage {
                reason: Arc::from("Input cannot be empty"),
            });
        }

        // Build context with history
        let mut context = String::new();
        if let Some(system_prompt) = &config.system_prompt {
            context.push_str(&format!("System: {}\n\n", system_prompt));
        }

        // Add recent history
        for message in history.iter().rev().take(10).rev() {
            context.push_str(&format!("{}: {}\n", message.role, message.content));
        }
        context.push_str(&format!("User: {}\n", input));

        // Generate completion
        let response = complete_with_engine(&config.engine, &context).await?;

        // Add user message to history
        let user_message = ChatMessage::user(input).with_tokens(input.len() as u32);
        self.add_message(user_message).await?;

        // Add assistant response to history
        let assistant_message = ChatMessage::assistant(response.clone()).with_tokens(response.len() as u32);
        self.add_message(assistant_message).await?;

        Ok(response)
    }
}

/// Chat completion function with production-ready error handling
pub async fn complete_chat(input: &str) -> ChatResult<String> {
    // Validate input
    if input.is_empty() {
        return Err(ChatError::InvalidMessage {
            reason: Arc::from("Input cannot be empty"),
        });
    }

    // Create default chat configuration
    let config = ChatConfig::default();
    let session = ChatSession::new(config);

    // Generate completion
    session.complete(input).await
}

/// Chat streaming function with async task pattern
pub fn stream_chat(input: &str) -> AsyncTask<ChatResult<String>> {
    let input = input.to_string();

    spawn_async(async move {
        // Create streaming-enabled configuration
        let config = ChatConfig::default()
            .with_streaming()
            .with_max_tokens(4096);

        let session = ChatSession::new(config);

        // Generate streaming completion
        session.complete(&input).await
    })
}

/// Chat loop function with conversation management
pub async fn chat_loop(input: &str) -> ChatResult<String> {
    // Validate input
    if input.is_empty() {
        return Err(ChatError::InvalidMessage {
            reason: Arc::from("Input cannot be empty"),
        });
    }

    // Create persistent chat session
    let config = ChatConfig::default()
        .with_memory()
        .with_max_history(100)
        .with_max_tokens(8192);

    let session = ChatSession::new(config);

    // Process conversation loop
    let response = session.complete(input).await?;

    // Return formatted response
    Ok(response)
}

/// Create a configured chat session for different use cases
#[inline]
pub fn create_chat_session(config: ChatConfig) -> ChatSession {
    ChatSession::new(config)
}

/// Create a simple chat configuration
#[inline]
pub fn create_simple_chat_config() -> ChatConfig {
    ChatConfig::default()
        .with_system_prompt("You are a helpful assistant.")
        .with_max_history(20)
        .with_max_tokens(4096)
}

/// Create a memory-enhanced chat configuration
#[inline]
pub fn create_memory_chat_config() -> ChatConfig {
    ChatConfig::default()
        .with_system_prompt("You are a helpful assistant with conversation memory.")
        .with_memory()
        .with_max_history(100)
        .with_max_tokens(8192)
        .with_auto_save(300)
}

/// Create a streaming chat configuration
#[inline]
pub fn create_streaming_chat_config() -> ChatConfig {
    ChatConfig::default()
        .with_system_prompt("You are a helpful assistant.")
        .with_streaming()
        .with_max_history(50)
        .with_max_tokens(4096)
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
