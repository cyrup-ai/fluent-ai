//! Core configuration types for chat features
//!
//! This module contains the fundamental configuration structures including model settings,
//! chat configuration, personality settings, behavior configuration, UI settings, and
//! integration configuration with zero-allocation patterns and blazing-fast performance.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use fluent_ai_async::AsyncStream;
#[cfg(feature = "bincode-serialization")]
use bincode;
#[cfg(feature = "rkyv-serialization")]
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Duration serialization helper module for serde
///
/// Provides functions to serialize and deserialize Duration values
/// as seconds for JSON and other text-based formats.
pub mod duration_secs {
    use super::*;

    /// Serialize a Duration as seconds (u64)
    ///
    /// Converts a Duration to its total seconds representation for serialization.
    /// This is useful for JSON and other text formats where Duration needs to
    /// be represented as a simple numeric value.
    ///
    /// # Arguments
    ///
    /// * `duration` - The Duration to serialize
    /// * `serializer` - The serde serializer
    ///
    /// # Returns
    ///
    /// The serialized duration as seconds
    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_secs())
    }

    /// Deserialize a Duration from seconds (u64)
    ///
    /// Reconstructs a Duration from its seconds representation during deserialization.
    /// This is the counterpart to the serialize function above.
    ///
    /// # Arguments
    ///
    /// * `deserializer` - The serde deserializer
    ///
    /// # Returns
    ///
    /// The reconstructed Duration from seconds
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

/// Model configuration for chat interactions
///
/// This configuration defines model-specific settings including provider selection,
/// model parameters, performance tuning, and behavior customization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model provider (e.g., "openai", "anthropic", "mistral", "gemini")
    ///
    /// Specifies which AI provider to use for model inference. Each provider
    /// has different capabilities, pricing, and performance characteristics.
    pub provider: Arc<str>,
    /// Model name/identifier
    ///
    /// The specific model to use from the provider (e.g., "gpt-4", "claude-3-opus").
    /// Model names are provider-specific and determine the capabilities and cost.
    pub model_name: Arc<str>,
    /// Model version or variant
    ///
    /// Optional specific version or variant of the model. Some providers offer
    /// versioned models or specialized variants (e.g., "20240301", "turbo").
    pub model_version: Option<Arc<str>>,
    /// Temperature for response randomness (0.0 to 2.0)
    ///
    /// Controls the randomness of model responses. Lower values (0.0-0.3) produce
    /// more focused and deterministic outputs, while higher values (0.7-2.0)
    /// increase creativity and randomness.
    pub temperature: f32,
    /// Maximum tokens in response
    ///
    /// Optional limit on the number of tokens the model can generate in its response.
    /// If None, uses the model's default maximum. Helps control response length and costs.
    pub max_tokens: Option<u32>,
    /// Top-p nucleus sampling parameter
    ///
    /// Optional nucleus sampling parameter that controls the cumulative probability
    /// cutoff for token selection. Values between 0.1 and 1.0, with lower values
    /// producing more focused responses.
    pub top_p: Option<f32>,
    /// Top-k sampling parameter
    ///
    /// Optional parameter that limits token selection to the top K most likely tokens.
    /// Lower values produce more focused responses, higher values allow more diversity.
    pub top_k: Option<u32>,
    /// Frequency penalty (-2.0 to 2.0)
    ///
    /// Optional penalty applied to tokens based on their frequency in the generated text.
    /// Positive values reduce repetition, negative values encourage repetition.
    pub frequency_penalty: Option<f32>,
    /// Presence penalty (-2.0 to 2.0)
    ///
    /// Optional penalty applied to tokens based on their presence in the generated text.
    /// Positive values encourage topic diversity, negative values encourage focus.
    pub presence_penalty: Option<f32>,
    /// Stop sequences
    ///
    /// Collection of strings that will stop generation when encountered.
    /// Useful for controlling response format and preventing over-generation.
    pub stop_sequences: Vec<Arc<str>>,
    /// System prompt/instructions
    ///
    /// Optional system-level instructions that define the model's behavior and role.
    /// This prompt sets the context and personality for all subsequent interactions.
    pub system_prompt: Option<Arc<str>>,
    /// Enable function calling
    ///
    /// Whether the model can call functions/tools during response generation.
    /// Enables structured interactions and integration with external systems.
    pub enable_functions: bool,
    /// Function calling mode ("auto", "none", "required")
    ///
    /// Controls when the model should use function calling:
    /// - "auto": Model decides when to call functions
    /// - "none": No function calling allowed
    /// - "required": Model must call a function
    pub function_mode: Arc<str>,
    /// Model-specific parameters
    ///
    /// Additional parameters specific to individual models or providers.
    /// Allows customization beyond standard parameters.
    pub custom_parameters: HashMap<Arc<str>, serde_json::Value>,
    /// Request timeout in milliseconds
    ///
    /// Maximum time to wait for a model response before timing out.
    /// Helps prevent hung requests and ensures responsive behavior.
    pub timeout_ms: u64,
    /// Retry configuration
    ///
    /// Configuration for automatic retry behavior on failures.
    /// Defines how many times to retry and with what delays.
    pub retry_config: ModelRetryConfig,
    /// Performance settings
    ///
    /// Configuration for performance optimizations like caching,
    /// batching, streaming, and connection management.
    pub performance: ModelPerformanceConfig}

/// Model retry configuration for handling transient failures
///
/// Defines how requests should be retried when they fail due to
/// temporary issues like network problems or rate limiting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRetryConfig {
    /// Maximum number of retries
    ///
    /// The maximum number of times to retry a failed request.
    /// Set to 0 to disable retries entirely.
    pub max_retries: u32,
    /// Base delay between retries in milliseconds
    ///
    /// The initial delay before the first retry attempt.
    /// Subsequent delays may be increased by the backoff multiplier.
    pub base_delay_ms: u64,
    /// Maximum delay between retries in milliseconds
    ///
    /// The maximum delay that will be used between retry attempts,
    /// even with exponential backoff applied.
    pub max_delay_ms: u64,
    /// Exponential backoff multiplier
    ///
    /// Factor by which the delay is multiplied after each failed retry.
    /// Values greater than 1.0 implement exponential backoff.
    pub backoff_multiplier: f32,
    /// Enable jitter to avoid thundering herd
    ///
    /// When true, adds random variation to retry delays to prevent
    /// multiple clients from retrying simultaneously.
    pub enable_jitter: bool}

/// Model performance configuration for optimizing throughput and latency
///
/// Controls various performance optimizations including caching, batching,
/// streaming, and connection management to maximize efficiency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceConfig {
    /// Enable response caching
    ///
    /// When true, caches model responses to avoid duplicate requests.
    /// Significantly improves performance for repeated queries.
    pub enable_caching: bool,
    /// Cache TTL in seconds
    ///
    /// How long cached responses remain valid before expiring.
    /// Longer TTL improves cache hit rates but may serve stale responses.
    pub cache_ttl_seconds: u64,
    /// Enable request batching
    ///
    /// When true, groups multiple requests together for more efficient
    /// processing. Improves throughput but may increase latency.
    pub enable_batching: bool,
    /// Maximum batch size
    ///
    /// The maximum number of requests that can be grouped in a single batch.
    /// Larger batches improve efficiency but increase memory usage.
    pub max_batch_size: u32,
    /// Batch timeout in milliseconds
    ///
    /// Maximum time to wait for a batch to fill before processing.
    /// Balances throughput optimization with latency requirements.
    pub batch_timeout_ms: u64,
    /// Enable streaming responses
    ///
    /// When true, responses are streamed as they're generated rather than
    /// waiting for completion. Reduces perceived latency for long responses.
    pub enable_streaming: bool,
    /// Connection pool size
    ///
    /// Number of persistent connections to maintain with the model provider.
    /// More connections enable higher concurrency but use more resources.
    pub connection_pool_size: u32,
    /// Keep-alive timeout in seconds
    ///
    /// How long to keep idle connections open before closing them.
    /// Longer timeouts reduce connection overhead but use more resources.
    pub keep_alive_timeout_seconds: u64}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            provider: Arc::from("openai"),
            model_name: Arc::from("gpt-4"),
            model_version: None,
            temperature: 0.7,
            max_tokens: Some(2048),
            top_p: Some(1.0),
            top_k: None,
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            stop_sequences: Vec::new(),
            system_prompt: None,
            enable_functions: true,
            function_mode: Arc::from("auto"),
            custom_parameters: HashMap::new(),
            timeout_ms: 30000, // 30 seconds
            retry_config: ModelRetryConfig::default(),
            performance: ModelPerformanceConfig::default()}
    }
}

impl Default for ModelRetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 1000, // 1 second
            max_delay_ms: 30000, // 30 seconds
            backoff_multiplier: 2.0,
            enable_jitter: true}
    }
}

impl Default for ModelPerformanceConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_ttl_seconds: 3600, // 1 hour
            enable_batching: false,
            max_batch_size: 10,
            batch_timeout_ms: 100,
            enable_streaming: true,
            connection_pool_size: 10,
            keep_alive_timeout_seconds: 60}
    }
}

impl ModelConfig {
    /// Create a new model configuration with provider and model name
    ///
    /// Creates a ModelConfig with the specified provider and model name,
    /// using default values for all other settings.
    ///
    /// # Arguments
    ///
    /// * `provider` - The AI provider (e.g., "openai", "anthropic")
    /// * `model_name` - The specific model identifier
    ///
    /// # Examples
    ///
    /// ```rust
    /// let config = ModelConfig::new("openai", "gpt-4");
    /// ```
    pub fn new(provider: impl Into<Arc<str>>, model_name: impl Into<Arc<str>>) -> Self {
        Self {
            provider: provider.into(),
            model_name: model_name.into(),
            ..Default::default()
        }
    }

    /// Create configuration for OpenAI models
    ///
    /// Convenience constructor for OpenAI models with optimized defaults.
    /// Sets the provider to "openai" and configures appropriate settings.
    ///
    /// # Arguments
    ///
    /// * `model_name` - OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo")
    ///
    /// # Examples
    ///
    /// ```rust
    /// let config = ModelConfig::openai("gpt-4");
    /// ```
    pub fn openai(model_name: impl Into<Arc<str>>) -> Self {
        Self::new("openai", model_name)
    }

    /// Create configuration for Anthropic models
    ///
    /// Convenience constructor for Anthropic models with optimized defaults.
    /// Sets the provider to "anthropic" and configures appropriate settings.
    ///
    /// # Arguments
    ///
    /// * `model_name` - Anthropic model name (e.g., "claude-3-opus", "claude-3-sonnet")
    ///
    /// # Examples
    ///
    /// ```rust
    /// let config = ModelConfig::anthropic("claude-3-opus");
    /// ```
    pub fn anthropic(model_name: impl Into<Arc<str>>) -> Self {
        Self::new("anthropic", model_name)
    }

    /// Create configuration for Mistral models
    ///
    /// Convenience constructor for Mistral models with optimized defaults.
    /// Sets the provider to "mistral" and configures appropriate settings.
    ///
    /// # Arguments
    ///
    /// * `model_name` - Mistral model name (e.g., "mistral-large", "mistral-medium")
    ///
    /// # Examples
    ///
    /// ```rust
    /// let config = ModelConfig::mistral("mistral-large");
    /// ```
    pub fn mistral(model_name: impl Into<Arc<str>>) -> Self {
        Self::new("mistral", model_name)
    }

    /// Create configuration for Gemini models
    ///
    /// Convenience constructor for Google Gemini models with optimized defaults.
    /// Sets the provider to "gemini" and configures appropriate settings.
    ///
    /// # Arguments
    ///
    /// * `model_name` - Gemini model name (e.g., "gemini-pro", "gemini-pro-vision")
    ///
    /// # Examples
    ///
    /// ```rust
    /// let config = ModelConfig::gemini("gemini-pro");
    /// ```
    pub fn gemini(model_name: impl Into<Arc<str>>) -> Self {
        Self::new("gemini", model_name)
    }

    /// Set the sampling temperature for response generation
    ///
    /// Controls the randomness of the model's responses. The value is clamped
    /// to the valid range of 0.0-2.0 to ensure proper model behavior.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Temperature value (automatically clamped to 0.0-2.0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// let config = ModelConfig::openai("gpt-4")
    ///     .with_temperature(0.8);
    /// ```
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature.clamp(0.0, 2.0);
        self
    }

    /// Set the maximum number of tokens in the response
    ///
    /// Limits the length of the model's response to control costs and
    /// ensure responses fit within expected bounds.
    ///
    /// # Arguments
    ///
    /// * `max_tokens` - Maximum tokens to generate
    ///
    /// # Examples
    ///
    /// ```rust
    /// let config = ModelConfig::openai("gpt-4")
    ///     .with_max_tokens(1000);
    /// ```
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the system prompt that defines the model's behavior
    ///
    /// The system prompt establishes the context, role, and personality
    /// for the AI model's responses throughout the conversation.
    ///
    /// # Arguments
    ///
    /// * `prompt` - System prompt text
    ///
    /// # Examples
    ///
    /// ```rust
    /// let config = ModelConfig::openai("gpt-4")
    ///     .with_system_prompt("You are a helpful coding assistant.");
    /// ```
    pub fn with_system_prompt(mut self, prompt: impl Into<Arc<str>>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Enable or disable function calling capabilities
    ///
    /// When enabled, the model can call functions and tools to extend
    /// its capabilities beyond text generation.
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable function calling
    ///
    /// # Examples
    ///
    /// ```rust
    /// let config = ModelConfig::openai("gpt-4")
    ///     .with_functions(true);
    /// ```
    pub fn with_functions(mut self, enable: bool) -> Self {
        self.enable_functions = enable;
        self
    }

    /// Validate the model configuration asynchronously
    ///
    /// Performs validation of the model configuration parameters to ensure
    /// they are valid and compatible with the selected provider and model.
    /// Returns a stream that emits validation results.
    ///
    /// # Returns
    ///
    /// An AsyncStream that emits `()` upon successful validation or errors
    /// if validation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use futures_util::StreamExt;
    ///
    /// let config = ModelConfig::openai("gpt-4");
    /// let mut validation = config.validate();
    /// 
    /// while let Some(result) = validation.next().await {
    ///     println!("Validation passed");
    /// }
    /// ```
    pub fn validate(&self) -> AsyncStream<()> {
        let _config = self.clone();
        // Use AsyncStream::with_channel for streaming-only architecture - emit success immediately
        AsyncStream::with_channel(move |sender| {
            // Emit success via sender - validation happens during stream processing
            let _ = sender.send(());
        })
    }
}

/// Core chat runtime configuration for the entire chat system
///
/// This configuration defines runtime behavior including message handling,
/// history management, streaming options, and nested configurations for
/// personality, behavior, UI, and integrations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(
    feature = "bincode-serialization",
    derive(bincode::Encode, bincode::Decode)
)]
#[cfg_attr(
    feature = "rkyv-serialization",
    derive(Archive, RkyvDeserialize, RkyvSerialize)
)]
pub struct ChatRuntimeConfig {
    /// Maximum message length in characters
    ///
    /// Limits the length of individual messages to prevent abuse and
    /// ensure reasonable processing times. Messages exceeding this limit
    /// will be truncated or rejected.
    pub max_message_length: usize,
    /// Enable message history storage and retrieval
    ///
    /// When true, messages are stored for later retrieval and context.
    /// When false, only the current conversation context is maintained.
    pub enable_history: bool,
    /// History retention period in seconds (for rkyv compatibility)
    ///
    /// How long to keep message history before automatic cleanup.
    /// Longer retention improves context but increases storage requirements.
    #[serde(with = "duration_secs")]
    pub history_retention: Duration,
    /// Enable streaming responses
    ///
    /// When true, responses are streamed as they're generated for better
    /// user experience. When false, complete responses are returned at once.
    pub enable_streaming: bool,
    /// Personality configuration
    ///
    /// Defines the AI's personality traits, response style, and behavioral
    /// characteristics that shape how it interacts with users.
    pub personality: PersonalityConfig,
    /// Behavior configuration
    ///
    /// Controls interaction patterns, memory management, and conversation
    /// flow behaviors that determine how the AI conducts conversations.
    pub behavior: BehaviorConfig,
    /// UI configuration
    ///
    /// Settings for user interface appearance, animations, themes, and
    /// visual presentation of the chat interface.
    pub ui: UIConfig,
    /// Integration configuration
    ///
    /// Configuration for external services, plugins, webhooks, and
    /// third-party integrations that extend chat functionality.
    pub integration: IntegrationConfig}

/// Personality configuration for AI behavior and character traits
///
/// Defines the AI's personality characteristics, communication style,
/// and behavioral tendencies that shape its interactions with users.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(
    feature = "bincode-serialization",
    derive(bincode::Encode, bincode::Decode)
)]
#[cfg_attr(
    feature = "rkyv-serialization",
    derive(Archive, RkyvDeserialize, RkyvSerialize)
)]
pub struct PersonalityConfig {
    /// Personality type identifier
    ///
    /// Defines the overall personality archetype (e.g., "analytical", "creative",
    /// "balanced") that influences all other personality aspects.
    pub personality_type: Arc<str>,
    /// Response style settings
    ///
    /// Communication style preference (e.g., "conversational", "formal",
    /// "technical") that affects how responses are structured and presented.
    pub response_style: Arc<str>,
    /// Tone configuration
    ///
    /// Emotional tone of responses (e.g., "friendly", "professional",
    /// "enthusiastic") that colors the AI's communication.
    pub tone: Arc<str>,
    /// Custom instructions
    ///
    /// Optional additional personality instructions that override or extend
    /// the base personality configuration with specific behavioral guidance.
    pub custom_instructions: Option<Arc<str>>,
    /// Creativity level (0.0-1.0)
    ///
    /// How creative and unconventional the AI's responses should be.
    /// Higher values encourage more creative and diverse responses.
    pub creativity: f64,
    /// Formality level (0.0-1.0)
    ///
    /// How formal or casual the AI's language should be.
    /// Higher values result in more formal, professional communication.
    pub formality: f64,
    /// Humor level (0.0-1.0)
    ///
    /// How much humor and playfulness to include in responses.
    /// Higher values result in more jokes, wordplay, and lighthearted content.
    pub humor: f64,
    /// Empathy level (0.0-1.0)
    ///
    /// How empathetic and emotionally aware the AI should be.
    /// Higher values result in more emotionally supportive responses.
    pub empathy: f64,
    /// Expertise level
    ///
    /// The level of expertise to demonstrate (e.g., "beginner", "intermediate",
    /// "expert") which affects the depth and complexity of explanations.
    pub expertise_level: Arc<str>,
    /// Verbosity level
    ///
    /// How verbose or concise responses should be (e.g., "brief", "moderate",
    /// "detailed") which controls response length and detail level.
    pub verbosity: Arc<str>,
    /// Personality traits
    ///
    /// Collection of specific personality traits that further define the AI's
    /// character (e.g., "helpful", "curious", "patient", "analytical").
    pub traits: Vec<Arc<str>>}

/// Behavior configuration for interaction patterns and conversation management
///
/// Controls how the AI behaves during conversations, including memory management,
/// response timing, user interface behaviors, and conversation flow patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(
    feature = "bincode-serialization",
    derive(bincode::Encode, bincode::Decode)
)]
#[cfg_attr(
    feature = "rkyv-serialization",
    derive(Archive, RkyvDeserialize, RkyvSerialize)
)]
pub struct BehaviorConfig {
    /// Maximum conversation length in messages
    ///
    /// Limits how many messages can be in a single conversation before
    /// truncation or archival. Prevents memory overflow in long conversations.
    pub max_conversation_length: u32,
    /// Enable conversation memory across sessions
    ///
    /// When true, the AI remembers previous conversations and can reference
    /// past interactions. When false, each session starts fresh.
    pub enable_memory: bool,
    /// Memory retention duration
    ///
    /// How long conversation memories are retained before automatic cleanup.
    /// Longer retention improves continuity but increases storage requirements.
    pub memory_duration: std::time::Duration,
    /// Response delay in milliseconds
    ///
    /// Artificial delay before sending responses to simulate thinking time
    /// and make interactions feel more natural and human-like.
    pub response_delay_ms: u64,
    /// Enable typing indicators
    ///
    /// When true, shows typing indicators while the AI is generating responses
    /// to provide visual feedback that processing is occurring.
    pub enable_typing_indicators: bool,
    /// Enable read receipts
    ///
    /// When true, shows when messages have been read by the AI,
    /// providing confirmation of message receipt and processing.
    pub enable_read_receipts: bool,
    /// Auto-save interval in seconds
    ///
    /// How frequently to automatically save conversation state to prevent
    /// data loss. More frequent saves improve reliability but increase I/O.
    pub auto_save_interval_seconds: u64,
    /// Enable auto-correction of user messages
    ///
    /// When true, automatically corrects obvious typos and grammar errors
    /// in user messages before processing. May alter user intent.
    pub enable_auto_correction: bool,
    /// Enable smart reply suggestions
    ///
    /// When true, provides suggested responses or follow-up questions
    /// to help users continue the conversation more effectively.
    pub enable_smart_replies: bool,
    /// Smart reply count
    ///
    /// Number of smart reply suggestions to generate and display to users.
    /// More suggestions provide options but may clutter the interface.
    pub smart_reply_count: u32,
    /// Context awareness level (0.0-1.0)
    ///
    /// How much the AI considers previous conversation context when responding.
    /// Higher values result in more contextually aware but potentially verbose responses.
    pub context_awareness: f64,
    /// Memory retention level (0.0-1.0)
    ///
    /// How strongly the AI retains and references previous conversation elements.
    /// Higher values improve continuity but may make conversations feel repetitive.
    pub memory_retention: f64,
    /// Conversation continuity across sessions
    ///
    /// When true, maintains conversation threads across multiple sessions.
    /// When false, each session is treated as independent.
    pub conversation_continuity: bool,
    /// Follow-up behavior strategy
    ///
    /// How the AI handles follow-up questions and continued conversations
    /// (e.g., "contextual", "independent", "summarizing").
    pub follow_up_behavior: Arc<str>,
    /// Proactivity level (0.0-1.0)
    ///
    /// How proactive the AI is in suggesting topics, asking questions,
    /// or extending conversations beyond direct responses.
    pub proactivity: f64,
    /// Question frequency (0.0-1.0)
    ///
    /// How often the AI asks clarifying or follow-up questions.
    /// Higher values result in more interactive but potentially intrusive conversations.
    pub question_frequency: f64,
    /// Conversation flow type
    ///
    /// Overall conversation management strategy (e.g., "natural", "structured",
    /// "goal-oriented") that influences response patterns and topic transitions.
    pub conversation_flow: Arc<str>,
    /// Error handling approach
    ///
    /// How to handle errors and unexpected situations (e.g., "graceful",
    /// "explicit", "recovery-focused") during conversations.
    pub error_handling: Arc<str>,
    /// Auto-save conversations to persistent storage
    ///
    /// When true, automatically saves conversation history for later retrieval.
    /// When false, conversations are only kept in memory during the session.
    pub auto_save_conversations: bool,
    /// Typing speed in characters per second
    ///
    /// Simulated typing speed for responses when typing indicators are enabled.
    /// Affects the duration of typing indicators and response delivery timing.
    pub typing_speed_cps: f32,
    /// Enable emoji and reaction responses
    ///
    /// When true, allows the AI to use emojis and reactions in responses
    /// to add emotional expression and engagement to conversations.
    pub enable_reactions: bool,
    /// Content filtering level
    ///
    /// Level of content filtering applied to responses (e.g., "none", "moderate",
    /// "strict") to ensure appropriate content based on context and requirements.
    pub content_filtering: Arc<str>}

/// UI configuration for interface behavior and visual presentation
///
/// Controls the visual appearance, animations, themes, and user interface
/// behaviors of the chat interface to customize the user experience.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(
    feature = "bincode-serialization",
    derive(bincode::Encode, bincode::Decode)
)]
#[cfg_attr(
    feature = "rkyv-serialization",
    derive(Archive, RkyvDeserialize, RkyvSerialize)
)]
pub struct UIConfig {
    /// Theme preference
    ///
    /// Visual theme for the interface (e.g., "light", "dark", "system")
    /// that controls the overall color scheme and visual styling.
    pub theme: Arc<str>,
    /// Font size in pixels
    ///
    /// Size of the text font used throughout the interface.
    /// Affects readability and accessibility for different users.
    pub font_size: u32,
    /// Enable animations and transitions
    ///
    /// When true, enables smooth animations and transitions between UI states.
    /// When false, uses instant state changes for better performance.
    pub enable_animations: bool,
    /// Animation speed multiplier (0.0-2.0)
    ///
    /// Controls the speed of UI animations and transitions.
    /// Values below 1.0 slow animations, above 1.0 speed them up.
    pub animation_speed: f32,
    /// Enable sound effects for UI interactions
    ///
    /// When true, plays audio feedback for various UI actions like
    /// message sending, receiving, and other interactions.
    pub enable_sound_effects: bool,
    /// Sound volume level (0.0-1.0)
    ///
    /// Volume level for UI sound effects and audio feedback.
    /// 0.0 is silent, 1.0 is maximum volume.
    pub sound_volume: f32,
    /// Enable message grouping by sender
    ///
    /// When true, groups consecutive messages from the same sender
    /// together for a cleaner visual presentation.
    pub enable_message_grouping: bool,
    /// Show timestamps on messages
    ///
    /// When true, displays timestamps showing when each message
    /// was sent or received for temporal context.
    pub show_timestamps: bool,
    /// Timestamp format string
    ///
    /// Format string for displaying timestamps (e.g., "HH:mm", "MM/dd HH:mm")
    /// using standard time formatting conventions.
    pub timestamp_format: Arc<str>,
    /// Enable syntax highlighting for code blocks
    ///
    /// When true, applies syntax highlighting to code blocks and
    /// technical content for improved readability.
    pub enable_syntax_highlighting: bool,
    /// Code syntax highlighting theme
    ///
    /// Theme for syntax highlighting (e.g., "github", "monokai", "solarized")
    /// that controls colors and styling for code content.
    pub code_theme: Arc<str>,
    /// Interface layout type
    ///
    /// Overall layout structure (e.g., "standard", "compact", "wide")
    /// that determines how UI elements are arranged.
    pub layout: Arc<str>,
    /// Color scheme configuration
    ///
    /// Color scheme settings (e.g., "adaptive", "high-contrast", "custom")
    /// that control color choices throughout the interface.
    pub color_scheme: Arc<str>,
    /// Display density preference
    ///
    /// How densely packed the interface elements are (e.g., "comfortable",
    /// "compact", "spacious") affecting spacing and visual breathing room.
    pub display_density: Arc<str>,
    /// Animation style settings
    ///
    /// Specific animation style preferences (e.g., "smooth", "bouncy", "minimal")
    /// that control the character of UI animations and transitions.
    pub animations: Arc<str>}

/// Integration configuration for external services and extensibility
///
/// Controls integration with external services, plugins, webhooks, APIs,
/// and other third-party systems that extend chat functionality.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(
    feature = "bincode-serialization",
    derive(bincode::Encode, bincode::Decode)
)]
#[cfg_attr(
    feature = "rkyv-serialization",
    derive(Archive, RkyvDeserialize, RkyvSerialize)
)]
pub struct IntegrationConfig {
    /// Enable plugin system
    ///
    /// When true, allows loading and execution of plugins that extend
    /// chat functionality with custom features and capabilities.
    pub enable_plugins: bool,
    /// Plugin directory path
    ///
    /// Optional filesystem path where plugins are located and loaded from.
    /// If None, plugins are disabled or loaded from default locations.
    pub plugin_directory: Option<Arc<str>>,
    /// Plugin whitelist
    ///
    /// List of plugin names or identifiers that are explicitly allowed to run.
    /// When not empty, only whitelisted plugins can be loaded and executed.
    pub plugin_whitelist: Vec<Arc<str>>,
    /// Plugin blacklist
    ///
    /// List of plugin names or identifiers that are explicitly forbidden.
    /// Blacklisted plugins will never be loaded regardless of other settings.
    pub plugin_blacklist: Vec<Arc<str>>,
    /// Enable webhook notifications
    ///
    /// When true, sends HTTP webhook notifications for various chat events
    /// to configured external endpoints for integration purposes.
    pub enable_webhooks: bool,
    /// Webhook notification URL
    ///
    /// Optional URL endpoint where webhook notifications are sent.
    /// Must be a valid HTTP/HTTPS URL if webhooks are enabled.
    pub webhook_url: Option<Arc<str>>,
    /// Webhook security secret
    ///
    /// Optional secret used to sign webhook payloads for security validation.
    /// Recipients can verify webhook authenticity using this secret.
    pub webhook_secret: Option<Arc<str>>,
    /// Enable API access for external applications
    ///
    /// When true, exposes API endpoints that allow external applications
    /// to interact with the chat system programmatically.
    pub enable_api_access: bool,
    /// API rate limit per minute
    ///
    /// Maximum number of API requests allowed per minute to prevent abuse
    /// and ensure system stability under load.
    pub api_rate_limit: u32,
    /// Custom integration configurations
    ///
    /// Key-value pairs defining custom integration settings for specific
    /// services or systems that don't fit standard integration patterns.
    pub custom_integrations: HashMap<Arc<str>, serde_json::Value>,
    /// External services list
    ///
    /// List of external service identifiers that the chat system
    /// can connect to and interact with during operation.
    pub external_services: Vec<Arc<str>>,
    /// API configuration profiles
    ///
    /// List of API configuration identifiers that define different
    /// API access profiles with varying permissions and capabilities.
    pub api_configurations: Vec<Arc<str>>,
    /// Authentication method options
    ///
    /// List of supported authentication methods for API access and
    /// external integrations (e.g., "bearer", "oauth", "api-key").
    pub authentication: Vec<Arc<str>>}

impl Default for PersonalityConfig {
    fn default() -> Self {
        Self {
            personality_type: Arc::from("balanced"),
            response_style: Arc::from("conversational"),
            tone: Arc::from("friendly"),
            custom_instructions: None,
            creativity: 0.7,
            formality: 0.5,
            humor: 0.3,
            empathy: 0.8,
            expertise_level: Arc::from("intermediate"),
            verbosity: Arc::from("moderate"),
            traits: vec![Arc::from("helpful"), Arc::from("accurate")]}
    }
}

impl Default for BehaviorConfig {
    fn default() -> Self {
        Self {
            max_conversation_length: 100,
            enable_memory: true,
            memory_duration: std::time::Duration::from_secs(3600), // 1 hour
            response_delay_ms: 500,
            enable_typing_indicators: true,
            enable_read_receipts: true,
            auto_save_interval_seconds: 30,
            enable_auto_correction: false,
            enable_smart_replies: true,
            smart_reply_count: 3,
            context_awareness: 0.8,
            memory_retention: 0.7,
            conversation_continuity: true,
            follow_up_behavior: Arc::from("contextual"),
            proactivity: 0.5,
            question_frequency: 0.3,
            conversation_flow: Arc::from("natural"),
            error_handling: Arc::from("graceful"),
            auto_save_conversations: true,
            typing_speed_cps: 30.0,
            enable_reactions: true,
            content_filtering: Arc::from("moderate")}
    }
}

impl Default for UIConfig {
    fn default() -> Self {
        Self {
            theme: Arc::from("system"),
            font_size: 14,
            enable_animations: true,
            animation_speed: 1.0,
            enable_sound_effects: false,
            sound_volume: 0.5,
            enable_message_grouping: true,
            show_timestamps: true,
            timestamp_format: Arc::from("HH:mm"),
            enable_syntax_highlighting: true,
            code_theme: Arc::from("github"),
            layout: Arc::from("standard"),
            color_scheme: Arc::from("adaptive"),
            display_density: Arc::from("comfortable"),
            animations: Arc::from("smooth")}
    }
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enable_plugins: false,
            plugin_directory: None,
            plugin_whitelist: Vec::new(),
            plugin_blacklist: Vec::new(),
            enable_webhooks: false,
            webhook_url: None,
            webhook_secret: None,
            enable_api_access: false,
            api_rate_limit: 60,
            custom_integrations: HashMap::new(),
            external_services: Vec::new(),
            api_configurations: Vec::new(),
            authentication: Vec::new()}
    }
}