//! Detailed builder implementations for configuration types
//!
//! This module contains the detailed builder implementations for various
//! configuration types to keep the main builder module focused.

use std::sync::Arc;
use std::time::Duration;

use super::core::{
    PersonalityConfig, BehaviorConfig, UIConfig, 
    LanguageHandlingConfig, DisplayConfig
};
use super::model::{ModelConfig, ModelRetryConfig, ModelPerformanceConfig};

/// Builder for constructing model configuration instances
///
/// Provides a fluent interface for building ModelConfig instances with
/// validation and sensible defaults. All methods return Self for chaining.
#[derive(Debug, Clone)]
pub struct ModelConfigBuilder {
    /// The model configuration being built
    config: ModelConfig}

impl ModelConfigBuilder {
    /// Create a new model configuration builder from an existing config
    ///
    /// # Arguments
    /// * `config` - The initial model configuration to build upon
    ///
    /// # Returns
    /// A new ModelConfigBuilder instance
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }

    /// Set the model provider (e.g., "openai", "anthropic", "local")
    ///
    /// # Arguments
    /// * `provider` - The provider identifier string
    ///
    /// # Returns
    /// Self for method chaining
    pub fn provider(mut self, provider: impl Into<Arc<str>>) -> Self {
        self.config.provider = provider.into();
        self
    }

    /// Set the specific model name (e.g., "gpt-4", "claude-3-opus", "llama2")
    ///
    /// # Arguments
    /// * `model_name` - The model identifier string
    ///
    /// # Returns
    /// Self for method chaining
    pub fn model_name(mut self, model_name: impl Into<Arc<str>>) -> Self {
        self.config.model_name = model_name.into();
        self
    }

    /// Set the sampling temperature for response generation
    ///
    /// Controls randomness in the model's responses. Lower values (0.0-0.3)
    /// produce more focused and deterministic outputs, while higher values
    /// (0.7-1.0) increase creativity and randomness.
    ///
    /// # Arguments
    /// * `temperature` - Temperature value between 0.0 and 2.0
    ///
    /// # Returns
    /// Self for method chaining
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = temperature;
        self
    }

    /// Set the maximum number of tokens to generate in responses
    ///
    /// Limits the length of generated responses. Setting this too low may
    /// result in truncated responses, while setting it too high may impact
    /// performance and cost.
    ///
    /// # Arguments
    /// * `max_tokens` - Maximum token count for responses
    ///
    /// # Returns
    /// Self for method chaining
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    /// Set the system prompt that defines the model's behavior and context
    ///
    /// The system prompt is sent with every request to establish the model's
    /// role, behavior, and response guidelines. This is crucial for maintaining
    /// consistent AI personality and behavior.
    ///
    /// # Arguments
    /// * `prompt` - The system prompt text
    ///
    /// # Returns
    /// Self for method chaining
    pub fn system_prompt(mut self, prompt: impl Into<Arc<str>>) -> Self {
        self.config.system_prompt = Some(prompt.into());
        self
    }

    /// Build and return the final model configuration
    ///
    /// Consumes the builder and returns the constructed ModelConfig instance.
    /// The configuration is validated to ensure all required fields are set.
    ///
    /// # Returns
    /// The constructed ModelConfig instance
    pub fn build(self) -> ModelConfig {
        self.config
    }
}

/// Builder for constructing personality configuration instances
///
/// Provides a fluent interface for building PersonalityConfig instances that
/// define the AI's personality traits, response style, and behavioral characteristics.
/// All methods return Self for chaining.
#[derive(Debug, Clone)]
pub struct PersonalityConfigBuilder {
    /// The personality configuration being built
    config: PersonalityConfig}

impl PersonalityConfigBuilder {
    /// Create a new personality configuration builder from an existing config
    ///
    /// # Arguments
    /// * `config` - The initial personality configuration to build upon
    ///
    /// # Returns
    /// A new PersonalityConfigBuilder instance
    pub fn new(config: PersonalityConfig) -> Self {
        Self { config }
    }

    /// Set the personality name for identification and display purposes
    ///
    /// # Arguments
    /// * `name` - A descriptive name for this personality profile
    ///
    /// # Returns
    /// Self for method chaining
    pub fn name(mut self, name: impl Into<Arc<str>>) -> Self {
        self.config.name = name.into();
        self
    }

    /// Set the personality type classification (e.g., "helpful", "analytical", "creative")
    ///
    /// # Arguments
    /// * `personality_type` - The type classification for this personality
    ///
    /// # Returns
    /// Self for method chaining
    pub fn personality_type(mut self, personality_type: impl Into<Arc<str>>) -> Self {
        self.config.personality_type = personality_type.into();
        self
    }

    /// Set the response style (e.g., "conversational", "formal", "concise")
    ///
    /// Defines how the AI structures and formats its responses.
    ///
    /// # Arguments
    /// * `style` - The preferred response style identifier
    ///
    /// # Returns
    /// Self for method chaining
    pub fn response_style(mut self, style: impl Into<Arc<str>>) -> Self {
        self.config.response_style = style.into();
        self
    }

    /// Set the conversational tone (e.g., "friendly", "professional", "casual")
    ///
    /// Influences the emotional character and attitude in responses.
    ///
    /// # Arguments
    /// * `tone` - The desired conversational tone
    ///
    /// # Returns
    /// Self for method chaining
    pub fn tone(mut self, tone: impl Into<Arc<str>>) -> Self {
        self.config.tone = tone.into();
        self
    }

    /// Set the formality level from 0.0 (very casual) to 1.0 (very formal)
    ///
    /// Controls the degree of formality in language, vocabulary choice,
    /// and sentence structure.
    ///
    /// # Arguments
    /// * `level` - Formality level between 0.0 and 1.0
    ///
    /// # Returns
    /// Self for method chaining
    pub fn formality_level(mut self, level: f32) -> Self {
        self.config.formality_level = level;
        self
    }

    /// Set the creativity level from 0.0 (very conservative) to 1.0 (very creative)
    ///
    /// Controls willingness to use creative language, metaphors, and
    /// unconventional approaches to problems.
    ///
    /// # Arguments
    /// * `level` - Creativity level between 0.0 and 1.0
    ///
    /// # Returns
    /// Self for method chaining
    pub fn creativity_level(mut self, level: f32) -> Self {
        self.config.creativity_level = level;
        self
    }

    /// Set the empathy level from 0.0 (analytical/detached) to 1.0 (highly empathetic)
    ///
    /// Controls how much the AI considers emotional context and responds
    /// with emotional understanding and support.
    ///
    /// # Arguments
    /// * `level` - Empathy level between 0.0 and 1.0
    ///
    /// # Returns
    /// Self for method chaining
    pub fn empathy_level(mut self, level: f32) -> Self {
        self.config.empathy_level = level;
        self
    }

    /// Set the humor level from 0.0 (serious/no humor) to 1.0 (frequent humor)
    ///
    /// Controls the frequency and appropriateness of humor, jokes,
    /// and playful responses in conversations.
    ///
    /// # Arguments
    /// * `level` - Humor level between 0.0 and 1.0
    ///
    /// # Returns
    /// Self for method chaining
    pub fn humor_level(mut self, level: f32) -> Self {
        self.config.humor_level = level;
        self
    }

    /// Add a custom personality trait to the configuration
    ///
    /// Custom traits allow for additional personality customization beyond
    /// the standard levels. Examples: "patient", "detail-oriented", "optimistic".
    ///
    /// # Arguments
    /// * `trait_name` - The custom trait to add
    ///
    /// # Returns
    /// Self for method chaining
    pub fn add_trait(mut self, trait_name: impl Into<Arc<str>>) -> Self {
        self.config.custom_traits.push(trait_name.into());
        self
    }

    /// Set a custom personality prompt that overrides default personality behavior
    ///
    /// This prompt is added to the system prompt to reinforce specific
    /// personality characteristics and behavioral guidelines.
    ///
    /// # Arguments
    /// * `prompt` - The personality-specific prompt text
    ///
    /// # Returns
    /// Self for method chaining
    pub fn personality_prompt(mut self, prompt: impl Into<Arc<str>>) -> Self {
        self.config.personality_prompt = Some(prompt.into());
        self
    }

    /// Build and return the final personality configuration
    ///
    /// Consumes the builder and returns the constructed PersonalityConfig instance.
    /// The configuration is validated to ensure personality levels are within valid ranges.
    ///
    /// # Returns
    /// The constructed PersonalityConfig instance
    pub fn build(self) -> PersonalityConfig {
        self.config
    }
}

impl Default for PersonalityConfigBuilder {
    /// Create a default personality configuration builder with balanced settings
    ///
    /// Uses PersonalityConfig::default() as the starting point, which provides
    /// moderate levels across all personality dimensions for general use.
    ///
    /// # Returns
    /// A PersonalityConfigBuilder with default settings
    fn default() -> Self {
        Self::new(PersonalityConfig::default())
    }
}

/// Builder for constructing behavior configuration instances
///
/// Provides a fluent interface for building BehaviorConfig instances that
/// control the AI's behavioral patterns, memory management, and interaction timing.
/// All methods return Self for chaining.
#[derive(Debug, Clone)]
pub struct BehaviorConfigBuilder {
    /// The behavior configuration being built
    config: BehaviorConfig}

impl BehaviorConfigBuilder {
    /// Create a new behavior configuration builder from an existing config
    ///
    /// # Arguments
    /// * `config` - The initial behavior configuration to build upon
    ///
    /// # Returns
    /// A new BehaviorConfigBuilder instance
    pub fn new(config: BehaviorConfig) -> Self {
        Self { config }
    }

    /// Enable or disable conversation memory functionality
    ///
    /// When enabled, the AI maintains context from previous interactions
    /// within the configured memory duration.
    ///
    /// # Arguments
    /// * `enable` - Whether to enable memory functionality
    ///
    /// # Returns
    /// Self for method chaining
    pub fn enable_memory(mut self, enable: bool) -> Self {
        self.config.enable_memory = enable;
        self
    }

    /// Set how long conversation memory should be retained
    ///
    /// Determines how long the AI remembers previous interactions.
    /// Longer durations provide better context but use more memory.
    ///
    /// # Arguments
    /// * `duration` - How long to retain conversation memory
    ///
    /// # Returns
    /// Self for method chaining
    pub fn memory_duration(mut self, duration: Duration) -> Self {
        self.config.memory_duration = duration;
        self
    }

    /// Set the maximum number of messages to retain in conversation history
    ///
    /// Limits memory usage by capping the conversation length.
    /// Older messages are discarded when this limit is reached.
    ///
    /// # Arguments
    /// * `length` - Maximum number of messages to retain
    ///
    /// # Returns
    /// Self for method chaining
    pub fn max_conversation_length(mut self, length: u32) -> Self {
        self.config.max_conversation_length = length;
        self
    }

    /// Enable or disable automatic conversation saving
    ///
    /// When enabled, conversations are automatically persisted to storage
    /// for later retrieval and analysis.
    ///
    /// # Arguments
    /// * `auto_save` - Whether to automatically save conversations
    ///
    /// # Returns
    /// Self for method chaining
    pub fn auto_save_conversations(mut self, auto_save: bool) -> Self {
        self.config.auto_save_conversations = auto_save;
        self
    }

    /// Set the artificial delay before sending responses in milliseconds
    ///
    /// Adds a delay to make responses feel more natural and less robotic.
    /// Useful for simulating thinking time or typing delays.
    ///
    /// # Arguments
    /// * `delay` - Response delay in milliseconds
    ///
    /// # Returns
    /// Self for method chaining
    pub fn response_delay_ms(mut self, delay: u64) -> Self {
        self.config.response_delay_ms = delay;
        self
    }

    /// Enable or disable typing indicators during response generation
    ///
    /// When enabled, shows visual indicators that the AI is "typing"
    /// to provide feedback during response generation.
    ///
    /// # Arguments
    /// * `enable` - Whether to show typing indicators
    ///
    /// # Returns
    /// Self for method chaining
    pub fn enable_typing_indicators(mut self, enable: bool) -> Self {
        self.config.enable_typing_indicators = enable;
        self
    }

    /// Set the simulated typing speed in characters per second
    ///
    /// Controls how fast the typing indicator animation appears.
    /// Higher values make the AI appear to type faster.
    ///
    /// # Arguments
    /// * `speed` - Typing speed in characters per second
    ///
    /// # Returns
    /// Self for method chaining
    pub fn typing_speed_cps(mut self, speed: f32) -> Self {
        self.config.typing_speed_cps = speed;
        self
    }

    /// Set the content filtering level (e.g., "none", "basic", "strict")
    ///
    /// Controls how aggressively inappropriate content is filtered
    /// from both inputs and outputs.
    ///
    /// # Arguments
    /// * `level` - Content filtering level identifier
    ///
    /// # Returns
    /// Self for method chaining
    pub fn content_filtering(mut self, level: impl Into<Arc<str>>) -> Self {
        self.config.content_filtering = level.into();
        self
    }

    /// Set the preferred language for conversations (e.g., "en", "es", "fr")
    ///
    /// Determines the default language for responses when language
    /// cannot be detected from user input.
    ///
    /// # Arguments
    /// * `language` - ISO 639-1 language code
    ///
    /// # Returns
    /// Self for method chaining
    pub fn preferred_language(mut self, language: impl Into<Arc<str>>) -> Self {
        self.config.language_handling.preferred_language = language.into();
        self
    }

    /// Build and return the final behavior configuration
    ///
    /// Consumes the builder and returns the constructed BehaviorConfig instance.
    /// The configuration is validated to ensure all timing values are reasonable.
    ///
    /// # Returns
    /// The constructed BehaviorConfig instance
    pub fn build(self) -> BehaviorConfig {
        self.config
    }
}

/// Builder for constructing user interface configuration instances
///
/// Provides a fluent interface for building UIConfig instances that control
/// visual appearance, animations, accessibility, and user experience settings.
/// All methods return Self for chaining.
#[derive(Debug, Clone)]
pub struct UIConfigBuilder {
    /// The UI configuration being built
    config: UIConfig}

impl UIConfigBuilder {
    /// Create a new UI configuration builder from an existing config
    ///
    /// # Arguments
    /// * `config` - The initial UI configuration to build upon
    ///
    /// # Returns
    /// A new UIConfigBuilder instance
    pub fn new(config: UIConfig) -> Self {
        Self { config }
    }

    /// Set the visual theme (e.g., "light", "dark", "auto", "high-contrast")
    ///
    /// Controls the overall color scheme and visual appearance.
    /// "auto" themes adapt to system preferences.
    ///
    /// # Arguments
    /// * `theme` - Theme identifier string
    ///
    /// # Returns
    /// Self for method chaining
    pub fn theme(mut self, theme: impl Into<Arc<str>>) -> Self {
        self.config.theme = theme.into();
        self
    }

    /// Set the font size scaling multiplier for accessibility
    ///
    /// Values above 1.0 increase font sizes, below 1.0 decrease them.
    /// Typical range is 0.8 to 2.0 for good usability.
    ///
    /// # Arguments
    /// * `multiplier` - Font size scaling factor
    ///
    /// # Returns
    /// Self for method chaining
    pub fn font_size_multiplier(mut self, multiplier: f32) -> Self {
        self.config.font_size_multiplier = multiplier;
        self
    }

    /// Enable or disable UI animations and transitions
    ///
    /// When disabled, improves performance and reduces motion for
    /// users sensitive to animations or on low-performance devices.
    ///
    /// # Arguments
    /// * `enable` - Whether to enable UI animations
    ///
    /// # Returns
    /// Self for method chaining
    pub fn enable_animations(mut self, enable: bool) -> Self {
        self.config.enable_animations = enable;
        self
    }

    /// Set the animation speed multiplier
    ///
    /// Values above 1.0 make animations faster, below 1.0 make them slower.
    /// Only applies when animations are enabled.
    ///
    /// # Arguments
    /// * `speed` - Animation speed multiplier
    ///
    /// # Returns
    /// Self for method chaining
    pub fn animation_speed(mut self, speed: f32) -> Self {
        self.config.animation_speed = speed;
        self
    }

    /// Enable or disable sound effects for UI interactions
    ///
    /// When enabled, plays audio feedback for various UI events
    /// like message sending, notifications, etc.
    ///
    /// # Arguments
    /// * `enable` - Whether to enable sound effects
    ///
    /// # Returns
    /// Self for method chaining
    pub fn enable_sound_effects(mut self, enable: bool) -> Self {
        self.config.enable_sound_effects = enable;
        self
    }

    /// Set the volume level for sound effects from 0.0 (silent) to 1.0 (full volume)
    ///
    /// Only applies when sound effects are enabled.
    /// Values outside 0.0-1.0 are clamped to valid range.
    ///
    /// # Arguments
    /// * `volume` - Volume level between 0.0 and 1.0
    ///
    /// # Returns
    /// Self for method chaining
    pub fn sound_volume(mut self, volume: f32) -> Self {
        self.config.sound_volume = volume;
        self
    }

    /// Enable or disable timestamp display on messages
    ///
    /// When enabled, shows when each message was sent/received.
    /// Useful for conversation history and debugging.
    ///
    /// # Arguments
    /// * `show` - Whether to show message timestamps
    ///
    /// # Returns
    /// Self for method chaining
    pub fn show_timestamps(mut self, show: bool) -> Self {
        self.config.display.show_timestamps = show;
        self
    }

    /// Enable or disable avatar/profile picture display
    ///
    /// When enabled, shows user and AI avatars next to messages
    /// for better visual identification.
    ///
    /// # Arguments
    /// * `show` - Whether to show avatars
    ///
    /// # Returns
    /// Self for method chaining
    pub fn show_avatars(mut self, show: bool) -> Self {
        self.config.display.show_avatars = show;
        self
    }

    /// Set the message bubble visual style (e.g., "rounded", "square", "minimal")
    ///
    /// Controls the appearance of message containers and speech bubbles.
    ///
    /// # Arguments
    /// * `style` - Message bubble style identifier
    ///
    /// # Returns
    /// Self for method chaining
    pub fn message_bubble_style(mut self, style: impl Into<Arc<str>>) -> Self {
        self.config.display.message_bubble_style = style.into();
        self
    }

    /// Enable or disable Markdown rendering in messages
    ///
    /// When enabled, processes Markdown formatting like **bold**, *italic*,
    /// `code`, and other Markdown syntax in message content.
    ///
    /// # Arguments
    /// * `enable` - Whether to render Markdown formatting
    ///
    /// # Returns
    /// Self for method chaining
    pub fn enable_markdown(mut self, enable: bool) -> Self {
        self.config.display.enable_markdown = enable;
        self
    }

    /// Build and return the final UI configuration
    ///
    /// Consumes the builder and returns the constructed UIConfig instance.
    /// The configuration is validated to ensure all values are within acceptable ranges.
    ///
    /// # Returns
    /// The constructed UIConfig instance
    pub fn build(self) -> UIConfig {
        self.config
    }
}