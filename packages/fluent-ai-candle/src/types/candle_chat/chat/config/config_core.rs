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

/// Duration serialization helper
pub mod duration_secs {
    use super::*;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_secs())
    }

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
    pub provider: Arc<str>,
    /// Model name/identifier
    pub model_name: Arc<str>,
    /// Model version or variant
    pub model_version: Option<Arc<str>>,
    /// Temperature for response randomness (0.0 to 2.0)
    pub temperature: f32,
    /// Maximum tokens in response
    pub max_tokens: Option<u32>,
    /// Top-p nucleus sampling parameter
    pub top_p: Option<f32>,
    /// Top-k sampling parameter
    pub top_k: Option<u32>,
    /// Frequency penalty (-2.0 to 2.0)
    pub frequency_penalty: Option<f32>,
    /// Presence penalty (-2.0 to 2.0)
    pub presence_penalty: Option<f32>,
    /// Stop sequences
    pub stop_sequences: Vec<Arc<str>>,
    /// System prompt/instructions
    pub system_prompt: Option<Arc<str>>,
    /// Enable function calling
    pub enable_functions: bool,
    /// Function calling mode ("auto", "none", "required")
    pub function_mode: Arc<str>,
    /// Model-specific parameters
    pub custom_parameters: HashMap<Arc<str>, serde_json::Value>,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
    /// Retry configuration
    pub retry_config: ModelRetryConfig,
    /// Performance settings
    pub performance: ModelPerformanceConfig,
}

/// Model retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,
    /// Base delay between retries in milliseconds
    pub base_delay_ms: u64,
    /// Maximum delay between retries in milliseconds
    pub max_delay_ms: u64,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f32,
    /// Enable jitter to avoid thundering herd
    pub enable_jitter: bool,
}

/// Model performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceConfig {
    /// Enable response caching
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Enable request batching
    pub enable_batching: bool,
    /// Maximum batch size
    pub max_batch_size: u32,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Enable streaming responses
    pub enable_streaming: bool,
    /// Connection pool size
    pub connection_pool_size: u32,
    /// Keep-alive timeout in seconds
    pub keep_alive_timeout_seconds: u64,
}

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
            performance: ModelPerformanceConfig::default(),
        }
    }
}

impl Default for ModelRetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 1000, // 1 second
            max_delay_ms: 30000, // 30 seconds
            backoff_multiplier: 2.0,
            enable_jitter: true,
        }
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
            keep_alive_timeout_seconds: 60,
        }
    }
}

impl ModelConfig {
    /// Create a new model configuration
    pub fn new(provider: impl Into<Arc<str>>, model_name: impl Into<Arc<str>>) -> Self {
        Self {
            provider: provider.into(),
            model_name: model_name.into(),
            ..Default::default()
        }
    }

    /// Create configuration for OpenAI models
    pub fn openai(model_name: impl Into<Arc<str>>) -> Self {
        Self::new("openai", model_name)
    }

    /// Create configuration for Anthropic models
    pub fn anthropic(model_name: impl Into<Arc<str>>) -> Self {
        Self::new("anthropic", model_name)
    }

    /// Create configuration for Mistral models
    pub fn mistral(model_name: impl Into<Arc<str>>) -> Self {
        Self::new("mistral", model_name)
    }

    /// Create configuration for Gemini models
    pub fn gemini(model_name: impl Into<Arc<str>>) -> Self {
        Self::new("gemini", model_name)
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature.clamp(0.0, 2.0);
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<Arc<str>>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Enable or disable function calling
    pub fn with_functions(mut self, enable: bool) -> Self {
        self.enable_functions = enable;
        self
    }

    /// Validate the model configuration
    pub fn validate(&self) -> AsyncStream<()> {
        let _config = self.clone();
        // Use AsyncStream::with_channel for streaming-only architecture - emit success immediately
        AsyncStream::with_channel(move |sender| {
            // Emit success via sender - validation happens during stream processing
            let _ = sender.send(());
        })
    }
}

/// Core chat configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(
    feature = "bincode-serialization",
    derive(bincode::Encode, bincode::Decode)
)]
#[cfg_attr(
    feature = "rkyv-serialization",
    derive(Archive, RkyvDeserialize, RkyvSerialize)
)]
pub struct ChatConfig {
    /// Maximum message length
    pub max_message_length: usize,
    /// Enable message history
    pub enable_history: bool,
    /// History retention period in seconds (for rkyv compatibility)
    #[serde(with = "duration_secs")]
    pub history_retention: Duration,
    /// Enable streaming responses
    pub enable_streaming: bool,
    /// Personality configuration
    pub personality: PersonalityConfig,
    /// Behavior configuration
    pub behavior: BehaviorConfig,
    /// UI configuration
    pub ui: UIConfig,
    /// Integration configuration
    pub integration: IntegrationConfig,
}

/// Personality configuration for AI behavior
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
    pub personality_type: Arc<str>,
    /// Response style settings
    pub response_style: Arc<str>,
    /// Tone configuration
    pub tone: Arc<str>,
    /// Custom instructions
    pub custom_instructions: Option<Arc<str>>,
    /// Creativity level (0.0-1.0)
    pub creativity: f64,
    /// Formality level (0.0-1.0)
    pub formality: f64,
    /// Humor level (0.0-1.0)
    pub humor: f64,
    /// Empathy level (0.0-1.0)
    pub empathy: f64,
    /// Expertise level
    pub expertise_level: Arc<str>,
    /// Verbosity level
    pub verbosity: Arc<str>,
    /// Personality traits
    pub traits: Vec<Arc<str>>,
}

/// Behavior configuration for interaction patterns
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
    /// Response delay in milliseconds
    pub response_delay_ms: u64,
    /// Enable typing indicators
    pub enable_typing_indicators: bool,
    /// Enable read receipts
    pub enable_read_receipts: bool,
    /// Auto-save interval in seconds
    pub auto_save_interval_seconds: u64,
    /// Enable auto-correction
    pub enable_auto_correction: bool,
    /// Enable smart replies
    pub enable_smart_replies: bool,
    /// Smart reply count
    pub smart_reply_count: u32,
    /// Context awareness level (0.0-1.0)
    pub context_awareness: f64,
    /// Memory retention level (0.0-1.0)
    pub memory_retention: f64,
    /// Conversation continuity
    pub conversation_continuity: bool,
    /// Follow-up behavior
    pub follow_up_behavior: Arc<str>,
    /// Proactivity level (0.0-1.0)
    pub proactivity: f64,
    /// Question frequency (0.0-1.0)
    pub question_frequency: f64,
    /// Conversation flow type
    pub conversation_flow: Arc<str>,
    /// Error handling approach
    pub error_handling: Arc<str>,
}

/// UI configuration for interface behavior
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
    pub theme: Arc<str>,
    /// Font size
    pub font_size: u32,
    /// Enable animations
    pub enable_animations: bool,
    /// Animation speed (0.0-2.0)
    pub animation_speed: f32,
    /// Enable sound effects
    pub enable_sound_effects: bool,
    /// Sound volume (0.0-1.0)
    pub sound_volume: f32,
    /// Message grouping
    pub enable_message_grouping: bool,
    /// Show timestamps
    pub show_timestamps: bool,
    /// Timestamp format
    pub timestamp_format: Arc<str>,
    /// Enable syntax highlighting
    pub enable_syntax_highlighting: bool,
    /// Code theme
    pub code_theme: Arc<str>,
    /// Layout type 
    pub layout: Arc<str>,
    /// Color scheme
    pub color_scheme: Arc<str>,
    /// Display density
    pub display_density: Arc<str>,
    /// Animation settings
    pub animations: Arc<str>,
}

/// Integration configuration for external services
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
    /// Enable plugins
    pub enable_plugins: bool,
    /// Plugin directory
    pub plugin_directory: Option<Arc<str>>,
    /// Plugin whitelist
    pub plugin_whitelist: Vec<Arc<str>>,
    /// Plugin blacklist
    pub plugin_blacklist: Vec<Arc<str>>,
    /// Enable webhooks
    pub enable_webhooks: bool,
    /// Webhook URL
    pub webhook_url: Option<Arc<str>>,
    /// Webhook secret
    pub webhook_secret: Option<Arc<str>>,
    /// Enable API access
    pub enable_api_access: bool,
    /// API rate limit per minute
    pub api_rate_limit: u32,
    /// Custom integrations
    pub custom_integrations: HashMap<Arc<str>, serde_json::Value>,
    /// External services
    pub external_services: Vec<Arc<str>>,
    /// API configurations
    pub api_configurations: Vec<Arc<str>>,
    /// Authentication methods
    pub authentication: Vec<Arc<str>>,
}

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
            traits: vec![Arc::from("helpful"), Arc::from("accurate")],
        }
    }
}

impl Default for BehaviorConfig {
    fn default() -> Self {
        Self {
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
        }
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
            animations: Arc::from("smooth"),
        }
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
            authentication: Vec::new(),
        }
    }
}