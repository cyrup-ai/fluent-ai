//! Core chat configuration types
//!
//! This module defines the main chat configuration structure and related types.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::model::ModelConfig;

/// Core chat configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatConfig {
    /// Unique configuration identifier
    pub id: Uuid,
    /// Configuration name
    pub name: Arc<str>,
    /// Configuration description
    pub description: Option<Arc<str>>,
    /// Model configuration
    pub model: ModelConfig,
    /// Personality configuration
    pub personality: PersonalityConfig,
    /// Behavior configuration
    pub behavior: BehaviorConfig,
    /// User interface configuration
    pub ui: UIConfig,
    /// Integration configuration
    pub integrations: IntegrationConfig,
}

/// Personality configuration for AI behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityConfig {
    /// AI assistant name
    pub name: Arc<str>,
    /// Personality type ("professional", "casual", "creative", "technical")
    pub personality_type: Arc<str>,
    /// Response style ("concise", "detailed", "conversational")
    pub response_style: Arc<str>,
    /// Tone ("formal", "friendly", "neutral", "enthusiastic")
    pub tone: Arc<str>,
    /// Formality level (0.0 to 1.0)
    pub formality_level: f32,
    /// Creativity level (0.0 to 1.0)
    pub creativity_level: f32,
    /// Empathy level (0.0 to 1.0)
    pub empathy_level: f32,
    /// Humor level (0.0 to 1.0)
    pub humor_level: f32,
    /// Custom personality traits
    pub custom_traits: Vec<Arc<str>>,
    /// Personality prompt additions
    pub personality_prompt: Option<Arc<str>>,
}

/// Behavior configuration for chat system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorConfig {
    /// Enable conversation memory
    pub enable_memory: bool,
    /// Memory retention duration
    pub memory_duration: Duration,
    /// Maximum conversation length
    pub max_conversation_length: u32,
    /// Auto-save conversations
    pub auto_save_conversations: bool,
    /// Response delay simulation in milliseconds
    pub response_delay_ms: u64,
    /// Enable typing indicators
    pub enable_typing_indicators: bool,
    /// Typing speed (characters per second)
    pub typing_speed_cps: f32,
    /// Enable message reactions
    pub enable_reactions: bool,
    /// Content filtering level ("none", "basic", "strict")
    pub content_filtering: Arc<str>,
    /// Language detection and handling
    pub language_handling: LanguageHandlingConfig,
}

/// Language handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageHandlingConfig {
    /// Auto-detect user language
    pub auto_detect: bool,
    /// Preferred response language
    pub preferred_language: Arc<str>,
    /// Fallback language
    pub fallback_language: Arc<str>,
    /// Enable translation for unsupported languages
    pub enable_translation: bool,
}

/// User interface configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIConfig {
    /// Theme ("light", "dark", "auto")
    pub theme: Arc<str>,
    /// Font size multiplier
    pub font_size_multiplier: f32,
    /// Enable animations
    pub enable_animations: bool,
    /// Animation speed multiplier
    pub animation_speed: f32,
    /// Enable sound effects
    pub enable_sound_effects: bool,
    /// Sound volume (0.0 to 1.0)
    pub sound_volume: f32,
    /// Display options
    pub display: DisplayConfig,
}

/// Display configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayConfig {
    /// Show timestamps
    pub show_timestamps: bool,
    /// Show avatars
    pub show_avatars: bool,
    /// Show typing indicators
    pub show_typing_indicators: bool,
    /// Message bubble style
    pub message_bubble_style: Arc<str>,
    /// Enable markdown rendering
    pub enable_markdown: bool,
    /// Enable code syntax highlighting
    pub enable_syntax_highlighting: bool,
}

/// Integration configuration for external services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Enable webhooks
    pub enable_webhooks: bool,
    /// Webhook URLs
    pub webhook_urls: Vec<Arc<str>>,
    /// API integrations
    pub api_integrations: Vec<ApiIntegrationConfig>,
    /// Plugin configurations
    pub plugins: Vec<PluginConfig>,
}

/// API integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiIntegrationConfig {
    /// Integration name
    pub name: Arc<str>,
    /// API endpoint
    pub endpoint: Arc<str>,
    /// Authentication method
    pub auth_method: Arc<str>,
    /// API key (if applicable)
    pub api_key: Option<Arc<str>>,
    /// Custom headers
    pub headers: std::collections::HashMap<Arc<str>, Arc<str>>,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
}

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Plugin name
    pub name: Arc<str>,
    /// Plugin version
    pub version: Arc<str>,
    /// Enable plugin
    pub enabled: bool,
    /// Plugin settings
    pub settings: std::collections::HashMap<Arc<str>, serde_json::Value>,
}

impl Default for PersonalityConfig {
    fn default() -> Self {
        Self {
            name: Arc::from("Assistant"),
            personality_type: Arc::from("professional"),
            response_style: Arc::from("conversational"),
            tone: Arc::from("friendly"),
            formality_level: 0.5,
            creativity_level: 0.6,
            empathy_level: 0.7,
            humor_level: 0.3,
            custom_traits: Vec::new(),
            personality_prompt: None,
        }
    }
}

impl Default for BehaviorConfig {
    fn default() -> Self {
        Self {
            enable_memory: true,
            memory_duration: Duration::from_secs(3600), // 1 hour
            max_conversation_length: 100,
            auto_save_conversations: true,
            response_delay_ms: 500,
            enable_typing_indicators: true,
            typing_speed_cps: 50.0,
            enable_reactions: true,
            content_filtering: Arc::from("basic"),
            language_handling: LanguageHandlingConfig::default(),
        }
    }
}

impl Default for LanguageHandlingConfig {
    fn default() -> Self {
        Self {
            auto_detect: true,
            preferred_language: Arc::from("en"),
            fallback_language: Arc::from("en"),
            enable_translation: false,
        }
    }
}

impl Default for UIConfig {
    fn default() -> Self {
        Self {
            theme: Arc::from("auto"),
            font_size_multiplier: 1.0,
            enable_animations: true,
            animation_speed: 1.0,
            enable_sound_effects: false,
            sound_volume: 0.5,
            display: DisplayConfig::default(),
        }
    }
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            show_timestamps: true,
            show_avatars: true,
            show_typing_indicators: true,
            message_bubble_style: Arc::from("modern"),
            enable_markdown: true,
            enable_syntax_highlighting: true,
        }
    }
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enable_webhooks: false,
            webhook_urls: Vec::new(),
            api_integrations: Vec::new(),
            plugins: Vec::new(),
        }
    }
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            name: Arc::from("Default Chat Configuration"),
            description: None,
            model: ModelConfig::default(),
            personality: PersonalityConfig::default(),
            behavior: BehaviorConfig::default(),
            ui: UIConfig::default(),
            integrations: IntegrationConfig::default(),
        }
    }
}