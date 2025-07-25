//! Core chat configuration types
//!
//! This module defines the main chat configuration structure and related types.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::model::ModelConfig;
use super::config_core::{BehaviorConfig, UIConfig};

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
    pub integrations: IntegrationConfig}

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
    /// Expertise level ("beginner", "intermediate", "advanced", "expert")
    pub expertise_level: Arc<str>,
    /// Verbosity level ("concise", "balanced", "detailed")
    pub verbosity: Arc<str>}

// BehaviorConfig moved to config_core.rs - use that version instead
// This duplicate has been eliminated to resolve "Legacy" confusion

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
    pub enable_translation: bool}

// UIConfig moved to config_core.rs - use that version instead
// This duplicate has been eliminated to resolve field mismatch issues

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
    pub enable_syntax_highlighting: bool}

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
    pub plugins: Vec<PluginConfig>}

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
    pub headers: HashMap<Arc<str>, Arc<str>>,
    /// Request timeout in milliseconds
    pub timeout_ms: u64}

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
    pub settings: HashMap<Arc<str>, serde_json::Value>}

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
            expertise_level: Arc::from("intermediate"),
            verbosity: Arc::from("balanced")}
    }
}

// BehaviorConfig Default implementation moved to config_core.rs to avoid duplication

impl Default for LanguageHandlingConfig {
    fn default() -> Self {
        Self {
            auto_detect: true,
            preferred_language: Arc::from("en"),
            fallback_language: Arc::from("en"),
            enable_translation: false}
    }
}

// Default implementation already exists in config_core.rs

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            show_timestamps: true,
            show_avatars: true,
            show_typing_indicators: true,
            message_bubble_style: Arc::from("modern"),
            enable_markdown: true,
            enable_syntax_highlighting: true}
    }
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enable_webhooks: false,
            webhook_urls: Vec::new(),
            api_integrations: Vec::new(),
            plugins: Vec::new()}
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
            integrations: IntegrationConfig::default()}
    }
}