//! Core chat configuration types and main ChatConfig struct
//!
//! This module provides the primary configuration structure for chat behavior,
//! integrating all configuration aspects with zero-allocation patterns.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::{
    SessionConfig, PersonalityConfig, BehaviorConfig, UIConfig, 
    IntegrationConfig, HistoryConfig, SecurityConfig, PerformanceConfig
};

/// Configuration for chat behavior and settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatConfig {
    /// Chat session settings
    pub session: SessionConfig,
    /// Personality configuration
    pub personality: PersonalityConfig,
    /// Behavior configuration
    pub behavior: BehaviorConfig,
    /// UI configuration
    pub ui: UIConfig,
    /// Integration settings
    pub integrations: IntegrationConfig,
    /// Custom chat settings
    pub custom_settings: HashMap<String, serde_json::Value>,
    /// Chat history settings
    pub history: HistoryConfig,
    /// Security settings
    pub security: SecurityConfig,
    /// Performance settings
    pub performance: PerformanceConfig,
}

/// Session configuration for chat management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Maximum session duration in minutes
    pub max_duration_minutes: Option<u64>,
    /// Auto-save interval in seconds
    pub auto_save_interval_seconds: u64,
    /// Session timeout in minutes
    pub timeout_minutes: u64,
    /// Enable session persistence
    pub enable_persistence: bool,
    /// Session storage location
    pub storage_path: Option<String>,
    /// Maximum concurrent sessions
    pub max_concurrent_sessions: usize,
    /// Session cleanup interval
    pub cleanup_interval_minutes: u64,
}

/// Personality configuration for AI behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityConfig {
    /// Personality name
    pub name: Arc<str>,
    /// Personality description
    pub description: Option<String>,
    /// Response style
    pub response_style: ResponseStyle,
    /// Formality level (0.0 = casual, 1.0 = formal)
    pub formality_level: f32,
    /// Enthusiasm level (0.0 = reserved, 1.0 = enthusiastic)
    pub enthusiasm_level: f32,
    /// Helpfulness level (0.0 = minimal, 1.0 = maximum)
    pub helpfulness_level: f32,
    /// Creativity level (0.0 = conservative, 1.0 = creative)
    pub creativity_level: f32,
    /// Verbosity level (0.0 = concise, 1.0 = verbose)
    pub verbosity_level: f32,
    /// Humor level (0.0 = serious, 1.0 = humorous)
    pub humor_level: f32,
    /// Empathy level (0.0 = analytical, 1.0 = empathetic)
    pub empathy_level: f32,
    /// Custom personality traits
    pub custom_traits: HashMap<String, f32>,
    /// Personality prompts
    pub prompts: Vec<String>,
    /// Personality examples
    pub examples: Vec<PersonalityExample>,
}

/// Response style options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseStyle {
    /// Professional and business-like
    Professional,
    /// Casual and friendly
    Casual,
    /// Academic and scholarly
    Academic,
    /// Creative and artistic
    Creative,
    /// Technical and precise
    Technical,
    /// Conversational and natural
    Conversational,
    /// Custom style with parameters
    Custom(HashMap<String, String>),
}

/// Example for personality training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityExample {
    /// Input prompt
    pub input: String,
    /// Expected response
    pub output: String,
    /// Context information
    pub context: Option<String>,
    /// Example weight
    pub weight: f32,
}

// Default implementations
impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            session: SessionConfig::default(),
            personality: PersonalityConfig::default(),
            behavior: BehaviorConfig::default(),
            ui: UIConfig::default(),
            integrations: IntegrationConfig::default(),
            custom_settings: HashMap::new(),
            history: HistoryConfig::default(),
            security: SecurityConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_duration_minutes: None,
            auto_save_interval_seconds: 300,
            timeout_minutes: 60,
            enable_persistence: true,
            storage_path: None,
            max_concurrent_sessions: 10,
            cleanup_interval_minutes: 60,
        }
    }
}

impl Default for PersonalityConfig {
    fn default() -> Self {
        Self {
            name: Arc::from("Assistant"),
            description: Some("A helpful AI assistant".to_string()),
            response_style: ResponseStyle::Conversational,
            formality_level: 0.5,
            enthusiasm_level: 0.6,
            helpfulness_level: 0.9,
            creativity_level: 0.7,
            verbosity_level: 0.5,
            humor_level: 0.3,
            empathy_level: 0.8,
            custom_traits: HashMap::new(),
            prompts: Vec::new(),
            examples: Vec::new(),
        }
    }
}