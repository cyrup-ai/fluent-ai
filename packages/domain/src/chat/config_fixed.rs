//! Configuration management system for chat features
//!
//! This module provides a comprehensive configuration management system with atomic updates,
//! validation, persistence, and change notifications using zero-allocation patterns and
//! lock-free operations for blazing-fast performance.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use atomic_counter::{AtomicCounter, ConsistentCounter};
use crossbeam_queue::SegQueue;
use serde::{Deserialize, Serialize, Deserializer, Serializer};
use tokio::sync::{RwLock, broadcast};
use uuid::Uuid;

/// Duration serialization helper for rkyv compatibility
mod duration_secs {
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

/// Core chat configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
#[derive(Debug, Clone, Serialize, Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
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

/// Behavior configuration for chat system
#[derive(Debug, Clone, Serialize, Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BehaviorConfig {
    /// Enable auto-responses
    pub auto_response: bool,
    /// Response delay settings
    pub response_delay: Duration,
    /// Enable message filtering
    pub enable_filtering: bool,
    /// Maximum concurrent conversations
    pub max_concurrent_chats: usize,
    /// Proactivity level (0.0-1.0)
    pub proactivity: f64,
    /// Question frequency (0.0-1.0)
    pub question_frequency: f64,
    /// Conversation flow style
    pub conversation_flow: Arc<str>,
    /// Follow-up behavior pattern
    pub follow_up_behavior: Arc<str>,
    /// Error handling approach
    pub error_handling: Arc<str>,
}

/// User interface configuration
#[derive(Debug, Clone, Serialize, Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct UIConfig {
    /// Theme settings
    pub theme: Arc<str>,
    /// Font size
    pub font_size: u32,
    /// Enable dark mode
    pub dark_mode: bool,
    /// Animation settings
    pub enable_animations: bool,
    /// Layout configuration
    pub layout: Arc<str>,
    /// Color scheme settings
    pub color_scheme: Arc<str>,
    /// Display density preference
    pub display_density: Arc<str>,
    /// Animation style configuration
    pub animations: Arc<str>,
}

/// Integration configuration for external services
#[derive(Debug, Clone, Serialize, Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct IntegrationConfig {
    /// Enabled integrations
    pub enabled_integrations: Vec<Arc<str>>,
    /// API keys and tokens
    pub credentials: HashMap<Arc<str>, Arc<str>>,
    /// Webhook configuration
    pub webhooks: Vec<Arc<str>>,
    /// External services configuration
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
            response_style: Arc::from("helpful"),
            tone: Arc::from("neutral"),
            custom_instructions: None,
            creativity: 0.5,
            formality: 0.5,
            humor: 0.3,
            empathy: 0.7,
            expertise_level: Arc::from("intermediate"),
            verbosity: Arc::from("balanced"),
            traits: Vec::new(),
        }
    }
}

impl Default for BehaviorConfig {
    fn default() -> Self {
        Self {
            auto_response: false,
            response_delay: Duration::from_millis(500),
            enable_filtering: true,
            max_concurrent_chats: 10,
            proactivity: 0.5,
            question_frequency: 0.3,
            conversation_flow: Arc::from("natural"),
            follow_up_behavior: Arc::from("contextual"),
            error_handling: Arc::from("graceful"),
        }
    }
}

impl Default for UIConfig {
    fn default() -> Self {
        Self {
            theme: Arc::from("default"),
            font_size: 14,
            dark_mode: false,
            enable_animations: true,
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
            enabled_integrations: Vec::new(),
            credentials: HashMap::new(),
            webhooks: Vec::new(),
            external_services: Vec::new(),
            api_configurations: Vec::new(),
            authentication: Vec::new(),
        }
    }
}