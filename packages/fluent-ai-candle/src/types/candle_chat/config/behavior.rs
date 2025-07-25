//! Behavior configuration for chat interactions
//!
//! This module defines configuration structures for chat behavior,
//! including typing indicators, error handling, content filtering,
//! conversation flow, and custom behavior rules.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Behavior configuration for chat interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorConfig {
    /// Enable proactive suggestions
    pub enable_suggestions: bool,
    /// Enable context awareness
    pub enable_context_awareness: bool,
    /// Enable learning from interactions
    pub enable_learning: bool,
    /// Response delay in milliseconds
    pub response_delay_ms: u64,
    /// Typing indicator settings
    pub typing_indicator: TypingIndicatorConfig,
    /// Error handling behavior
    pub error_handling: ErrorHandlingConfig,
    /// Content filtering settings
    pub content_filtering: ContentFilteringConfig,
    /// Conversation flow settings
    pub conversation_flow: ConversationFlowConfig,
    /// Custom behavior rules
    pub custom_rules: Vec<BehaviorRule>,
}

/// Typing indicator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypingIndicatorConfig {
    /// Enable typing indicator
    pub enabled: bool,
    /// Typing speed (characters per second)
    pub typing_speed_cps: f32,
    /// Show indicator for responses longer than N characters
    pub min_response_length: usize,
    /// Maximum indicator duration in seconds
    pub max_duration_seconds: u64,
}

/// Error handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorHandlingConfig {
    /// Show detailed error messages
    pub show_detailed_errors: bool,
    /// Enable error recovery suggestions
    pub enable_recovery_suggestions: bool,
    /// Maximum retry attempts
    pub max_retry_attempts: usize,
    /// Error logging level
    pub logging_level: ErrorLoggingLevel,
    /// Custom error messages
    pub custom_error_messages: HashMap<String, String>,
}

/// Error logging level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorLoggingLevel {
    /// No error logging
    None,
    /// Log only critical errors
    Critical,
    /// Log all errors
    All,
    /// Log errors with full context
    Verbose,
}

/// Content filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentFilteringConfig {
    /// Enable content filtering
    pub enabled: bool,
    /// Filtering strictness (0.0 = permissive, 1.0 = strict)
    pub strictness_level: f32,
    /// Custom filter rules
    pub custom_filters: Vec<ContentFilter>,
    /// Blocked words/phrases
    pub blocked_content: Vec<String>,
    /// Allowed domains for links
    pub allowed_domains: Vec<String>,
}

/// Content filter rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentFilter {
    /// Filter name
    pub name: String,
    /// Filter pattern (regex)
    pub pattern: String,
    /// Filter action
    pub action: FilterAction,
    /// Filter weight
    pub weight: f32,
}

/// Action to take when filter matches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterAction {
    /// Block the content
    Block,
    /// Warn about the content
    Warn,
    /// Replace with alternative
    Replace(String),
    /// Flag for review
    Flag,
}

/// Conversation flow configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationFlowConfig {
    /// Enable conversation threading
    pub enable_threading: bool,
    /// Maximum conversation depth
    pub max_depth: usize,
    /// Enable topic tracking
    pub enable_topic_tracking: bool,
    /// Context window size
    pub context_window_size: usize,
    /// Enable conversation summarization
    pub enable_summarization: bool,
    /// Summary trigger threshold
    pub summary_trigger_length: usize,
}

/// Custom behavior rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorRule {
    /// Rule name
    pub name: String,
    /// Rule condition
    pub condition: String,
    /// Rule action
    pub action: String,
    /// Rule priority
    pub priority: i32,
    /// Rule enabled flag
    pub enabled: bool,
}

// Default implementations
impl Default for BehaviorConfig {
    fn default() -> Self {
        Self {
            enable_suggestions: true,
            enable_context_awareness: true,
            enable_learning: false,
            response_delay_ms: 500,
            typing_indicator: TypingIndicatorConfig::default(),
            error_handling: ErrorHandlingConfig::default(),
            content_filtering: ContentFilteringConfig::default(),
            conversation_flow: ConversationFlowConfig::default(),
            custom_rules: Vec::new(),
        }
    }
}

impl Default for TypingIndicatorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            typing_speed_cps: 50.0,
            min_response_length: 100,
            max_duration_seconds: 10,
        }
    }
}

impl Default for ErrorHandlingConfig {
    fn default() -> Self {
        Self {
            show_detailed_errors: false,
            enable_recovery_suggestions: true,
            max_retry_attempts: 3,
            logging_level: ErrorLoggingLevel::All,
            custom_error_messages: HashMap::new(),
        }
    }
}

impl Default for ContentFilteringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strictness_level: 0.5,
            custom_filters: Vec::new(),
            blocked_content: Vec::new(),
            allowed_domains: Vec::new(),
        }
    }
}

impl Default for ConversationFlowConfig {
    fn default() -> Self {
        Self {
            enable_threading: true,
            max_depth: 10,
            enable_topic_tracking: true,
            context_window_size: 20,
            enable_summarization: false,
            summary_trigger_length: 1000,
        }
    }
}