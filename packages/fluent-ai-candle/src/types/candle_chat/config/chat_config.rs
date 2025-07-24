//! Chat configuration types and utilities
//!
//! This module provides configuration structures for chat behavior,
//! personality settings, and UI preferences with zero-allocation patterns.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

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

/// UI configuration for chat interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIConfig {
    /// Theme settings
    pub theme: ThemeConfig,
    /// Layout settings
    pub layout: LayoutConfig,
    /// Animation settings
    pub animations: AnimationConfig,
    /// Accessibility settings
    pub accessibility: AccessibilityConfig,
    /// Custom UI elements
    pub custom_elements: HashMap<String, serde_json::Value>,
}

/// Theme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeConfig {
    /// Theme name
    pub name: String,
    /// Color scheme
    pub colors: HashMap<String, String>,
    /// Font settings
    pub fonts: FontConfig,
    /// Dark mode enabled
    pub dark_mode: bool,
    /// Custom CSS
    pub custom_css: Option<String>,
}

/// Font configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontConfig {
    /// Primary font family
    pub primary_font: String,
    /// Secondary font family
    pub secondary_font: String,
    /// Font size in pixels
    pub font_size: u16,
    /// Line height multiplier
    pub line_height: f32,
    /// Font weight
    pub font_weight: FontWeight,
}

/// Font weight options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontWeight {
    /// Thin (100)
    Thin,
    /// Light (300)
    Light,
    /// Normal (400)
    Normal,
    /// Medium (500)
    Medium,
    /// Bold (700)
    Bold,
    /// Extra Bold (800)
    ExtraBold,
}

/// Layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutConfig {
    /// Sidebar position
    pub sidebar_position: SidebarPosition,
    /// Chat width percentage
    pub chat_width_percent: u8,
    /// Message spacing in pixels
    pub message_spacing: u16,
    /// Enable compact mode
    pub compact_mode: bool,
    /// Show timestamps
    pub show_timestamps: bool,
    /// Show avatars
    pub show_avatars: bool,
}

/// Sidebar position options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SidebarPosition {
    /// Left side
    Left,
    /// Right side
    Right,
    /// Hidden
    Hidden,
    /// Floating
    Floating,
}

/// Animation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationConfig {
    /// Enable animations
    pub enabled: bool,
    /// Animation duration in milliseconds
    pub duration_ms: u64,
    /// Animation easing function
    pub easing: EasingFunction,
    /// Reduce motion for accessibility
    pub reduce_motion: bool,
}

/// Animation easing function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EasingFunction {
    /// Linear easing
    Linear,
    /// Ease in
    EaseIn,
    /// Ease out
    EaseOut,
    /// Ease in-out
    EaseInOut,
    /// Custom cubic bezier
    CubicBezier(f32, f32, f32, f32),
}

/// Accessibility configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilityConfig {
    /// Enable screen reader support
    pub screen_reader_support: bool,
    /// High contrast mode
    pub high_contrast: bool,
    /// Large text mode
    pub large_text: bool,
    /// Keyboard navigation only
    pub keyboard_only: bool,
    /// Focus indicators
    pub focus_indicators: bool,
    /// Alternative text for images
    pub alt_text_enabled: bool,
}

/// Integration configuration for external services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Enabled integrations
    pub enabled_integrations: Vec<String>,
    /// Integration settings
    pub integration_settings: HashMap<String, serde_json::Value>,
    /// API keys for integrations
    pub api_keys: HashMap<String, String>,
    /// Webhook configurations
    pub webhooks: Vec<WebhookConfig>,
    /// Plugin configurations
    pub plugins: Vec<PluginConfig>,
}

/// Webhook configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    /// Webhook name
    pub name: String,
    /// Webhook URL
    pub url: String,
    /// Events to trigger on
    pub events: Vec<String>,
    /// Authentication method
    pub auth_method: WebhookAuth,
    /// Enabled flag
    pub enabled: bool,
}

/// Webhook authentication method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebhookAuth {
    /// No authentication
    None,
    /// Bearer token
    Bearer(String),
    /// API key in header
    ApiKey(String, String),
    /// Basic authentication
    Basic(String, String),
}

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin settings
    pub settings: HashMap<String, serde_json::Value>,
    /// Plugin enabled flag
    pub enabled: bool,
    /// Plugin priority
    pub priority: i32,
}

/// History configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryConfig {
    /// Enable history storage
    pub enabled: bool,
    /// Maximum history entries
    pub max_entries: usize,
    /// History retention days
    pub retention_days: u64,
    /// Enable search in history
    pub enable_search: bool,
    /// Enable history export
    pub enable_export: bool,
    /// Compression for old entries
    pub compress_old_entries: bool,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable encryption
    pub enable_encryption: bool,
    /// Encryption algorithm
    pub encryption_algorithm: String,
    /// Enable audit logging
    pub enable_audit_logging: bool,
    /// Session security settings
    pub session_security: SessionSecurityConfig,
    /// Content security policy
    pub content_security_policy: Option<String>,
}

/// Session security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSecurityConfig {
    /// Require authentication
    pub require_auth: bool,
    /// Session token expiry minutes
    pub token_expiry_minutes: u64,
    /// Enable CSRF protection
    pub enable_csrf_protection: bool,
    /// Secure cookie settings
    pub secure_cookies: bool,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable response caching
    pub enable_caching: bool,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// Enable lazy loading
    pub enable_lazy_loading: bool,
    /// Message batch size
    pub message_batch_size: usize,
    /// Debounce delay for typing
    pub debounce_delay_ms: u64,
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

impl Default for UIConfig {
    fn default() -> Self {
        Self {
            theme: ThemeConfig::default(),
            layout: LayoutConfig::default(),
            animations: AnimationConfig::default(),
            accessibility: AccessibilityConfig::default(),
            custom_elements: HashMap::new(),
        }
    }
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enabled_integrations: Vec::new(),
            integration_settings: HashMap::new(),
            api_keys: HashMap::new(),
            webhooks: Vec::new(),
            plugins: Vec::new(),
        }
    }
}

// Additional default implementations for nested structs
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

impl Default for ThemeConfig {
    fn default() -> Self {
        Self {
            name: "Default".to_string(),
            colors: HashMap::new(),
            fonts: FontConfig::default(),
            dark_mode: false,
            custom_css: None,
        }
    }
}

impl Default for FontConfig {
    fn default() -> Self {
        Self {
            primary_font: "Inter, sans-serif".to_string(),
            secondary_font: "JetBrains Mono, monospace".to_string(),
            font_size: 14,
            line_height: 1.5,
            font_weight: FontWeight::Normal,
        }
    }
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            sidebar_position: SidebarPosition::Left,
            chat_width_percent: 70,
            message_spacing: 12,
            compact_mode: false,
            show_timestamps: true,
            show_avatars: true,
        }
    }
}

impl Default for AnimationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            duration_ms: 200,
            easing: EasingFunction::EaseInOut,
            reduce_motion: false,
        }
    }
}

impl Default for AccessibilityConfig {
    fn default() -> Self {
        Self {
            screen_reader_support: true,
            high_contrast: false,
            large_text: false,
            keyboard_only: false,
            focus_indicators: true,
            alt_text_enabled: true,
        }
    }
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: 10000,
            retention_days: 365,
            enable_search: true,
            enable_export: true,
            compress_old_entries: true,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_encryption: true,
            encryption_algorithm: "AES-256-GCM".to_string(),
            enable_audit_logging: true,
            session_security: SessionSecurityConfig::default(),
            content_security_policy: None,
        }
    }
}

impl Default for SessionSecurityConfig {
    fn default() -> Self {
        Self {
            require_auth: false,
            token_expiry_minutes: 60,
            enable_csrf_protection: true,
            secure_cookies: true,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_size_limit: 1000,
            enable_lazy_loading: true,
            message_batch_size: 50,
            debounce_delay_ms: 300,
        }
    }
}