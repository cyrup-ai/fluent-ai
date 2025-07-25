//! UI configuration for chat interface
//!
//! This module defines configuration structures for the chat user interface,
//! including theme, layout, animation, and accessibility settings.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

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

// Default implementations
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