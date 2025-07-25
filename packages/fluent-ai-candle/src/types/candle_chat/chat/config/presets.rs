//! Configuration presets for common use cases
//!
//! This module provides pre-configured chat configurations for common scenarios.

use super::core::ChatConfig;
use super::builder::ConfigurationBuilder;

/// Professional assistant configuration
pub fn professional() -> ChatConfig {
    ConfigurationBuilder::new()
        .name("Professional Assistant")
        .personality(|p| {
            p.personality_type("professional")
                .response_style("detailed")
                .tone("formal")
                .formality_level(0.8)
                .creativity_level(0.3)
                .empathy_level(0.5)
                .humor_level(0.1)
        })
        .behavior(|b| {
            b.content_filtering("strict")
                .typing_speed_cps(75.0)
                .response_delay_ms(800)
        })
        .build()
}

/// Casual friend configuration
pub fn casual() -> ChatConfig {
    ConfigurationBuilder::new()
        .name("Casual Friend")
        .personality(|p| {
            p.personality_type("casual")
                .response_style("conversational")
                .tone("friendly")
                .formality_level(0.2)
                .creativity_level(0.7)
                .empathy_level(0.8)
                .humor_level(0.6)
        })
        .behavior(|b| {
            b.content_filtering("basic")
                .typing_speed_cps(60.0)
                .response_delay_ms(300)
        })
        .build()
}

/// Creative partner configuration
pub fn creative() -> ChatConfig {
    ConfigurationBuilder::new()
        .name("Creative Partner")
        .personality(|p| {
            p.personality_type("creative")
                .response_style("detailed")
                .tone("enthusiastic")
                .formality_level(0.3)
                .creativity_level(0.9)
                .empathy_level(0.7)
                .humor_level(0.5)
        })
        .model(|m| {
            m.temperature(0.9)
                .max_tokens(4096)
        })
        .build()
}

/// Technical expert configuration
pub fn technical() -> ChatConfig {
    ConfigurationBuilder::new()
        .name("Technical Expert")
        .personality(|p| {
            p.personality_type("technical")
                .response_style("detailed")
                .tone("neutral")
                .formality_level(0.6)
                .creativity_level(0.4)
                .empathy_level(0.5)
                .humor_level(0.2)
        })
        .model(|m| {
            m.temperature(0.3)
                .max_tokens(4096)
                .system_prompt("You are a technical expert. Provide accurate, detailed, and well-structured technical information.")
        })
        .build()
}

/// Customer support configuration
pub fn customer_support() -> ChatConfig {
    ConfigurationBuilder::new()
        .name("Customer Support")
        .personality(|p| {
            p.personality_type("professional")
                .response_style("concise")
                .tone("friendly")
                .formality_level(0.5)
                .creativity_level(0.3)
                .empathy_level(0.9)
                .humor_level(0.2)
        })
        .behavior(|b| {
            b.content_filtering("basic")
                .response_delay_ms(200)
                .enable_typing_indicators(true)
                .typing_speed_cps(80.0)
        })
        .build()
}

/// Gaming companion configuration
pub fn gaming_companion() -> ChatConfig {
    ConfigurationBuilder::new()
        .name("Gaming Buddy")
        .personality(|p| {
            p.personality_type("casual")
                .response_style("conversational")
                .tone("enthusiastic")
                .formality_level(0.1)
                .creativity_level(0.8)
                .empathy_level(0.6)
                .humor_level(0.8)
        })
        .behavior(|b| {
            b.content_filtering("none")
                .typing_speed_cps(90.0)
                .response_delay_ms(100)
        })
        .build()
}

/// Educational tutor configuration
pub fn educational_tutor() -> ChatConfig {
    ConfigurationBuilder::new()
        .name("Educational Tutor")
        .personality(|p| {
            p.personality_type("professional")
                .response_style("detailed")
                .tone("friendly")
                .formality_level(0.4)
                .creativity_level(0.6)
                .empathy_level(0.8)
                .humor_level(0.3)
        })
        .model(|m| {
            m.temperature(0.5)
                .max_tokens(3000)
                .system_prompt("You are an educational tutor. Explain concepts clearly, provide examples, and encourage learning.")
        })
        .behavior(|b| {
            b.content_filtering("strict")
                .typing_speed_cps(65.0)
        })
        .build()
}

/// Therapy assistant configuration
pub fn therapy_assistant() -> ChatConfig {
    ConfigurationBuilder::new()
        .name("Therapy Assistant")
        .personality(|p| {
            p.personality_type("professional")
                .response_style("conversational")
                .tone("neutral")
                .formality_level(0.3)
                .creativity_level(0.4)
                .empathy_level(0.95)
                .humor_level(0.1)
        })
        .model(|m| {
            m.temperature(0.6)
                .system_prompt("You are a supportive assistant. Listen actively, ask thoughtful questions, and provide empathetic responses.")
        })
        .behavior(|b| {
            b.content_filtering("strict")
                .typing_speed_cps(55.0)
                .response_delay_ms(1000)
        })
        .build()
}