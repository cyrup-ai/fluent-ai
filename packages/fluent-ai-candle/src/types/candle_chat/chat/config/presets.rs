//! Configuration presets for common use cases
//!
//! This module provides pre-configured chat configurations for common scenarios.

use super::core::ChatConfig;
use super::config_builder::{ConfigurationBuilder, PersonalityConfigBuilder, BehaviorConfigBuilder, ModelConfigBuilder};

/// Professional assistant configuration for business and formal interactions
///
/// Creates a chat configuration optimized for professional environments, business
/// communications, and formal assistance scenarios. This preset emphasizes accuracy,
/// detailed responses, and maintains appropriate professional boundaries.
///
/// # Personality Characteristics
/// - **Formality**: High (0.8) - Uses professional language and structure
/// - **Creativity**: Low (0.3) - Prioritizes accuracy over creative expression
/// - **Empathy**: Moderate (0.5) - Professional understanding without over-personalization  
/// - **Humor**: Minimal (0.1) - Maintains serious, business-appropriate tone
///
/// # Behavior Configuration
/// - **Content Filtering**: Strict - Ensures workplace-appropriate responses
/// - **Typing Speed**: 75 CPS - Professional communication pace
/// - **Response Delay**: 800ms - Thoughtful response timing
///
/// # Ideal Use Cases
/// - Business consulting and advisory roles
/// - Corporate customer service and support
/// - Professional documentation and communication
/// - Executive assistant functionality
/// - Legal and financial advisory contexts
///
/// # Example Usage
/// ```rust
/// let config = professional();
/// let chat = ChatSession::with_config(config);
/// // Chat will maintain professional tone and provide detailed, accurate responses
/// ```
pub fn professional() -> ChatConfig {
    ConfigurationBuilder::new()
        .name("Professional Assistant")
        .personality_with(|p: PersonalityConfigBuilder| {
            p.personality_type("professional")
                .response_style("detailed")
                .tone("formal")
                .formality(0.8)
                .creativity(0.3)
                .empathy(0.5)
                .humor(0.1)
        })
        .behavior_with(|b: BehaviorConfigBuilder| {
            b.content_filtering("strict")
                .typing_speed_cps(75.0)
                .response_delay_ms(800)
        })
        .build()
}

/// Casual friend configuration for relaxed, informal conversations
///
/// Creates a chat configuration that mimics a friendly, approachable companion
/// for everyday conversations, social interactions, and casual assistance.
/// This preset emphasizes warmth, creativity, and natural conversational flow.
///
/// # Personality Characteristics
/// - **Formality**: Low (0.2) - Uses relaxed, informal language
/// - **Creativity**: High (0.7) - Encourages creative and varied responses
/// - **Empathy**: High (0.8) - Shows genuine care and emotional understanding
/// - **Humor**: Moderate (0.6) - Incorporates appropriate humor and wit
///
/// # Behavior Configuration
/// - **Content Filtering**: Basic - Allows casual language while maintaining safety
/// - **Typing Speed**: 60 CPS - Natural, unhurried conversation pace
/// - **Response Delay**: 300ms - Quick, responsive interaction
///
/// # Ideal Use Cases
/// - Personal assistant for daily tasks and reminders
/// - Social companion for entertainment and conversation
/// - Creative brainstorming and idea generation
/// - Casual learning and exploration
/// - Gaming and entertainment contexts
///
/// # Example Usage
/// ```rust
/// let config = casual();
/// let chat = ChatSession::with_config(config);
/// // Chat will be friendly, creative, and use informal language
/// ```
pub fn casual() -> ChatConfig {
    ConfigurationBuilder::new()
        .name("Casual Friend")
        .personality_with(|p: PersonalityConfigBuilder| {
            p.personality_type("casual")
                .response_style("conversational")
                .tone("friendly")
                .formality(0.2)
                .creativity(0.7)
                .empathy(0.8)
                .humor(0.6)
        })
        .behavior_with(|b: BehaviorConfigBuilder| {
            b.content_filtering("basic")
                .typing_speed_cps(60.0)
                .response_delay_ms(300)
        })
        .build()
}

/// Creative partner configuration for artistic and innovative collaboration
///
/// Creates a chat configuration optimized for creative work, artistic projects,
/// brainstorming sessions, and innovative problem-solving. This preset maximizes
/// creative expression while maintaining coherent, detailed responses.
///
/// # Personality Characteristics
/// - **Formality**: Low (0.3) - Relaxed tone that encourages creative expression
/// - **Creativity**: Maximum (0.9) - Highly imaginative and original responses
/// - **Empathy**: High (0.7) - Understanding of creative processes and artistic vision
/// - **Humor**: Moderate (0.5) - Playful and witty when appropriate
///
/// # Model Configuration
/// - **Temperature**: High (0.9) - Maximum creativity and variation in responses
/// - **Max Tokens**: Extended (4096) - Allows for detailed creative outputs
/// - **Response Style**: Detailed - Comprehensive exploration of ideas
///
/// # Ideal Use Cases
/// - Creative writing and storytelling assistance
/// - Art and design concept development
/// - Marketing and advertising brainstorming
/// - Innovation workshops and ideation sessions
/// - Music, poetry, and creative content generation
/// - Product design and creative problem-solving
///
/// # Example Usage
/// ```rust
/// let config = creative();
/// let chat = ChatSession::with_config(config);
/// // Chat will provide highly creative, imaginative responses for artistic projects
/// ```
pub fn creative() -> ChatConfig {
    ConfigurationBuilder::new()
        .name("Creative Partner")
        .personality_with(|p: PersonalityConfigBuilder| {
            p.personality_type("creative")
                .response_style("detailed")
                .tone("enthusiastic")
                .formality(0.3)
                .creativity(0.9)
                .empathy(0.7)
                .humor(0.5)
        })
        .model_with(|m: ModelConfigBuilder| {
            m.temperature(0.9)
                .max_tokens(4096)
        })
        .build()
}

/// Technical expert configuration for programming and engineering assistance
///
/// Creates a chat configuration specialized for technical discussions, programming
/// help, engineering problem-solving, and detailed technical documentation.
/// This preset prioritizes accuracy, precision, and comprehensive technical explanations.
///
/// # Personality Characteristics
/// - **Formality**: Moderate-High (0.6) - Professional technical communication
/// - **Creativity**: Moderate (0.4) - Balanced approach to technical solutions
/// - **Empathy**: Moderate (0.5) - Understanding without emotional bias
/// - **Humor**: Low (0.2) - Minimal humor to maintain technical focus
///
/// # Model Configuration
/// - **Temperature**: Low (0.3) - Precise, consistent technical responses
/// - **Max Tokens**: Extended (4096) - Detailed technical explanations
/// - **System Prompt**: Technical expert guidance with structured information
/// - **Response Style**: Detailed - Comprehensive technical coverage
///
/// # Ideal Use Cases
/// - Software development and debugging assistance
/// - Code review and optimization recommendations
/// - Technical architecture and design discussions
/// - Engineering problem-solving and troubleshooting
/// - Documentation and technical writing
/// - System administration and DevOps guidance
/// - Scientific and mathematical computations
///
/// # Example Usage
/// ```rust
/// let config = technical();
/// let chat = ChatSession::with_config(config);
/// // Chat will provide precise, detailed technical assistance
/// ```
pub fn technical() -> ChatConfig {
    ConfigurationBuilder::new()
        .name("Technical Expert")
        .personality_with(|p: PersonalityConfigBuilder| {
            p.personality_type("technical")
                .response_style("detailed")
                .tone("neutral")
                .formality(0.6)
                .creativity(0.4)
                .empathy(0.5)
                .humor(0.2)
        })
        .model_with(|m: ModelConfigBuilder| {
            m.temperature(0.3)
                .max_tokens(4096)
                .system_prompt("You are a technical expert. Provide accurate, detailed, and well-structured technical information.")
        })
        .build()
}

/// Customer support configuration for service and assistance interactions
///
/// Creates a chat configuration optimized for customer service scenarios,
/// help desk operations, and user support contexts. This preset balances
/// professionalism with approachability to provide excellent customer experiences.
///
/// # Personality Characteristics
/// - **Formality**: Moderate (0.5) - Professional yet approachable tone
/// - **Creativity**: Low (0.3) - Focus on clear, helpful responses
/// - **Empathy**: Maximum (0.9) - High emotional intelligence and understanding
/// - **Humor**: Minimal (0.2) - Light, appropriate humor when suitable
///
/// # Behavior Configuration
/// - **Content Filtering**: Basic - Safe for customer interactions
/// - **Response Delay**: 200ms - Quick response for customer satisfaction
/// - **Typing Indicators**: Enabled - Shows active engagement
/// - **Typing Speed**: 80 CPS - Professional communication pace
///
/// # Ideal Use Cases
/// - Customer service and support tickets
/// - Product help and troubleshooting
/// - Order management and inquiries
/// - Technical support for non-technical users
/// - Complaint resolution and feedback handling
/// - Pre-sales and product information
///
/// # Example Usage
/// ```rust
/// let config = customer_support();
/// let chat = ChatSession::with_config(config);
/// // Chat will provide empathetic, helpful customer service responses
/// ```
pub fn customer_support() -> ChatConfig {
    ConfigurationBuilder::new()
        .name("Customer Support")
        .personality_with(|p: PersonalityConfigBuilder| {
            p.personality_type("professional")
                .response_style("concise")
                .tone("friendly")
                .formality(0.5)
                .creativity(0.3)
                .empathy(0.9)
                .humor(0.2)
        })
        .behavior_with(|b: BehaviorConfigBuilder| {
            b.content_filtering("basic")
                .response_delay_ms(200)
                .enable_typing_indicators(true)
                .typing_speed_cps(80.0)
        })
        .build()
}

/// Gaming companion configuration for entertainment and casual gaming
///
/// Creates a chat configuration designed for gaming contexts, entertainment
/// interactions, and casual social gaming experiences. This preset maximizes
/// fun, engagement, and energy while maintaining a relaxed, friendly atmosphere.
///
/// # Personality Characteristics
/// - **Formality**: Minimal (0.1) - Very casual, gaming-appropriate language
/// - **Creativity**: High (0.8) - Creative and varied gaming-related responses
/// - **Empathy**: Moderate (0.6) - Understanding of gaming experiences and emotions
/// - **Humor**: High (0.8) - Frequent, appropriate gaming humor and references
///
/// # Behavior Configuration
/// - **Content Filtering**: None - Allows gaming language and expressions
/// - **Typing Speed**: 90 CPS - Fast-paced interaction for gaming contexts
/// - **Response Delay**: 100ms - Immediate, responsive gaming interaction
///
/// # Ideal Use Cases
/// - Gaming session companion and commentary
/// - Multiplayer game coordination and strategy
/// - Game recommendation and discovery
/// - Gaming community interaction and chat
/// - Esports discussion and analysis
/// - Casual gaming entertainment and banter
/// - Game streaming and content creation support
///
/// # Example Usage
/// ```rust
/// let config = gaming_companion();
/// let chat = ChatSession::with_config(config);
/// // Chat will be energetic, fun, and gaming-focused
/// ```
pub fn gaming_companion() -> ChatConfig {
    ConfigurationBuilder::new()
        .name("Gaming Buddy")
        .personality_with(|p: PersonalityConfigBuilder| {
            p.personality_type("casual")
                .response_style("conversational")
                .tone("enthusiastic")
                .formality(0.1)
                .creativity(0.8)
                .empathy(0.6)
                .humor(0.8)
        })
        .behavior_with(|b: BehaviorConfigBuilder| {
            b.content_filtering("none")
                .typing_speed_cps(90.0)
                .response_delay_ms(100)
        })
        .build()
}

/// Educational tutor configuration for learning and academic assistance
///
/// Creates a chat configuration optimized for educational contexts, tutoring
/// sessions, and academic learning support. This preset balances expertise
/// with patience and encouragement to create an effective learning environment.
///
/// # Personality Characteristics
/// - **Formality**: Moderate (0.4) - Professional yet approachable for learning
/// - **Creativity**: Moderate-High (0.6) - Creative teaching methods and examples
/// - **Empathy**: High (0.8) - Patient understanding of learning challenges
/// - **Humor**: Light (0.3) - Appropriate educational humor to aid learning
///
/// # Model Configuration
/// - **Temperature**: Moderate (0.5) - Balanced creativity and accuracy for teaching
/// - **Max Tokens**: Extended (3000) - Detailed explanations and examples
/// - **System Prompt**: Educational guidance with clear explanations and encouragement
/// - **Response Style**: Detailed - Comprehensive educational coverage
///
/// # Behavior Configuration
/// - **Content Filtering**: Strict - Ensures appropriate educational content
/// - **Typing Speed**: 65 CPS - Thoughtful, educational pace
///
/// # Ideal Use Cases
/// - Academic tutoring and homework assistance
/// - Concept explanation and clarification
/// - Study planning and learning strategies
/// - Educational content creation and review
/// - Skill development and practice guidance
/// - Exam preparation and test-taking strategies
/// - Research assistance and academic writing support
///
/// # Example Usage
/// ```rust
/// let config = educational_tutor();
/// let chat = ChatSession::with_config(config);
/// // Chat will provide patient, encouraging educational assistance
/// ```
pub fn educational_tutor() -> ChatConfig {
    ConfigurationBuilder::new()
        .name("Educational Tutor")
        .personality_with(|p: PersonalityConfigBuilder| {
            p.personality_type("professional")
                .response_style("detailed")
                .tone("friendly")
                .formality(0.4)
                .creativity(0.6)
                .empathy(0.8)
                .humor(0.3)
        })
        .model_with(|m: ModelConfigBuilder| {
            m.temperature(0.5)
                .max_tokens(3000)
                .system_prompt("You are an educational tutor. Explain concepts clearly, provide examples, and encourage learning.")
        })
        .behavior_with(|b: BehaviorConfigBuilder| {
            b.content_filtering("strict")
                .typing_speed_cps(65.0)
        })
        .build()
}

/// Therapy assistant configuration for supportive and empathetic interactions
///
/// Creates a chat configuration designed for therapeutic support, mental health
/// assistance, and emotionally supportive conversations. This preset maximizes
/// empathy and active listening while maintaining appropriate professional boundaries.
///
/// # Important Note
/// This configuration is designed for supportive assistance only and does not
/// replace professional mental health services. Always recommend professional
/// help for serious mental health concerns.
///
/// # Personality Characteristics
/// - **Formality**: Low (0.3) - Warm, approachable therapeutic communication
/// - **Creativity**: Moderate (0.4) - Thoughtful, varied therapeutic responses
/// - **Empathy**: Maximum (0.95) - Exceptional emotional understanding and validation
/// - **Humor**: Minimal (0.1) - Serious, supportive tone with rare light moments
///
/// # Model Configuration
/// - **Temperature**: Moderate (0.6) - Balanced responses for therapeutic context
/// - **System Prompt**: Supportive guidance emphasizing active listening and empathy
///
/// # Behavior Configuration
/// - **Content Filtering**: Strict - Ensures safe, appropriate therapeutic responses
/// - **Typing Speed**: 55 CPS - Slow, thoughtful therapeutic pace
/// - **Response Delay**: 1000ms - Reflective timing that shows deep consideration
///
/// # Ideal Use Cases
/// - Emotional support and active listening
/// - Stress management and coping strategies
/// - Personal reflection and self-awareness
/// - Goal setting and motivation support
/// - Relationship and communication guidance
/// - Mindfulness and mental wellness practices
///
/// # Example Usage
/// ```rust
/// let config = therapy_assistant();
/// let chat = ChatSession::with_config(config);
/// // Chat will provide empathetic, supportive therapeutic-style responses
/// ```
pub fn therapy_assistant() -> ChatConfig {
    ConfigurationBuilder::new()
        .name("Therapy Assistant")
        .personality_with(|p: PersonalityConfigBuilder| {
            p.personality_type("professional")
                .response_style("conversational")
                .tone("neutral")
                .formality(0.3)
                .creativity(0.4)
                .empathy(0.95)
                .humor(0.1)
        })
        .model_with(|m: ModelConfigBuilder| {
            m.temperature(0.6)
                .system_prompt("You are a supportive assistant. Listen actively, ask thoughtful questions, and provide empathetic responses.")
        })
        .behavior_with(|b: BehaviorConfigBuilder| {
            b.content_filtering("strict")
                .typing_speed_cps(55.0)
                .response_delay_ms(1000)
        })
        .build()
}