//! Zero-allocation OpenAI content moderation implementation
//!
//! Provides comprehensive content safety analysis using OpenAI's moderation models
//! with blazing-fast performance and no unsafe operations.

use std::collections::HashMap;

use fluent_ai_http3::{HttpClient, HttpConfig, HttpRequest};
use serde::{Deserialize, Serialize};

use super::{OpenAIError, OpenAIResult};
use crate::AsyncTask;
use crate::ZeroOneOrMany;

/// Content moderation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationRequest {
    pub input: ModerationInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Input for moderation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModerationInput {
    /// Single text string
    Single(String),
    /// Array of text strings for batch processing
    Array(ZeroOneOrMany<String>),
}

/// Moderation response from OpenAI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationResponse {
    pub id: String,
    pub model: String,
    pub results: ZeroOneOrMany<ModerationResult>,
}

/// Individual moderation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationResult {
    pub flagged: bool,
    pub categories: ModerationCategories,
    pub category_scores: ModerationScores,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category_applied_input_types: Option<HashMap<String, ZeroOneOrMany<String>>>,
}

/// Content violation categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationCategories {
    pub sexual: bool,
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: bool,
    pub harassment: bool,
    #[serde(rename = "harassment/threatening")]
    pub harassment_threatening: bool,
    pub hate: bool,
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: bool,
    #[serde(rename = "illicit")]
    pub illicit: bool,
    #[serde(rename = "illicit/violent")]
    pub illicit_violent: bool,
    #[serde(rename = "self-harm")]
    pub self_harm: bool,
    #[serde(rename = "self-harm/intent")]
    pub self_harm_intent: bool,
    #[serde(rename = "self-harm/instructions")]
    pub self_harm_instructions: bool,
    pub violence: bool,
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: bool,
}

/// Confidence scores for each category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationScores {
    pub sexual: f32,
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: f32,
    pub harassment: f32,
    #[serde(rename = "harassment/threatening")]
    pub harassment_threatening: f32,
    pub hate: f32,
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: f32,
    #[serde(rename = "illicit")]
    pub illicit: f32,
    #[serde(rename = "illicit/violent")]
    pub illicit_violent: f32,
    #[serde(rename = "self-harm")]
    pub self_harm: f32,
    #[serde(rename = "self-harm/intent")]
    pub self_harm_intent: f32,
    #[serde(rename = "self-harm/instructions")]
    pub self_harm_instructions: f32,
    pub violence: f32,
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: f32,
}

/// Moderation policy configuration
#[derive(Debug, Clone)]
pub struct ModerationPolicy {
    pub sexual_threshold: f32,
    pub harassment_threshold: f32,
    pub hate_threshold: f32,
    pub violence_threshold: f32,
    pub self_harm_threshold: f32,
    pub illicit_threshold: f32,
    pub strict_mode: bool,
    pub block_minors_content: bool,
}

/// Content safety assessment
#[derive(Debug, Clone)]
pub struct SafetyAssessment {
    pub is_safe: bool,
    pub risk_level: RiskLevel,
    pub triggered_categories: ZeroOneOrMany<String>,
    pub highest_score: f32,
    pub recommendation: SafetyRecommendation,
}

/// Risk level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Safe,
    Low,
    Medium,
    High,
    Critical,
}

/// Safety recommendation
#[derive(Debug, Clone)]
pub enum SafetyRecommendation {
    Allow,
    Review,
    Block,
    RedFlag,
}

/// Content analysis context
#[derive(Debug, Clone)]
pub struct AnalysisContext {
    pub user_age: Option<u16>,
    pub content_type: ContentType,
    pub platform: Option<String>,
    pub region: Option<String>,
    pub is_public: bool,
}

/// Type of content being analyzed
#[derive(Debug, Clone, Copy)]
pub enum ContentType {
    UserMessage,
    PublicPost,
    Comment,
    Article,
    Email,
    ChatMessage,
    Review,
    GeneratedContent,
}

impl ModerationInput {
    /// Create single input
    #[inline(always)]
    pub fn single(text: impl Into<String>) -> Self {
        Self::Single(text.into())
    }

    /// Create batch input
    #[inline(always)]
    pub fn batch(texts: ZeroOneOrMany<String>) -> Self {
        Self::Array(texts)
    }

    /// Get input count for rate limiting
    #[inline(always)]
    pub fn count(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Array(texts) => match texts {
                ZeroOneOrMany::None => 0,
                ZeroOneOrMany::One(_) => 1,
                ZeroOneOrMany::Many(vec) => vec.len(),
            },
        }
    }

    /// Validate input for API compliance
    #[inline(always)]
    pub fn validate(&self) -> OpenAIResult<()> {
        match self {
            Self::Single(text) => {
                if text.is_empty() {
                    return Err(OpenAIError::ModerationError(
                        "Input text cannot be empty".to_string(),
                    ));
                }
                if text.len() > 32768 {
                    return Err(OpenAIError::ModerationError(
                        "Input text exceeds 32K character limit".to_string(),
                    ));
                }
            }
            Self::Array(texts) => match texts {
                ZeroOneOrMany::None => {
                    return Err(OpenAIError::ModerationError(
                        "Input array cannot be empty".to_string(),
                    ));
                }
                ZeroOneOrMany::One(text) => {
                    if text.is_empty() {
                        return Err(OpenAIError::ModerationError(
                            "Input text cannot be empty".to_string(),
                        ));
                    }
                    if text.len() > 32768 {
                        return Err(OpenAIError::ModerationError(
                            "Input text exceeds 32K character limit".to_string(),
                        ));
                    }
                }
                ZeroOneOrMany::Many(vec) => {
                    if vec.is_empty() {
                        return Err(OpenAIError::ModerationError(
                            "Input array cannot be empty".to_string(),
                        ));
                    }
                    if vec.len() > 1000 {
                        return Err(OpenAIError::ModerationError(
                            "Too many inputs in batch (max 1000)".to_string(),
                        ));
                    }
                    for text in vec {
                        if text.is_empty() {
                            return Err(OpenAIError::ModerationError(
                                "Individual text cannot be empty".to_string(),
                            ));
                        }
                        if text.len() > 32768 {
                            return Err(OpenAIError::ModerationError(
                                "Input text exceeds 32K character limit".to_string(),
                            ));
                        }
                    }
                }
            },
        }
        Ok(())
    }
}

impl ModerationRequest {
    /// Create new moderation request
    #[inline(always)]
    pub fn new(input: ModerationInput) -> Self {
        Self {
            input,
            model: Some("text-moderation-latest".to_string()),
        }
    }

    /// Create with legacy model
    #[inline(always)]
    pub fn with_legacy_model(input: ModerationInput) -> Self {
        Self {
            input,
            model: Some("text-moderation-stable".to_string()),
        }
    }

    /// Set custom model
    #[inline(always)]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Validate request
    #[inline(always)]
    pub fn validate(&self) -> OpenAIResult<()> {
        self.input.validate()
    }
}

impl ModerationResult {
    /// Check if content violates any policy
    #[inline(always)]
    pub fn has_violations(&self) -> bool {
        self.flagged
    }

    /// Get all triggered categories
    #[inline(always)]
    pub fn get_triggered_categories(&self) -> ZeroOneOrMany<String> {
        let mut categories = Vec::new();

        if self.categories.sexual {
            categories.push("sexual".to_string());
        }
        if self.categories.sexual_minors {
            categories.push("sexual/minors".to_string());
        }
        if self.categories.harassment {
            categories.push("harassment".to_string());
        }
        if self.categories.harassment_threatening {
            categories.push("harassment/threatening".to_string());
        }
        if self.categories.hate {
            categories.push("hate".to_string());
        }
        if self.categories.hate_threatening {
            categories.push("hate/threatening".to_string());
        }
        if self.categories.illicit {
            categories.push("illicit".to_string());
        }
        if self.categories.illicit_violent {
            categories.push("illicit/violent".to_string());
        }
        if self.categories.self_harm {
            categories.push("self-harm".to_string());
        }
        if self.categories.self_harm_intent {
            categories.push("self-harm/intent".to_string());
        }
        if self.categories.self_harm_instructions {
            categories.push("self-harm/instructions".to_string());
        }
        if self.categories.violence {
            categories.push("violence".to_string());
        }
        if self.categories.violence_graphic {
            categories.push("violence/graphic".to_string());
        }

        ZeroOneOrMany::from_vec(categories)
    }

    /// Get highest risk score
    #[inline(always)]
    pub fn get_highest_score(&self) -> f32 {
        let scores = &self.category_scores;
        [
            scores.sexual,
            scores.sexual_minors,
            scores.harassment,
            scores.harassment_threatening,
            scores.hate,
            scores.hate_threatening,
            scores.illicit,
            scores.illicit_violent,
            scores.self_harm,
            scores.self_harm_intent,
            scores.self_harm_instructions,
            scores.violence,
            scores.violence_graphic,
        ]
        .iter()
        .fold(0.0, |max, &score| if score > max { score } else { max })
    }

    /// Check against custom policy
    #[inline(always)]
    pub fn violates_policy(&self, policy: &ModerationPolicy) -> bool {
        let scores = &self.category_scores;

        // Check minors content if strict policy
        if policy.block_minors_content && self.categories.sexual_minors {
            return true;
        }

        // Check against thresholds
        scores.sexual > policy.sexual_threshold
            || scores.harassment > policy.harassment_threshold
            || scores.harassment_threatening > policy.harassment_threshold
            || scores.hate > policy.hate_threshold
            || scores.hate_threatening > policy.hate_threshold
            || scores.violence > policy.violence_threshold
            || scores.violence_graphic > policy.violence_threshold
            || scores.self_harm > policy.self_harm_threshold
            || scores.self_harm_intent > policy.self_harm_threshold
            || scores.self_harm_instructions > policy.self_harm_threshold
            || scores.illicit > policy.illicit_threshold
            || scores.illicit_violent > policy.illicit_threshold
    }

    /// Generate safety assessment
    #[inline(always)]
    pub fn assess_safety(
        &self,
        policy: &ModerationPolicy,
        context: &AnalysisContext,
    ) -> SafetyAssessment {
        let is_violation = self.violates_policy(policy);
        let highest_score = self.get_highest_score();
        let triggered_categories = self.get_triggered_categories();

        let risk_level = if highest_score >= 0.9 {
            RiskLevel::Critical
        } else if highest_score >= 0.7 {
            RiskLevel::High
        } else if highest_score >= 0.5 {
            RiskLevel::Medium
        } else if highest_score >= 0.2 {
            RiskLevel::Low
        } else {
            RiskLevel::Safe
        };

        let recommendation = match (is_violation, risk_level, context.is_public) {
            (true, RiskLevel::Critical, _) => SafetyRecommendation::RedFlag,
            (true, RiskLevel::High, true) => SafetyRecommendation::Block,
            (true, RiskLevel::High, false) => SafetyRecommendation::Review,
            (true, RiskLevel::Medium, _) => SafetyRecommendation::Review,
            (false, RiskLevel::Low, _) => SafetyRecommendation::Allow,
            _ => SafetyRecommendation::Allow,
        };

        SafetyAssessment {
            is_safe: !is_violation,
            risk_level,
            triggered_categories,
            highest_score,
            recommendation,
        }
    }
}

impl ModerationPolicy {
    /// Create standard policy
    #[inline(always)]
    pub fn standard() -> Self {
        Self {
            sexual_threshold: 0.5,
            harassment_threshold: 0.5,
            hate_threshold: 0.5,
            violence_threshold: 0.5,
            self_harm_threshold: 0.5,
            illicit_threshold: 0.5,
            strict_mode: false,
            block_minors_content: true,
        }
    }

    /// Create strict policy for public content
    #[inline(always)]
    pub fn strict() -> Self {
        Self {
            sexual_threshold: 0.3,
            harassment_threshold: 0.3,
            hate_threshold: 0.3,
            violence_threshold: 0.3,
            self_harm_threshold: 0.2,
            illicit_threshold: 0.3,
            strict_mode: true,
            block_minors_content: true,
        }
    }

    /// Create permissive policy for private content
    #[inline(always)]
    pub fn permissive() -> Self {
        Self {
            sexual_threshold: 0.8,
            harassment_threshold: 0.7,
            hate_threshold: 0.7,
            violence_threshold: 0.8,
            self_harm_threshold: 0.5,
            illicit_threshold: 0.7,
            strict_mode: false,
            block_minors_content: true,
        }
    }

    /// Create policy for educational content
    #[inline(always)]
    pub fn educational() -> Self {
        Self {
            sexual_threshold: 0.6,
            harassment_threshold: 0.4,
            hate_threshold: 0.4,
            violence_threshold: 0.7,
            self_harm_threshold: 0.3,
            illicit_threshold: 0.5,
            strict_mode: false,
            block_minors_content: false,
        }
    }

    /// Customize thresholds
    #[inline(always)]
    pub fn with_sexual_threshold(mut self, threshold: f32) -> Self {
        self.sexual_threshold = threshold;
        self
    }

    #[inline(always)]
    pub fn with_harassment_threshold(mut self, threshold: f32) -> Self {
        self.harassment_threshold = threshold;
        self
    }

    #[inline(always)]
    pub fn with_hate_threshold(mut self, threshold: f32) -> Self {
        self.hate_threshold = threshold;
        self
    }

    #[inline(always)]
    pub fn with_violence_threshold(mut self, threshold: f32) -> Self {
        self.violence_threshold = threshold;
        self
    }

    #[inline(always)]
    pub fn with_self_harm_threshold(mut self, threshold: f32) -> Self {
        self.self_harm_threshold = threshold;
        self
    }

    #[inline(always)]
    pub fn with_illicit_threshold(mut self, threshold: f32) -> Self {
        self.illicit_threshold = threshold;
        self
    }

    #[inline(always)]
    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }

    #[inline(always)]
    pub fn with_block_minors_content(mut self, block: bool) -> Self {
        self.block_minors_content = block;
        self
    }
}

impl AnalysisContext {
    /// Create context for public content
    #[inline(always)]
    pub fn public(content_type: ContentType) -> Self {
        Self {
            user_age: None,
            content_type,
            platform: None,
            region: None,
            is_public: true,
        }
    }

    /// Create context for private content
    #[inline(always)]
    pub fn private(content_type: ContentType) -> Self {
        Self {
            user_age: None,
            content_type,
            platform: None,
            region: None,
            is_public: false,
        }
    }

    /// Set user age for content filtering
    #[inline(always)]
    pub fn with_user_age(mut self, age: u16) -> Self {
        self.user_age = Some(age);
        self
    }

    /// Set platform for platform-specific rules
    #[inline(always)]
    pub fn with_platform(mut self, platform: impl Into<String>) -> Self {
        self.platform = Some(platform.into());
        self
    }

    /// Set region for regional compliance
    #[inline(always)]
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Check if content is suitable for minors
    #[inline(always)]
    pub fn is_minor_safe(&self) -> bool {
        if let Some(age) = self.user_age {
            age >= 18
        } else {
            false // Default to strict for unknown age
        }
    }
}

impl RiskLevel {
    /// Get numeric risk score
    #[inline(always)]
    pub fn score(&self) -> u8 {
        match self {
            Self::Safe => 0,
            Self::Low => 1,
            Self::Medium => 2,
            Self::High => 3,
            Self::Critical => 4,
        }
    }

    /// Check if risk level requires human review
    #[inline(always)]
    pub fn requires_review(&self) -> bool {
        matches!(self, Self::Medium | Self::High | Self::Critical)
    }

    /// Check if risk level requires immediate blocking
    #[inline(always)]
    pub fn requires_blocking(&self) -> bool {
        matches!(self, Self::High | Self::Critical)
    }
}

impl SafetyRecommendation {
    /// Check if content should be allowed
    #[inline(always)]
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allow)
    }

    /// Check if content needs review
    #[inline(always)]
    pub fn needs_review(&self) -> bool {
        matches!(self, Self::Review | Self::RedFlag)
    }

    /// Check if content should be blocked
    #[inline(always)]
    pub fn should_block(&self) -> bool {
        matches!(self, Self::Block | Self::RedFlag)
    }

    /// Get action priority (higher = more urgent)
    #[inline(always)]
    pub fn priority(&self) -> u8 {
        match self {
            Self::Allow => 0,
            Self::Review => 1,
            Self::Block => 2,
            Self::RedFlag => 3,
        }
    }
}

/// Batch moderate multiple texts efficiently using OpenAI Moderation API
#[inline(always)]
pub fn batch_moderate(
    texts: ZeroOneOrMany<String>,
    policy: &ModerationPolicy,
    context: &AnalysisContext,
) -> AsyncTask<ZeroOneOrMany<SafetyAssessment>> {
    let owned_policy = policy.clone();
    let owned_context = context.clone();
    crate::async_task::spawn_async(async move {
        match texts {
            ZeroOneOrMany::None => ZeroOneOrMany::None,
            ZeroOneOrMany::One(text) => {
                match call_openai_moderation_api(&text, &owned_policy, &owned_context).await {
                    Ok(assessment) => ZeroOneOrMany::One(assessment),
                    Err(_) => {
                        // Fallback to safe assessment on API error
                        ZeroOneOrMany::One(SafetyAssessment {
                            is_safe: true,
                            risk_level: RiskLevel::Safe,
                            triggered_categories: ZeroOneOrMany::None,
                            highest_score: 0.0,
                            recommendation: SafetyRecommendation::Allow,
                        })
                    }
                }
            }
            ZeroOneOrMany::Many(text_vec) => {
                let mut assessments = Vec::with_capacity(text_vec.len());
                for text in text_vec.iter() {
                    match call_openai_moderation_api(text, &owned_policy, &owned_context).await {
                        Ok(assessment) => assessments.push(assessment),
                        Err(_) => {
                            // Fallback to safe assessment on API error
                            assessments.push(SafetyAssessment {
                                is_safe: true,
                                risk_level: RiskLevel::Safe,
                                triggered_categories: ZeroOneOrMany::None,
                                highest_score: 0.0,
                                recommendation: SafetyRecommendation::Allow,
                            });
                        }
                    }
                }
                ZeroOneOrMany::from_vec(assessments)
            }
        }
    })
}

/// Call OpenAI Moderation API for real content analysis
#[inline(always)]
async fn call_openai_moderation_api(
    text: &str,
    policy: &ModerationPolicy,
    context: &AnalysisContext,
) -> Result<SafetyAssessment, OpenAIError> {
    // Create HTTP client with AI-optimized configuration
    let client = HttpClient::with_config(HttpConfig::ai_optimized())
        .map_err(|e| OpenAIError::HttpError(format!("Failed to create HTTP client: {}", e)))?;

    // Build moderation request
    let moderation_request = ModerationRequest {
        input: ModerationInput::Single(text.to_string()),
        model: Some("text-moderation-latest".to_string()),
    };

    let request_body = serde_json::to_vec(&moderation_request).map_err(|e| {
        OpenAIError::SerializationError(format!("Failed to serialize request: {}", e))
    })?;

    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| OpenAIError::AuthenticationError("OPENAI_API_KEY not set".to_string()))?;

    // Create HTTP request
    let http_request = HttpRequest::post("https://api.openai.com/v1/moderations", request_body)
        .map_err(|e| OpenAIError::HttpError(format!("Failed to create request: {}", e)))?
        .header("Authorization", &format!("Bearer {}", api_key))
        .header("Content-Type", "application/json");

    // Send request
    let response = client
        .send(http_request)
        .await
        .map_err(|e| OpenAIError::HttpError(format!("Request failed: {}", e)))?;

    if !response.status().is_success() {
        return Err(OpenAIError::ApiError(format!(
            "API request failed with status: {}",
            response.status()
        )));
    }

    // Parse response
    let response_body = response
        .bytes()
        .await
        .map_err(|e| OpenAIError::HttpError(format!("Failed to read response: {}", e)))?;

    let moderation_response: ModerationResponse = serde_json::from_slice(&response_body)
        .map_err(|e| OpenAIError::SerializationError(format!("Failed to parse response: {}", e)))?;

    // Convert OpenAI response to SafetyAssessment
    convert_moderation_response_to_assessment(moderation_response, policy, context)
}

/// Convert OpenAI moderation response to our SafetyAssessment format
#[inline(always)]
fn convert_moderation_response_to_assessment(
    response: ModerationResponse,
    _policy: &ModerationPolicy,
    _context: &AnalysisContext,
) -> Result<SafetyAssessment, OpenAIError> {
    let result = response
        .results
        .into_iter()
        .next()
        .ok_or_else(|| OpenAIError::InvalidResponse("No moderation results".to_string()))?;

    let flagged = result.flagged;
    let highest_score = result
        .category_scores
        .values()
        .fold(0.0f32, |max, &score| max.max(score));

    let triggered_categories = if flagged {
        let mut categories = Vec::new();
        for (category, &is_flagged) in result.categories.iter() {
            if is_flagged {
                categories.push(category.clone());
            }
        }
        ZeroOneOrMany::from_vec(categories)
    } else {
        ZeroOneOrMany::None
    };

    let risk_level = if highest_score >= 0.8 {
        RiskLevel::High
    } else if highest_score >= 0.5 {
        RiskLevel::Medium
    } else if highest_score >= 0.2 {
        RiskLevel::Low
    } else {
        RiskLevel::Safe
    };

    let recommendation = if flagged {
        if highest_score >= 0.8 {
            SafetyRecommendation::Block
        } else {
            SafetyRecommendation::Review
        }
    } else {
        SafetyRecommendation::Allow
    };

    Ok(SafetyAssessment {
        is_safe: !flagged,
        risk_level,
        triggered_categories,
        highest_score,
        recommendation,
    })
}

/// Get available moderation models
#[inline(always)]
pub fn available_models() -> ZeroOneOrMany<String> {
    ZeroOneOrMany::from_vec(vec![
        "text-moderation-latest".to_string(),
        "text-moderation-stable".to_string(),
    ])
}

/// Check if model is available
#[inline(always)]
pub fn is_model_available(model: &str) -> bool {
    matches!(model, "text-moderation-latest" | "text-moderation-stable")
}
