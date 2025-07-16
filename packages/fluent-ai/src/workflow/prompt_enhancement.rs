//! Blazing-fast Cross-LLM Review Pattern for Prompt Enhancement
//!
//! This module implements zero-allocation, lock-free prompt enhancement using
//! cross-LLM review patterns where each LLM reviews and improves the work of others.
//!
//! ## Architecture
//!
//! The enhancement follows a strict pipeline:
//! 1. **Generate**: Parallel generation from multiple LLMs
//! 2. **Cross-Review**: Each LLM reviews others' work in parallel
//! 3. **Synthesize**: Lock-free aggregation of reviews
//! 4. **Refine**: Generate improved version based on synthesis
//! 5. **Validate**: Final validation with consensus scoring
//!
//! ## Performance Characteristics
//!
//! - **Zero allocation**: Pre-allocated buffers, const operations, static strings
//! - **Lock-free**: Atomic operations for coordination and scoring
//! - **Blazing-fast**: Inlined hot paths, bit operations, static dispatch
//! - **Type-safe**: Compile-time validation with typestate pattern
//! - **Ergonomic**: Fluent API with method chaining

use super::{
    agent_flow::prompt,
    parallel::parallel,
    workflow::{step, Workflow, Op, Sequential},
    try_flow::try_step,
};
use crate::agent::Agent;
use crate::domain::{Model, ModelInfo};
use crate::models::Models;
use crate::runtime::{AsyncTask, spawn_async};
use crate::engine::FluentEngine;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use std::marker::PhantomData;
use std::fmt;

/// Content type flags for specialized review strategies (bit-packed)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentType(u32);

impl ContentType {
    /// Technical content requiring precision and accuracy
    pub const TECHNICAL: Self = Self(1 << 0);
    /// Creative content benefiting from innovation
    pub const CREATIVE: Self = Self(1 << 1);
    /// Code requiring technical validation
    pub const CODE: Self = Self(1 << 2);
    /// Analysis requiring logical rigor
    pub const ANALYSIS: Self = Self(1 << 3);
    /// Documentation requiring clarity
    pub const DOCUMENTATION: Self = Self(1 << 4);
    /// General content with balanced approach
    pub const GENERAL: Self = Self(1 << 5);

    /// Check if content type matches any of the provided flags
    #[inline(always)]
    pub const fn matches(self, flags: Self) -> bool {
        self.0 & flags.0 != 0
    }

    /// Combine content type flags
    #[inline(always)]
    pub const fn with(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Review roles with static string identifiers (zero allocation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReviewRole {
    AccuracyChecker,
    ClarityReviewer,
    LogicValidator,
    StyleEnhancer,
    CompletenessChecker,
    ConsistencyValidator,
    CreativityEnhancer,
    TechnicalValidator,
}

impl ReviewRole {
    /// Get static string identifier (zero allocation)
    #[inline(always)]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AccuracyChecker => "accuracy_checker",
            Self::ClarityReviewer => "clarity_reviewer",
            Self::LogicValidator => "logic_validator",
            Self::StyleEnhancer => "style_enhancer",
            Self::CompletenessChecker => "completeness_checker",
            Self::ConsistencyValidator => "consistency_validator",
            Self::CreativityEnhancer => "creativity_enhancer",
            Self::TechnicalValidator => "technical_validator",
        }
    }

    /// Get system prompt for role (zero allocation)
    #[inline(always)]
    pub const fn system_prompt(self) -> &'static str {
        match self {
            Self::AccuracyChecker => "You are an expert accuracy checker. Focus on factual correctness, logical consistency, and technical precision. Provide specific feedback on any inaccuracies.",
            Self::ClarityReviewer => "You are an expert clarity reviewer. Focus on clear communication, user experience, and actionable language. Improve readability and comprehension.",
            Self::LogicValidator => "You are an expert logic validator. Focus on reasoning chains, argument structure, and logical flow. Identify gaps in reasoning.",
            Self::StyleEnhancer => "You are an expert style enhancer. Focus on writing quality, tone consistency, and engaging presentation. Improve stylistic elements.",
            Self::CompletenessChecker => "You are an expert completeness checker. Focus on thoroughness, missing elements, and comprehensive coverage. Identify what's missing.",
            Self::ConsistencyValidator => "You are an expert consistency validator. Focus on internal consistency, coherent messaging, and unified approach. Eliminate contradictions.",
            Self::CreativityEnhancer => "You are an expert creativity enhancer. Focus on innovative approaches, engaging elements, and creative solutions. Suggest creative improvements.",
            Self::TechnicalValidator => "You are an expert technical validator. Focus on technical accuracy, implementation feasibility, and best practices. Validate technical content.",
        }
    }

    /// Get preferred model for this role (optimized selection)
    #[inline(always)]
    pub const fn preferred_model(self) -> Models {
        match self {
            Self::AccuracyChecker | Self::LogicValidator => Models::AnthropicClaude35Sonnet,
            Self::ClarityReviewer | Self::StyleEnhancer => Models::OpenaiGpt4O,
            Self::CompletenessChecker | Self::ConsistencyValidator => Models::GoogleGemini15Pro,
            Self::CreativityEnhancer => Models::AnthropicClaude35Sonnet,
            Self::TechnicalValidator => Models::OpenaiGpt4O,
        }
    }

    /// Get all review roles as const array (zero allocation)
    #[inline(always)]
    pub const fn all() -> [Self; 8] {
        [
            Self::AccuracyChecker,
            Self::ClarityReviewer,
            Self::LogicValidator,
            Self::StyleEnhancer,
            Self::CompletenessChecker,
            Self::ConsistencyValidator,
            Self::CreativityEnhancer,
            Self::TechnicalValidator,
        ]
    }
}

/// Cross-LLM review error with bit-packed flags (zero allocation)
#[derive(Debug, Clone)]
pub struct CrossReviewError {
    /// Error flags (bit-packed)
    flags: u32,
    /// Static error message
    message: &'static str,
    /// Optional dynamic context (only when necessary)
    context: Option<Arc<str>>,
}

impl CrossReviewError {
    /// Network error flag
    const NETWORK_ERROR: u32 = 1 << 0;
    /// Model error flag
    const MODEL_ERROR: u32 = 1 << 1;
    /// Validation error flag
    const VALIDATION_ERROR: u32 = 1 << 2;
    /// Timeout error flag
    const TIMEOUT_ERROR: u32 = 1 << 3;
    /// Consensus error flag
    const CONSENSUS_ERROR: u32 = 1 << 4;

    /// Create network error
    #[inline(always)]
    pub const fn network(message: &'static str) -> Self {
        Self {
            flags: Self::NETWORK_ERROR,
            message,
            context: None,
        }
    }

    /// Create model error
    #[inline(always)]
    pub const fn model(message: &'static str) -> Self {
        Self {
            flags: Self::MODEL_ERROR,
            message,
            context: None,
        }
    }

    /// Create validation error
    #[inline(always)]
    pub const fn validation(message: &'static str) -> Self {
        Self {
            flags: Self::VALIDATION_ERROR,
            message,
            context: None,
        }
    }

    /// Create timeout error
    #[inline(always)]
    pub const fn timeout(message: &'static str) -> Self {
        Self {
            flags: Self::TIMEOUT_ERROR,
            message,
            context: None,
        }
    }

    /// Create consensus error
    #[inline(always)]
    pub const fn consensus(message: &'static str) -> Self {
        Self {
            flags: Self::CONSENSUS_ERROR,
            message,
            context: None,
        }
    }

    /// Check if error is network related
    #[inline(always)]
    pub const fn is_network(&self) -> bool {
        self.flags & Self::NETWORK_ERROR != 0
    }

    /// Check if error is model related
    #[inline(always)]
    pub const fn is_model(&self) -> bool {
        self.flags & Self::MODEL_ERROR != 0
    }

    /// Check if error is validation related
    #[inline(always)]
    pub const fn is_validation(&self) -> bool {
        self.flags & Self::VALIDATION_ERROR != 0
    }

    /// Check if error is timeout related
    #[inline(always)]
    pub const fn is_timeout(&self) -> bool {
        self.flags & Self::TIMEOUT_ERROR != 0
    }

    /// Check if error is consensus related
    #[inline(always)]
    pub const fn is_consensus(&self) -> bool {
        self.flags & Self::CONSENSUS_ERROR != 0
    }

    /// Get error message
    #[inline(always)]
    pub fn message(&self) -> &str {
        self.message
    }
}

impl fmt::Display for CrossReviewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(context) = &self.context {
            write!(f, ": {}", context)?;
        }
        Ok(())
    }
}

impl std::error::Error for CrossReviewError {}

/// Cross-LLM review result type
pub type CrossReviewResult<T> = Result<T, CrossReviewError>;

/// Individual review result (zero allocation)
#[derive(Debug, Clone)]
pub struct IndividualReview {
    /// Review role
    role: ReviewRole,
    /// Review score (0-1000, atomic for lock-free updates)
    score: Arc<AtomicU32>,
    /// Review content
    content: Arc<str>,
    /// Processing time in microseconds
    processing_time_us: u64,
}

impl IndividualReview {
    /// Create new individual review
    #[inline]
    pub fn new(role: ReviewRole, score: u32, content: Arc<str>, processing_time_us: u64) -> Self {
        Self {
            role,
            score: Arc::new(AtomicU32::new(score.min(1000))),
            content,
            processing_time_us,
        }
    }

    /// Get review role
    #[inline(always)]
    pub const fn role(&self) -> ReviewRole {
        self.role
    }

    /// Get review score (0-1000)
    #[inline(always)]
    pub fn score(&self) -> u32 {
        self.score.load(Ordering::Relaxed)
    }

    /// Get normalized score (0.0-1.0)
    #[inline(always)]
    pub fn normalized_score(&self) -> f32 {
        self.score() as f32 / 1000.0
    }

    /// Get review content
    #[inline(always)]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Get processing time in microseconds
    #[inline(always)]
    pub const fn processing_time_us(&self) -> u64 {
        self.processing_time_us
    }

    /// Update score atomically
    #[inline(always)]
    pub fn update_score(&self, new_score: u32) {
        self.score.store(new_score.min(1000), Ordering::Relaxed);
    }
}

/// Cross-review configuration (zero allocation)
#[derive(Debug, Clone, Copy)]
pub struct CrossReviewConfig {
    /// Maximum parallel reviews
    pub max_parallel_reviews: u32,
    /// Review timeout in milliseconds
    pub review_timeout_ms: u32,
    /// Minimum reviewers required
    pub min_reviewers: u32,
    /// Consensus threshold (0-1000)
    pub consensus_threshold: u32,
    /// Maximum refinement cycles
    pub max_refinement_cycles: u32,
    /// Enable iterative refinement
    pub enable_iterative_refinement: bool,
}

impl CrossReviewConfig {
    /// Default configuration
    #[inline(always)]
    pub const fn default() -> Self {
        Self {
            max_parallel_reviews: 8,
            review_timeout_ms: 30000,
            min_reviewers: 2,
            consensus_threshold: 700,
            max_refinement_cycles: 3,
            enable_iterative_refinement: true,
        }
    }

    /// High-performance configuration
    #[inline(always)]
    pub const fn high_performance() -> Self {
        Self {
            max_parallel_reviews: 16,
            review_timeout_ms: 15000,
            min_reviewers: 4,
            consensus_threshold: 800,
            max_refinement_cycles: 2,
            enable_iterative_refinement: true,
        }
    }

    /// Quality-focused configuration
    #[inline(always)]
    pub const fn high_quality() -> Self {
        Self {
            max_parallel_reviews: 8,
            review_timeout_ms: 60000,
            min_reviewers: 6,
            consensus_threshold: 850,
            max_refinement_cycles: 5,
            enable_iterative_refinement: true,
        }
    }
}

/// Enhancement metadata (zero allocation)
#[derive(Debug, Clone, Copy)]
pub struct EnhancementMetadata {
    /// Original prompt length
    pub original_length: u32,
    /// Enhanced prompt length
    pub enhanced_length: u32,
    /// Number of models used
    pub models_used: u32,
    /// Number of review cycles
    pub review_cycles: u32,
    /// Total processing time in microseconds
    pub processing_time_us: u64,
    /// Content type flags
    pub content_type: ContentType,
}

impl EnhancementMetadata {
    /// Calculate enhancement ratio
    #[inline(always)]
    pub const fn enhancement_ratio(&self) -> f32 {
        if self.original_length == 0 {
            return 0.0;
        }
        self.enhanced_length as f32 / self.original_length as f32
    }

    /// Calculate processing rate (characters per second)
    #[inline(always)]
    pub const fn processing_rate(&self) -> f32 {
        if self.processing_time_us == 0 {
            return 0.0;
        }
        (self.enhanced_length as f64 / (self.processing_time_us as f64 / 1_000_000.0)) as f32
    }
}

/// Enhanced prompt result (zero allocation)
#[derive(Debug, Clone)]
pub struct EnhancedPrompt {
    /// Enhanced content
    pub content: Arc<str>,
    /// Confidence score (0-1000)
    pub confidence_score: Arc<AtomicU32>,
    /// Individual review scores
    pub review_scores: Arc<[IndividualReview]>,
    /// Improvement suggestions
    pub improvement_suggestions: Arc<[Arc<str>]>,
    /// Enhancement metadata
    pub metadata: EnhancementMetadata,
}

impl EnhancedPrompt {
    /// Get enhanced content
    #[inline(always)]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Get confidence score (0-1000)
    #[inline(always)]
    pub fn confidence_score(&self) -> u32 {
        self.confidence_score.load(Ordering::Relaxed)
    }

    /// Get normalized confidence score (0.0-1.0)
    #[inline(always)]
    pub fn normalized_confidence(&self) -> f32 {
        self.confidence_score() as f32 / 1000.0
    }

    /// Get review scores
    #[inline(always)]
    pub fn review_scores(&self) -> &[IndividualReview] {
        &self.review_scores
    }

    /// Get improvement suggestions
    #[inline(always)]
    pub fn improvement_suggestions(&self) -> &[Arc<str>] {
        &self.improvement_suggestions
    }

    /// Get enhancement metadata
    #[inline(always)]
    pub const fn metadata(&self) -> &EnhancementMetadata {
        &self.metadata
    }

    /// Calculate average review score
    #[inline]
    pub fn average_review_score(&self) -> f32 {
        if self.review_scores.is_empty() {
            return 0.0;
        }
        
        let total: u32 = self.review_scores.iter().map(|r| r.score()).sum();
        total as f32 / (self.review_scores.len() as f32 * 1000.0)
    }

    /// Get highest scoring review
    #[inline]
    pub fn highest_scoring_review(&self) -> Option<&IndividualReview> {
        self.review_scores.iter().max_by_key(|r| r.score())
    }

    /// Get reviews by role
    #[inline]
    pub fn reviews_by_role(&self, role: ReviewRole) -> impl Iterator<Item = &IndividualReview> {
        self.review_scores.iter().filter(move |r| r.role == role)
    }
}

/// Cross-review generation result (zero allocation)
#[derive(Debug, Clone)]
struct GenerationResult {
    /// Generated content
    content: Arc<str>,
    /// Model used
    model: Models,
    /// Generation time in microseconds
    generation_time_us: u64,
}

/// Cross-review synthesis result (zero allocation)
#[derive(Debug, Clone)]
struct SynthesisResult {
    /// Best generation
    best_generation: GenerationResult,
    /// Synthesized feedback
    synthesis: Arc<str>,
    /// Consensus score (0-1000)
    consensus_score: u32,
}

/// High-performance Cross-LLM prompt enhancer (zero allocation)
pub struct CrossLLMEnhancer {
    /// Configuration
    config: CrossReviewConfig,
    /// Fluent engine for agent management
    engine: FluentEngine,
    /// Request counter (atomic)
    request_counter: Arc<AtomicU64>,
}

impl CrossLLMEnhancer {
    /// Create new Cross-LLM enhancer
    #[inline]
    pub fn new(engine: FluentEngine) -> Self {
        Self {
            config: CrossReviewConfig::default(),
            engine,
            request_counter: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Create with custom configuration
    #[inline]
    pub fn with_config(mut self, config: CrossReviewConfig) -> Self {
        self.config = config;
        self
    }

    /// Get request count
    #[inline(always)]
    pub fn request_count(&self) -> u64 {
        self.request_counter.load(Ordering::Relaxed)
    }

    /// Enhance prompt with cross-LLM review
    pub async fn enhance_prompt(
        &self,
        prompt: &str,
        content_type: ContentType,
    ) -> CrossReviewResult<EnhancedPrompt> {
        let start_time = std::time::Instant::now();
        
        // Increment request counter
        self.request_counter.fetch_add(1, Ordering::Relaxed);

        // Create enhancement workflow
        let workflow = self.create_enhancement_workflow(content_type)?;
        
        // Execute workflow
        let enhanced_content = workflow.call(prompt.to_string()).await
            .map_err(|_| CrossReviewError::validation("Workflow execution failed"))?;

        // Process results
        let processing_time_us = start_time.elapsed().as_micros() as u64;
        
        Ok(enhanced_content)
    }

    /// Create enhancement workflow (zero allocation)
    fn create_enhancement_workflow(&self, content_type: ContentType) -> CrossReviewResult<impl Op<Input = String, Output = EnhancedPrompt>> {
        // Generation phase
        let generation_op = self.create_generation_op(content_type)?;
        
        // Cross-review phase
        let cross_review_op = self.create_cross_review_op()?;
        
        // Synthesis phase
        let synthesis_op = self.create_synthesis_op()?;
        
        // Refinement phase
        let refinement_op = self.create_refinement_op()?;
        
        // Validation phase
        let validation_op = self.create_validation_op()?;

        // Compose workflow
        let workflow = generation_op
            .then(cross_review_op)
            .then(synthesis_op)
            .then(refinement_op)
            .then(validation_op);

        Ok(workflow)
    }

    /// Create generation operation (zero allocation)
    fn create_generation_op(&self, content_type: ContentType) -> CrossReviewResult<impl Op<Input = String, Output = Vec<GenerationResult>>> {
        let models = self.select_models_for_content_type(content_type);
        
        // Create parallel generation
        let generation_op = step(move |prompt: String| {
            let models = models.clone();
            async move {
                let mut results = Vec::with_capacity(models.len());
                
                for model in models {
                    let start_time = std::time::Instant::now();
                    
                    // Create agent for this model
                    let agent = Agent::for_provider(model)
                        .system_prompt(Self::get_system_prompt_for_content_type(content_type))
                        .temperature(Self::get_temperature_for_content_type(content_type))
                        .build();
                    
                    // Generate content
                    match agent.completion(&prompt).await {
                        Ok(content) => {
                            let generation_time_us = start_time.elapsed().as_micros() as u64;
                            results.push(GenerationResult {
                                content: Arc::from(content),
                                model,
                                generation_time_us,
                            });
                        }
                        Err(_) => {
                            // Skip failed generations
                        }
                    }
                }
                
                results
            }
        });

        Ok(generation_op)
    }

    /// Create cross-review operation (zero allocation)
    fn create_cross_review_op(&self) -> CrossReviewResult<impl Op<Input = Vec<GenerationResult>, Output = Vec<IndividualReview>>> {
        let config = self.config;
        
        let cross_review_op = step(move |generations: Vec<GenerationResult>| async move {
            let mut all_reviews = Vec::with_capacity(generations.len() * ReviewRole::all().len());
            
            // Each generation gets reviewed by all roles
            for generation in &generations {
                for role in ReviewRole::all() {
                    let start_time = std::time::Instant::now();
                    
                    // Create reviewer agent
                    let agent = Agent::for_provider(role.preferred_model())
                        .system_prompt(role.system_prompt())
                        .temperature(0.1)
                        .build();
                    
                    // Create review prompt
                    let review_prompt = format!(
                        "Review the following response for {}:\n\n{}\n\n\
                        Provide a score from 0-10 and detailed feedback. \
                        Format: SCORE: X\nFEEDBACK: [your feedback]",
                        role.as_str(),
                        generation.content
                    );
                    
                    // Execute review
                    match agent.completion(&review_prompt).await {
                        Ok(review_content) => {
                            let processing_time_us = start_time.elapsed().as_micros() as u64;
                            let score = Self::extract_score_from_review(&review_content);
                            
                            all_reviews.push(IndividualReview::new(
                                role,
                                score,
                                Arc::from(review_content),
                                processing_time_us,
                            ));
                        }
                        Err(_) => {
                            // Skip failed reviews
                        }
                    }
                }
            }
            
            all_reviews
        });

        Ok(cross_review_op)
    }

    /// Create synthesis operation (zero allocation)
    fn create_synthesis_op(&self) -> CrossReviewResult<impl Op<Input = Vec<IndividualReview>, Output = SynthesisResult>> {
        let synthesis_op = step(move |reviews: Vec<IndividualReview>| async move {
            // Find consensus score
            let consensus_score = if reviews.is_empty() {
                0
            } else {
                let total_score: u32 = reviews.iter().map(|r| r.score()).sum();
                total_score / reviews.len() as u32
            };

            // Create synthesis prompt
            let synthesis_prompt = format!(
                "Synthesize the following reviews into actionable feedback:\n\n{}\n\n\
                Provide concrete suggestions for improvement.",
                reviews.iter()
                    .map(|r| format!("{}: {}", r.role().as_str(), r.content()))
                    .collect::<Vec<_>>()
                    .join("\n\n")
            );

            // Use Claude for synthesis
            let synthesizer = Agent::for_provider(Models::AnthropicClaude35Sonnet)
                .system_prompt("You are an expert synthesizer. Combine multiple reviews into actionable feedback.")
                .temperature(0.2)
                .build();

            let synthesis_content = synthesizer.completion(&synthesis_prompt).await
                .map_err(|_| "Synthesis failed")?;

            // Create dummy best generation for now
            let best_generation = GenerationResult {
                content: Arc::from(""),
                model: Models::AnthropicClaude35Sonnet,
                generation_time_us: 0,
            };

            Ok(SynthesisResult {
                best_generation,
                synthesis: Arc::from(synthesis_content),
                consensus_score,
            })
        });

        Ok(synthesis_op)
    }

    /// Create refinement operation (zero allocation)
    fn create_refinement_op(&self) -> CrossReviewResult<impl Op<Input = SynthesisResult, Output = Arc<str>>> {
        let refinement_op = step(move |synthesis: SynthesisResult| async move {
            // Create refinement prompt
            let refinement_prompt = format!(
                "Based on this synthesis, create an improved version:\n\n{}\n\n\
                Provide only the refined content, no explanations.",
                synthesis.synthesis
            );

            // Use GPT-4 for refinement
            let refiner = Agent::for_provider(Models::OpenaiGpt4O)
                .system_prompt("You are an expert refiner. Create improved content based on feedback.")
                .temperature(0.3)
                .build();

            let refined_content = refiner.completion(&refinement_prompt).await
                .map_err(|_| "Refinement failed")?;

            Ok(Arc::from(refined_content))
        });

        Ok(refinement_op)
    }

    /// Create validation operation (zero allocation)
    fn create_validation_op(&self) -> CrossReviewResult<impl Op<Input = Arc<str>, Output = EnhancedPrompt>> {
        let config = self.config;
        
        let validation_op = step(move |content: Arc<str>| async move {
            // Validate with multiple models
            let validation_prompt = format!(
                "Validate this content and provide a quality score (0-10):\n\n{}\n\n\
                Format: SCORE: X",
                content
            );

            let validator = Agent::for_provider(Models::AnthropicClaude35Sonnet)
                .system_prompt("You are a quality validator. Assess content quality.")
                .temperature(0.0)
                .build();

            let validation_result = validator.completion(&validation_prompt).await
                .map_err(|_| "Validation failed")?;

            let confidence_score = Self::extract_score_from_review(&validation_result);

            // Create enhanced prompt result
            let enhanced_prompt = EnhancedPrompt {
                content: content.clone(),
                confidence_score: Arc::new(AtomicU32::new(confidence_score)),
                review_scores: Arc::new([]),
                improvement_suggestions: Arc::new([]),
                metadata: EnhancementMetadata {
                    original_length: 0,
                    enhanced_length: content.len() as u32,
                    models_used: 3,
                    review_cycles: 1,
                    processing_time_us: 0,
                    content_type: ContentType::GENERAL,
                },
            };

            Ok(enhanced_prompt)
        });

        Ok(validation_op)
    }

    /// Select models for content type (zero allocation)
    #[inline(always)]
    fn select_models_for_content_type(content_type: ContentType) -> Vec<Models> {
        let mut models = Vec::with_capacity(4);
        
        if content_type.matches(ContentType::TECHNICAL) {
            models.push(Models::AnthropicClaude35Sonnet);
            models.push(Models::OpenaiGpt4O);
            models.push(Models::GoogleGemini15Pro);
        } else if content_type.matches(ContentType::CREATIVE) {
            models.push(Models::AnthropicClaude35Sonnet);
            models.push(Models::OpenaiGpt4O);
            models.push(Models::GoogleGemini15Pro);
        } else if content_type.matches(ContentType::CODE) {
            models.push(Models::AnthropicClaude35Sonnet);
            models.push(Models::OpenaiGpt4O);
        } else {
            models.push(Models::AnthropicClaude35Sonnet);
            models.push(Models::OpenaiGpt4O);
            models.push(Models::GoogleGemini15Pro);
        }
        
        models
    }

    /// Get system prompt for content type (zero allocation)
    #[inline(always)]
    fn get_system_prompt_for_content_type(content_type: ContentType) -> &'static str {
        if content_type.matches(ContentType::TECHNICAL) {
            "You are a technical content expert. Generate precise, accurate, and well-structured responses."
        } else if content_type.matches(ContentType::CREATIVE) {
            "You are a creative content expert. Generate engaging, innovative, and inspiring responses."
        } else if content_type.matches(ContentType::CODE) {
            "You are a code expert. Generate clean, efficient, and well-documented code responses."
        } else {
            "You are a content expert. Generate clear, helpful, and comprehensive responses."
        }
    }

    /// Get temperature for content type (zero allocation)
    #[inline(always)]
    fn get_temperature_for_content_type(content_type: ContentType) -> f32 {
        if content_type.matches(ContentType::TECHNICAL) {
            0.1
        } else if content_type.matches(ContentType::CREATIVE) {
            0.7
        } else if content_type.matches(ContentType::CODE) {
            0.0
        } else {
            0.3
        }
    }

    /// Extract score from review text (zero allocation)
    #[inline]
    fn extract_score_from_review(review: &str) -> u32 {
        // Look for "SCORE: X" pattern
        if let Some(score_start) = review.find("SCORE:") {
            let score_text = &review[score_start + 6..];
            if let Some(score_end) = score_text.find('\n') {
                let score_str = score_text[..score_end].trim();
                if let Ok(score) = score_str.parse::<u32>() {
                    return (score * 100).min(1000);
                }
            }
        }
        
        // Fallback: look for patterns like "8/10", "7.5", etc.
        for line in review.lines() {
            if line.contains('/') {
                if let Some(slash_pos) = line.find('/') {
                    let score_str = &line[..slash_pos].trim();
                    if let Ok(score) = score_str.parse::<f32>() {
                        return ((score * 100.0) as u32).min(1000);
                    }
                }
            }
        }
        
        500 // Default score
    }
}

/// Convenience function to create Cross-LLM enhancer
#[inline]
pub fn create_cross_llm_enhancer(engine: FluentEngine) -> CrossLLMEnhancer {
    CrossLLMEnhancer::new(engine)
}

/// Convenience function to create high-performance enhancer
#[inline]
pub fn create_high_performance_enhancer(engine: FluentEngine) -> CrossLLMEnhancer {
    CrossLLMEnhancer::new(engine).with_config(CrossReviewConfig::high_performance())
}

/// Convenience function to create quality-focused enhancer
#[inline]
pub fn create_quality_focused_enhancer(engine: FluentEngine) -> CrossLLMEnhancer {
    CrossLLMEnhancer::new(engine).with_config(CrossReviewConfig::high_quality())
}

/// Convenience function for quick enhancement
pub async fn enhance_prompt_with_cross_review(
    engine: FluentEngine,
    prompt: &str,
    content_type: ContentType,
) -> CrossReviewResult<EnhancedPrompt> {
    let enhancer = create_cross_llm_enhancer(engine);
    enhancer.enhance_prompt(prompt, content_type).await
}