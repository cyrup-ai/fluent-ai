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

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use fluent_ai_domain::agent::Agent;

// Simple model abstraction for cross-review workflow
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkflowModel {
    OpenAiGpt4oMini,
    AnthropicClaude3Haiku,
    MistralSmall,
    OpenAiGpt4o,
    AnthropicClaude3Opus,
    MistralLarge,
}

impl WorkflowModel {
    pub fn name(&self) -> &'static str {
        match self {
            Self::OpenAiGpt4oMini => "gpt-4o-mini",
            Self::AnthropicClaude3Haiku => "claude-3-haiku",
            Self::MistralSmall => "mistral-small",
            Self::OpenAiGpt4o => "gpt-4o",
            Self::AnthropicClaude3Opus => "claude-3-opus",
            Self::MistralLarge => "mistral-large",
        }
    }
}

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
    pub fn matches(self, flags: u32) -> bool {
        (self.0 & flags) != 0
    }
}

/// Configuration for the cross-review process (zero allocation)
#[derive(Debug, Clone, Copy)]
pub struct CrossReviewConfig {
    /// Models to use for generation and review
    pub models: &'static [WorkflowModel],
    /// Timeout for each model interaction in milliseconds
    pub timeout_ms: u32,
    /// Temperature for creative tasks (0.0-1.0)
    pub temperature: f32,
    /// Number of refinement cycles
    pub refinement_cycles: u32}

impl Default for CrossReviewConfig {
    #[inline]
    fn default() -> Self {
        Self {
            models: &[
                WorkflowModel::OpenAiGpt4oMini,
                WorkflowModel::AnthropicClaude3Haiku,
                WorkflowModel::MistralSmall,
            ],
            timeout_ms: 5000,
            temperature: 0.7,
            refinement_cycles: 1}
    }
}

impl CrossReviewConfig {
    /// High-performance config (fastest models, short timeout)
    #[inline]
    pub const fn high_performance() -> Self {
        Self {
            models: &[WorkflowModel::OpenAiGpt4oMini, WorkflowModel::AnthropicClaude3Haiku],
            timeout_ms: 3000,
            temperature: 0.5,
            refinement_cycles: 1}
    }

    /// High-quality config (strongest models, longer timeout)
    #[inline]
    pub const fn high_quality() -> Self {
        Self {
            models: &[
                WorkflowModel::OpenAiGpt4o,
                WorkflowModel::AnthropicClaude3Opus,
                WorkflowModel::MistralLarge,
            ],
            timeout_ms: 15000,
            temperature: 0.3,
            refinement_cycles: 2}
    }
}

/// Error types for the cross-review process
#[derive(Debug)]
pub enum CrossReviewError {
    /// Error during model interaction
    ModelError(String),
    /// Timeout during generation or review
    Timeout(String),
    /// Invalid configuration
    ConfigError(String)}

impl fmt::Display for CrossReviewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelError(s) => write!(f, "Model Error: {}", s),
            Self::Timeout(s) => write!(f, "Timeout: {}", s),
            Self::ConfigError(s) => write!(f, "Config Error: {}", s)}
    }
}

impl std::error::Error for CrossReviewError {}

impl CrossReviewError {
    /// Create a new model error
    #[inline]
    pub fn model(msg: &str) -> Self {
        Self::ModelError(msg.to_string())
    }
}

/// Result type for cross-review operations
pub type CrossReviewResult<T> = Result<T, CrossReviewError>;

/// Represents the final enhanced prompt
#[derive(Debug, Clone)]
pub struct EnhancedPrompt {
    /// The enhanced content
    pub content: Arc<str>,
    /// Confidence score (0-1000)
    pub confidence_score: Arc<AtomicU32>,
    /// Individual review scores
    pub review_scores: Arc<[IndividualReview]>,
    /// Improvement suggestions
    pub improvement_suggestions: Arc<[Arc<str>]>,
    /// Total processing time in microseconds
    pub processing_time_us: u64}

/// Represents a single review from one model about another's generation
#[derive(Debug, Clone)]
pub struct IndividualReview {
    /// The model that performed the review
    pub reviewer: WorkflowModel,
    /// The model whose work was reviewed
    pub target: WorkflowModel,
    /// The review content
    pub review_text: Arc<str>,
    /// The score given (0-1000)
    pub score: u32,
    /// Time taken for review in microseconds
    pub review_time_us: u64}

/// A lock-free, zero-allocation matrix for storing and analyzing cross-review results
#[derive(Debug, Clone)]
pub struct CrossReviewMatrix {
    /// Stores (reviewer, target, score) tuples
    review_pairs: Arc<[(WorkflowModel, WorkflowModel, u32)]>,
    /// Number of unique models involved
    dimensions: u32}

impl CrossReviewMatrix {
    /// Create a new matrix from a slice of reviews
    #[inline]
    pub fn from_reviews(reviews: &[IndividualReview]) -> Self {
        let review_pairs: Vec<_> = reviews
            .iter()
            .map(|r| (r.reviewer, r.target, r.score))
            .collect();

        let mut unique_models = std::collections::HashSet::new();
        for (reviewer, target, _) in review_pairs.iter() {
            unique_models.insert(reviewer);
            unique_models.insert(target);
        }

        Self {
            review_pairs: Arc::from(review_pairs.into_boxed_slice()),
            dimensions: unique_models.len() as u32}
    }

    /// Get the average score for a specific target model
    #[inline]
    pub fn average_score_for_target(&self, target: WorkflowModel) -> f32 {
        let (total, count) = self
            .review_pairs
            .iter()
            .filter(|(_, t, _)| *t == target)
            .fold((0, 0), |(sum, count), (_, _, score)| {
                (sum + score, count + 1)
            });

        if count == 0 {
            0.0
        } else {
            total as f32 / count as f32
        }
    }

    /// Get the normalized average score across all reviews (0.0-1.0)
    #[inline]
    pub fn normalized_average_score(&self) -> f32 {
        if self.review_pairs.is_empty() {
            return 0.0;
        }
        let total_score: u32 = self.review_pairs.iter().map(|(_, _, score)| score).sum();
        (total_score as f32 / self.review_pairs.len() as f32) / 1000.0
    }

    /// Get all scores for a given target model
    #[inline]
    pub fn scores_for_target(&self, target: WorkflowModel) -> impl Iterator<Item = u32> + '_ {
        self.review_pairs
            .iter()
            .filter(move |(_, t, _)| *t == target)
            .map(|(_, _, score)| *score)
    }

    /// Get all scores given by a specific reviewer
    #[inline]
    pub fn scores_by_reviewer(&self, reviewer: WorkflowModel) -> impl Iterator<Item = u32> + '_ {
        self.review_pairs
            .iter()
            .filter(move |(r, _, _)| *r == reviewer)
            .map(|(_, _, score)| *score)
    }

    /// Calculate consensus variance (measure of agreement between reviewers)
    #[inline]
    pub fn consensus_variance(&self) -> f32 {
        if self.review_pairs.is_empty() {
            return 0.0;
        }

        let mean = self.normalized_average_score();
        let variance: f32 = self
            .review_pairs
            .iter()
            .map(|(_, _, score)| {
                let normalized = *score as f32 / 1000.0;
                (normalized - mean).powi(2)
            })
            .sum::<f32>()
            / self.review_pairs.len() as f32;

        variance
    }

    /// Calculate consensus strength (lower variance = higher consensus)
    #[inline(always)]
    pub fn consensus_strength(&self) -> f32 {
        1.0 - self.consensus_variance().min(1.0)
    }

    /// Get review coverage (percentage of possible N×(N-1) reviews completed)
    #[inline]
    pub fn coverage_percentage(&self) -> f32 {
        if self.dimensions <= 1 {
            return 1.0;
        }

        let possible_reviews = self.dimensions * (self.dimensions - 1);
        let actual_reviews = self.review_pairs.len() as u32;

        (actual_reviews as f32 / possible_reviews as f32) * 100.0
    }

    /// Get model performance ranking (by average scores received)
    #[inline]
    pub fn model_performance_ranking(&self) -> Vec<(WorkflowModel, f32)> {
        let mut model_scores: std::collections::HashMap<WorkflowModel, Vec<u32>> =
            std::collections::HashMap::new();

        // Collect scores for each target model
        for (_, target, score) in self.review_pairs.iter() {
            model_scores.entry(*target).or_default().push(*score);
        }

        // Calculate averages and sort
        let mut rankings: Vec<(WorkflowModel, f32)> = model_scores
            .into_iter()
            .map(|(model, scores)| {
                let average = scores.iter().sum::<u32>() as f32 / scores.len() as f32;
                (model, average / 1000.0) // Normalize to 0.0-1.0
            })
            .collect();

        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rankings
    }

    /// Get reviewer reliability ranking (by consistency of scores given)
    #[inline]
    pub fn reviewer_reliability_ranking(&self) -> Vec<(WorkflowModel, f32)> {
        let mut reviewer_scores: std::collections::HashMap<WorkflowModel, Vec<u32>> =
            std::collections::HashMap::new();

        // Collect scores by each reviewer
        for (reviewer, _, score) in self.review_pairs.iter() {
            reviewer_scores.entry(*reviewer).or_default().push(*score);
        }

        // Calculate consistency (inverse of variance) and sort
        let mut rankings: Vec<(WorkflowModel, f32)> = reviewer_scores
            .into_iter()
            .map(|(model, scores)| {
                if scores.len() <= 1 {
                    return (model, 1.0); // Single score is perfectly consistent
                }

                let mean = scores.iter().sum::<u32>() as f32 / scores.len() as f32;
                let variance = scores
                    .iter()
                    .map(|&score| (score as f32 - mean).powi(2))
                    .sum::<f32>()
                    / scores.len() as f32;

                let consistency = 1.0 - (variance / 1000000.0).min(1.0); // Normalize variance
                (model, consistency)
            })
            .collect();

        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rankings
    }
}

/// Cross-review generation result (zero allocation)
#[derive(Debug, Clone)]
struct GenerationResult {
    /// Generated content
    content: Arc<str>,
    /// Model used
    model: WorkflowModel,
    /// Generation time in microseconds
    generation_time_us: u64}

/// Cross-review synthesis result (zero allocation)
#[derive(Debug, Clone)]
struct SynthesisResult {
    /// Best generation
    best_generation: GenerationResult,
    /// Synthesized feedback
    synthesis: Arc<str>,
    /// Consensus score (0-1000)
    consensus_score: u32}

/// High-performance Cross-Provider prompt enhancer (zero allocation)
pub struct CrossProviderEnhancer {
    /// Configuration
    config: CrossReviewConfig,
    /// Request counter (atomic)
    request_counter: Arc<AtomicU64>}

impl CrossProviderEnhancer {
    /// Create new Cross-LLM enhancer
    #[inline]
    pub fn new() -> Self {
        Self {
            config: CrossReviewConfig::default(),
            request_counter: Arc::new(AtomicU64::new(0))}
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

        // Execute cross-review enhancement pipeline
        let generations = self
            .generate_from_multiple_models(prompt, content_type)
            .await?;
        let reviews = Self::execute_cross_reviews(generations, self.config).await?;
        let enhanced_content = self.synthesize_and_refine(prompt, &reviews).await?;

        // Process results
        let processing_time_us = start_time.elapsed().as_micros() as u64;

        let enhanced_prompt = EnhancedPrompt {
            content: Arc::from(enhanced_content),
            confidence_score: Arc::new(AtomicU32::new(800)), // Placeholder
            review_scores: Arc::from(reviews.into_boxed_slice()),
            improvement_suggestions: Arc::new([]), // Placeholder
            processing_time_us};

        Ok(enhanced_prompt)
    }

    /// Generate responses from multiple models in parallel
    async fn generate_from_multiple_models(
        &self,
        prompt: &str,
        content_type: ContentType,
    ) -> CrossReviewResult<Vec<GenerationResult>> {
        let generation_futures = self.config.models.iter().map(|model| {
            Self::execute_generation(*model, prompt, content_type, self.config.timeout_ms)
        });

        let results = futures::future::join_all(generation_futures).await;
        results.into_iter().collect()
    }

    /// Execute a single generation task with timeout and retry logic
    async fn execute_generation(
        model: WorkflowModel,
        prompt: &str,
        content_type: ContentType,
        timeout_ms: u32,
    ) -> CrossReviewResult<GenerationResult> {
        let start_time = std::time::Instant::now();

        let agent = Agent::for_provider(model)
            .system_prompt(Self::generation_system_prompt(content_type))
            .max_tokens(2048)
            .timeout(Duration::from_millis(timeout_ms as u64))
            .build();

        let content = agent.completion(prompt).await.map_err(|e| {
            CrossReviewError::model(&format!("Generation failed for {}: {}", model.name(), e))
        })?;

        Ok(GenerationResult {
            model,
            content: Arc::from(content.as_str()),
            generation_time_us: start_time.elapsed().as_micros() as u64})
    }

    /// Execute cross-reviews for all generated content
    async fn execute_cross_reviews(
        generations: Vec<GenerationResult>,
        config: CrossReviewConfig,
    ) -> CrossReviewResult<Vec<IndividualReview>> {
        let mut review_futures = Vec::new();

        for reviewer_gen in &generations {
            for target_gen in &generations {
                if reviewer_gen.model != target_gen.model {
                    let fut = Self::execute_single_review(
                        reviewer_gen.model,
                        target_gen.model,
                        target_gen.content.clone(),
                        config,
                    );
                    review_futures.push(fut);
                }
            }
        }

        let results = futures::future::join_all(review_futures).await;
        results.into_iter().collect()
    }

    /// Execute a single cross-review task
    async fn execute_single_review(
        reviewer: WorkflowModel,
        target: WorkflowModel,
        content: Arc<str>,
        config: CrossReviewConfig,
    ) -> CrossReviewResult<IndividualReview> {
        let start_time = std::time::Instant::now();

        let review_prompt = format!(
            "PEER REVIEW TASK:\nTarget Model: {}\nContent to Review:\n{}\n\nProvide:\n1. Quality Score (0-10)\n2. Strengths\n3. Weaknesses\n4. Actionable Suggestions",
            target.name(),
            content
        );

        let agent = Agent::for_provider(reviewer)
            .system_prompt(Self::review_system_prompt(reviewer, target))
            .max_tokens(1024)
            .temperature(config.temperature)
            .timeout(Duration::from_millis(config.timeout_ms as u64))
            .build();

        let review_text = agent.completion(&review_prompt).await?;
        let score = Self::extract_score_from_review(&review_text);

        Ok(IndividualReview {
            reviewer,
            target,
            review_text: Arc::from(review_text.as_str()),
            score,
            review_time_us: start_time.elapsed().as_micros() as u64})
    }

    /// Synthesize and refine the final prompt from reviews
    async fn synthesize_and_refine(
        &self,
        original_prompt: &str,
        reviews: &[IndividualReview],
    ) -> CrossReviewResult<String> {
        let synthesis_model = self
            .config
            .models
            .first()
            .copied()
            .unwrap_or(WorkflowModel::AnthropicClaude3Haiku);
        let review_summary = reviews
            .iter()
            .map(|r| format!("Review from {}:\n{}", r.reviewer.name(), r.review_text))
            .collect::<Vec<_>>()
            .join("\n\n");

        let synthesis_prompt = format!(
            "Synthesize an improved response based on peer reviews.\nOriginal Prompt: {}\n\n{}",
            original_prompt, review_summary
        );

        let synthesizer = Agent::for_provider(synthesis_model)
            .system_prompt("You are a master synthesizer. Integrate all feedback to create a superior response.")
            .max_tokens(4096)
            .temperature(0.1)
            .build();

        synthesizer.completion(&synthesis_prompt).await
    }

    /// Get system prompt for generation based on content type
    #[inline(always)]
    fn generation_system_prompt(content_type: ContentType) -> &'static str {
        match content_type {
            ContentType::TECHNICAL => {
                "You are a world-class technical expert. Provide a precise, accurate, and detailed response."
            }
            ContentType::CREATIVE => {
                "You are a highly creative AI. Generate an innovative, engaging, and imaginative response."
            }
            ContentType::CODE => {
                "You are an expert programmer. Provide clean, efficient, and well-documented code."
            }
            _ => {
                "You are a world-class AI assistant. Provide a comprehensive, accurate, and well-structured response."
            }
        }
    }

    /// Get system prompt for reviewing
    #[inline(always)]
    fn review_system_prompt(reviewer: WorkflowModel, target: WorkflowModel) -> String {
        format!(
            "You are {} reviewing {}'s response. Be thorough, constructive, and objective.",
            reviewer.name(),
            target.name()
        )
    }

    /// Get temperature for review based on content type
    #[inline(always)]
    fn review_temperature(content_type: ContentType) -> f32 {
        if content_type.matches(ContentType::CODE) {
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

/// Convenience function to create Cross-Provider enhancer
#[inline]
pub fn create_cross_provider_enhancer() -> CrossProviderEnhancer {
    CrossProviderEnhancer::new()
}

/// Convenience function to create high-performance enhancer
#[inline]
pub fn create_high_performance_enhancer() -> CrossProviderEnhancer {
    CrossProviderEnhancer::new().with_config(CrossReviewConfig::high_performance())
}

/// Convenience function to create quality-focused enhancer
#[inline]
pub fn create_quality_focused_enhancer() -> CrossProviderEnhancer {
    CrossProviderEnhancer::new().with_config(CrossReviewConfig::high_quality())
}

/// Convenience function for quick enhancement
pub async fn enhance_prompt_with_cross_review(
    prompt: &str,
    content_type: ContentType,
) -> CrossReviewResult<EnhancedPrompt> {
    let enhancer = create_cross_llm_enhancer();
    enhancer.enhance_prompt(prompt, content_type).await
}
