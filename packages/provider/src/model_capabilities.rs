//! Model capability definitions and capability-based filtering
//! 
//! This module defines the core model capabilities structure and provides
//! functionality for filtering and querying models based on their capabilities.

use serde::{Serialize, Deserialize};

/// Core model information and capabilities
/// 
/// This structure contains all the essential information about an AI model,
/// including its capabilities, limits, and pricing information. It serves
/// as the canonical representation of model metadata in the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfoData {
    /// Provider name (e.g., "openai", "anthropic", "google")
    pub provider_name: String,
    
    /// Model name as used by the provider
    pub name: String,
    
    /// Maximum number of input tokens the model can accept
    pub max_input_tokens: Option<u64>,
    
    /// Maximum number of output tokens the model can generate
    pub max_output_tokens: Option<u64>,
    
    /// Input pricing per 1M tokens in USD
    pub input_price: Option<f64>,
    
    /// Output pricing per 1M tokens in USD
    pub output_price: Option<f64>,
    
    /// Whether the model supports vision/image inputs
    pub supports_vision: Option<bool>,
    
    /// Whether the model supports function/tool calling
    pub supports_function_calling: Option<bool>,
    
    /// Whether the model requires explicit max_tokens parameter
    pub require_max_tokens: Option<bool>,
    
    /// Whether the model supports thinking/reasoning modes
    pub supports_thinking: Option<bool>,
    
    /// Optimal thinking budget for reasoning tasks
    pub optimal_thinking_budget: Option<u32>,
}

/// Model capability flags for filtering and selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCapabilities {
    /// Model supports vision/image processing
    pub vision: bool,
    
    /// Model supports function/tool calling
    pub function_calling: bool,
    
    /// Model supports advanced reasoning/thinking
    pub thinking: bool,
    
    /// Model supports audio input/output
    pub audio: bool,
    
    /// Model supports streaming responses
    pub streaming: bool,
    
    /// Model supports system prompts
    pub system_prompts: bool,
}

/// Model performance characteristics
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelPerformance {
    /// Context window size in tokens
    pub context_length: u32,
    
    /// Maximum output tokens per request
    pub max_output: u32,
    
    /// Relative speed compared to baseline (1.0 = baseline)
    pub relative_speed: f32,
    
    /// Relative quality compared to baseline (1.0 = baseline)
    pub relative_quality: f32,
    
    /// Cost per 1M input tokens in USD
    pub input_cost: f32,
    
    /// Cost per 1M output tokens in USD
    pub output_cost: f32,
}

impl ModelInfoData {
    /// Extract capability flags from model info
    /// 
    /// # Returns
    /// * ModelCapabilities struct with boolean flags for each capability
    pub fn get_capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            vision: self.supports_vision.unwrap_or(false),
            function_calling: self.supports_function_calling.unwrap_or(false),
            thinking: self.supports_thinking.unwrap_or(false),
            audio: false, // Not yet supported by any models in the dataset
            streaming: true, // Most models support streaming
            system_prompts: true, // Most models support system prompts
        }
    }
    
    /// Extract performance characteristics from model info
    /// 
    /// # Returns
    /// * ModelPerformance struct with performance metrics
    pub fn get_performance(&self) -> ModelPerformance {
        ModelPerformance {
            context_length: self.max_input_tokens.unwrap_or(128000) as u32,
            max_output: self.max_output_tokens.unwrap_or(4096) as u32,
            relative_speed: 1.0, // Default baseline speed
            relative_quality: 1.0, // Default baseline quality
            input_cost: self.input_price.unwrap_or(0.0) as f32,
            output_cost: self.output_price.unwrap_or(0.0) as f32,
        }
    }
    
    /// Check if model supports a specific capability
    /// 
    /// # Arguments
    /// * `capability` - The capability to check for
    /// 
    /// # Returns
    /// * `true` if the model supports the capability, `false` otherwise
    pub fn supports_capability(&self, capability: Capability) -> bool {
        match capability {
            Capability::Vision => self.supports_vision.unwrap_or(false),
            Capability::FunctionCalling => self.supports_function_calling.unwrap_or(false),
            Capability::Thinking => self.supports_thinking.unwrap_or(false),
            Capability::LargeContext => self.max_input_tokens.unwrap_or(0) >= 100000,
            Capability::HighOutput => self.max_output_tokens.unwrap_or(0) >= 8192,
            Capability::LowCost => {
                let input_cost = self.input_price.unwrap_or(f64::INFINITY);
                let output_cost = self.output_price.unwrap_or(f64::INFINITY);
                input_cost <= 1.0 && output_cost <= 3.0 // Thresholds for "low cost"
            }
        }
    }
    
    /// Calculate total cost for a given token usage
    /// 
    /// # Arguments
    /// * `input_tokens` - Number of input tokens
    /// * `output_tokens` - Number of output tokens
    /// 
    /// # Returns
    /// * Total cost in USD, or None if pricing is not available
    pub fn calculate_cost(&self, input_tokens: u64, output_tokens: u64) -> Option<f64> {
        let input_price = self.input_price?;
        let output_price = self.output_price?;
        
        let input_cost = (input_tokens as f64 / 1_000_000.0) * input_price;
        let output_cost = (output_tokens as f64 / 1_000_000.0) * output_price;
        
        Some(input_cost + output_cost)
    }
    
    /// Check if model is suitable for a given use case
    /// 
    /// # Arguments
    /// * `use_case` - The use case to evaluate suitability for
    /// 
    /// # Returns
    /// * Suitability score from 0.0 to 1.0, or None if not suitable
    pub fn suitability_score(&self, use_case: UseCase) -> Option<f32> {
        match use_case {
            UseCase::CodeGeneration => {
                if self.supports_function_calling.unwrap_or(false) {
                    Some(0.9)
                } else {
                    Some(0.6)
                }
            }
            UseCase::ImageAnalysis => {
                if self.supports_vision.unwrap_or(false) {
                    Some(0.95)
                } else {
                    None
                }
            }
            UseCase::Reasoning => {
                if self.supports_thinking.unwrap_or(false) {
                    Some(0.95)
                } else {
                    Some(0.7)
                }
            }
            UseCase::LongContext => {
                let context_length = self.max_input_tokens.unwrap_or(0);
                if context_length >= 1_000_000 {
                    Some(1.0)
                } else if context_length >= 200_000 {
                    Some(0.8)
                } else if context_length >= 100_000 {
                    Some(0.6)
                } else {
                    Some(0.3)
                }
            }
            UseCase::BudgetSensitive => {
                let input_cost = self.input_price.unwrap_or(f64::INFINITY);
                let output_cost = self.output_price.unwrap_or(f64::INFINITY);
                
                if input_cost <= 0.5 && output_cost <= 1.5 {
                    Some(1.0)
                } else if input_cost <= 1.0 && output_cost <= 3.0 {
                    Some(0.7)
                } else if input_cost <= 2.0 && output_cost <= 6.0 {
                    Some(0.4)
                } else {
                    Some(0.1)
                }
            }
        }
    }
}

/// Specific capabilities that models can support
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    /// Vision/image processing capability
    Vision,
    
    /// Function/tool calling capability
    FunctionCalling,
    
    /// Advanced reasoning/thinking capability
    Thinking,
    
    /// Large context window (>100k tokens)
    LargeContext,
    
    /// High output token limit (>8k tokens)
    HighOutput,
    
    /// Low cost model
    LowCost,
}

/// Common use cases for model selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UseCase {
    /// Code generation and programming tasks
    CodeGeneration,
    
    /// Image analysis and vision tasks
    ImageAnalysis,
    
    /// Complex reasoning and problem solving
    Reasoning,
    
    /// Long document processing
    LongContext,
    
    /// Cost-sensitive applications
    BudgetSensitive,
}

/// Filter models by required capabilities
/// 
/// # Arguments
/// * `models` - Iterator of model info data
/// * `required_capabilities` - Slice of required capabilities
/// 
/// # Returns
/// * Vector of models that support all required capabilities
pub fn filter_models_by_capabilities<'a, I>(
    models: I,
    required_capabilities: &[Capability],
) -> Vec<&'a ModelInfoData>
where
    I: Iterator<Item = &'a ModelInfoData>,
{
    models
        .filter(|model| {
            required_capabilities
                .iter()
                .all(|&cap| model.supports_capability(cap))
        })
        .collect()
}

/// Rank models by suitability for a use case
/// 
/// # Arguments
/// * `models` - Iterator of model info data
/// * `use_case` - The use case to rank models for
/// 
/// # Returns
/// * Vector of (model, score) pairs sorted by suitability score (descending)
pub fn rank_models_by_use_case<'a, I>(
    models: I,
    use_case: UseCase,
) -> Vec<(&'a ModelInfoData, f32)>
where
    I: Iterator<Item = &'a ModelInfoData>,
{
    let mut ranked: Vec<_> = models
        .filter_map(|model| {
            model.suitability_score(use_case).map(|score| (model, score))
        })
        .collect();
    
    // Sort by score descending
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    ranked
}

/// Find the most cost-effective model for given requirements
/// 
/// # Arguments
/// * `models` - Iterator of model info data
/// * `required_capabilities` - Required capabilities
/// * `expected_input_tokens` - Expected input tokens per request
/// * `expected_output_tokens` - Expected output tokens per request
/// 
/// # Returns
/// * Option containing the most cost-effective model and its cost per request
pub fn find_most_cost_effective<'a, I>(
    models: I,
    required_capabilities: &[Capability],
    expected_input_tokens: u64,
    expected_output_tokens: u64,
) -> Option<(&'a ModelInfoData, f64)>
where
    I: Iterator<Item = &'a ModelInfoData>,
{
    let suitable_models = filter_models_by_capabilities(models, required_capabilities);
    
    suitable_models
        .into_iter()
        .filter_map(|model| {
            model
                .calculate_cost(expected_input_tokens, expected_output_tokens)
                .map(|cost| (model, cost))
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}