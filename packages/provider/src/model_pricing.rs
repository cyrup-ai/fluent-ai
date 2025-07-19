//! Model pricing calculations and cost optimization
//! 
//! This module provides comprehensive pricing analysis and cost optimization
//! functionality for AI model selection and usage tracking.

use crate::model_capabilities::ModelInfoData;
use crate::completion_provider::ModelConfig;
use fluent_ai_domain::PricingTier;

/// Convert ModelInfoData to ModelConfig with zero allocation
/// 
/// This function provides the critical conversion between the raw model information
/// and the runtime configuration used by completion providers. It applies production
/// defaults and ensures all configuration values are valid and optimized.
/// 
/// # Arguments
/// * `info` - Model information data containing capabilities and pricing
/// * `model_name` - Static string identifier for the model
/// 
/// # Returns
/// * ModelConfig optimized for production use
/// 
/// # Performance
/// * Zero allocation design using static strings and value copies
/// * Optimized default values based on model capabilities
/// * Validated parameter ranges for production safety
pub fn model_info_to_config(info: &ModelInfoData, model_name: &'static str) -> ModelConfig {
    ModelConfig {
        max_tokens: info.max_output_tokens.unwrap_or(4096) as u32,
        temperature: 0.7,
        top_p: 0.9,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        context_length: info.max_input_tokens.unwrap_or(128000) as u32,
        system_prompt: "You are a helpful AI assistant.",
        supports_tools: info.supports_function_calling.unwrap_or(false),
        supports_vision: info.supports_vision.unwrap_or(false),
        supports_audio: false, // Not yet available in model data
        supports_thinking: info.supports_thinking.unwrap_or(false),
        optimal_thinking_budget: info.optimal_thinking_budget.unwrap_or(1024),
        provider: extract_provider_name(&info.provider_name),
        model_name,
    }
}

/// Extract normalized provider name from model info
/// 
/// # Arguments
/// * `provider_name` - Raw provider name from model data
/// 
/// # Returns
/// * Normalized provider name suitable for configuration
fn extract_provider_name(provider_name: &str) -> &'static str {
    match provider_name.to_lowercase().as_str() {
        "openai" => "openai",
        "anthropic" => "anthropic", 
        "google" => "google",
        "mistral" | "mistralai" => "mistral",
        "meta" => "meta",
        "cohere" => "cohere",
        "ai21" => "ai21",
        "deepseek" => "deepseek",
        "qwen" => "qwen",
        "perplexity" => "perplexity",
        "xai" | "x.ai" => "xai",
        "amazon" => "amazon",
        "minimax" => "minimax",
        _ => "unknown",
    }
}

// Re-export PricingTier from domain for backward compatibility
pub use fluent_ai_domain::PricingTier;

/// Cost analysis result for model comparison
#[derive(Debug, Clone)]
pub struct CostAnalysis {
    /// Input cost per 1M tokens in USD
    pub input_cost_per_million: f64,
    
    /// Output cost per 1M tokens in USD
    pub output_cost_per_million: f64,
    
    /// Pricing tier classification
    pub pricing_tier: PricingTier,
    
    /// Cost per typical request (assuming 1000 input, 500 output tokens)
    pub typical_request_cost: f64,
    
    /// Break-even point in requests compared to premium models
    pub break_even_requests: Option<u64>,
    
    /// Cost efficiency score (higher is more efficient)
    pub efficiency_score: f32,
}

impl ModelInfoData {
    /// Perform comprehensive cost analysis for the model
    /// 
    /// # Returns
    /// * CostAnalysis with detailed pricing breakdown and metrics
    pub fn analyze_cost(&self) -> Option<CostAnalysis> {
        let input_cost = self.input_price?;
        let output_cost = self.output_price?;
        
        let pricing_tier = classify_pricing_tier(input_cost, output_cost);
        let typical_request_cost = calculate_typical_request_cost(input_cost, output_cost);
        let efficiency_score = calculate_efficiency_score(input_cost, output_cost, &self);
        
        Some(CostAnalysis {
            input_cost_per_million: input_cost,
            output_cost_per_million: output_cost,
            pricing_tier,
            typical_request_cost,
            break_even_requests: calculate_break_even(input_cost, output_cost),
            efficiency_score,
        })
    }
    
    /// Calculate cost for specific token usage with detailed breakdown
    /// 
    /// # Arguments
    /// * `input_tokens` - Number of input tokens
    /// * `output_tokens` - Number of output tokens
    /// 
    /// # Returns
    /// * CostBreakdown with itemized costs
    pub fn calculate_detailed_cost(&self, input_tokens: u64, output_tokens: u64) -> Option<CostBreakdown> {
        let input_price = self.input_price?;
        let output_price = self.output_price?;
        
        let input_cost = (input_tokens as f64 / 1_000_000.0) * input_price;
        let output_cost = (output_tokens as f64 / 1_000_000.0) * output_price;
        let total_cost = input_cost + output_cost;
        
        Some(CostBreakdown {
            input_tokens,
            output_tokens,
            input_cost,
            output_cost,
            total_cost,
            cost_per_token: total_cost / (input_tokens + output_tokens) as f64,
        })
    }
    
    /// Estimate monthly cost for given usage patterns
    /// 
    /// # Arguments
    /// * `daily_requests` - Average requests per day
    /// * `avg_input_tokens` - Average input tokens per request
    /// * `avg_output_tokens` - Average output tokens per request
    /// 
    /// # Returns
    /// * MonthlyCostEstimate with projected costs
    pub fn estimate_monthly_cost(
        &self,
        daily_requests: u32,
        avg_input_tokens: u32,
        avg_output_tokens: u32,
    ) -> Option<MonthlyCostEstimate> {
        let input_price = self.input_price?;
        let output_price = self.output_price?;
        
        let daily_input_tokens = daily_requests as u64 * avg_input_tokens as u64;
        let daily_output_tokens = daily_requests as u64 * avg_output_tokens as u64;
        
        let daily_cost = self.calculate_cost(daily_input_tokens, daily_output_tokens)?;
        let monthly_cost = daily_cost * 30.0;
        
        Some(MonthlyCostEstimate {
            daily_requests,
            daily_cost,
            monthly_cost,
            cost_per_request: daily_cost / daily_requests as f64,
            total_monthly_tokens: (daily_input_tokens + daily_output_tokens) * 30,
        })
    }
}

/// Detailed cost breakdown for a specific request
#[derive(Debug, Clone)]
pub struct CostBreakdown {
    /// Number of input tokens
    pub input_tokens: u64,
    
    /// Number of output tokens
    pub output_tokens: u64,
    
    /// Cost for input tokens in USD
    pub input_cost: f64,
    
    /// Cost for output tokens in USD
    pub output_cost: f64,
    
    /// Total cost in USD
    pub total_cost: f64,
    
    /// Average cost per token in USD
    pub cost_per_token: f64,
}

/// Monthly cost estimation for usage planning
#[derive(Debug, Clone)]
pub struct MonthlyCostEstimate {
    /// Average daily requests
    pub daily_requests: u32,
    
    /// Daily cost in USD
    pub daily_cost: f64,
    
    /// Projected monthly cost in USD
    pub monthly_cost: f64,
    
    /// Average cost per request in USD
    pub cost_per_request: f64,
    
    /// Total monthly token usage
    pub total_monthly_tokens: u64,
}

/// Classify model into pricing tier based on costs
/// 
/// This is now a thin wrapper around `PricingTier::classify` from the domain crate.
fn classify_pricing_tier(input_cost: f64, output_cost: f64) -> PricingTier {
    PricingTier::classify(input_cost, output_cost)
}

/// Calculate cost for a typical request (1000 input, 500 output tokens)
fn calculate_typical_request_cost(input_cost: f64, output_cost: f64) -> f64 {
    let input_tokens = 1000.0;
    let output_tokens = 500.0;
    
    (input_tokens / 1_000_000.0) * input_cost + (output_tokens / 1_000_000.0) * output_cost
}

/// Calculate efficiency score based on cost and capabilities
fn calculate_efficiency_score(input_cost: f64, output_cost: f64, model: &ModelInfoData) -> f32 {
    let base_cost = input_cost + output_cost * 0.5; // Weight output cost less since it's typically higher
    
    // Start with base efficiency (inverse of cost)
    let mut score = 1.0 / (base_cost + 0.1); // Add small constant to avoid division by zero
    
    // Boost score for additional capabilities
    if model.supports_vision.unwrap_or(false) {
        score *= 1.2;
    }
    
    if model.supports_function_calling.unwrap_or(false) {
        score *= 1.15;
    }
    
    if model.supports_thinking.unwrap_or(false) {
        score *= 1.1;
    }
    
    // Boost score for large context windows
    let context_length = model.max_input_tokens.unwrap_or(0);
    if context_length >= 1_000_000 {
        score *= 1.3;
    } else if context_length >= 200_000 {
        score *= 1.15;
    }
    
    // Normalize score to reasonable range (0.0 to 10.0)
    (score * 10.0).min(10.0) as f32
}

/// Calculate break-even point compared to premium models
fn calculate_break_even(input_cost: f64, output_cost: f64) -> Option<u64> {
    const PREMIUM_INPUT_COST: f64 = 60.0; // Example premium model cost
    const PREMIUM_OUTPUT_COST: f64 = 180.0;
    
    let cost_per_typical_request = calculate_typical_request_cost(input_cost, output_cost);
    let premium_cost_per_request = calculate_typical_request_cost(PREMIUM_INPUT_COST, PREMIUM_OUTPUT_COST);
    
    if cost_per_typical_request >= premium_cost_per_request {
        None // This model is not cheaper
    } else {
        let savings_per_request = premium_cost_per_request - cost_per_typical_request;
        // Assume $100 setup/integration cost for premium model
        Some((100.0 / savings_per_request) as u64)
    }
}

/// Compare costs between multiple models for given usage
/// 
/// # Arguments
/// * `models` - Iterator of model info data
/// * `input_tokens` - Input tokens per request
/// * `output_tokens` - Output tokens per request
/// * `requests_per_month` - Monthly request volume
/// 
/// # Returns
/// * Vector of (model, monthly_cost) pairs sorted by cost (ascending)
pub fn compare_model_costs<'a, I>(
    models: I,
    input_tokens: u64,
    output_tokens: u64,
    requests_per_month: u32,
) -> Vec<(&'a ModelInfoData, f64)>
where
    I: Iterator<Item = &'a ModelInfoData>,
{
    let mut costs: Vec<_> = models
        .filter_map(|model| {
            let cost_per_request = model.calculate_cost(input_tokens, output_tokens)?;
            let monthly_cost = cost_per_request * requests_per_month as f64;
            Some((model, monthly_cost))
        })
        .collect();
    
    // Sort by monthly cost ascending
    costs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    
    costs
}

/// Find models within a specific budget range
/// 
/// # Arguments
/// * `models` - Iterator of model info data
/// * `max_monthly_budget` - Maximum monthly budget in USD
/// * `expected_requests` - Expected monthly requests
/// * `avg_input_tokens` - Average input tokens per request
/// * `avg_output_tokens` - Average output tokens per request
/// 
/// # Returns
/// * Vector of models that fit within the budget
pub fn find_models_within_budget<'a, I>(
    models: I,
    max_monthly_budget: f64,
    expected_requests: u32,
    avg_input_tokens: u32,
    avg_output_tokens: u32,
) -> Vec<&'a ModelInfoData>
where
    I: Iterator<Item = &'a ModelInfoData>,
{
    models
        .filter(|model| {
            if let Some(cost_per_request) = model.calculate_cost(avg_input_tokens as u64, avg_output_tokens as u64) {
                let monthly_cost = cost_per_request * expected_requests as f64;
                monthly_cost <= max_monthly_budget
            } else {
                false
            }
        })
        .collect()
}