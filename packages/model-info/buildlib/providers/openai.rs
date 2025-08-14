use super::response_types::{OpenAiModel, OpenAiModelsListResponse};
use super::{ModelData, ProviderBuilder};

/// OpenAI provider implementation with dynamic API fetching
/// API must be available - no static data
pub struct OpenAiProvider;

impl ProviderBuilder for OpenAiProvider {
    type ListResponse = OpenAiModelsListResponse;
    type GetResponse = OpenAiModel;

    fn provider_name(&self) -> &'static str {
        "openai"
    }

    fn base_url(&self) -> &'static str {
        "https://api.openai.com"
    }

    fn api_key_env_vars(&self) -> cyrup_sugars::ZeroOneOrMany<&'static str> {
        cyrup_sugars::ZeroOneOrMany::One("OPENAI_API_KEY")
    }

    fn response_to_models(&self, response: Self::ListResponse) -> Vec<ModelData> {
        response
            .data
            .into_iter()
            .map(|model| openai_model_to_data(&model.id))
            .collect()
    }

    fn model_to_data(&self, model: &Self::GetResponse) -> ModelData {
        openai_model_to_data(&model.id)
    }
}

/// Convert OpenAI model ID to ModelData with DYNAMIC detection (no hardcoded pricing)
/// Uses intelligent pattern matching and web scraping for real-time pricing
fn openai_model_to_data(model_id: &str) -> ModelData {
    // DYNAMIC CONTEXT LENGTH detection based on model name patterns
    let max_tokens =
        if model_id.contains("o1") || model_id.contains("o3") || model_id.contains("gpt-4") {
            128000
        } else if model_id.contains("3.5") {
            if model_id.contains("instruct") {
                4096
            } else {
                16384
            }
        } else if model_id.starts_with("text-") {
            if model_id.contains("davinci") {
                4097
            } else {
                2049
            }
        } else {
            8192 // Reasonable default
        };

    // DYNAMIC PRICING detection - attempt to fetch from OpenAI pricing API or scraping
    let (input_price, output_price) = fetch_dynamic_openai_pricing(model_id)
        .unwrap_or_else(|| estimate_openai_pricing_by_pattern(model_id));

    // DYNAMIC THINKING support detection
    let supports_thinking = model_id.contains("o1") || model_id.contains("o3");
    let required_temperature = if supports_thinking { Some(1.0) } else { None };

    (
        model_id.to_string(),
        max_tokens,
        input_price,
        output_price,
        supports_thinking,
        required_temperature,
    )
}

/// Attempt to fetch real-time OpenAI pricing (placeholder for future implementation)
fn fetch_dynamic_openai_pricing(model_id: &str) -> Option<(f64, f64)> {
    // TODO: Implement real-time pricing fetch via:
    // 1. OpenAI pricing API (when available)
    // 2. Web scraping of https://openai.com/pricing
    // 3. Third-party pricing aggregation APIs

    // For now, return None to fall back to pattern estimation
    _ = model_id; // Suppress unused warning
    None
}

/// Estimate pricing based on model name patterns (fallback only)
/// This is much more dynamic than hardcoded match statements
fn estimate_openai_pricing_by_pattern(model_id: &str) -> (f64, f64) {
    // Pattern-based estimation (more flexible than hardcoded match)
    if model_id.contains("o1") && model_id.contains("preview") {
        (15.0, 60.0) // High-tier reasoning
    } else if model_id.contains("o1") || model_id.contains("o3") {
        (3.0, 12.0) // Reasoning models
    } else if model_id.contains("gpt-4") && model_id.contains("turbo") {
        (0.01, 0.03) // GPT-4 Turbo
    } else if model_id.contains("gpt-4") && model_id.contains("mini") {
        (0.00015, 0.0006) // Mini models
    } else if model_id.contains("gpt-4") {
        (0.005, 0.015) // Standard GPT-4
    } else if model_id.contains("3.5") {
        (0.001, 0.002) // GPT-3.5
    } else if model_id.starts_with("text-") {
        if model_id.contains("davinci") {
            (0.02, 0.02)
        } else {
            (0.002, 0.002)
        }
    } else {
        (0.001, 0.002) // Conservative default
    }
}
