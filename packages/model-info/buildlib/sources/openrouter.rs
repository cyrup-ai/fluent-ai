use super::ModelSpec;
use fluent_ai_http3::{Http3, HttpStreamExt};
use anyhow::{Context, Result};
use serde::Deserialize;

/// OpenRouter.ai API response structures
#[derive(Debug, Deserialize, Default)]
struct OpenRouterResponse {
    data: Vec<OpenRouterModel>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterModel {
    id: String,
    name: String,
    created: u64,
    description: Option<String>,
    context_length: Option<u32>,
    architecture: Option<OpenRouterArchitecture>,
    pricing: Option<OpenRouterPricing>,
    top_provider: Option<OpenRouterTopProvider>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterArchitecture {
    modality: Option<String>,
    tokenizer: Option<String>,
    instruct_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterPricing {
    prompt: String,    // Price per 1M tokens as string
    completion: String, // Price per 1M tokens as string
}

#[derive(Debug, Deserialize)]
struct OpenRouterTopProvider {
    context_length: Option<u32>,
    max_completion_tokens: Option<u32>,
    is_moderated: Option<bool>,
}

/* TODO: Update to use Http3 when needed
/// Fetch comprehensive model specifications from OpenRouter.ai
/// This is our primary data source as it has the most complete database
pub async fn fetch_model_spec(
    http_client: &Http3,
    model_id: &str,
    provider: &str,
) -> Result<ModelSpec> {
    // OpenRouter.ai models endpoint
    const OPENROUTER_API_URL: &str = "https://openrouter.ai/api/v1/models";
    
    let response: OpenRouterResponse = http_client
        .fetch_json_public(OPENROUTER_API_URL)
        .await
        .context("Failed to fetch models from OpenRouter.ai API")?;
    
    // Find the specific model we're looking for
    let openrouter_model = response
        .data
        .into_iter()
        .find(|model| {
            // Try exact match first
            if model.id == model_id {
                return true;
            }
            
            // Try provider-prefixed match (e.g., "openai/gpt-4o")
            if model.id == format!("{}/{}", provider, model_id) {
                return true;
            }
            
            // Try reverse match (e.g., model_id might be "openai/gpt-4o")
            if model_id.contains('/') && model.id.contains(&model_id.split('/').last().unwrap_or("")) {
                return true;
            }
            
            false
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Model {} for provider {} not found in OpenRouter.ai database",
                model_id,
                provider
            )
        })?;
    
    // Convert OpenRouter data to our ModelSpec format
    let mut spec = ModelSpec {
        id: model_id.to_string(),
        provider: provider.to_string(),
        description: openrouter_model.description,
        context_window: openrouter_model.context_length,
        ..Default::default()
    };
    
    // Extract pricing information
    if let Some(pricing) = openrouter_model.pricing {
        spec.pricing_input = parse_price_string(&pricing.prompt);
        spec.pricing_output = parse_price_string(&pricing.completion);
    }
    
    // Extract token limits from various sources
    if let Some(context_length) = openrouter_model.context_length {
        // Apply 75/25 split for input/output tokens as reasonable default
        spec.max_input_tokens = Some((context_length as f32 * 0.75) as u32);
        spec.max_output_tokens = Some((context_length as f32 * 0.25) as u32);
    }
    
    // Override with more specific token limits if available
    if let Some(top_provider) = openrouter_model.top_provider {
        if let Some(max_completion) = top_provider.max_completion_tokens {
            spec.max_output_tokens = Some(max_completion);
            // Recalculate input tokens
            if let Some(total_context) = spec.context_window {
                spec.max_input_tokens = Some(total_context.saturating_sub(max_completion));
            }
        }
        
        if let Some(context_length) = top_provider.context_length {
            spec.context_window = Some(context_length);
        }
    }
    
    // Infer capabilities from model architecture and name
    if let Some(arch) = openrouter_model.architecture {
        if let Some(modality) = arch.modality {
            spec.supports_vision = modality.contains("image") || modality.contains("vision");
        }
    }
    
    // Infer capabilities from model name patterns
    spec.supports_function_calling = infer_function_calling_support(&spec.id, &spec.provider);
    spec.supports_thinking = infer_thinking_support(&spec.id, &spec.provider);
    spec.supports_embeddings = infer_embeddings_support(&spec.id, &spec.provider);
    
    // Set thinking-specific parameters
    if spec.supports_thinking {
        spec.required_temperature = Some(1.0);
        spec.optimal_thinking_budget = Some(100000);
    }
    
    // Add timestamp for freshness tracking
    spec.last_updated = Some(chrono::Utc::now().to_rfc3339());
    
    Ok(spec)
}

/// Parse OpenRouter price string (e.g., "0.000005" -> 5.0 per 1M tokens)
fn parse_price_string(price_str: &str) -> Option<f64> {
    price_str
        .parse::<f64>()
        .ok()
        .map(|price_per_token| price_per_token * 1_000_000.0) // Convert to price per 1M tokens
}

/// Infer function calling support based on model patterns
fn infer_function_calling_support(model_id: &str, provider: &str) -> bool {
    match provider {
        "openai" => {
            !model_id.contains("embedding") && 
            !model_id.starts_with("dall-e") &&
            !model_id.starts_with("whisper")
        }
        "anthropic" => true, // All Claude models support function calling
        "mistral" => !model_id.contains("embed") && !model_id.contains("codestral"),
        "xai" => true, // All Grok models support function calling
        "together" => model_id.contains("mixtral") || model_id.contains("llama"),
        "huggingface" => false, // Most HF models don't support function calling
        _ => false, // Conservative default
    }
}

/// Infer thinking/reasoning support based on model patterns
fn infer_thinking_support(model_id: &str, provider: &str) -> bool {
    match provider {
        "openai" => model_id.starts_with('o') && (model_id.contains('1') || model_id.contains('3')),
        "xai" => !model_id.contains("mini"), // Grok models except mini support thinking
        _ => false,
    }
}

/// Infer embeddings support based on model patterns
fn infer_embeddings_support(model_id: &str, _provider: &str) -> bool {
    model_id.contains("embed") || model_id.contains("embedding")
}
*/