use super::ModelSpec;
use fluent_ai_http3::{Http3, HttpStreamExt};
use anyhow::{Context, Result};
use serde::Deserialize;

/// LiteLLM model database structures
#[derive(Debug, Deserialize, Default)]
struct LiteLLMModelList {
    data: Vec<LiteLLMModel>,
}

#[derive(Debug, Deserialize)]
struct LiteLLMModel {
    #[serde(rename = "model_name")]
    model_name: String,
    #[serde(rename = "litellm_provider")]
    litellm_provider: String,
    #[serde(rename = "max_tokens")]
    max_tokens: Option<u32>,
    #[serde(rename = "max_input_tokens")]
    max_input_tokens: Option<u32>,
    #[serde(rename = "max_output_tokens")]
    max_output_tokens: Option<u32>,
    #[serde(rename = "input_cost_per_token")]
    input_cost_per_token: Option<f64>,
    #[serde(rename = "output_cost_per_token")]
    output_cost_per_token: Option<f64>,
    #[serde(rename = "supports_function_calling")]
    supports_function_calling: Option<bool>,
    #[serde(rename = "supports_vision")]
    supports_vision: Option<bool>,
    mode: Option<Vec<String>>,
}

/* TODO: Update to use Http3 when needed
/// Fetch model specifications from LiteLLM's model database
/// This is our secondary data source with community-maintained specifications
pub async fn fetch_model_spec(
    http_client: &Http3,
    model_id: &str,
    provider: &str,
) -> Result<ModelSpec> {
    // LiteLLM model database endpoint
    const LITELLM_API_URL: &str = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";
    
    let response: LiteLLMModelList = http_client
        .fetch_json_public(LITELLM_API_URL)
        .await
        .context("Failed to fetch models from LiteLLM database")?;
    
    // Find the specific model we're looking for
    let litellm_model = response
        .data
        .into_iter()
        .find(|model| {
            // Try exact match first
            if model.model_name == model_id {
                return true;
            }
            
            // Try provider match
            if model.litellm_provider.to_lowercase() == provider.to_lowercase() && 
               model.model_name.contains(model_id) {
                return true;
            }
            
            // Try partial match with provider prefix
            if model.model_name == format!("{}/{}", provider, model_id) {
                return true;
            }
            
            // Try removing common prefixes/suffixes
            let normalized_model_name = normalize_model_name(&model.model_name);
            let normalized_model_id = normalize_model_name(model_id);
            if normalized_model_name == normalized_model_id {
                return true;
            }
            
            false
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Model {} for provider {} not found in LiteLLM database",
                model_id,
                provider
            )
        })?;
    
    // Convert LiteLLM data to our ModelSpec format
    let mut spec = ModelSpec {
        id: model_id.to_string(),
        provider: provider.to_string(),
        ..Default::default()
    };
    
    // Extract token limits
    if let Some(max_tokens) = litellm_model.max_tokens {
        // If specific input/output limits are provided, use them
        if let Some(max_input) = litellm_model.max_input_tokens {
            spec.max_input_tokens = Some(max_input);
        } else {
            // Apply 75/25 split for input/output tokens
            spec.max_input_tokens = Some((max_tokens as f32 * 0.75) as u32);
        }
        
        if let Some(max_output) = litellm_model.max_output_tokens {
            spec.max_output_tokens = Some(max_output);
        } else {
            spec.max_output_tokens = Some((max_tokens as f32 * 0.25) as u32);
        }
        
        spec.context_window = Some(max_tokens);
    } else {
        // Use individual limits if provided
        spec.max_input_tokens = litellm_model.max_input_tokens;
        spec.max_output_tokens = litellm_model.max_output_tokens;
    }
    
    // Extract pricing (convert from per-token to per-1M-tokens)
    if let Some(input_cost) = litellm_model.input_cost_per_token {
        spec.pricing_input = Some(input_cost * 1_000_000.0);
    }
    
    if let Some(output_cost) = litellm_model.output_cost_per_token {
        spec.pricing_output = Some(output_cost * 1_000_000.0);
    }
    
    // Extract capabilities
    spec.supports_function_calling = litellm_model.supports_function_calling.unwrap_or(false);
    spec.supports_vision = litellm_model.supports_vision.unwrap_or(false);
    
    // Infer additional capabilities from mode field
    if let Some(modes) = litellm_model.mode {
        spec.supports_embeddings = modes.iter().any(|mode| mode.contains("embedding"));
    }
    
    // Infer thinking support from model patterns
    spec.supports_thinking = infer_thinking_support(&spec.id, &spec.provider);
    if spec.supports_thinking {
        spec.required_temperature = Some(1.0);
        spec.optimal_thinking_budget = Some(100000);
    }
    
    // Add timestamp for freshness tracking
    spec.last_updated = Some(chrono::Utc::now().to_rfc3339());
    
    Ok(spec)
}

/// Normalize model names for better matching
fn normalize_model_name(name: &str) -> String {
    name.to_lowercase()
        .replace("--", "-")
        .replace("_", "-")
        .replace(".", "-")
        .trim_start_matches("openai/")
        .trim_start_matches("anthropic/")
        .trim_start_matches("mistral/")
        .trim_start_matches("huggingface/")
        .trim_start_matches("together/")
        .trim_start_matches("xai/")
        .to_string()
}

/// Infer thinking/reasoning support from model patterns
fn infer_thinking_support(model_id: &str, provider: &str) -> bool {
    match provider {
        "openai" => {
            model_id.starts_with('o') && 
            (model_id.contains('1') || model_id.contains('3')) &&
            !model_id.contains("embedding")
        }
        "xai" => {
            model_id.contains("grok") && !model_id.contains("mini")
        }
        _ => false,
    }
}
*/