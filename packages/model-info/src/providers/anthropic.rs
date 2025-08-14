use std::sync::OnceLock;

use fluent_ai_async::AsyncStream;
use hashbrown::HashMap;

use crate::common::{ModelInfo, ProviderTrait};

// Type alias for complex provider data tuple to improve readability
type ProviderModelData = (u32, u32, f64, f64, bool, bool, bool, bool);

#[derive(Clone, Debug, PartialEq)]
pub struct AnthropicProvider;

impl ProviderTrait for AnthropicProvider {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        let model_name = model.to_string();

        AsyncStream::with_channel(move |sender| {
            let model_info = adapt_anthropic_to_model_info(&model_name);
            let _ = sender.send(model_info);
        })
    }

    fn list_models(&self) -> AsyncStream<ModelInfo> {
        // Anthropic doesn't have a public models API, so we use hardcoded models
        AsyncStream::with_channel(move |sender| {
            let models = vec![
                "claude-3-5-sonnet-20240620",
                "claude-3-haiku-20240307",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
            ];

            for model in models {
                let model_info = adapt_anthropic_to_model_info(model);
                let _ = sender.send(model_info);
            }
        })
    }

    fn provider_name(&self) -> &'static str {
        "anthropic"
    }
}

fn adapt_anthropic_to_model_info(model: &str) -> ModelInfo {
    static MAP: OnceLock<HashMap<&'static str, ProviderModelData>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        // (max_input, max_output, input_price, output_price, vision, function_calling, embeddings, thinking)
        m.insert(
            "claude-3-5-sonnet-20240620",
            (200000, 50000, 3.0, 15.0, true, true, false, false),
        );
        m.insert(
            "claude-3-haiku-20240307",
            (200000, 50000, 0.25, 1.25, true, true, false, false),
        );
        m.insert(
            "claude-3-opus-20240229",
            (200000, 50000, 15.0, 75.0, true, true, false, false),
        );
        m.insert(
            "claude-3-sonnet-20240229",
            (200000, 50000, 3.0, 15.0, true, true, false, false),
        );
        m
    });

    let (
        max_input,
        max_output,
        pricing_input,
        pricing_output,
        supports_vision,
        supports_function_calling,
        supports_embeddings,
        supports_thinking,
    ) = map
        .get(model)
        .copied()
        .unwrap_or((200000, 50000, 0.0, 0.0, true, true, false, false));

    ModelInfo {
        // Core identification
        provider_name: "anthropic",
        name: Box::leak(model.to_string().into_boxed_str()),

        // Token limits
        max_input_tokens: std::num::NonZeroU32::new(max_input),
        max_output_tokens: std::num::NonZeroU32::new(max_output),

        // Pricing (per 1M tokens)
        input_price: Some(pricing_input),
        output_price: Some(pricing_output),

        // Capability flags
        supports_vision,
        supports_function_calling,
        supports_embeddings,
        requires_max_tokens: false,
        supports_thinking,

        // Advanced features
        optimal_thinking_budget: if supports_thinking { Some(75000) } else { None },
        system_prompt_prefix: None,
        real_name: None,
        model_type: None,
        patch: None,
        required_temperature: None,
    }
}
