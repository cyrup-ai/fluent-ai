use crate::common::{ModelInfo, ProviderTrait};
use fluent_ai_async::AsyncStream;
use serde::Deserialize;

#[derive(Deserialize, Default)]
pub struct TogetherModelData {
    pub id: String,
    pub context_length: u64,
    pub pricing: TogetherPricingData,
    pub description: String,
}

#[derive(Deserialize, Default)]
pub struct TogetherPricingData {
    pub input: f64,
    pub output: f64,
}

#[derive(Clone)]
pub struct TogetherProvider;

impl ProviderTrait for TogetherProvider {
    fn get_model_info(&self, model: &str) -> AsyncStream<ModelInfo> {
        let model_name = model.to_string();
        
        AsyncStream::with_channel(move |sender| {
            // For now, create dummy data to test the adapt function
            let dummy_data = TogetherModelData {
                id: model_name.clone(),
                context_length: 32768,
                pricing: TogetherPricingData {
                    input: 0.0,
                    output: 0.0,
                },
                description: "Default model".to_string(),
            };
            
            let model_info = adapt_together_to_model_info(&dummy_data);
            let _ = sender.send(model_info);
        })
    }
}

fn adapt_together_to_model_info(data: &TogetherModelData) -> ModelInfo {
    let is_thinking = data.description.to_lowercase().contains("reasoning") || data.id.contains("reasoning");
    let required_temperature = if is_thinking { Some(1.0) } else { None };
    
    ModelInfo {
        name: data.id.clone(),
        max_context: data.context_length,
        pricing_input: data.pricing.input,
        pricing_output: data.pricing.output,
        is_thinking,
        required_temperature,
    }
}