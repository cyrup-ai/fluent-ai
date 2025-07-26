pub trait Model {
    fn name(&self) -> &'static str;
    fn max_context_length(&self) -> u64;
    fn pricing_input(&self) -> f64;
    fn pricing_output(&self) -> f64;
    fn is_thinking(&self) -> bool;
    fn required_temperature(&self) -> Option<f64>;
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub max_context: u64,
    pub pricing_input: f64,
    pub pricing_output: f64,
    pub is_thinking: bool,
    pub required_temperature: Option<f64>,
}

pub trait ProviderTrait {
    async fn get_model_info(&self, model: &str) -> anyhow::Result<ModelInfo>;
}